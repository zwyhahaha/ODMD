import numpy as np
from datetime import datetime
import coptpy as cp 
from coptpy import COPT
from scipy.optimize import minimize
from pulp import *
from src.copt_pulp import *
from src.utils import get_reward,get_resource,evaluate_solution

def offline_policy_sqrt(env_dict,T_type='unknown', runtime = False):
    env = cp.Envr()
    model = env.createModel("offline_policy")
    model.setParam('Logging', False)

    B,bmax,fun = env_dict['B'],env_dict['bmax'],env_dict['f']
    T = env_dict['N'] if T_type == 'known' else env_dict['N_max']
    x = model.addMVar(T, lb=0, ub=np.sqrt(bmax),vtype=COPT.CONTINUOUS, nameprefix="x")
    f = np.array([fun[t]['params']['f1'] for t in range(T)])
    if T_type == 'known':
        def loss(x):
            return f@x
    elif T_type == 'unknown':
        def loss(x):
            return f@np.diag(env_dict['Qs'][:T])@x
    start_time = datetime.now()
    model.addQConstr(x@x<=B)
    model.setObjective(loss(x), sense=COPT.MAXIMIZE)
    model.solve()
    if model.status == COPT.OPTIMAL:
        solution=[(x[t].X)**2 for t in range(T)]
        fun = model.objval
    else:
        print("Optimization was stopped with status:", model.status)
        solution=None
        fun=None
    if runtime:
        print('optimization time for N={T}:', datetime.now()-start_time)
    return solution, fun

def offline_policy_linear(env_dict,T_type='unknown', runtime = False):
    env = cp.Envr()
    model = env.createModel("offline_policy")
    model.setParam('Logging', False)

    B,bmax,fun = env_dict['B'],env_dict['bmax'],env_dict['f']
    T = env_dict['N'] if T_type == 'known' else env_dict['N_max']
    x = model.addMVar(T, lb=0, ub=bmax,vtype=COPT.CONTINUOUS, nameprefix="x")
    f = np.array([fun[t]['params']['f1'] for t in range(T)])
    if T_type == 'known':
        def loss(x):
            return f@x
    elif T_type == 'unknown':
        def loss(x):
            return f@np.diag(env_dict['Qs'][:T])@x
    start_time = datetime.now()
    model.addConstr(x@np.ones(T)<=B)
    model.setObjective(loss(x), sense=COPT.MAXIMIZE)
    model.solve()
    if model.status == COPT.OPTIMAL:
        solution=[(x[t].X) for t in range(T)]
        fun = model.objval
    else:
        print("Optimization was stopped with status:", model.status)
        solution=None
        fun=None
    if runtime:
        print('optimization time for N={T}:', datetime.now()-start_time)
    return solution, fun

def offline_policy_sqrt_manual(env_dict,T_type='unknown', runtime = False):
    coeffs = [env_dict['Qs'][i] * env_dict['f'][i]['params']['f1'] for i in range(env_dict['N_max'])]
    B = env_dict['B']
    deno = np.dot(coeffs, coeffs)
    solution = [coeffs[i]**2*B / deno for i in range(len(coeffs))]
    fun = np.dot(coeffs, np.sqrt(solution))
    return solution, fun

def offline_policy_ufunc(env_dict,T_type='unknown', runtime = False):
    B,bmax = env_dict['B'],env_dict['bmax']
    T = env_dict['N'] if T_type == 'known' else env_dict['N_max']
    if T_type == 'known':
        def loss(x):
            res = np.array([get_reward(x[t],env_dict['f'][t]) for t in range(T)])
            return -res.sum()
    elif T_type == 'unknown':
        def loss(x):
            Qs = env_dict['Qs'][:T]
            res = np.array([get_reward(x[t],env_dict['f'][t]) for t in range(T)])
            return -np.dot(Qs, res)
    def constr(x):
        nx = np.array(x)
        return nx.dot(nx)
    cons = (
        {'type': 'ineq', 'fun': lambda x: B - constr(x)},
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: bmax - x}
    )
    x0 = [B/T] * T
    start_time = datetime.now()
    result = minimize(loss, x0, constraints=cons)
    if runtime:
        print('optimization time for N={T}:', datetime.now()-start_time)
    # fun = evaluate_solution(result.x, env_dict)
    return result.x, -result.fun

def offline_policy(env_dict,T_type='unknown', runtime = False):
    if env_dict['f'][0]['type']=='sqrt':
        # return offline_policy_sqrt(env_dict,T_type,runtime)
        return offline_policy_sqrt_manual(env_dict,T_type,runtime)
    elif env_dict['f'][0]['type']=='linear':
        return offline_policy_linear(env_dict,T_type,runtime)
    else:
        return offline_policy_ufunc(env_dict,T_type,runtime)
    
def make_decision(ft, coef, mu, bmax):
    """ 
    primal update: argmax_{0<=xt<=bmax} (coef*ft - mu*xt)
    """
    if mu == 0:
        return bmax 
    else:
        if ft['type'] == 'linear':
            xt = bmax if coef*ft['params']['f1']-mu > 0 else 0
        elif ft['type'] == 'sqrt':
            opt = (coef * ft['params']['f1']) / (2 * mu)
            xt = np.clip(opt, 0, np.sqrt(bmax))
            xt = xt*xt 
        elif ft['type'] == 'log':
            opt = coef / mu - 1
            xt = np.clip(opt, 0, bmax)
        elif ft['type'] == 'power':
            opt= (coef * ft['params']['f1']) / mu 
            xt = np.clip(opt, 0, bmax**(1-ft['params']['f1']))
            xt=xt**(1/(1-ft['params']['f1']))
        else:
            raise NotImplementedError
        return xt
    
def primal_update(method, env_dict, t, mu):
    ft, bmax = env_dict['f'][t], env_dict['bmax']
    xt = make_decision(ft,1,mu,bmax)
    # if 'naive_new' in method:
    #     Qs = env_dict['Qs']
    #     xt = make_decision(ft,1,mu/Qs[t],bmax)
    # else:
    #     xt = make_decision(ft,1,mu,bmax)
    return xt
    
def dual_subgrad(bxt, rho):
    return rho - bxt

def guess_dual_subgrad(mu, Bt, t, env_dict, guess):
    bmax, Qs,  N_max = env_dict['bmax'], env_dict['Qs'], env_dict['N_max']
    Qst = [Qs[s]/Qs[t] for s in range(env_dict['N_max'])]
    if guess == 0:
        bxs = 0
        for s in range(t,N_max):
            coef_s = Qst[s]
            Xs = [make_decision(env_dict['f'][i],coef_s,mu,bmax) for i in range(t + 1)]
            bs = np.array(Xs[:t+1])
            bxs += bs.mean()
        return (Bt-bxs)/sum(Qst[t:N_max])
    elif guess == 1:
        bxs = 0
        for s in range(N_max):
            coef_s = Qst[s]
            Xs = [make_decision(env_dict['f'][i],coef_s,mu,bmax) for i in range(t + 1)]
            bs = np.array(Xs[:t+1])
            bxs += bs.mean()
        return (env_dict['B']-bxs)/Qs.sum()
    elif guess == 2:
        Xs = [make_decision(env_dict['f'][t],Qst[s],mu,bmax) for s in range(t, N_max)]
        bs = np.array(Xs[:N_max - t])
        return (Bt-sum(bs))/sum(Qst[t:N_max])
    elif guess ==3:
        return (Bt)/sum(Qst[t:N_max])

def dual_update(method, env_dict, t, mu, bxt, Bt, lamda):
    if 'guess' in method:
        if 'wang_guess0' in method:
            rho = Bt/(env_dict['N_max']-(t + 1) + 1)
            gt = dual_subgrad(bxt, rho)
        elif 'he_guess0' in method:
            gt = guess_dual_subgrad(mu, Bt, t, env_dict, 0)
        elif 'he_guess1' in method:
            gt = guess_dual_subgrad(mu, Bt, t, env_dict, 1)
        elif 'wang_guess1' in method:
            gt = guess_dual_subgrad(mu, Bt, t, env_dict, 2)
    else:
        if 'balseiro' in method:
            rho = lamda
        elif 'naive_new' in method:
            # rho = env_dict['B'] / env_dict['N_max']
            Qs = env_dict['Qs']
            rho = env_dict['B'] * Qs[t] / env_dict['N_mean']
        elif 'naive+' in method:
            # rho= guess_dual_subgrad(mu, Bt, t, env_dict, 3)
            Qs = env_dict['Qs']
            Qst = [Qs[s]/Qs[t] for s in range(env_dict['N_max'])]
            rho = (Bt)/sum(Qst[t:env_dict['N_max']])
        elif 'naive' in method:
            rho = env_dict['B'] / env_dict['N_mean']
        gt = dual_subgrad(bxt, rho)
    return gt

def online_policy(eta,env_dict,method,lamda=None):
    Bt = env_dict['B']
    mu = 0
    xt = [0] * env_dict['N_max']
    reward = [0] * env_dict['N_max']
    mu_record,gt_record=[],[]
    first_stop = 1
    stop = env_dict['N_max']
    for t in range(env_dict['N_max']):
        xt[t] = primal_update(method, env_dict, t, mu)
        rho = lamda[t] if lamda else 0
        gt = dual_update(method, env_dict, t, mu, xt[t], Bt, rho)
        gt_record.append(gt)
        if xt[t] <= Bt:
            Bt = Bt - xt[t]
        else:
            xt[t] = 0
            if first_stop:
                first_stop = 0
                stop = env_dict['N_mean'] - t
        mu = max(0,mu - eta*gt)
        mu_record.append(mu)
        reward[t],_ = evaluate_solution(xt, env_dict, t)
    Qs = env_dict['Qs']
    probs = [Qs[n]-Qs[n+1] for n in range(env_dict['N_max'])]
    probs.append(Qs[-1])
    reward_mean = sum([probs[i] * reward[i] for i in range(env_dict['N_max'])])
    stop_mean = sum([min(stop,i) * probs[i] for i in range(env_dict['N_max'])])
    return reward_mean,stop_mean,mu_record,gt_record

def compute_lambda(B, T_min, T_max):
        
    horizons = range(T_min, T_max + 1)
    max_horizon = range(1, T_max + 1)

    prob = LpProblem("Maximize_Z", LpMaximize)

    # decision variables
    z = LpVariable('z')
    y = LpVariable.dicts('y', (horizons, max_horizon), upBound=1)
    lambda_vars = LpVariable.dicts('lambda', max_horizon, lowBound=0)

    z.setInitialValue(T_min/T_max)

    for T in horizons:
        for t in range(1,T+1):
            y[T][t].setInitialValue(T/T_max)

    for t in max_horizon:
        lambda_vars[t].setInitialValue(B/T_max)

    # objective
    prob += z

    # constraints
    for T in horizons:
        prob += lpSum([y[T][t] for t in range(1,T+1)]) >= z * T#,  f"constraint_sum_y_{T}"

    prob += lpSum([lambda_vars[t] for t in max_horizon]) <= B#, "constraint_sum_lambda"

    for T in horizons:
        for t in range(1,T+1):
            prob += y[T][t] <= T * lambda_vars[t] / B#, f"constraint_upper_bound_of_y_{T}{t}"

    prob.solve(COPT_DLL(msg=False))

    # print(f"Optimal Value of z: {z.varValue}")
    # for t in max_horizon:
    #     print(f"lambda[{t}] = {lambda_vars[t].varValue}")
    lambda_values_list = [lambda_vars[t].varValue for t in max_horizon]
    lambda_values_list.sort(reverse=True)
    # print(lambda_values_list)

    return lambda_values_list