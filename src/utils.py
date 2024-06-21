import numpy as np
from scipy.stats import rv_discrete

### RV Utils
def rv_cont2dsct(rv_continuous,N_min,N_max):
    # Generate a range of integer values
    values = np.arange(N_min - 0.5, N_max + 0.51, 1)
    # Calculate probabilities for each value using the CDF
    probabilities = np.diff(rv_continuous.cdf(values))
    probabilities = probabilities / np.sum(probabilities)
    # Create a discrete distribution
    rv = rv_discrete(values=(np.arange(N_min, N_max + 1, 1), probabilities))
    return rv 

def rv_qtl2dsct(Qs,N_min,N_max):
    pmf = rv_qtl2pmf(Qs,N_min,N_max)
    rv = rv_discrete(values=(np.arange(N_min, N_max + 1, 1), pmf))
    return rv 

def rv_qtl2pmf(Qs,N_min,N_max):
    # Qs[-1]=0
    # length(pmf)=N_max-N_min+1
    pmf = [Qs[i] - Qs[i+1] for i in range(N_min-1,N_max)]
    pmf[-1] = 1-sum(pmf[:-1])
    return pmf

def get_Qexpr(args,t):
    if args['N_type']=='Q_exp':
        return args['Q_beta']**t # args['Q_beta']**t
    elif args['N_type'] == 'Q_power':
        return t**(-args['Q_alpha']) # t**args['Q_alpha']
    else:
        raise NotImplementedError

### Reward Utils
def get_reward(x, func_dict):
    if func_dict['type'] == 'linear':
        return func_dict['params']['f1'] * x 
    if func_dict['type'] == 'sqrt':
        return func_dict['params']['f1'] * np.sqrt(x)
    if func_dict['type'] == 'log':
        return func_dict['params']['f1'] * np.log(x+1)
    if func_dict['type'] == 'power':
        return  (x**func_dict['params']['f1'])
    else:
        pass 

def get_resource(x):
    return x 

def evaluate_solution(sol, env_dict, T = None):
    if T is None:
        T = env_dict['N']
    rewards = np.array([get_reward(sol[t],env_dict['f'][t]) for t in range(T)])
    resources = np.array([get_resource(sol[t]) for t in range(T)])
    return np.sum(rewards), env_dict['B']-np.sum(resources)

def conditional_mean(t,env_dict):
    Qs,N_max = env_dict['Qs'], env_dict['N_max']
    Qst = [Qs[s]/Qs[t] for s in range(N_max)]
    return sum(Qst[t:N_max])

def expected_consumption(t,Bt,env_dict,method):
    if "LP_naive+" in method:
        d = env_dict['B'] / env_dict['N_mean']
    elif "LP_naive" in method:
        d = Bt / conditional_mean(t,env_dict)
    else:
        raise KeyError
    return d