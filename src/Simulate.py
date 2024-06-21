from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.Env import Envr
from src.Policy import offline_policy, online_policy, compute_lambda

def simulate_single_setting(eta0,args,methods,runtime = True):
    env = Envr(args)
    eta = {method:eta0[method]/np.sqrt(env.N_mean) for method in methods}
    if 'on_naive_new' in methods:
        eta['on_naive_new'] = eta0['on_naive_new']/np.sqrt(env.N_max)
    start_time = datetime.now()
    lamda = None
    if 'on_balseiro' in methods:
        lamda = compute_lambda(env.B, env.N_min, env.N_max)
    # print(f'RUNTIME: lambda={datetime.now()-start_time}')

    reward = {method: 0 for method in methods}
    stop = {method: env.N_max for method in methods}
    for i in range(args['rounds']):
        env_dict = env.draw_instance()
        reward['off_unknown'] += offline_policy(env_dict,'unknown')[1]
        for method in methods:
            if 'off' in method:
                continue
            reward[method] += online_policy(eta[method],env_dict,method,lamda)[0]
            stop[method] += online_policy(eta[method],env_dict,method,lamda)[1]
    reward = {method: value/args['rounds'] for method,value in reward.items()}
    stop = {method: value/args['rounds'] for method,value in stop.items()}
    N = env.N_max if 'max' in args['B_type'] else env.N_mean 
    return reward,stop,N 

def simulate_all_setting(eta0, args, methods):
    rewards = []
    stops = []
    Ns = []
    try:
        for scale in range(args['max_scale']):
            args['as_scale'] = scale 
            start_time = datetime.now()
            reward,stop,N_mean = simulate_single_setting(eta0, args, methods)
            rewards.append(reward)
            stops.append(stop)
            Ns.append(N_mean)
            print(f'N_mean={N_mean} finished. Runtime={datetime.now()-start_time}')
    except KeyboardInterrupt:
        print('Interupted')
    rewards = pd.DataFrame(rewards)
    rewards['N'] = Ns 
    rewards.set_index('N', inplace=True)
    stops = pd.DataFrame(stops)
    stops['N'] = Ns 
    stops.set_index('N', inplace=True)
    return rewards, stops

def visualize_rewards(rewards,name,methods):
    benchmark_name = 'off_unknown'
    benchmark = rewards[benchmark_name]
    methods = methods[1:]
    rel_names = [method + "_rel" for method in methods]
    rgt_names = [method + "_rgt" for method in methods]
    for i,method in enumerate(methods):
        rel_name,rgt_name = rel_names[i],rgt_names[i]
        rewards[rel_name]=rewards[method]/benchmark
        rewards[rgt_name]=benchmark-rewards[method]

    labels = {
        'on_he_guess0':r'$g_t^2=(B_t-\sum_{s\geq t} h_s^{(t)})/(\sum_{s\geq t} Q_s/Q_t)$',
        'on_he_guess1':r'$g_t^3=(B_t-\sum_{s\geq 1} h_s^{(t)})/(\sum_{s\geq 1} Q_s/Q_t)$',
        'on_wang_guess0':r'$g_t^0=B_t/(T_{max}-t+1)-\tilde x_t$',
        'on_wang_guess1':r'$g_t^1=(B_t-\sum_{s\geq t} \tilde x_s^{(t)})/(\sum_{s\geq t} Q_s/Q_t)$',
        'on_balseiro':'VT_DMD',
        'on_naive':'naive',
        'on_naive+':'naive+',
        'on_naive_new':'naive_new'
    }
    styles = ['o-', 's--', '^:', '*-', 'd--', 'o:', 's-']
    
    relative_df = rewards[rel_names]
    relative_df.plot(style=styles[:len(methods)],markersize=3)
    plt.legend([labels[method] for method in methods])
    plt.xlabel('Horizon Mean')
    plt.ylabel('Competitive Ratio')
    plt.title(name)
    plt.savefig(f"figure/{name}-relative.png",dpi=300)
    plt.show()
    
    regret_df = rewards[rgt_names]
    regret_df.plot(style=styles[:len(methods)],markersize=3)
    plt.legend([labels[method] for method in methods])
    plt.xlabel('Horizon Mean')
    plt.ylabel('Regret')
    plt.title(name)
    plt.savefig(f"figure/{name}-regret.png",dpi=300)
    plt.show()

def visualize_stops(stops,name,methods):

    labels = {
        'on_he_guess0':r'$g_t^2=(B_t-\sum_{s\geq t} h_s^{(t)})/(\sum_{s\geq t} Q_s/Q_t)$',
        'on_he_guess1':r'$g_t^3=(B_t-\sum_{s\geq 1} h_s^{(t)})/(\sum_{s\geq 1} Q_s/Q_t)$',
        'on_wang_guess0':r'$g_t^0=B_t/(T_{max}-t+1)-\tilde x_t$',
        'on_wang_guess1':r'$g_t^1=(B_t-\sum_{s\geq t} \tilde x_s^{(t)})/(\sum_{s\geq t} Q_s/Q_t)$',
        'on_balseiro':'VT_DMD',
        'on_naive':'naive',
        'on_naive+':'naive+',
        'on_naive_new':'naive_new'
    }
    styles = ['o-', 's--', '^:', '*-', 'd--', 'o:', 's-']
    
    methods.remove('off_unknown')
    stops_df = stops[methods]
    stops_df.plot(style=styles[:len(methods)],markersize=3)
    plt.legend([labels[method] for method in methods])
    plt.xlabel('Horizon Mean')
    plt.ylabel(r'$E(T)-\tau$')
    plt.title(name)
    plt.savefig(f"figure/{name}-stop.png",dpi=300)
    plt.show()