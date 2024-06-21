'''
NaiveApproach TEST CASE1
'''

from src.Simulate import simulate_all_setting,visualize_rewards,visualize_stops
from scipy.stats import rv_discrete, uniform

args = {
        'N_type': 'normal',
        'N_min': 5,
        'N_max':200, # NOTE: when Q_exp, set large N_max(200)!
                    # NOTE: when Q_power, set large N_max(70)!
        'N_mean':5,
        'N_std':1,
        'B_type':'mean',
        'astype':'multiplicative',
        'as_scale':1,
        'as_stepsize':5,
        'max_scale':30,
        'rounds':3,
        'Q_beta':0.99,
        'Q_alpha':0.75,
        'f_shape':'linear',
        # 'f_coef':rv_discrete(values=([1,3], [0.5,0.5])),# x**0.9
        'f_coef':uniform(loc=2, scale=3)
    }


eta0 = {
    'off_unknown':1,'on_he_guess0':1,'on_he_guess1':1,\
    'on_wang_guess0':1,'on_wang_guess1':1,'on_balseiro':1,\
    'on_naive':1.5,'on_naive+':1,'on_naive_new':1.2
}
# methods = ['off_unknown','on_wang_guess1',\
#             'on_naive','on_naive+']
methods = ['off_unknown','on_naive',\
            'on_naive+','on_wang_guess1']

# methods = ['off_unknown','on_naive','on_naive+','on_naive_new']

N_type,f_shape,astype = args['N_type'],args['f_shape'],args['astype']
name = f"{N_type}_{f_shape}_{astype}"
rewards,stops=simulate_all_setting(eta0,args,methods)
rewards.to_csv(f"reward_data/{name}.csv")
print(rewards)
visualize_rewards(rewards,name,methods)
visualize_stops(stops,name,methods)