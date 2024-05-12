'''
NaiveApproach TEST CASE1
'''

from src.Simulate import simulate_all_setting,visualize_rewards
from scipy.stats import rv_discrete

args = {
        'N_type': 'Q_exp',
        'N_min': 5,
        'N_max':200, # NOTE: when Q_exp, set large N_max(200)!
                    # NOTE: when Q_power, set large N_max(70)!
        'N_mean':5,
        'N_std':1,
        'astype':'multiplicative',
        'as_scale':1,
        'as_stepsize':5,
        'max_scale':2,
        'rounds':1,
        'Q_beta':0.99,
        'Q_alpha':0.75,
        'f_shape':'power',
        'f_coef':rv_discrete(values=([0.9], [1])),# x**0.9
    }


eta0 = {
    'off_unknown':1,'on_he_guess0':1,'on_he_guess1':1,\
    'on_wang_guess0':1,'on_wang_guess1':1,'on_balseiro':1,\
    'on_naive':1,'on_naive+':1
}
methods = ['off_unknown','on_wang_guess1',\
            'on_naive','on_naive+']

rewards=simulate_all_setting(eta0,args,methods)
print(rewards)
visualize_rewards(rewards,'exp1',methods)