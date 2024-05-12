'''
Environment Setup
'''

import numpy as np
from scipy.stats import truncnorm, truncexpon, rv_discrete
from src.utils import rv_cont2dsct, rv_qtl2dsct, get_Qexpr

class Envr():
    def __init__(self, args):
        self.Qs,self.rv,self.N_mean,self.N_min,self.N_max = self._set_rv(args)
        self.B = self.N_mean
        self.bmax = 2*self.B / self.N_mean
        self.f_shape = args['f_shape']
        self.f_coef = args['f_coef']

    @staticmethod
    def _set_rv(args):
        astype,k,ss = args['astype'],args['as_scale'],args['as_stepsize']
        if args['N_type'] == 'normal':
            if astype=='additive':
                N_mean, N_std = args['N_mean']+k*ss, args['N_std']
            elif astype=='multiplicative':
                N_mean, N_std = args['N_mean']*(k+1), args['N_std']*(k+1)
            elif astype=='vanish':
                N_mean, N_std = args['N_mean']*(k+1), args['N_std']*np.sqrt(k+1)
            else:
                raise NotImplementedError
            N_min, N_max = N_mean-2*N_std, N_mean+2*N_std
            a, b = (N_min - N_mean) / N_std, (N_max - N_mean) / N_std
            rv_continuous = truncnorm(a, b, loc=N_mean, scale=N_std) 
            rv = rv_cont2dsct(rv_continuous,N_min,N_max)
            Qs = np.array([1 - rv.cdf(t - 0.001) for t in range(1, N_max + 1)])+[0]
        elif args['N_type'] == 'exponential':
            assert astype=='multiplicative'
            N_mean = args['N_mean']*(k+1)
            N_std = N_mean 
            N_min = 1
            N_max = 2*N_mean 
            b = (N_max - N_min) / N_mean
            rv_continuous = truncexpon(b=b, loc=N_min, scale=N_mean)
            rv = rv_cont2dsct(rv_continuous,N_min,N_max)
            Qs = np.array([1 - rv.cdf(t - 0.001) for t in range(1, N_max + 1)])+[0]
        elif args['N_type'].startswith('Q_'):
            if astype=='additive':
                N_min, N_max = args['N_min']+k*ss, args['N_max']+k*ss
                # length(Qs)=N_max+1
                Qs=np.array([1]*N_min+[get_Qexpr(args,t) for t in range(1,N_max-N_min+1)]+[0])
            elif astype=='multiplicative':
                N_min, N_max = args['N_min']*(k+1), args['N_max']*(k+1)
                Q_base= [1]*args['N_min']+\
                    [get_Qexpr(args,t) for t in range(1,args['N_max']-args['N_min']+1)]+[0]
                Qs=np.array([element for element in Q_base for _ in range(k+1)])
            elif astype=='trunc_max':
                N_min, N_max = args['N_min'],args['N_max']+k*ss
                Qs=np.array([1]*N_min+[get_Qexpr(args,t) for t in range(1,N_max-N_min+1)]+[0])
            rv = rv_qtl2dsct(Qs,N_min,N_max)
            N_mean = Qs.sum()
        else:
            raise NotImplementedError
        return Qs,rv,N_mean,N_min,N_max 
    def draw_reward(self, N):
        # only consider the case that f is homogeneous with coef drawn from f_coef
        def random_reward(self):
            func_type = self.f_shape
            params = {'f1': self.f_coef.rvs()}
            return {'type': func_type, 'params': params}
        reward_funs = [random_reward(self) for _ in range(N)]
        return reward_funs
    
    def draw_instance(self):
        env_dict = {}
        N = self.rv.rvs()
        env_dict['N'] = N
        env_dict['N_max'] = self.N_max
        env_dict['N_min'] = self.N_min
        env_dict['N_mean'] = self.N_mean 
        env_dict['B'] = self.B 
        env_dict['bmax'] = self.bmax
        env_dict['f'] = self.draw_reward(self.N_max)
        env_dict['Qs'] = self.Qs
        return env_dict

if __name__ == '__main__':
    args = {
        'N_type': 'Q_power',
        'N_min': 5,
        'N_max':70, # NOTE: when Q_exp, set large N_max(200)!
                    # NOTE: when Q_power, set large N_max(70)!
        'N_mean':5,
        'N_std':1,
        'astype':'multiplicative',
        'as_scale':0,
        'as_stepsize':5,
        'Q_beta':0.99,
        'Q_alpha':0.75,
        'f_shape':'sqrt',
        'f_coef':rv_discrete(values=([1,1], [0.5, 0.5])),
    }
    env = Envr(args)
    instance = env.draw_instance()
    print(instance['N'])