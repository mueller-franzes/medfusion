

import torch 
import torch.nn as nn 


class BasicNoiseScheduler(nn.Module):
    def __init__(
        self,
        timesteps=1000,
        T=None,
        ):
        super().__init__()
        self.timesteps = timesteps 
        self.T = timesteps if  T is None else T 

        self.register_buffer('timesteps_array', torch.linspace(0, self.T-1, self.timesteps, dtype=torch.long))   # NOTE: End is inclusive therefore use -1 to get [0, T-1]  
    
    def __len__(self):
        return len(self.timesteps)

    def sample(self, x_0):
        """Randomly sample t from [0,T] and return x_t and x_T based on x_0"""
        t = torch.randint(0, self.T, (x_0.shape[0],), dtype=torch.long, device=x_0.device) # NOTE: High is exclusive, therefore [0, T-1]
        x_T = self.x_final(x_0) 
        return self.estimate_x_t(x_0, t, x_T), x_T, t
    
    def estimate_x_t_prior_from_x_T(self, x_T, t, **kwargs):
        raise NotImplemented
    
    def estimate_x_t_prior_from_x_0(self, x_0, t, **kwargs):
        raise NotImplemented

    def estimate_x_t(self, x_0, t, x_T=None, **kwargs):
        """Get x_t at time t"""
        raise NotImplemented

    @classmethod
    def x_final(cls, x):
        """Get noise that should be obtained for t->T """
        raise NotImplemented

    @staticmethod 
    def extract(x, t, ndim):
        """Extract values from x at t and reshape them to n-dim tensor"""
        return x.gather(0, t).reshape(-1, *((1,)*(ndim-1))) 
    


