
import torch 
import torch.nn.functional as F


from medical_diffusion.models.noise_schedulers import BasicNoiseScheduler

class GaussianNoiseScheduler(BasicNoiseScheduler):
    def __init__(
        self,
        timesteps=1000,
        T = None, 
        schedule_strategy='cosine',
        beta_start = 0.0001, # default 1e-4, stable-diffusion ~ 1e-3
        beta_end = 0.02,
        betas = None,
        ):
        super().__init__(timesteps, T)

        self.schedule_strategy = schedule_strategy

        if betas is not None:
            betas = torch.as_tensor(betas, dtype = torch.float64)
        elif schedule_strategy == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
        elif schedule_strategy == "scaled_linear": # proposed as "quadratic" in https://arxiv.org/abs/2006.11239, used in stable-diffusion 
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64)**2
        elif schedule_strategy == "cosine":
            s = 0.008
            x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64) # [0, T]
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas =  torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError(f"{schedule_strategy} does is not implemented for {self.__class__}")


        alphas = 1-betas 
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)


        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas) # (0 , 1)

        register_buffer('alphas', alphas) # (1 , 0)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',  torch.sqrt(1. / alphas_cumprod - 1))

        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
            

    def estimate_x_t(self, x_0, t, x_T=None):
        # NOTE: t == 0 means diffused for 1 step (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py#L108)
        # NOTE: t == 0 means not diffused for cold-diffusion (in contradiction to the above comment) https://github.com/arpitbansal297/Cold-Diffusion-Models/blob/c828140b7047ca22f995b99fbcda360bc30fc25d/denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L361
        x_T = self.x_final(x_0) if x_T is None else x_T 
        # ndim = x_0.ndim
        # x_t = (self.extract(self.sqrt_alphas_cumprod, t, ndim)*x_0 + 
        #         self.extract(self.sqrt_one_minus_alphas_cumprod, t, ndim)*x_T)
        def clipper(b):
            tb = t[b]
            if tb<0:
                return x_0[b]
            elif tb>=self.T:
                return x_T[b] 
            else:
                return self.sqrt_alphas_cumprod[tb]*x_0[b]+self.sqrt_one_minus_alphas_cumprod[tb]*x_T[b]
        x_t = torch.stack([clipper(b) for b in range(t.shape[0])]) 
        return x_t 
    

    def estimate_x_t_prior_from_x_T(self, x_t, t, x_T, use_log=True, clip_x0=True,  var_scale=0, cold_diffusion=False): 
        x_0 = self.estimate_x_0(x_t, x_T, t, clip_x0)
        return self.estimate_x_t_prior_from_x_0(x_t, t, x_0, use_log, clip_x0, var_scale, cold_diffusion)


    def estimate_x_t_prior_from_x_0(self, x_t, t, x_0, use_log=True, clip_x0=True, var_scale=0, cold_diffusion=False):
        x_0 = self._clip_x_0(x_0) if clip_x0 else x_0

        if cold_diffusion: # see https://arxiv.org/abs/2208.09392 
            x_T_est =  self.estimate_x_T(x_t, x_0, t) # or use x_T estimated by UNet if available? 
            x_t_est = self.estimate_x_t(x_0, t, x_T=x_T_est) 
            x_t_prior = self.estimate_x_t(x_0, t-1, x_T=x_T_est) 
            noise_t = x_t_est-x_t_prior
            x_t_prior = x_t-noise_t 
        else:
            mean = self.estimate_mean_t(x_t, x_0, t)
            variance = self.estimate_variance_t(t, x_t.ndim, use_log, var_scale)
            std = torch.exp(0.5*variance) if use_log else torch.sqrt(variance)
            std[t==0] = 0.0 
            x_T = self.x_final(x_t)
            x_t_prior =  mean+std*x_T
        return x_t_prior, x_0 

    
    def estimate_mean_t(self, x_t, x_0, t):
        ndim = x_t.ndim
        return (self.extract(self.posterior_mean_coef1, t, ndim)*x_0+
                self.extract(self.posterior_mean_coef2, t, ndim)*x_t) 
    

    def estimate_variance_t(self, t, ndim, log=True, var_scale=0, eps=1e-20):
        min_variance = self.extract(self.posterior_variance, t, ndim)
        max_variance = self.extract(self.betas, t, ndim)
        if log:
            min_variance = torch.log(min_variance.clamp(min=eps))
            max_variance = torch.log(max_variance.clamp(min=eps))
        return var_scale * max_variance + (1 - var_scale) * min_variance 
    

    def estimate_x_0(self, x_t, x_T, t, clip_x0=True):
        ndim = x_t.ndim
        x_0 = (self.extract(self.sqrt_recip_alphas_cumprod, t, ndim)*x_t - 
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, ndim)*x_T)
        x_0 = self._clip_x_0(x_0) if clip_x0 else x_0
        return x_0


    def estimate_x_T(self, x_t, x_0, t, clip_x0=True):
        ndim = x_t.ndim
        x_0 = self._clip_x_0(x_0) if clip_x0 else x_0
        return ((self.extract(self.sqrt_recip_alphas_cumprod, t, ndim)*x_t - x_0)/ 
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, ndim))
    
    
    @classmethod
    def x_final(cls, x):
        return torch.randn_like(x)

    @classmethod
    def _clip_x_0(cls, x_0):
        # See "static/dynamic thresholding" in Imagen https://arxiv.org/abs/2205.11487 

        # "static thresholding"
        m = 1 # Set this to about 4*sigma = 4 if latent diffusion is used  
        x_0 = x_0.clamp(-m, m)

        # "dynamic thresholding"
        # r = torch.stack([torch.quantile(torch.abs(x_0_b), 0.997) for x_0_b in x_0])
        # r = torch.maximum(r, torch.full_like(r,m))
        # x_0 =  torch.stack([x_0_b.clamp(-r_b, r_b)/r_b*m for x_0_b, r_b in zip(x_0, r) ] )
        
        return x_0



