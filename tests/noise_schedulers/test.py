

from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
import torch 
from pathlib import Path

from torchvision.utils import save_image


device = torch.device('cuda')

scheduler = GaussianNoiseScheduler()
# scheduler.to(device)
path_out = Path.cwd()/'results/test'


# print(scheduler.posterior_mean_coef1)
torch.manual_seed(0)
# x_0 = torch.ones((2, 3, 64, 64))
x_0 = torch.rand((2, 3, 64, 64))
noise = torch.randn_like(x_0)
t = torch.tensor([0, 999]) 

x_t = scheduler.estimate_x_t(x_0=x_0, t=t, x_T=noise)

# x_0_pred = scheduler.estimate_x_t(x_0=x_0, t=torch.full_like(t, 0) , noise=noise)
# assert (x_0_pred == x_0).all(), "For t=0, function should return x_0"
# x_t, noise, t = scheduler.sample(x_0)
# print(x_t)


# x_0 = scheduler.estimate_x_0(x_t, noise, t)
# print(x_0)
# print(x_0.shape)



pred = torch.randn_like(x_t)
x_t_prior, _ = scheduler.estimate_x_t_prior_from_x_T(x_t, t, pred, clip_x0=False)
print(x_t_prior)

# save_image(x_t_prior, path_out/'test2.png')

