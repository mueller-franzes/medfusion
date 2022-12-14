

from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.data.datasets import SimpleDataset2D, AIROGSDataset
from medical_diffusion.models.embedders.latent_embedders import VQVAE
import torch 
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns 
from math import ceil


import statsmodels.api as sm





device = torch.device('cuda')
path_out = Path.cwd()/'results/test'
torch.manual_seed(0)

ds = AIROGSDataset(
        crawler_ext='jpg',
        image_resize=(256, 256),
        image_crop=(256, 256),
        path_root='/home/gustav/Documents/datasets/AIROGS/dataset', # '/home/gustav/Documents/datasets/AIROGS/dataset',  /mnt/hdd/datasets/eye/AIROGS/data/
    )
x_0 = ds[0]['source'][None] # [B, C, H, W]


scheduler = GaussianNoiseScheduler(timesteps=500, schedule_strategy='scaled_linear')


# embedder = VQVAE.load_from_checkpoint('runs/2022_10_06_233542_vqvae_eye/last.ckpt')
# with torch.no_grad():
#     x_0 = embedder.encode(x_0)

noise = torch.randn_like(x_0)

step=100
binrange=(-2.5,2.5)
bins = 50

ncols=8
nelem = (scheduler.T+step)//step+2
nrows = ceil(nelem/8)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
ax_iter = iter(ax.flatten())
for axis in ax_iter:
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
ax_iter = iter(ax.flatten())



axis = next(ax_iter)
sm.qqplot(x_0.flatten(), line='45', ax=axis)

for t in range(0, scheduler.T+step, step):
    print(t)
    t = torch.tensor([t]) 
    x_t = scheduler.estimate_x_t(x_0=x_0, t=t, x_T=noise) # [B, C, H, W]
    axis = next(ax_iter)
    sm.qqplot(x_t.flatten(), line='45', ax=axis)

axis = next(ax_iter)
sm.qqplot(noise.flatten(), line='45', ax=axis)

fig.tight_layout()
fig.savefig(path_out/'scheduler_nosing_qq.png')



