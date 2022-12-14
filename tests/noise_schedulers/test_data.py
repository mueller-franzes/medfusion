

from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.data.datasets import SimpleDataset2D, AIROGSDataset, CheXpert_Dataset, MSIvsMSS_2_Dataset
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN
import torch 
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns 
from math import ceil


from torchvision.utils import save_image

# ds = SimpleDataset2D(
#     crawler_ext='jpg',
#     image_resize=(352, 528),
#     image_crop=(192, 288),
#     path_root='/home/gustav/Documents/datasets/AIROGS/dataset',
# )

# ds = AIROGSDataset(
#         crawler_ext='jpg',
#         image_resize=(256, 256),
#         image_crop=(256, 256),
#         path_root='/home/gustav/Documents/datasets/AIROGS/dataset', # '/home/gustav/Documents/datasets/AIROGS/dataset',  /mnt/hdd/datasets/eye/AIROGS/data/
#     )
# ds = CheXpert_Dataset(
#         crawler_ext='jpg',
#         augment_horizontal_flip=False,
#         augment_vertical_flip=False,
#         path_root='/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed/valid',
#     )

ds = MSIvsMSS_2_Dataset(
        crawler_ext='jpg',
        image_resize=None,
        image_crop=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False, 
        # path_root='/home/gustav/Documents/datasets/Kather_2/train',
        path_root='/mnt/hdd/datasets/pathology/kather_msi_mss_2/train/',
    )

device = torch.device('cuda')

scheduler = GaussianNoiseScheduler(timesteps=1000, beta_start=1e-4, schedule_strategy='scaled_linear')
# scheduler.to(device)
path_out = Path.cwd()/'results/test/scheduler'
path_out.mkdir(parents=True, exist_ok=True)


# print(scheduler.posterior_mean_coef1)
torch.manual_seed(0)
x_0 = ds[0]['source'][None] # [B, C, H, W]



embedder = VAE.load_from_checkpoint('runs/2022_11_25_232957_patho_vaegan/last_vae.ckpt')
with torch.no_grad():
    x_0 = embedder.encode(x_0)

# x_0 = (x_0-x_0.min())/(x_0.max()-x_0.min())
# x_0 = x_0*2-1
# x*2-1 = (x-0.5)*2

noise = torch.randn_like(x_0)

x_ts = [] 
step=100


for t in range(0, scheduler.T+step, step):
    t = torch.tensor([t]) 
    x_t = scheduler.estimate_x_t(x_0=x_0, t=t, x_T=noise) # [B, C, H, W]
    print(t, x_t.mean(), x_t.std())
    x_ts.append(x_t)

x_ts = torch.cat(x_ts)
# save_image(x_ts, path_out/'scheduler_nosing.png', normalize=True, scale_each=True)




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
sns.histplot(x=x_0.flatten(), bins=bins, binrange=binrange, ax=axis)

for t in range(0, scheduler.T+step, step):
    print(t)
    t = torch.tensor([t]) 
    x_t = scheduler.estimate_x_t(x_0=x_0, t=t, x_T=noise) # [B, C, H, W]
    axis = next(ax_iter)
    sns.histplot(x=x_t.flatten(), bins=bins, binrange=binrange, ax=axis)

axis = next(ax_iter)
sns.histplot(x=noise.flatten(), bins=bins, binrange=binrange, ax=axis)

fig.tight_layout()
fig.savefig(path_out/'scheduler_nosing_histo.png')



