

from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.data.datasets import SimpleDataset2D
from medical_diffusion.models.pipelines import DiffusionPipeline
import torch 
from pathlib import Path

from torchvision.utils import save_image

ds = SimpleDataset2D(
    crawler_ext='jpg',
    image_resize=(352, 528),
    image_crop=(192, 288),
    path_root='/home/gustav/Documents/datasets/AIROGS/dataset',
)

device = torch.device('cuda')

pipeline = DiffusionPipeline.load_from_checkpoint('runs/2022_09_22_153738/last.ckpt')
pipeline.to(device)

scheduler = GaussianNoiseScheduler()
scheduler.to(device)


path_out = Path.cwd()/'results/test'
torch.manual_seed(0)


x_0 = ds[0]['source'][None] # [B, C, H, W]
x_0 = x_0.to(device)
x_0 = x_0*2-1
noise = torch.rand_like(x_0)

x_ts = [] 
x_0_preds = []
for t in range(0, 1000, 100):
    time = torch.tensor([t], device=device) 
    x_t = scheduler.estimate_x_t(x_0=x_0, t=time, noise=noise) # [B, C, H, W]
    x_0_pred = pipeline.denoise(x_t, i=t)
    x_t = x_t/2+0.5
    x_0_pred = x_0_pred/2+0.5
    x_ts.append(x_t)
    x_0_preds.append(x_0_pred)
# print(x_t)
x_ts = torch.cat(x_ts)
save_image(x_ts, path_out/'test2.png')

x_0_preds = torch.cat(x_0_preds)
save_image(x_0_preds, path_out/'test3.png')

# x_0 = scheduler.estimate_x_0(x_t, noise, t)
# # print(x_0)

# x_t_prior = scheduler.estimate_x_t_prior_from_noise(x_t, t, noise, noise=noise)



