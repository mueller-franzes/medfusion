from pathlib import Path
import math 

import torch 
from torchvision.utils import save_image

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import AIROGSDataset, SimpleDataset2D
from medical_diffusion.models.embedders.latent_embedders import VQVAE, VQGAN


path_out = Path.cwd()/'results/test'
path_out.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')
torch.manual_seed(0)

ds = AIROGSDataset(
    crawler_ext='jpg',
    image_resize=(256, 256),
    image_crop=(256, 256),
    path_root='/home/gustav/Documents/datasets/AIROGS/dataset', # '/home/gustav/Documents/datasets/AIROGS/dataset', '/mnt/hdd/datasets/eye/AIROGS/data/'
)

x = ds[0]['source'][None].to(device) # [B, C, H, W]

# v_min = x.min() 
# v_max = x.max()
# x = (x-v_min)/(v_max-v_min)
# x = x*2-1

# x = (x+1)/2
# x = x*(v_max-v_min)+v_min

embedder = VQVAE.load_from_checkpoint('runs/2022_10_06_233542_vqvae_eye/last.ckpt')
embedder.to(device)


with torch.no_grad():
    z = embedder.encode(x)

x_pred = embedder.decode(z)


images = torch.cat([x, x_pred])
save_image(images, path_out/'test_latent_embedder.png', nrwos=int(math.sqrt(images.shape[0])), normalize=True, scale_each=True)