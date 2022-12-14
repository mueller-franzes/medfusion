from medical_diffusion.data.datasets import SimpleDataset2D, AIROGSDataset

import torch.nn.functional as F 

import matplotlib.pyplot as plt 
from pathlib import Path 
from torchvision.utils import save_image

path_out = Path().cwd()/'results'/'test'
path_out.mkdir(parents=True, exist_ok=True)

ds = AIROGSDataset(
    crawler_ext='jpg',
    image_resize=(256, 256),
    image_crop=(256, 256),
    path_root='/mnt/hdd/datasets/eye/AIROGS/data/', # '/home/gustav/Documents/datasets/AIROGS/dataset', '/mnt/hdd/datasets/eye/AIROGS/data/'
)

weights = ds.get_weights()
images = [ds[n]['source'] for n in range(4)]

interpolation_mode = 'bilinear'
images = [F.interpolate(img[None], size=[128, 128], mode=interpolation_mode, align_corners=None)[0]  for img in images]

images = [img/2+0.5 for img in images]

save_image(images, path_out/'test.png')