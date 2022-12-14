from medical_diffusion.data.datasets import SimpleDataset3D

import matplotlib.pyplot as plt 
from pathlib import Path 
from torchvision.utils import save_image
import torch 

path_out = Path().cwd()/'results'/'test'
path_out.mkdir(parents=True, exist_ok=True)


ds = SimpleDataset3D(
    crawler_ext='nii.gz',
    image_resize=None,
    image_crop=None,
    path_root='/mnt/hdd/datasets/breast/DUKE/dataset_lr_256_256_32',
    use_znorm=False
)

image = ds[0]['source'] # [C, D, H, W]

image = image.swapaxes(0, 1) # [D, C, H, W] -> treat D as Batch Dimension 
image = image/2+0.5

save_image(image, path_out/'test.png')