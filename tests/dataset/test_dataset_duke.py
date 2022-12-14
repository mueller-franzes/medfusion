from medical_diffusion.data.datasets import  DUKEDataset

import matplotlib.pyplot as plt 
from pathlib import Path 
from torchvision.utils import save_image
from pathlib import Path 

path_out = Path().cwd()/'results'/'test'
path_out.mkdir(parents=True, exist_ok=True)

ids = [int(path_file.stem.split('_')[-1]) for path_file in Path('/mnt/hdd/datasets/breast/Diffusion2D/images').glob('*.png')]
print(min(ids), max(ids)) # [0, 53]

ds = DUKEDataset(
    crawler_ext='png',
    image_resize=None,
    image_crop=None,
    path_root='/mnt/hdd/datasets/breast/Diffusion2D/images',
)

print(ds[0])
images = [ds[n]['source'] for n in range(4)]




save_image(images, path_out/'test.png')