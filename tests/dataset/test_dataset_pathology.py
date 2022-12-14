from medical_diffusion.data.datasets import  MSIvsMSS_Dataset

import matplotlib.pyplot as plt 
from pathlib import Path 
from torchvision.utils import save_image
from pathlib import Path 

path_out = Path().cwd()/'results'/'test'
path_out.mkdir(parents=True, exist_ok=True)


ds = MSIvsMSS_Dataset(
    crawler_ext='png',
    image_resize=None,
    image_crop=None,
    path_root='/home/gustav/Documents/datasets/Kather/data/CRC/train',
)

print(ds[0])
images = [ds[n]['source']/2+0.5 for n in range(4)]




save_image(images, path_out/'test.png')