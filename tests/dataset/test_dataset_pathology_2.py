from medical_diffusion.data.datasets import  MSIvsMSS_2_Dataset

import matplotlib.pyplot as plt 
from pathlib import Path 
from torchvision.utils import save_image
from pathlib import Path 

path_out = Path().cwd()/'results'/'test'/'patho2'
path_out.mkdir(parents=True, exist_ok=True)


ds = MSIvsMSS_2_Dataset(
    crawler_ext='jpg',
    image_resize=None,
    image_crop=None,
    # path_root='/home/gustav/Documents/datasets/Kather_2/train',
    path_root='/mnt/hdd/datasets/pathology/kather_msi_mss_2/train/'
)

print(ds[0])
images = [ds[n]['source']/2+0.5 for n in range(4)]




save_image(images, path_out/'test.png')