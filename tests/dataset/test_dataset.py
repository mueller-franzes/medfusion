from medical_diffusion.data.datasets import SimpleDataset2D

import matplotlib.pyplot as plt 
from pathlib import Path 
from torchvision.utils import save_image

path_out = Path().cwd()/'results'/'test'
path_out.mkdir(parents=True, exist_ok=True)

# ds = SimpleDataset2D(
#     crawler_ext='jpg',
#     image_resize=(352, 528),
#     image_crop=(192, 288),
#     path_root='/home/gustav/Documents/datasets/AIROGS/dataset',
# )

ds = SimpleDataset2D(
    crawler_ext='tif',
    image_resize=None,
    image_crop=None,
    path_root='/home/gustav/Documents/datasets/BREAST-DIAGNOSIS/dataset_lr2d/'
)

images = [ds[n]['source'] for n in range(4)]


save_image(images, path_out/'test.png')