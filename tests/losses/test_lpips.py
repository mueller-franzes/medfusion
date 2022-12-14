

import torch 
from medical_diffusion.loss.perceivers import LPIPS
from medical_diffusion.data.datasets import AIROGSDataset, SimpleDataset3D

loss = LPIPS(normalize=False)
torch.manual_seed(0)

# input = torch.randn((1, 3, 16, 128, 128)) # 3D - 1 channel 
# input = torch.randn((1, 1, 128, 128)) # 2D - 1 channel 
# input = torch.randn((1, 3, 128, 128)) # 2D - 3 channel 

# target = input/2 

# print(loss(input, target))


# ds = AIROGSDataset(
#     crawler_ext='jpg',
#     image_resize=(256, 256),
#     image_crop=(256, 256),
#     path_root='/mnt/hdd/datasets/eye/AIROGS/data/', # '/home/gustav/Documents/datasets/AIROGS/dataset', '/mnt/hdd/datasets/eye/AIROGS/data/'
# )
ds = SimpleDataset3D(
    crawler_ext='nii.gz',
    image_resize=None,
    image_crop=None,
    flip=True, 
    path_root='/mnt/hdd/datasets/breast/DUKE/dataset_lr_256_256_32',
    use_znorm=True
    )

input = ds[0]['source'][None]

target = torch.randn_like(input)
print(loss(input, target))