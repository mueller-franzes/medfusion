from pathlib import Path

import torch 
import numpy as np 
from PIL import Image
from torchvision.utils import save_image





# class_2 = 'RG'
# class_1 = 'NRG'
# path_out = Path().cwd()/'results'/'AIROGS'/'generated_images'
# path_root = Path('/mnt/hdd/datasets/eye/AIROGS/data_generated_diffusion/')
# path_root = Path('/mnt/hdd/datasets/eye/AIROGS/data_generated_stylegan3')
# path_root = Path('/mnt/hdd/datasets/eye/AIROGS/data_256x256_ref/')

class_2 = 'Cardiomegaly'
class_1 = 'No_Cardiomegaly'
path_out = Path().cwd()/'results'/'CheXpert'/'generated_images'
path_root = Path('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_diffusion3_150/')
# path_root = Path('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_progan/')
# path_root = Path('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/reference/')

# class_2 = 'MSIH'
# class_1 = 'nonMSIH'
# path_out = Path().cwd()/'results'/'MSIvsMSS_2'/'generated_images'
# path_root = Path('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/diffusion2_150/')
# path_root = Path('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/SYNTH-CRC-10K/')
# path_root = Path('/mnt/hdd/datasets/pathology/kather_msi_mss_2/train')

num = 2
np.random.seed(2)
a = np.random.randint(0, 1000)
b = np.random.randint(0, 1000)
print(a, b)

path_out.mkdir(parents=True, exist_ok=True)
paths_class_1 = [path_img for n, path_img in enumerate((path_root/class_1).iterdir()) if a<=n<a+num]
paths_class_2 = [path_img for n, path_img in enumerate((path_root/class_2).iterdir()) if b<=n<b+num]
paths_imgs = paths_class_1+paths_class_2

pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0)/255.0 # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)

images = [pil2torch(Image.open(path_img).convert('RGB')) for path_img in paths_imgs ]

images = torch.stack(images)

save_image(images, path_out/'img_row.png', nrwos=1, normalize=True, scale_each=True)
