from pathlib import Path
import math 

import torch 
import torch.nn.functional as F
from torchvision.utils import save_image

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import AIROGSDataset, MSIvsMSS_2_Dataset, CheXpert_2_Dataset
from medical_diffusion.models.embedders.latent_embedders import VQVAE, VQGAN, VAE, VAEGAN
import matplotlib.pyplot as plt 
import seaborn as sns 

path_out = Path.cwd()/'results/test/latent_embedder'
path_out.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')
torch.manual_seed(0)

# ds = AIROGSDataset( #  256x256
#     crawler_ext='jpg',
#     augment_horizontal_flip=True,
#     augment_vertical_flip=True,
#     # path_root='/home/gustav/Documents/datasets/AIROGS/dataset',
#     path_root='/mnt/hdd/datasets/eye/AIROGS/data_256x256',
# )

# ds = MSIvsMSS_2_Dataset( #  512x512
#     # image_resize=256,
#     crawler_ext='jpg',
#     augment_horizontal_flip=False,
#     augment_vertical_flip=False,
#     # path_root='/home/gustav/Documents/datasets/Kather_2/train'
#     path_root='/mnt/hdd/datasets/pathology/kather_msi_mss_2/train/'
# )

ds = CheXpert_2_Dataset( #  256x256
    augment_horizontal_flip=False,
    augment_vertical_flip=False,
    path_root = '/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed_tianyu'
)

dm = SimpleDataModule(
    ds_train = ds,
    batch_size=4, 
    num_workers=0,
) 


# ------------------ Load Model -------------------
model = VAE.load_from_checkpoint('runs/2022_12_12_133315_chest_vaegan/last_vae.ckpt')

# from diffusers import StableDiffusionPipeline
# with open('auth_token.txt', 'r') as file:
#     auth_token = file.read()
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32,  use_auth_token=auth_token)
# model = pipe.vae

model = model.to(device)

# ------------- Reset Seed ------------
torch.manual_seed(0)

# ------------ Prepare Data ----------------
date_iter = iter(dm.train_dataloader())
for k in range(1):
    batch = next(date_iter) 
x = batch['source']
x = x.to(device) #.to(torch.float16)

# ------------- Run Model ----------------
with torch.no_grad():
    # ------------- Encode ----------
    z = model.encode(x)
    # z = z.latent_dist.sample() # Only for stable-diffusion 

    # ------------- Decode -----------
    sns.histplot(z.flatten().detach().cpu().numpy())
    plt.savefig('test.png')
    x_pred = model.decode(z)
    # x_pred = x_pred.sample # Only for stable-diffusion 
    x_pred = x_pred.clamp(-1, 1)

images =  x_pred[0] #torch.cat([x, x_pred])
save_image(images, path_out/'latent_embedder_vaegan.png', nrow=x.shape[0], normalize=True, scale_each=True)


