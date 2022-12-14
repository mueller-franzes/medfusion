from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore as IS

from medical_diffusion.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall


# ----------------Settings --------------
batch_size = 100
max_samples = None # set to None for all 
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd()/'results'/'CheXpert'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)


# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

# -------------- Helpers ---------------------
pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)


# ---------------- Dataset/Dataloader ----------------
# ds_real = ImageFolder('/mnt/hdd/datasets/pathology/kather_msi_mss_2/train', transform=pil2torch)
# ds_fake = ImageFolder('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/SYNTH-CRC-10K/', transform=pil2torch) 
# ds_fake = ImageFolder('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/diffusion2_250', transform=pil2torch) 

# ds_real = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_256x256_ref/', transform=pil2torch)
# ds_fake = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_generated_stylegan3/', transform=pil2torch) 
# ds_fake = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_generated_diffusion', transform=pil2torch) 

ds_real = ImageFolder('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/reference/', transform=pil2torch)
# ds_fake = ImageFolder('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_progan/', transform=pil2torch) 
ds_fake = ImageFolder('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_diffusion3_250/', transform=pil2torch) 

ds_real.samples = ds_real.samples[slice(max_samples)]
ds_fake.samples = ds_fake.samples[slice(max_samples)]


# --------- Select specific class ------------
# target_class = 'MSIH'
# ds_real = Subset(ds_real, [i for i in range(len(ds_real)) if ds_real.samples[i][1] == ds_real.class_to_idx[target_class]])
# ds_fake = Subset(ds_fake, [i for i in range(len(ds_fake)) if ds_fake.samples[i][1] == ds_fake.class_to_idx[target_class]])

# Only for testing metrics against OpenAI implementation 
# ds_real = TensorDataset(torch.from_numpy(np.load('/home/gustav/Documents/code/guided-diffusion/data/VIRTUAL_imagenet64_labeled.npz')['arr_0']).swapaxes(1,-1))
# ds_fake = TensorDataset(torch.from_numpy(np.load('/home/gustav/Documents/code/guided-diffusion/data/biggan_deep_imagenet64.npz')['arr_0']).swapaxes(1,-1))


dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")
logger.info(f"Samples Fake: {len(ds_fake)}")

# ------------- Init Metrics ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc_fid = FID().to(device) # requires uint8
# calc_is = IS(splits=1).to(device) # requires uint8, features must be 1008 see https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L603 
calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)




# --------------- Start Calculation -----------------
for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch[0].to(device)

    # -------------- FID -------------------
    calc_fid.update(imgs_real_batch, real=True)

    # ------ Improved Precision/Recall--------
    calc_pr.update(imgs_real_batch, real=True)

# torch.save(torch.concat(calc_fid.real_features), 'real_fid.pt')
# torch.save(torch.concat(calc_pr.real_features), 'real_ipr.pt')


for fake_batch in tqdm(dm_fake):
    imgs_fake_batch = fake_batch[0].to(device)

    # -------------- FID -------------------
    calc_fid.update(imgs_fake_batch, real=False)

    # -------------- IS -------------------
    # calc_is.update(imgs_fake_batch)

    # ---- Improved Precision/Recall--------
    calc_pr.update(imgs_fake_batch, real=False)

# torch.save(torch.concat(calc_fid.fake_features), 'fake_fid.pt')
# torch.save(torch.concat(calc_pr.fake_features), 'fake_ipr.pt')

# --------------- Load features --------------
# real_fid = torch.as_tensor(torch.load('real_fid.pt'), device=device)
# real_ipr = torch.as_tensor(torch.load('real_ipr.pt'), device=device)
# fake_fid = torch.as_tensor(torch.load('fake_fid.pt'), device=device)
# fake_ipr = torch.as_tensor(torch.load('fake_ipr.pt'), device=device)

# calc_fid.real_features = real_fid.chunk(batch_size)
# calc_pr.real_features = real_ipr.chunk(batch_size)
# calc_fid.fake_features = fake_fid.chunk(batch_size)
# calc_pr.fake_features = fake_ipr.chunk(batch_size)



# -------------- Summary -------------------
fid = calc_fid.compute()
logger.info(f"FID Score: {fid}")

# is_mean, is_std = calc_is.compute()
# logger.info(f"IS Score: mean {is_mean} std {is_std}") 

precision, recall = calc_pr.compute()
logger.info(f"Precision: {precision}, Recall {recall} ")

