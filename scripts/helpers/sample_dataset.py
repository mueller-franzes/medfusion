
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
import numpy as np 
from PIL import Image
import time

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ------------ Load Model ------------
device = torch.device('cuda')
# pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
pipeline = DiffusionPipeline.load_from_checkpoint('runs/2022_12_12_171357_chest_diffusion/last.ckpt')
pipeline.to(device)

if __name__ == "__main__":
    # {'NRG':0, 'RG':1} 3270, {'MSIH':0, 'nonMSIH':1} :9979 {'No_Cardiomegaly':0, 'Cardiomegaly':1} 7869
    for steps in [50, 100, 150, 200, 250]:
        for name, label in  {'No_Cardiomegaly':0, 'Cardiomegaly':1}.items(): 
            n_samples = 7869
            sample_batch = 200
            cfg = 1
        
            # path_out = Path(f'/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/diffusion2_{steps}/')/name
            path_out = Path(f'/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_diffusion3_{steps}')/name
            # path_out = Path('/mnt/hdd/datasets/eye/AIROGS/data_generated_diffusion')/name
            path_out.mkdir(parents=True, exist_ok=True)

            # --------- Generate Samples  -------------------
            torch.manual_seed(0)
            counter = 0
            for chunk in chunks(list(range(n_samples)), sample_batch):
                condition = torch.tensor([label]*len(chunk), device=device) if label is not None else None 
                un_cond = torch.tensor([1-label]*len(chunk), device=device)  if label is not None else None # Might be None, or 1-condition or specific label 
                results = pipeline.sample(len(chunk), (8, 32, 32), guidance_scale=cfg, condition=condition, un_cond=un_cond, steps=steps)
                # results = pipeline.sample(len(chunk), (4, 64, 64), guidance_scale=cfg, condition=condition, un_cond=un_cond, steps=steps )

                results = results.cpu().numpy()
                # --------- Save result ----------------
                for image in results:
                    image =  image.clip(-1, 1) # or (image-image.min())/(image.max()-image.min()) 
                    image = (image+1)/2*255  # Transform from [-1, 1] to [0, 1] to [0, 255]
                    image = np.moveaxis(image, 0, -1)
                    image = image.astype(np.uint8)
                    image = np.squeeze(image, axis=-1) if image.shape[-1]==1 else image 
                    Image.fromarray(image).convert("RGB").save(path_out/f'fake_{counter}.png')
                    counter += 1

            
            torch.cuda.empty_cache()
            time.sleep(3)
