
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline

def rgb2gray(img):
    # img [B, C, H, W]
    return  ((0.3 * img[:,0]) + (0.59 * img[:,1]) + (0.11 * img[:,2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]

def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])

if __name__ == "__main__":
    path_out = Path.cwd()/'results/CheXpert/samples'
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device('cuda')

    # ------------ Load Model ------------
    # pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
    pipeline = DiffusionPipeline.load_from_checkpoint('runs/2022_12_12_171357_chest_diffusion/last.ckpt')
    pipeline.to(device)

    
    # --------- Generate Samples  -------------------
    steps = 150
    use_ddim = True 
    images = {}
    n_samples = 16

    for cond in [0,1,None]:
        torch.manual_seed(0)
 
        # --------- Conditioning ---------
        condition = torch.tensor([cond]*n_samples, device=device) if cond is not None else None 
        # un_cond = torch.tensor([1-cond]*n_samples, device=device)
        un_cond = None 

        # ----------- Run --------
        results = pipeline.sample(n_samples, (8, 32, 32), guidance_scale=8, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )
        # results = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=1, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )

        # --------- Save result ---------------
        results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
        results = results.clamp(0, 1)
        utils.save_image(results, path_out/f'test_{cond}.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
        images[cond] = results


    diff = torch.abs(normalize(rgb2gray(images[1]))-normalize(rgb2gray(images[0]))) # [0,1] -> [0, 1]
    # diff = torch.abs(images[1]-images[0])
    utils.save_image(diff, path_out/'diff.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
    

        