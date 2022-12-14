
from pathlib import Path
from PIL import Image
import numpy as np 



if __name__ == "__main__":
    path_out = Path.cwd()/'media/'
    path_out.mkdir(parents=True, exist_ok=True)

    # imgs = [] 
    # for img_i in range(50):
    #     for label_a, label_b, label_c in [('NRG', 'No_Cardiomegaly', 'nonMSIH'), ('RG', 'Cardiomegaly', 'MSIH')]:
    #         img_a = Image.open(f'/mnt/hdd/datasets/eye/AIROGS/data_generated_diffusion/{label_a}/fake_{img_i}.png').quantize(200, 0).convert('RGB')
    #         img_b = Image.open(f'/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_diffusion2_150/{label_b}/fake_{img_i}.png').quantize(50, 0).convert('RGB')
    #         img_c = Image.open(f'/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/diffusion2_150/{label_c}/fake_{img_i}.png').resize((256, 256)).quantize(10, 0).convert('RGB')

    #         img = Image.fromarray(np.concatenate([np.array(img_a), np.array(img_b), np.array(img_c)], axis=1), 'RGB').quantize(256, 1) 
    #         imgs.append(img)

    # imgs[0].save(fp=path_out/f'animation.gif', format='GIF', append_images=imgs[1:], optimize=False, save_all=True, duration=500, loop=0)

    imgs = [] 
    path_root = Path('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/diffusion2_150')
    for img_i in range(50):
        for path_label in path_root.iterdir():
            img = Image.open(path_label/f'fake_{img_i}.png').resize((256, 256))
            imgs.append(img)

    imgs[0].save(fp=path_out/f'animation_histo.gif', format='GIF', append_images=imgs[1:], optimize=False, save_all=True, duration=500, loop=0)


        