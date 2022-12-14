

from pathlib import Path 
from torchvision.utils import save_image
import pandas as pd 
import torch 
import torch.nn.functional as F
from medical_diffusion.data.datasets import CheXpert_Dataset, CheXpert_2_Dataset
import math

path_out = Path().cwd()/'results'/'test'/'CheXpert_2'
path_out.mkdir(parents=True, exist_ok=True)

path_root = Path('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed_tianyu')
labels = pd.read_csv(path_root/'labels/cheXPert_label.csv', index_col='Path')


# Get patients 
# labels['patient'] = labels.index.str.split('/').str[2]
# labels.set_index('patient',drop=True, append=True, inplace=True)

# for c in labels.columns:
#     print(labels[c].value_counts(dropna=False))

ds = CheXpert_2_Dataset(
    path_root=path_root,
)


weights = ds.get_weights()

x = torch.stack([ds[n]['source'] for n in range(4)])
b = x.shape[0]
save_image(x, path_out/'samples_down_0.png', nrwos=int(math.sqrt(b)), normalize=True, scale_each=True )

size_0 = torch.tensor(x.shape[2:])

for i in range(3): 
    new_size = torch.div(size_0, 2**(i+1), rounding_mode='floor' )
    x_i = F.interpolate(x, size=tuple(new_size), mode='nearest', align_corners=None)  
    print(x_i.shape)
    save_image(x_i, path_out/f'samples_down_{i+1}.png', nrwos=int(math.sqrt(b)), normalize=True, scale_each=True)