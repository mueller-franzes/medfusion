
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.embedders import LabelEmbedder

import torch 

cond_embedder = LabelEmbedder
cond_embedder_kwargs = {
    'emb_dim': 64,
    'num_classes':2
}

noise_estimator = UNet
noise_estimator_kwargs = {
    'in_ch':3, 
    'out_ch':3, 
    'spatial_dims':2,
    'hid_chs':     [32, 64, 128,  256],
    'kernel_sizes': [ 1,  3,   3,    3],
    'strides':    [ 1,  2,   2,   2],
    # 'kernel_sizes':[(1,3,3), (1,3,3), (1,3,3),    3,   3],
    # 'strides':[  1,     (1,2,2), (1,2,2),    2,   2],
    # 'kernel_sizes':[3, 3, 3, 3, 3],
    # 'strides':     [1, 2, 2, 2, 2],
    'cond_embedder':cond_embedder,
    'cond_embedder_kwargs': cond_embedder_kwargs,
    'use_attention': 'linear', #['none', 'spatial', 'spatial', 'spatial', 'linear'],
    }


model  = UNet(**noise_estimator_kwargs)
# print(model)

input = torch.randn((1,3,256,256))
time = torch.randn([1,])
cond = torch.tensor([0,]) 
out_hor, out_ver = model(input, time, cond)
# print(out_hor)