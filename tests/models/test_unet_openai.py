
from medical_diffusion.external.stable_diffusion.unet_openai import UNetModel
from medical_diffusion.models.embedders import LabelEmbedder

import torch 


noise_estimator = UNetModel
noise_estimator_kwargs = {}


model  = noise_estimator(**noise_estimator_kwargs)
print(model)

input = torch.randn((1,4,32,32))
time = torch.randn([1,])
cond = None #torch.tensor([0,]) 
out_hor, out_ver = model(input, time, cond)
print(out_hor)