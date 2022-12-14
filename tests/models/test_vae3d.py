import torch 
from medical_diffusion.models.embedders.latent_embedders import VQVAE, VQGAN


input = torch.randn((1, 3, 16, 128, 128)) # [B, C, H, W]


model = VQVAE(in_channels=3, out_channels=3, spatial_dims = 3, emb_channels=1, deep_supervision=True)
# output = model(input)
# print(output)
loss = model._step({'source':input}, 1, 'train', 1, 1)
print(loss)


# model = VQGAN(in_channels=3, out_channels=3, spatial_dims = 3, emb_channels=1, deep_supervision=True)
# # output = model(input)
# # print(output)
# loss = model._step({'source':input}, 1, 'train', 1, 1)
# print(loss)
