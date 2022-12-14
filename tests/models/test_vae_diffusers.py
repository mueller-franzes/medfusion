
import torch 
from medical_diffusion.external.diffusers.vae import VQModel, VQVAEWrapper, VAEWrapper


# model = AutoencoderKL(in_channels=3, out_channels=3)

input = torch.randn((1, 3, 128, 128)) # [B, C, H, W]

# model = VQModel(in_channels=3, out_channels=3)
# output = model(input, sample_posterior=True)
# print(output)

model = VQVAEWrapper(in_ch=3, out_ch=3)
output = model(input)
print(output)




# model = VAEWrapper(in_ch=3, out_ch=3)
# output = model(input)
# print(output)