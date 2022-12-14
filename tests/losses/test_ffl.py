from medical_diffusion.loss.ffl_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class

import torch
fake = torch.randn(4, 3, 64, 64)  # replace it with the predicted tensor of shape (N, C, H, W)
real = torch.randn(4, 3, 64, 64)  # replace it with the target tensor of shape (N, C, H, W)

loss = ffl(fake, real)  # calculate focal frequency loss
print(loss)
