from pathlib import Path 
import torch 
from medical_diffusion.models.embedders.latent_embedders import VQVAE, VQGAN, VAE, VAEGAN
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

path_root = Path('runs/2022_12_01_210017_patho_vaegan')

# Load model 
model = VAEGAN.load_from_checkpoint(path_root/'last.ckpt')
# model = torch.load(path_root/'last.ckpt') 



# Save model-part 
# torch.save(model.vqvae, path_root/'last_vae.ckpt') # Not working 
# ------ Ugly workaround ----------
checkpointing = ModelCheckpoint()
trainer = Trainer(callbacks=[checkpointing])
trainer.strategy._lightning_module = model.vqvae 
trainer.model = model.vqvae 
trainer.save_checkpoint(path_root/'last_vae.ckpt')
# -----------------

model = VAE.load_from_checkpoint(path_root/'last_vae.ckpt')
# model = torch.load(path_root/'last_vae.ckpt')  # load_state_dict