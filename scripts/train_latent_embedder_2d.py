




from pathlib import Path
from datetime import datetime

import torch 
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import AIROGSDataset, MSIvsMSS_2_Dataset, CheXpert_2_Dataset
from medical_diffusion.models.embedders.latent_embedders import VQVAE, VQGAN, VAE, VAEGAN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None


    # ------------ Load Data ----------------
    # ds_1 = AIROGSDataset( #  256x256
    #     crawler_ext='jpg',
    #     augment_horizontal_flip=True,
    #     augment_vertical_flip=True,
    #     # path_root='/home/gustav/Documents/datasets/AIROGS/dataset',
    #     path_root='/mnt/hdd/datasets/eye/AIROGS/data_256x256',
    # )

    # ds_2 = MSIvsMSS_2_Dataset( #  512x512
    #     # image_resize=256,    
    #     crawler_ext='jpg',
    #     augment_horizontal_flip=True,
    #     augment_vertical_flip=True,
    #     # path_root='/home/gustav/Documents/datasets/Kather_2/train'
    #     path_root='/mnt/hdd/datasets/pathology/kather_msi_mss_2/train/'
    # )

    ds_3 = CheXpert_2_Dataset( #  256x256
        # image_resize=128, 
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        # path_root = '/home/gustav/Documents/datasets/CheXpert/preprocessed_tianyu'
        path_root = '/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed_tianyu'
    )

    # ds = ConcatDataset([ds_1, ds_2, ds_3])
   
    dm = SimpleDataModule(
        ds_train = ds_3,
        batch_size=8, 
        # num_workers=0,
        pin_memory=True
    ) 
    

    # ------------ Initialize Model ------------
    model = VAE(
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs =    [ 64, 128, 256,  512], 
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,    2],
        deep_supervision=1,
        use_attention= 'none',
        loss = torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-6},
        embedding_loss_weight=1e-6
    )

    # model.load_pretrained(Path.cwd()/'runs/2022_12_01_183752_patho_vae/last.ckpt', strict=True)

    # model = VAEGAN(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=8,
    #     spatial_dims=2,
    #     hid_chs =    [ 64, 128, 256,  512],
    #     deep_supervision=1,
    #     use_attention= 'none',
    #     start_gan_train_step=-1,
    #     embedding_loss_weight=1e-6
    # )

    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_11_25_082209_chest_vae/last.ckpt')
    # model.load_pretrained(Path.cwd()/'runs/2022_11_25_232957_patho_vaegan/last.ckpt')


    # model = VQVAE(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=2,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention = 'none',
    # )


    # model = VQGAN(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=2,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     start_gan_train_step=-1,
    #     pixel_loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention='none',
    # )
    
    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_12_13_093727_patho_vqvae/last.ckpt')
    

    # -------------- Training Initialization ---------------
    to_monitor = "train/L1"  # "val/loss" 
    min_max = "min"
    save_and_sample_every = 50

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


