
from pathlib import Path 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.utils import save_image
from monai.networks.blocks import UnetOutBlock


from medical_diffusion.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from medical_diffusion.loss.gan_losses import hinge_d_loss
from medical_diffusion.loss.perceivers import LPIPS
from medical_diffusion.models.model_base import BasicModel, VeryBasicModel


from pytorch_msssim import SSIM, ssim


class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl 






class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, emb_channels, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_channels = emb_channels
        self.beta = beta
  
        self.embedder = nn.Embedding(num_embeddings, emb_channels)
        self.embedder.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        assert z.shape[1] == self.emb_channels, "Channels of z and codebook don't match"
        z_ch = torch.moveaxis(z, 1, -1) # [B, C, *] -> [B, *, C]
        z_flattened = z_ch.reshape(-1, self.emb_channels) # [B, *, C] -> [Bx*, C], Note: or use contiguous() and view()

        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        dist = (    torch.sum(z_flattened**2, dim=1, keepdim=True) 
                 +  torch.sum(self.embedder.weight**2, dim=1)
                -2* torch.einsum("bd,dn->bn", z_flattened, self.embedder.weight.t())
        ) # [Bx*, num_embeddings]

        min_encoding_indices = torch.argmin(dist, dim=1) # [Bx*]
        z_q = self.embedder(min_encoding_indices) # [Bx*, C]
        z_q = z_q.view(z_ch.shape) # [Bx*, C] -> [B, *, C] 
        z_q = torch.moveaxis(z_q, -1, 1) # [B, *, C] -> [B, C, *]
 
        # Compute Embedding Loss 
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
     
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss


  
class Discriminator(nn.Module):
    def __init__(self, 
        in_channels=1, 
        spatial_dims = 3,
        hid_chs =    [32,       64,      128,      256,  512],
        kernel_sizes=[(1,3,3), (1,3,3), (1,3,3),    3,   3],
        strides =    [  1,     (1,2,2), (1,2,2),    2,   2],
        act_name=("Swish", {}),
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=None
        ):
        super().__init__()

        self.inc =  BasicBlock(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=hid_chs[0],  
            kernel_size=kernel_sizes[0], # 2*pad = kernel-stride -> kernel = 2*pad + stride => 1 = 2*0+1, 3, =2*1+1, 2 = 2*0+2, 4 = 2*1+2 
            stride=strides[0], 
            norm_name=norm_name, 
            act_name=act_name, 
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            BasicBlock(
                spatial_dims=spatial_dims, 
                in_channels=hid_chs[i-1], 
                out_channels=hid_chs[i],  
                kernel_size=kernel_sizes[i], 
                stride=strides[i], 
                act_name=act_name, 
                norm_name=norm_name, 
                dropout=dropout)
            for i in range(1, len(hid_chs))
        ])

 
        self.outc =  BasicBlock( 
            spatial_dims=spatial_dims, 
            in_channels=hid_chs[-1], 
            out_channels=1,  
            kernel_size=3, 
            stride=1, 
            act_name=None, 
            norm_name=None, 
            dropout=None,
            zero_conv=True
        )

    

    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        return self.outc(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, 
        in_channels=1, 
        spatial_dims = 3,
        hid_chs =    [64, 128, 256, 512, 512],
        kernel_sizes=[4,   4,   4,  4,   4],
        strides =    [2,   2,   2,  1,   1],
        act_name=("LeakyReLU", {'negative_slope': 0.2}),
        norm_name = ("BATCH", {}),
        dropout=None
        ):
        super().__init__()

        self.inc =  BasicBlock(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=hid_chs[0],  
            kernel_size=kernel_sizes[0], 
            stride=strides[0], 
            norm_name=None, 
            act_name=act_name, 
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            BasicBlock(
                spatial_dims=spatial_dims, 
                in_channels=hid_chs[i-1], 
                out_channels=hid_chs[i],  
                kernel_size=kernel_sizes[i], 
                stride=strides[i], 
                act_name=act_name, 
                norm_name=norm_name, 
                dropout=dropout)
            for i in range(1, len(strides))
        ])


        self.outc =  BasicBlock( 
            spatial_dims=spatial_dims, 
            in_channels=hid_chs[-1], 
            out_channels=1,  
            kernel_size=4, 
            stride=1, 
            norm_name=None, 
            act_name=None, 
            dropout=False,
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        return self.outc(x)




class VQVAE(BasicModel):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims = 2,
        emb_channels = 4,
        num_embeddings = 8192,
        hid_chs =    [32, 64, 128,  256],
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        act_name=("Swish", {}),
        dropout=0.0,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        beta = 0.25,
        embedding_loss_weight=1.0,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        

        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        loss = torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},

        sample_every_n_steps = 1000

    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock 
        self.inc = ConvBlock(spatial_dims, in_channels, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0],
                                  act_name=act_name, norm_name=norm_name)

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims, 
                hid_chs[i-1], 
                hid_chs[i], 
                kernel_sizes[i], 
                strides[i],
                kernel_sizes[i], 
                norm_name,
                act_name,
                dropout,
                use_res_block,
                learnable_interpolation,
                use_attention[i])
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = BasicBlock(spatial_dims, hid_chs[-1], emb_channels, 1)


        # ----------- Quantizer --------------
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, 
            emb_channels=emb_channels,
            beta=beta
        )    

        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name)

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims, 
                hid_chs[i+1], 
                hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=strides[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                skip_channels=0)
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])

    
    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        return z 
            
    def decode(self, z):
        z, _ = self.quantizer(z)
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x 

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)
   
        return out, out_hor[::-1], emb_loss 
    
    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth<2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0 

    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False, 
                       nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))


    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'
        weights = [1/2**i for i in range(1+len(pred_vertical))] # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]

        # Loss
        loss = 0
        loss += torch.mean(self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target))*weights[0]  
        
        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            loss += torch.mean(self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i))*weights[i+1]  

        return loss 

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss*self.embedding_loss_weight

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss':loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images 
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)]) 
            save_image(images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)
    
        return loss



class VQGAN(VeryBasicModel):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims = 2,
        emb_channels = 4,
        num_embeddings = 8192,
        hid_chs =    [ 64, 128,  256, 512],
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        act_name=("Swish", {}),
        dropout=0.0,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        beta = 0.25,
        embedding_loss_weight=1.0,
        perceiver = LPIPS,
        perceiver_kwargs = {},
        perceptual_loss_weight: float = 1.0,
        
        
        start_gan_train_step = 50000, # NOTE step increase with each optimizer 
        gan_loss_weight: float = 1.0, # = discriminator  
        
        optimizer_vqvae=torch.optim.Adam, 
        optimizer_gan=torch.optim.Adam, 
        optimizer_vqvae_kwargs={'lr':1e-6},
        optimizer_gan_kwargs={'lr':1e-6}, 
        lr_scheduler_vqvae= None, 
        lr_scheduler_vqvae_kwargs={},
        lr_scheduler_gan= None, 
        lr_scheduler_gan_kwargs={},

        pixel_loss = torch.nn.L1Loss,
        pixel_loss_kwargs={'reduction':'none'},
        gan_loss_fct = hinge_d_loss,

        sample_every_n_steps = 1000

    ):
        super().__init__()
        self.sample_every_n_steps=sample_every_n_steps
        self.start_gan_train_step = start_gan_train_step
        self.gan_loss_weight = gan_loss_weight
        self.embedding_loss_weight = embedding_loss_weight

        self.optimizer_vqvae = optimizer_vqvae
        self.optimizer_gan = optimizer_gan
        self.optimizer_vqvae_kwargs = optimizer_vqvae_kwargs
        self.optimizer_gan_kwargs = optimizer_gan_kwargs
        self.lr_scheduler_vqvae = lr_scheduler_vqvae
        self.lr_scheduler_vqvae_kwargs = lr_scheduler_vqvae_kwargs
        self.lr_scheduler_gan = lr_scheduler_gan
        self.lr_scheduler_gan_kwargs = lr_scheduler_gan_kwargs

        self.pixel_loss_fct = pixel_loss(**pixel_loss_kwargs)
        self.gan_loss_fct = gan_loss_fct

        self.vqvae = VQVAE(in_channels, out_channels, spatial_dims, emb_channels, num_embeddings, hid_chs, kernel_sizes,
            strides, norm_name, act_name, dropout, use_res_block, deep_supervision, learnable_interpolation, use_attention, 
            beta, embedding_loss_weight, perceiver, perceiver_kwargs, perceptual_loss_weight)

        self.discriminator = nn.ModuleList([Discriminator(in_channels, spatial_dims, hid_chs, kernel_sizes, strides, 
                                            act_name, norm_name, dropout) for i in range(len(self.vqvae.outc_ver)+1)])

    
        # self.discriminator = nn.ModuleList([NLayerDiscriminator(in_channels, spatial_dims) 
        #                                     for _ in range(len(self.vqvae.decoder.outc_ver)+1)])


    
    def encode(self, x):
        return self.vqvae.encode(x) 
            
    def decode(self, z):
        return self.vqvae.decode(z)

    def forward(self, x):
        return self.vqvae.forward(x)
    
    
    def vae_img_loss(self, pred, target, dec_out_layer, step, discriminator, depth=0):
        # ------ VQVAE -------
        rec_loss =  self.vqvae.rec_loss(pred, [], target)

        # ------- GAN ----- 
        if step > self.start_gan_train_step:
            gan_loss = -torch.mean(discriminator[depth](pred))  
            lambda_weight = self.compute_lambda(rec_loss, gan_loss, dec_out_layer)
            gan_loss = gan_loss*lambda_weight

            with torch.no_grad():
                self.log(f"train/gan_loss_{depth}", gan_loss, on_step=True, on_epoch=True)
                self.log(f"train/lambda_{depth}", lambda_weight, on_step=True, on_epoch=True) 
        else:
            gan_loss = 0 #torch.tensor([0.0], requires_grad=True, device=target.device)
    
        return self.gan_loss_weight*gan_loss+rec_loss
    

    def gan_img_loss(self, pred, target, step, discriminators, depth):
        if (step > self.start_gan_train_step) and (depth<len(discriminators)):
            logits_real = discriminators[depth](target.detach())
            logits_fake = discriminators[depth](pred.detach())
            loss = self.gan_loss_fct(logits_real, logits_fake)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=target.device)
        
        with torch.no_grad():
            self.log(f"train/loss_1_{depth}", loss, on_step=True, on_epoch=True) 
        return loss 

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x 

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        weights = [1/2**i for i in range(1+len(pred_vertical))] # horizontal + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]
        logging_dict = {}

        if optimizer_idx == 0:
            # Horizontal/Top Layer 
            img_loss = self.vae_img_loss(pred, target, self.vqvae.outc.conv, step, self.discriminator, 0)*weights[0]

            # Vertical/Deep Layer 
            for i, pred_i in enumerate(pred_vertical): 
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
                img_loss += self.vae_img_loss(pred_i, target_i, self.vqvae.outc_ver[i].conv, step, self.discriminator, i+1)*weights[i+1]
            loss =  img_loss+self.embedding_loss_weight*emb_loss

            with torch.no_grad():
                logging_dict[f'img_loss'] = img_loss
                logging_dict[f'emb_loss'] = emb_loss
                logging_dict['loss_0'] = loss

        elif optimizer_idx == 1:
            # Horizontal/Top Layer 
            loss = self.gan_img_loss(pred, target, step, self.discriminator, 0)*weights[0]

            # Vertical/Deep Layer 
            for i, pred_i in enumerate(pred_vertical):  
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
                loss += self.gan_img_loss(pred_i, target_i, step, self.discriminator, i+1)*weights[i+1]
            
            with torch.no_grad():
                logging_dict['loss_1'] = loss


        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict['loss'] = loss
            logging_dict[f'L2'] = torch.nn.functional.mse_loss(pred, x)
            logging_dict[f'L1'] = torch.nn.functional.l1_loss(pred, x)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0: # NOTE: step 1 (opt1) , step=2 (opt2), step=3 (opt1), ...
            
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images 
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)]) 
            save_image(images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)
        
        return loss 
    
    def configure_optimizers(self):
        opt_vqvae = self.optimizer_vqvae(self.vqvae.parameters(), **self.optimizer_vqvae_kwargs)
        opt_gan = self.optimizer_gan(self.discriminator.parameters(), **self.optimizer_gan_kwargs)
        schedulers = []
        if self.lr_scheduler_vqvae is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_vqvae(opt_vqvae, **self.lr_scheduler_vqvae_kwargs), 
                'interval': 'step',
                'frequency': 1
            })
        if self.lr_scheduler_gan is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_gan(opt_gan, **self.lr_scheduler_gan_kwargs),
                'interval': 'step',
                'frequency': 1
            })
        return [opt_vqvae, opt_gan], schedulers
    
    def compute_lambda(self, rec_loss, gan_loss, dec_out_layer, eps=1e-4):
        """Computes adaptive weight as proposed in eq. 7 of https://arxiv.org/abs/2012.09841"""
        rec_grads = torch.autograd.grad(rec_loss, dec_out_layer.weight, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, dec_out_layer.weight, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + eps) 
        d_weight = torch.clamp(d_weight, 0.0, 1e4)
        return d_weight.detach()



class VAE(BasicModel):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims = 2,
        emb_channels = 4,
        hid_chs =    [ 64, 128,  256, 512],
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name=("Swish", {}),
        dropout=None,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        embedding_loss_weight=1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        

        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        loss = torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},

        sample_every_n_steps = 1000

    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        # self.ssim_fct = SSIM(data_range=1, size_average=False, channel=out_channels, spatial_dims=spatial_dims, nonnegative_ssim=True)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        # -------- Loss-Reg---------
        # self.logvar = nn.Parameter(torch.zeros(size=()) )

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name, 
            norm_name=norm_name,
            emb_channels=None
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = norm_name,
                act_name = act_name,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = None
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2*emb_channels, 3),
            BasicBlock(spatial_dims, 2*emb_channels, 2*emb_channels, 1)
        )


        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()    


        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])
        # self.logvar_ver = nn.ParameterList([
        #     nn.Parameter(torch.zeros(size=()) )
        #     for _ in range(1, deep_supervision+1)
        # ])

    
    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z 
            
    def decode(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x 

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)
   
        return out, out_hor[::-1], emb_loss 
    
    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth<2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0 
    
    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False, 
                        nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))
    
    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0
        rec_loss = self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)
        # rec_loss = rec_loss/ torch.exp(self.logvar) + self.logvar # Note this is include in Stable-Diffusion but logvar is not used in optimizer 
        loss += torch.sum(rec_loss)/pred.shape[0]  
        

        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            rec_loss_i = self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i] 
            loss += torch.sum(rec_loss_i)/pred.shape[0]  

        return loss 

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss*self.embedding_loss_weight
         
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss':loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)
            # logging_dict['logvar'] = self.logvar

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images 
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)]) 
            save_image(images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)
    
        return loss




class VAEGAN(VeryBasicModel):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims = 2,
        emb_channels = 4,
        hid_chs =    [ 64, 128,  256, 512],
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name=("Swish", {}),
        dropout=0.0,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        embedding_loss_weight=1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        
        
        start_gan_train_step = 50000, # NOTE step increase with each optimizer 
        gan_loss_weight: float = 1.0, # = discriminator  
        
        optimizer_vqvae=torch.optim.Adam, 
        optimizer_gan=torch.optim.Adam, 
        optimizer_vqvae_kwargs={'lr':1e-6}, # 'weight_decay':1e-2, {'lr':1e-6, 'betas':(0.5, 0.9)}
        optimizer_gan_kwargs={'lr':1e-6}, # 'weight_decay':1e-2,
        lr_scheduler_vqvae= None, 
        lr_scheduler_vqvae_kwargs={},
        lr_scheduler_gan= None, 
        lr_scheduler_gan_kwargs={},

        pixel_loss = torch.nn.L1Loss,
        pixel_loss_kwargs={'reduction':'none'},
        gan_loss_fct = hinge_d_loss,

        sample_every_n_steps = 1000

    ):
        super().__init__()
        self.sample_every_n_steps=sample_every_n_steps
        self.start_gan_train_step = start_gan_train_step
        self.gan_loss_weight = gan_loss_weight
        self.embedding_loss_weight = embedding_loss_weight

        self.optimizer_vqvae = optimizer_vqvae
        self.optimizer_gan = optimizer_gan
        self.optimizer_vqvae_kwargs = optimizer_vqvae_kwargs
        self.optimizer_gan_kwargs = optimizer_gan_kwargs
        self.lr_scheduler_vqvae = lr_scheduler_vqvae
        self.lr_scheduler_vqvae_kwargs = lr_scheduler_vqvae_kwargs
        self.lr_scheduler_gan = lr_scheduler_gan
        self.lr_scheduler_gan_kwargs = lr_scheduler_gan_kwargs

        self.pixel_loss_fct = pixel_loss(**pixel_loss_kwargs)
        self.gan_loss_fct = gan_loss_fct

        self.vqvae = VAE(in_channels, out_channels, spatial_dims, emb_channels, hid_chs, kernel_sizes,
            strides, norm_name, act_name, dropout, use_res_block, deep_supervision, learnable_interpolation, use_attention, 
            embedding_loss_weight, perceiver, perceiver_kwargs, perceptual_loss_weight)

        self.discriminator = nn.ModuleList([Discriminator(in_channels, spatial_dims, hid_chs, kernel_sizes, strides, 
                                            act_name, norm_name, dropout) for i in range(len(self.vqvae.outc_ver)+1)])

    
        # self.discriminator = nn.ModuleList([NLayerDiscriminator(in_channels, spatial_dims) 
        #                                     for _ in range(len(self.vqvae.outc_ver)+1)])


    
    def encode(self, x):
        return self.vqvae.encode(x) 
            
    def decode(self, z):
        return self.vqvae.decode(z)

    def forward(self, x):
        return self.vqvae.forward(x)
    
    
    def vae_img_loss(self, pred, target, dec_out_layer, step, discriminator, depth=0):
         # ------ VQVAE -------
        rec_loss = self.vqvae.rec_loss(pred, [], target)

        # ------- GAN ----- 
        if (step > self.start_gan_train_step) and (depth<2):
            gan_loss = -torch.sum(discriminator[depth](pred))  # clamp(..., None, 0) => only punish areas that were rated as fake (<0) by discriminator => ensures loss >0 and +- don't cannel out in sum
            lambda_weight = self.compute_lambda(rec_loss, gan_loss, dec_out_layer)
            gan_loss = gan_loss*lambda_weight

            with torch.no_grad():
                self.log(f"train/gan_loss_{depth}", gan_loss, on_step=True, on_epoch=True)
                self.log(f"train/lambda_{depth}", lambda_weight, on_step=True, on_epoch=True) 
        else:
            gan_loss = 0 #torch.tensor([0.0], requires_grad=True, device=target.device)
    


        return self.gan_loss_weight*gan_loss+rec_loss
    
    def gan_img_loss(self, pred, target, step, discriminators, depth):
        if (step > self.start_gan_train_step) and (depth<len(discriminators)):  
            logits_real = discriminators[depth](target.detach())
            logits_fake = discriminators[depth](pred.detach())
            loss = self.gan_loss_fct(logits_real, logits_fake)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=target.device)
        
        with torch.no_grad():
            self.log(f"train/loss_1_{depth}", loss, on_step=True, on_epoch=True) 
        return loss 

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x 

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        logging_dict = {}

        if optimizer_idx == 0:
            # Horizontal/Top Layer 
            img_loss = self.vae_img_loss(pred, target, self.vqvae.outc.conv, step, self.discriminator, 0)

            # Vertical/Deep Layer 
            for i, pred_i in enumerate(pred_vertical): 
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
                img_loss += self.vae_img_loss(pred_i, target_i, self.vqvae.outc_ver[i].conv, step, self.discriminator, i+1)
            loss =  img_loss+self.embedding_loss_weight*emb_loss

            with torch.no_grad():
                logging_dict[f'img_loss'] = img_loss
                logging_dict[f'emb_loss'] = emb_loss
                logging_dict['loss_0'] = loss

        elif optimizer_idx == 1:
            # Horizontal/Top Layer 
            loss = self.gan_img_loss(pred, target, step, self.discriminator, 0)

            # Vertical/Deep Layer 
            for i, pred_i in enumerate(pred_vertical):  
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
                loss += self.gan_img_loss(pred_i, target_i, step, self.discriminator, i+1)
            
            with torch.no_grad():
                logging_dict['loss_1'] = loss


        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict['loss'] = loss
            logging_dict[f'L2'] = torch.nn.functional.mse_loss(pred, x)
            logging_dict[f'L1'] = torch.nn.functional.l1_loss(pred, x)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)
            # logging_dict['logvar'] = self.vqvae.logvar

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0: # NOTE: step 1 (opt1) , step=2 (opt2), step=3 (opt1), ...
            
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images 
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)]) 
            save_image(images, path_out/f'sample_{log_step}.png',  nrow=x.shape[0], normalize=True)
        
        return loss 
    
    def configure_optimizers(self):
        opt_vqvae = self.optimizer_vqvae(self.vqvae.parameters(), **self.optimizer_vqvae_kwargs)
        opt_gan = self.optimizer_gan(self.discriminator.parameters(), **self.optimizer_gan_kwargs)
        schedulers = []
        if self.lr_scheduler_vqvae is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_vqvae(opt_vqvae, **self.lr_scheduler_vqvae_kwargs), 
                'interval': 'step',
                'frequency': 1
            })
        if self.lr_scheduler_gan is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_gan(opt_gan, **self.lr_scheduler_gan_kwargs),
                'interval': 'step',
                'frequency': 1
            })
        return [opt_vqvae, opt_gan], schedulers
    
    def compute_lambda(self, rec_loss, gan_loss, dec_out_layer, eps=1e-4):
        """Computes adaptive weight as proposed in eq. 7 of https://arxiv.org/abs/2012.09841"""
        rec_grads = torch.autograd.grad(rec_loss, dec_out_layer.weight, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, dec_out_layer.weight, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + eps) 
        d_weight = torch.clamp(d_weight, 0.0, 1e4)
        return d_weight.detach()