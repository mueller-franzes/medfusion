

from typing import Optional, Tuple, Union
from pathlib import Path 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from .unet_blocks import UNetMidBlock2D, get_down_block, get_up_block
from .taming_discriminator import NLayerDiscriminator
from medical_diffusion.models import BasicModel
from torchvision.utils import save_image

from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i+1]
            is_final_block = False #i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i+1]

            is_final_block = False # i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        sample = z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.batch_size = parameters.shape[0]
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        device = self.parameters.device
        sample_device = "cpu" if device.type == "mps" else device
        sample = torch.randn(self.mean.shape, generator=generator, device=sample_device).to(device)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)/self.batch_size
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                )/self.batch_size

        # q_z_x = Normal(self.mean, self.logvar.mul(.5).exp())
        # p_z = Normal(torch.zeros_like(self.mean), torch.ones_like(self.logvar))
        # kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        # return kl_div

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class VQModel(nn.Module):
    r"""VQ-VAE model from the paper Neural Discrete Representation Learning by Aaron van den Oord, Oriol Vinyals and Koray
    Kavukcuoglu.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
    """


    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )

        self.quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)
        self.quantize = VectorQuantizer(
            num_vq_embeddings, latent_channels, beta=0.25, remap=None, sane_index_shape=False
        )
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

    # def encode(self, x: torch.FloatTensor):
    #     z = self.encoder(x)
    #     z = self.quant_conv(z)
    #     return z
    
    def encode(self, x, return_loss=True, force_quantize= True):
        z = self.encoder(x)
        z = self.quant_conv(z)

        if force_quantize:
            z_q, emb_loss, _ = self.quantize(z)
        else:
            z_q, emb_loss = z, None 

        if return_loss:
            return z_q, emb_loss
        else:
            return z_q
        
    def decode(self, z_q)  -> torch.FloatTensor:
        z_q = self.post_quant_conv(z_q)
        x = self.decoder(z_q)
        return x 

    # def decode(self, z: torch.FloatTensor, return_loss=True, force_quantize: bool = True)  -> torch.FloatTensor:
    #     if force_quantize:
    #         z_q, emb_loss, _ = self.quantize(z)
    #     else:
    #         z_q, emb_loss = z, None 

    #     z_q = self.post_quant_conv(z_q)
    #     x = self.decoder(z_q)

    #     if return_loss:
    #         return x, emb_loss
    #     else:
    #         return x

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
        """
        # h = self.encode(sample)
        h, emb_loss = self.encode(sample)
        dec = self.decode(h)
        # dec, emb_loss = self.decode(h)

        return dec, emb_loss


class AutoencoderKL(nn.Module):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
    """


    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D","DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (32, 64, 128, 128),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        sample_size: int = 32,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = torch.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x: torch.FloatTensor):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
        """
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        dec = self.decode(z)
        return dec, kl_loss



class VQVAEWrapper(BasicModel):
    def __init__(
        self, 
        in_ch: int = 3,
        out_ch: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (32, 64, 128, 256, ),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 64,
        norm_num_groups: int = 32, 

        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={}, 
        lr_scheduler=None, 
        lr_scheduler_kwargs={}, 
        loss=torch.nn.MSELoss, 
        loss_kwargs={}
        ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, loss, loss_kwargs)
        self.model = VQModel(in_ch, out_ch, down_block_types, up_block_types, block_out_channels,
                            layers_per_block, act_fn, latent_channels, sample_size, num_vq_embeddings, norm_num_groups) 
    
    def forward(self, sample):
        return self.model(sample) 

    def encode(self, x):
        z = self.model.encode(x, return_loss=False)  
        return z
    
    def decode(self, z):
        x = self.model.decode(z)
        return x

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x

        # ------------------------- Run Model ---------------------------
        pred, vq_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.loss_fct(pred, target)
        loss += vq_loss
         
        # --------------------- Compute Metrics  -------------------------------
        results = {'loss':loss}
        with torch.no_grad():
            results['L2'] = torch.nn.functional.mse_loss(pred, target)
            results['L1'] = torch.nn.functional.l1_loss(pred, target)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            def norm(x):
                return (x-x.min())/(x.max()-x.min())

            images = [x, pred]
            log_step = self.global_step // self.trainer.log_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            images = torch.cat([norm(img) for img in images])
            save_image(images, path_out/f'sample_{log_step}.png')
    
        return loss

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake)))
    return d_loss

class VQGAN(BasicModel):
    def __init__(
        self, 
        in_ch: int = 3,
        out_ch: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (32, 64, 128, 256, ),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 64,
        norm_num_groups: int = 32, 

        start_gan_train_step = 50000, # NOTE step increase with each optimizer 
        gan_loss_weight: float = 1.0, # alias discriminator  
        perceptual_loss_weight: float = 1.0,
        embedding_loss_weight: float = 1.0,
                
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={}, 
        lr_scheduler=None, 
        lr_scheduler_kwargs={}, 
        loss=torch.nn.MSELoss, 
        loss_kwargs={}
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, loss, loss_kwargs)
        self.vqvae = VQModel(in_ch, out_ch, down_block_types, up_block_types, block_out_channels, layers_per_block, act_fn, 
                                  latent_channels, sample_size, num_vq_embeddings, norm_num_groups)
        self.discriminator = NLayerDiscriminator(in_ch) 
        # self.perceiver = ... # Currently not supported, would require another trained NN 

        self.start_gan_train_step = start_gan_train_step
        self.perceptual_loss_weight = perceptual_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.embedding_loss_weight = embedding_loss_weight
    
    def forward(self, x, condition=None):
        return self.vqvae(x)

    def encode(self, x):
        z = self.vqvae.encode(x, return_loss=False)  
        return z
    
    def decode(self, z):
        x = self.vqvae.decode(z)
        return x


    def compute_lambda(self, rec_loss, gan_loss, eps=1e-4):
        """Computes adaptive weight as proposed in eq. 7 of https://arxiv.org/abs/2012.09841"""
        last_layer = self.vqvae.decoder.conv_out.weight
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + eps) 
        d_weight = torch.clamp(d_weight, 0.0, 1e4)
        return d_weight.detach()



    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        x = batch['source']
        # condition = batch.get('target', None)

        pred, vq_emb_loss = self.vqvae(x)

        if optimizer_idx == 0:
            # ------ VAE -------
            vq_img_loss = F.mse_loss(pred, x)
            vq_per_loss = 0.0 #self.perceiver(pred, x) 
            rec_loss = vq_img_loss+self.perceptual_loss_weight*vq_per_loss

            # ------- GAN ----- 
            if step > self.start_gan_train_step:
                gan_loss = -torch.mean(self.discriminator(pred))
                lambda_weight = self.compute_lambda(rec_loss, gan_loss)
                gan_loss = gan_loss*lambda_weight
            else:
                gan_loss = torch.tensor([0.0], requires_grad=True, device=x.device)

            loss =  self.gan_loss_weight*gan_loss+rec_loss+self.embedding_loss_weight*vq_emb_loss

        elif optimizer_idx == 1:
            if step > self.start_gan_train_step//2:
                logits_real = self.discriminator(x.detach())
                logits_fake = self.discriminator(pred.detach())
                loss = hinge_d_loss(logits_real, logits_fake)
            else:
                loss = torch.tensor([0.0], requires_grad=True, device=x.device)

        # --------------------- Compute Metrics  -------------------------------
        results = {'loss':loss.detach(), f'loss_{optimizer_idx}':loss.detach()}
        with torch.no_grad():
            results[f'L2'] = torch.nn.functional.mse_loss(pred, x)
            results[f'L1'] = torch.nn.functional.l1_loss(pred, x)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.trainer.log_every_n_steps == 0: # NOTE: step 1 (opt1) , step=2 (opt2), step=3 (opt1), ...
            def norm(x):
                return (x-x.min())/(x.max()-x.min())

            images = torch.cat([x, pred])
            log_step = self.global_step // self.trainer.log_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            images = torch.stack([norm(img) for img in images])
            save_image(images, path_out/f'sample_{log_step}.png')
        
        return loss 
    
    def configure_optimizers(self):
        opt_vae = self.optimizer(self.vqvae.parameters(), **self.optimizer_kwargs)
        opt_disc = self.optimizer(self.discriminator.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            scheduler = [
                {
                    'scheduler': self.lr_scheduler(opt_vae, **self.lr_scheduler_kwargs), 
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': self.lr_scheduler(opt_disc, **self.lr_scheduler_kwargs),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
        else:
            scheduler = []

        return [opt_vae, opt_disc], scheduler
    
class VAEWrapper(BasicModel):
    def __init__(
        self, 
        in_ch: int = 3,
        out_ch: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"), # "DownEncoderBlock2D", "DownEncoderBlock2D",
        up_block_types: Tuple[str] = ("UpDecoderBlock2D", "UpDecoderBlock2D","UpDecoderBlock2D" ), # "UpDecoderBlock2D", "UpDecoderBlock2D",
        block_out_channels: Tuple[int] = (32, 64, 128, 256), #  128, 256
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        sample_size: int = 32,

        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-3, 'amsgrad':True}, 
        lr_scheduler=None, 
        lr_scheduler_kwargs={}, 
        # loss=torch.nn.MSELoss, # WARNING: No Effect 
        # loss_kwargs={'reduction': 'mean'}
        ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs ) # loss, loss_kwargs
        self.model = AutoencoderKL(in_ch, out_ch, down_block_types, up_block_types, block_out_channels,
                            layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size) 
    
        self.logvar = nn.Parameter(torch.zeros(size=())) # Better weighting between KL and MSE, see (https://arxiv.org/abs/1903.05789), also used by Taming-Transfomer/Stable Diffusion 

    def forward(self, sample):
        return self.model(sample) 

    def encode(self, x):
        z = self.model.encode(x) # Latent space but not yet mapped to discrete embedding vectors  
        return z.sample(generator=None)
    
    def decode(self, z):
        x = self.model.decode(z)
        return x

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x
        HALF_LOG_TWO_PI = 0.91893 # log(2pi)/2

        # ------------------------- Run Model ---------------------------
        pred, kl_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = torch.sum( torch.square(pred-target))/x.shape[0]  #torch.sum( torch.square((pred-target)/torch.exp(self.logvar))/2 + self.logvar + HALF_LOG_TWO_PI )/x.shape[0] 
        loss += kl_loss
         
        # --------------------- Compute Metrics  -------------------------------
        results = {'loss':loss.detach()}
        with torch.no_grad():
            results['L2'] = torch.nn.functional.mse_loss(pred, target)
            results['L1'] = torch.nn.functional.l1_loss(pred, target)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            def norm(x):
                return (x-x.min())/(x.max()-x.min())

            images = torch.cat([x, pred])
            log_step = self.global_step // self.trainer.log_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            images = torch.stack([norm(img) for img in images])
            save_image(images, path_out/f'sample_{log_step}.png')
    
        return loss