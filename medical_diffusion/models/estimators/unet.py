
import torch 
import torch.nn as nn 
from monai.networks.blocks import UnetOutBlock

from medical_diffusion.models.utils.conv_blocks import BasicBlock, UpBlock, DownBlock, UnetBasicBlock, UnetResBlock, save_add
from medical_diffusion.models.embedders import TimeEmbbeding
from medical_diffusion.models.utils.attention_blocks import SpatialTransformer, LinearTransformer






class UNet(nn.Module):

    def __init__(self, 
            in_ch=1, 
            out_ch=1, 
            spatial_dims = 3,
            hid_chs =    [32, 64, 128,  256],
            kernel_sizes=[ 1,  3,   3,    3],
            strides =    [ 1,  2,   2,   2],
            downsample_kernel_sizes = None, 
            upsample_kernel_sizes = None, 
            act_name=("SWISH", {}),
            norm_name = ("GROUP", {'num_groups':32, "affine": True}),
            time_embedder=TimeEmbbeding,
            time_embedder_kwargs={},
            cond_embedder=None,
            cond_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning = False, 
            dropout=0.0, 
            learnable_interpolation=True,
            use_attention='none',
        ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        if downsample_kernel_sizes is None:
            downsample_kernel_sizes = kernel_sizes
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = strides
        

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 

        # ------------- Condition-Embedder-----------
        if cond_embedder is not None:
            self.cond_embedder=cond_embedder(**cond_embedder_kwargs)
        else:
            self.cond_embedder = None 

        # ----------- In-Convolution ------------
        in_ch = in_ch*2 if self.use_self_conditioning else in_ch 
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims = spatial_dims, 
            in_channels = in_ch, 
            out_channels = hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name,
            norm_name=norm_name,
            emb_channels=time_emb_dim
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
                emb_channels = time_emb_dim
            )
            for i in range(1, self.depth)
        ])

      
     
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
                emb_channels=time_emb_dim,
                skip_channels=hid_chs[i]
            )
            for i in range(self.depth-1)
        ])
        

        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            UnetOutBlock(spatial_dims, hid_chs[i], out_ch, dropout=None) 
            for i in range(1, deep_supervision+1)
        ])
 

    def forward(self, x_t, t=None, condition=None, self_cond=None):
        # x_t [B, C, *]
        # t [B,]
        # condition [B,]
        # self_cond [B, C, *]
        x = [ None for _ in range(len(self.encoders)+1) ]

        # -------- Time Embedding (Global) -----------
        if t is None:
            time_emb = None 
        else:
            time_emb = self.time_embedder(t) # [B, C]

        # -------- Condition Embedding (Global) -----------
        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None  
        else:
            cond_emb = self.cond_embedder(condition) # [B, C]
        
        # ----------- Embedding Summation -------- 
        emb = save_add(time_emb, cond_emb)
       
        # ---------- Self-conditioning-----------
        if self.use_self_conditioning:
            self_cond =  torch.zeros_like(x_t) if self_cond is None else x_t 
            x_t = torch.cat([x_t, self_cond], dim=1)  
    
        # -------- In-Convolution --------------
        x[0] = self.inc(x_t, emb)

        # --------- Encoder --------------
        for i in range(len(self.encoders)):
            x[i+1] = self.encoders[i](x[i], emb)

        # -------- Decoder -----------
        for i in range(len(self.decoders), 0, -1):
            x[i-1] = self.decoders[i-1](x[i], x[i-1], emb)

        # ---------Out-Convolution ------------
        y = self.outc(x[0])
        y_ver = [outc_ver_i(x[i+1]) for i, outc_ver_i in  enumerate(self.outc_ver)]

        return y, y_ver




if __name__=='__main__':
    model = UNet(in_ch=3, use_res_block=False, learnable_interpolation=False)
    input = torch.randn((1,3,16,128,128))
    time = torch.randn((1,))
    out_hor, out_ver = model(input, time)
    print(out_hor[0].shape)