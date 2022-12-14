from typing import Optional, Sequence, Tuple, Union, Type

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 


from monai.networks.blocks.dynunet_block import get_padding, get_output_padding
from monai.networks.layers import Pool, Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer, get_dropout_layer
from monai.utils.misc import ensure_tuple_rep

from medical_diffusion.models.utils.attention_blocks import Attention, zero_module

def save_add(*args):
    args = [arg for arg in args if arg is not None]
    return sum(args) if len(args)>0 else None 


class SequentialEmb(nn.Sequential):
    def forward(self, input, emb):
        for module in self:
            input = module(input, emb)
        return input


class BasicDown(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        learnable_interpolation=True,
        use_res=False
        ) -> None:
        super().__init__()

        if learnable_interpolation:
            Convolution = Conv[Conv.CONV, spatial_dims]
            self.down_op = Convolution(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=get_padding(kernel_size, stride),
                dilation=1,
                groups=1,
                bias=True,
            )
            
            if use_res:
                self.down_skip = nn.PixelUnshuffle(2) # WARNING: Only supports 2D, , out_channels == 4*in_channels
    
        else:
            Pooling = Pool['avg', spatial_dims]
            self.down_op = Pooling(
                kernel_size=kernel_size,
                stride=stride,
                padding=get_padding(kernel_size, stride)
            )


    def forward(self, x, emb=None):
        y = self.down_op(x)
        if hasattr(self, 'down_skip'):
            y = y+self.down_skip(x)
        return y 

class BasicUp(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        learnable_interpolation=True,
        use_res=False,
        ) -> None:
        super().__init__()
        self.learnable_interpolation = learnable_interpolation
        if learnable_interpolation:
            # TransConvolution = Conv[Conv.CONVTRANS, spatial_dims]
            # padding = get_padding(kernel_size, stride)
            # output_padding = get_output_padding(kernel_size, stride, padding)
            # self.up_op = TransConvolution(
            #     in_channels,
            #     out_channels,
            #     kernel_size=kernel_size,
            #     stride=stride,
            #     padding=padding,
            #     output_padding=output_padding,
            #     groups=1,
            #     bias=True,
            #     dilation=1
            # )

            self.calc_shape = lambda x: tuple((np.asarray(x)-1)*np.atleast_1d(stride)+np.atleast_1d(kernel_size)
                                            -2*np.atleast_1d(get_padding(kernel_size, stride)))
            Convolution = Conv[Conv.CONV, spatial_dims]
            self.up_op = Convolution(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
        )

            if use_res:
                self.up_skip = nn.PixelShuffle(2) # WARNING: Only supports 2D, out_channels == in_channels/4
        else:
            self.calc_shape = lambda x: tuple((np.asarray(x)-1)*np.atleast_1d(stride)+np.atleast_1d(kernel_size)
                                            -2*np.atleast_1d(get_padding(kernel_size, stride)))
    
    def forward(self, x, emb=None):
        if self.learnable_interpolation:
            new_size = self.calc_shape(x.shape[2:]) 
            x_res = F.interpolate(x, size=new_size, mode='nearest-exact')
            y = self.up_op(x_res)
            if hasattr(self, 'up_skip'):
                y = y+self.up_skip(x)
            return y 
        else:
            new_size = self.calc_shape(x.shape[2:]) 
            return F.interpolate(x, size=new_size, mode='nearest-exact')


class BasicBlock(nn.Module):
    """
    A block that consists of Conv-Norm-Drop-Act, similar to blocks.Convolution. 
    
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        zero_conv: zero out the parameters of the convolution.  
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int]=1,
        norm_name: Union[Tuple, str, None]=None,
        act_name: Union[Tuple, str, None] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        zero_conv: bool = False,
    ):
        super().__init__()
        Convolution = Conv[Conv.CONV, spatial_dims]
        conv = Convolution(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=get_padding(kernel_size, stride),
            dilation=1,
            groups=1,
            bias=True,
        )
        self.conv = zero_module(conv) if zero_conv else conv 
        
        if norm_name is not None:
            self.norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)  
        if dropout is not None:
            self.drop = get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        if act_name is not None:
            self.act = get_act_layer(name=act_name)
        

    def forward(self, inp):
        out = self.conv(inp)
        if hasattr(self, "norm"):
            out = self.norm(out)  
        if hasattr(self, 'drop'):
            out = self.drop(out)
        if hasattr(self, "act"):
            out = self.act(out) 
        return out

class BasicResBlock(nn.Module):
    """
        A block that consists of Conv-Act-Norm + skip. 
    
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        zero_conv: zero out the parameters of the convolution.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int]=1,
        norm_name: Union[Tuple, str, None]=None,
        act_name: Union[Tuple, str, None] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        zero_conv: bool = False
    ):
        super().__init__()
        self.basic_block = BasicBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout, zero_conv) 
        Convolution = Conv[Conv.CONV, spatial_dims]
        self.conv_res = Convolution(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=get_padding(1, stride),
            dilation=1,
            groups=1,
            bias=True,
        ) if in_channels != out_channels else nn.Identity()

    
    def forward(self, inp):
        out = self.basic_block(inp)
        residual = self.conv_res(inp)
        out = out+residual
        return out



class UnetBasicBlock(nn.Module):
    """
    A modified version of monai.networks.blocks.UnetBasicBlock with additional embedding

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        emb_channels: Number of embedding channels 
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int]=1,
        norm_name: Union[Tuple, str]=None,
        act_name: Union[Tuple, str]=None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        emb_channels: int = None,
        blocks = 2
    ):
        super().__init__()
        self.block_seq = nn.ModuleList([
            BasicBlock(spatial_dims, in_channels if i==0 else out_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout, i==blocks-1)
            for i in range(blocks)
        ])
        
        if emb_channels is not None:
            self.local_embedder = nn.Sequential(
                get_act_layer(name=act_name),
                nn.Linear(emb_channels, out_channels),  
            ) 

    def forward(self, x, emb=None):
        # ------------ Embedding ----------
        if emb is not None:
            emb = self.local_embedder(emb) 
            b,c, *_ = emb.shape 
            sp_dim = x.ndim-2
            emb = emb.reshape(b, c, *((1,)*sp_dim) )
            # scale, shift = emb.chunk(2, dim = 1)
            # x = x * (scale + 1) + shift
            # x = x+emb

        # ----------- Convolution ---------
        n_blocks = len(self.block_seq)
        for i, block in enumerate(self.block_seq):
            x = block(x)
            if (emb is not None) and i<n_blocks:
                x += emb 
        return x 


class UnetResBlock(nn.Module):
    """
    A modified version of monai.networks.blocks.UnetResBlock with additional skip connection and embedding

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        emb_channels: Number of embedding channels 
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int]=1,
        norm_name: Union[Tuple, str]=None,
        act_name: Union[Tuple, str]=None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        emb_channels: int = None,
        blocks = 2
    ):
        super().__init__()
        self.block_seq = nn.ModuleList([
            BasicResBlock(spatial_dims, in_channels if i==0 else out_channels, out_channels, kernel_size, stride, norm_name, act_name, dropout, i==blocks-1)
            for i in range(blocks)
        ])

        if emb_channels is not None:
            self.local_embedder = nn.Sequential(
                get_act_layer(name=act_name),
                nn.Linear(emb_channels, out_channels),  
            ) 


    def forward(self, x, emb=None):
        # ------------ Embedding ----------
        if emb is not None:
            emb = self.local_embedder(emb) 
            b,c, *_ = emb.shape 
            sp_dim = x.ndim-2
            emb = emb.reshape(b, c, *((1,)*sp_dim) )
            # scale, shift = emb.chunk(2, dim = 1)
            # x = x * (scale + 1) + shift
            # x = x+emb

        # ----------- Convolution ---------
        n_blocks = len(self.block_seq)
        for i, block in enumerate(self.block_seq):
            x = block(x)
            if (emb is not None) and i<n_blocks-1:
                x += emb 
        return x 



class DownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        downsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_res_block: bool = False,
        learnable_interpolation: bool = True,
        use_attention: str = 'none',
        emb_channels: int = None
    ):
        super(DownBlock, self).__init__()
        enable_down = ensure_tuple_rep(stride, spatial_dims) != ensure_tuple_rep(1, spatial_dims)
        down_out_channels = out_channels if learnable_interpolation and enable_down else in_channels
      
        # -------------- Down ----------------------
        self.down_op = BasicDown(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=downsample_kernel_size,
            stride=stride,
            learnable_interpolation=learnable_interpolation,
            use_res=False
        ) if enable_down else nn.Identity()
       

        # ---------------- Attention -------------
        self.attention = Attention(
            spatial_dims=spatial_dims,
            in_channels=down_out_channels,
            out_channels=down_out_channels,
            num_heads=8,
            ch_per_head=down_out_channels//8,
            depth=1,
            norm_name=norm_name,
            dropout=dropout,
            emb_dim=emb_channels,
            attention_type=use_attention
        )
       
        # -------------- Convolution ----------------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.conv_block = ConvBlock(
            spatial_dims,
            down_out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            emb_channels=emb_channels 
        )

        
    def forward(self, x, emb=None):  
        # ----------- Down ---------
        x = self.down_op(x)
        
        # ----------- Attention -------------
        if self.attention is not None: 
            x = self.attention(x, emb) 

        # ------------- Convolution --------------
        x = self.conv_block(x, emb)

        return x


class UpBlock(nn.Module):
    def __init__(
        self, 
        spatial_dims, 
        in_channels: int, 
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_res_block: bool = False,
        learnable_interpolation: bool = True,
        use_attention: str = 'none',
        emb_channels: int = None, 
        skip_channels: int = 0
    ):
        super(UpBlock, self).__init__()
        enable_up = ensure_tuple_rep(stride, spatial_dims) != ensure_tuple_rep(1, spatial_dims)
        skip_out_channels = out_channels if learnable_interpolation and enable_up else in_channels+skip_channels
        self.learnable_interpolation = learnable_interpolation     
        

        # -------------- Up ----------------------
        self.up_op = BasicUp(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=stride,
            learnable_interpolation=learnable_interpolation,
            use_res=False
        ) if enable_up else nn.Identity()

        # ---------------- Attention -------------
        self.attention = Attention(
            spatial_dims=spatial_dims,
            in_channels=skip_out_channels,
            out_channels=skip_out_channels,
            num_heads=8,
            ch_per_head=skip_out_channels//8,
            depth=1,
            norm_name=norm_name,
            dropout=dropout,
            emb_dim=emb_channels,
            attention_type=use_attention
        )

    
        # -------------- Convolution ----------------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.conv_block = ConvBlock(
            spatial_dims,
            skip_out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            emb_channels=emb_channels
        )



    def forward(self, x_enc, x_skip=None, emb=None): 
        # ----------- Up -------------
        x = self.up_op(x_enc)

        # ----------- Skip Connection ------------
        if x_skip is not None:
            if self.learnable_interpolation: # Channel of x_enc and x_skip are equal and summation is possible 
                x = x+x_skip
            else:
                x = torch.cat((x, x_skip), dim=1)

        # ----------- Attention -------------
        if self.attention is not None: 
            x = self.attention(x, emb)      

        # ----------- Convolution ------------
        x = self.conv_block(x, emb)

        return x
