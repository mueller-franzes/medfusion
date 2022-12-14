
import torch 

from medical_diffusion.models.utils.attention_blocks import LinearTransformer,LinearTransformerNd, SpatialTransformer

from medical_diffusion.external.stable_diffusion.unet_openai import AttentionBlock
from medical_diffusion.external.stable_diffusion.attention import SpatialSelfAttention # similar/equal to Attention used SD-UNet implementation 



torch.manual_seed(0)
input = torch.randn((1, 32, 64, 64)) # 2D 

b, ch, *_ = input.shape 
dim = input.ndim 
# attention  = SpatialTransformer(dim-2, in_channels=ch, out_channels=ch, num_heads=8)
# attention(input)

embedding = input 

torch.manual_seed(0)
attention_a  = LinearTransformer(input.ndim-2, in_channels=ch, out_channels=ch, num_heads=1, ch_per_head=ch, emb_dim=None)
torch.manual_seed(0)
attention_a2  = LinearTransformerNd(input.ndim-2, in_channels=ch, out_channels=ch, num_heads=1, ch_per_head=ch, emb_dim=None)
torch.manual_seed(0)
attention_b  = SpatialSelfAttention(in_channels=ch)
torch.manual_seed(0)
attention_c = AttentionBlock(ch, num_heads=1, num_head_channels=ch)

a = attention_a(input)
a2 = attention_a2(input)
b = attention_b(input)
c = attention_c(input)

print(torch.abs(a-b).max(), torch.abs(a-a2).max(), torch.abs(a-c).max())