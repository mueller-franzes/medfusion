
import torch 

from medical_diffusion.models.utils.attention_blocks import LinearTransformer, SpatialTransformer


input = torch.randn((1, 32, 16, 64, 64)) # 3D 
input = torch.randn((1, 32, 64, 64)) # 2D 

b, ch, *_ = input.shape 
dim = input.ndim 
# attention  = SpatialTransformer(dim-2, in_channels=ch, out_channels=ch, num_heads=8)
# attention(input)

embedding = input 
embedding = None 
emb_dim = embedding.shape[1] if embedding is not None else None 
attention  = LinearTransformer(input.ndim-2, in_channels=ch, out_channels=ch, num_heads=3, emb_dim=emb_dim)
attention  = SpatialTransformer(input.ndim-2, in_channels=ch, out_channels=ch, num_heads=3, emb_dim=emb_dim)

print(attention(input, embedding))