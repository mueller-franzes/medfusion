import torch 
from medical_diffusion.models.embedders import TimeEmbbeding, SinusoidalPosEmb, LabelEmbedder

cond_emb = LabelEmbedder(10, num_classes=2)
c = torch.tensor([[0,], [1,]])
v = cond_emb(c)
print(v)


tim_emb = SinusoidalPosEmb(20, max_period=10)
t = torch.tensor([1,2,3, 1000])
v = tim_emb(t)
print(v)

tim_emb = TimeEmbbeding(4*4, SinusoidalPosEmb, {'max_period':10})
t = torch.tensor([1,2,3, 1000])
v = tim_emb(t)
print(v)