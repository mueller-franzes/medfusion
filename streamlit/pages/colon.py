import streamlit as st
import torch 
import numpy as np 

from medical_diffusion.models.pipelines import DiffusionPipeline

st.title("Colon histology images", anchor=None)
st.sidebar.markdown("Medfusion for colon histology image generation")
st.header('Information')
st.markdown('Medfusion was trained on the [CRC-DX](https://zenodo.org/record/3832231#.Y29uInbMKbg) dataset')



st.header('Settings')
n_samples = st.number_input("Samples", min_value=1, max_value=25, value=4)
steps = st.number_input("Sampling steps", min_value=1, max_value=999, value=50)
guidance_scale = st.number_input("Guidance scale", min_value=1, max_value=10, value=1)
seed = st.number_input("Seed", min_value=0, max_value=None, value=1)
cond_str = st.radio("Microsatellite stable", ('Yes', 'No'), index=1, help="Conditioned on 'microsatellite stable (MSS)' or 'microsatellite instable (MSI)'", horizontal=True)
torch.manual_seed(seed)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

@st.cache(allow_output_mutation = True)  
def init_pipeline():
    pipeline = DiffusionPipeline.load_from_checkpoint('runs/2022_12_02_174623_patho_diffusion/last.ckpt')
    return pipeline

if st.button(f'Sample (using {device_str})'):
    cond = {'Yes':1, 'No':0}[cond_str]
    condition = torch.tensor([cond]*n_samples, device=device)
    un_cond = torch.tensor([1-cond]*n_samples, device=device)

    pipeline = init_pipeline()
    pipeline.to(device)
    images = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=guidance_scale, condition=condition, un_cond=un_cond, steps=steps, use_ddim=True )

    images = images.clamp(-1, 1)
    images = images.cpu().numpy() # [B, C, H, W]
    images = (images+1)/2  # Transform from [-1, 1] to [0, 1]

    images = [np.moveaxis(img, 0, -1) for img in images]
    st.image(images, channels="RGB", output_format='png') # expects (w,h,3) 