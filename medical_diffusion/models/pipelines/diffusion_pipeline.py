

from pathlib import Path 
from tqdm import tqdm

import torch 
import torch.nn.functional as F 
from torchvision.utils import save_image 
import streamlit as st

from medical_diffusion.models import BasicModel
from medical_diffusion.utils.train_utils import EMAModel
from medical_diffusion.utils.math_utils import kl_gaussians






class DiffusionPipeline(BasicModel):
    def __init__(self, 
        noise_scheduler,
        noise_estimator,
        latent_embedder=None,
        noise_scheduler_kwargs={},
        noise_estimator_kwargs={},
        latent_embedder_checkpoint='',
        estimator_objective = 'x_T', # 'x_T' or 'x_0'
        estimate_variance=False, 
        use_self_conditioning=False, 
        classifier_free_guidance_dropout=0.5, # Probability to drop condition during training, has only an effect for label-conditioned training 
        num_samples = 4,
        do_input_centering = True, # Only for training
        clip_x0=True, # Has only an effect during traing if use_self_conditioning=True, import for inference/sampling  
        use_ema = False,
        ema_kwargs = {},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4}, # stable-diffusion ~ 1e-4
        lr_scheduler= None, # stable-diffusion - LambdaLR
        lr_scheduler_kwargs={}, 
        loss=torch.nn.L1Loss,
        loss_kwargs={},
        sample_every_n_steps = 1000
        ):
        # self.save_hyperparameters(ignore=['noise_estimator', 'noise_scheduler']) 
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.loss_fct = loss(**loss_kwargs)
        self.sample_every_n_steps=sample_every_n_steps

        noise_estimator_kwargs['estimate_variance'] = estimate_variance
        noise_estimator_kwargs['use_self_conditioning'] = use_self_conditioning

        self.noise_scheduler = noise_scheduler(**noise_scheduler_kwargs)
        self.noise_estimator = noise_estimator(**noise_estimator_kwargs)
        
        with torch.no_grad():
            if latent_embedder is not None:
                self.latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint)
                for param in self.latent_embedder.parameters():
                    param.requires_grad = False
            else:
                self.latent_embedder = None 

        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.num_samples = num_samples
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.do_input_centering = do_input_centering
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)



    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        results = {}
        x_0 = batch['source']
        condition = batch.get('target', None) 

        # Embed into latent space or normalize 
        if self.latent_embedder is not None:
            self.latent_embedder.eval() 
            with torch.no_grad():
                x_0 = self.latent_embedder.encode(x_0)
        
        if self.do_input_centering:
            x_0 = 2*x_0-1 # [0, 1] -> [-1, 1]

        # if self.clip_x0:
        #     x_0 = torch.clamp(x_0, -1, 1)
        

        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0,T-1] and compute x_t (noisy version of x_0 at t)
            x_t, x_T, t = self.noise_scheduler.sample(x_0) 
                
        # Use EMA Model
        if self.use_ema and (state != 'train'):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Re-estimate x_T or x_0, self-conditioned on previous estimate 
        self_cond = None 
        if self.use_self_conditioning:
            with torch.no_grad():
                pred, pred_vertical = noise_estimator(x_t, t, condition, None) 
                if self.estimate_variance:
                    pred, _ =  pred.chunk(2, dim = 1)  # Seperate actual prediction and variance estimation 
                if self.estimator_objective == "x_T": # self condition on x_0 
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0": # self condition on x_T 
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")
            
        # Classifier free guidance 
        if torch.rand(1)<self.classifier_free_guidance_dropout:
            condition = None 
       
        # Run Denoise 
        pred, pred_vertical = noise_estimator(x_t, t, condition, self_cond) 
        
        # Separate variance (scale) if it was learned 
        if self.estimate_variance:
            pred, pred_var =  pred.chunk(2, dim = 1)  # Separate actual prediction and variance estimation 

        # Specify target 
        if self.estimator_objective == "x_T":
            target = x_T 
        elif self.estimator_objective == "x_0":
            target = x_0 
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        
        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        loss = 0
        weights = [1/2**i for i in range(1+len(pred_vertical))] # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]

        # ----------------- MSE/L1, ... ----------------------
        loss += self.loss_fct(pred, target)*weights[0]

        # ----------------- Variance Loss --------------
        if self.estimate_variance:
            # var_scale = var_scale.clamp(-1, 1) # Should not be necessary 
            var_scale = (pred_var+1)/2 # Assumed to be in [-1, 1] -> [0, 1] 
            pred_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=var_scale)
            # pred_logvar = pred_var  # If variance is estimated directly 

            if  self.estimator_objective == 'x_T':
                pred_x_0 = self.noise_scheduler.estimate_x_0(x_t, x_T, t, clip_x0=self.clip_x0)
            elif self.estimator_objective == "x_0":
                pred_x_0 = pred 
            else:
                raise NotImplementedError()

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, x_0, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=0)
            
            kl_loss = torch.mean(kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar), dim=list(range(1, x_0.ndim)))
            nnl_loss = torch.mean(F.gaussian_nll_loss(pred_x_0, x_0, torch.exp(pred_logvar), reduction='none'), dim=list(range(1, x_0.ndim)))
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss
            
            results['variance_scale'] = torch.mean(var_scale)
            results['variance_loss'] = var_loss

            
        # ----------------------------- Deep Supervision -------------------------
        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            loss += self.loss_fct(pred_i, target_i)*weights[i+1]
        results['loss']  = loss

       
       
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            results['L2'] = F.mse_loss(pred, target)
            results['L1'] = F.l1_loss(pred, target)
            # results['SSIM'] = SSIMMetric(data_range=pred.max()-pred.min(), spatial_dims=source.ndim-2)(pred, target)

            # for i, pred_i in enumerate(pred_vertical):
            #     target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            #     results[f'L1_{i}'] = F.l1_loss(pred_i, target_i).detach()
              
       

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x_0.shape[0], on_step=True, on_epoch=True)           
        
        
        #------------------ Log Image -----------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            dataformats =  'NHWC' if x_0.ndim == 5 else 'HWC'
            def norm(x):
                return (x-x.min())/(x.max()-x.min())

            sample_cond = condition[0:self.num_samples] if condition is not None else None
            sample_img = self.sample(num_samples=self.num_samples, img_size=x_0.shape[1:], condition=sample_cond).detach()
             
            log_step = self.global_step // self.sample_every_n_steps
            # self.logger.experiment.add_images("predict_img", norm(torch.moveaxis(pred[0,-1:], 0,-1)), global_step=self.current_epoch, dataformats=dataformats) 
            # self.logger.experiment.add_images("target_img", norm(torch.moveaxis(target[0,-1:], 0,-1)), global_step=self.current_epoch, dataformats=dataformats) 
            
            # self.logger.experiment.add_images("source_img", norm(torch.moveaxis(x_0[0,-1:], 0,-1)), global_step=log_step, dataformats=dataformats) 
            # self.logger.experiment.add_images("sample_img", norm(torch.moveaxis(sample_img[0,-1:], 0,-1)), global_step=log_step, dataformats=dataformats) 
            
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 32 images 
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = depth2batch(sample_img)[:32]
            save_image(images, path_out/f'sample_{log_step}.png', normalize=True)
        
        
        return loss

    
    def forward(self, x_t, t, condition=None, self_cond=None, guidance_scale=1.0, cold_diffusion=False, un_cond=None):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition is not None) and (guidance_scale != 1.0):
            # Model prediction 
            pred_uncond, _ = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond)
            pred_cond, _ = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond =  pred_uncond.chunk(2, dim = 1)  
                pred_cond,   pred_var_cond =  pred_cond.chunk(2, dim = 1) 
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            pred, _ =  noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var =  pred.chunk(2, dim = 1)  

        if self.estimate_variance:
            pred_var_scale = pred_var/2+0.5 # [-1, 1] -> [0, 1]
            pred_var_value = pred_var  
        else:
            pred_var_scale = 0
            pred_var_value = None 

        # pred_var_scale = pred_var_scale.clamp(0, 1)

        if  self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T 
        elif self.estimator_objective == 'x_T':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = pred 
            self_cond = x_0 
        else:
            raise ValueError("Unknown Objective")
        
        return x_t_prior, x_0, x_T, self_cond 


    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition=None, use_ddim=True, **kwargs):
        self_cond = None 

        # ---------- run denoise loop ---------------
        if use_ddim:
            steps = self.noise_scheduler.timesteps if steps is None else steps
            timesteps_array = torch.linspace(0, self.noise_scheduler.T-1, steps, dtype=torch.long, device=x_t.device) # [0, 1, 2, ..., T-1] if steps = T 
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)] # [0, ...,T-1] (target time not time of x_t)
            
        st_prog_bar = st.progress(0)
        for i, t in tqdm(enumerate(reversed(timesteps_array))):
            st_prog_bar.progress((i+1)/len(timesteps_array))

            # UNet prediction 
            x_t, x_0, x_T, self_cond = self(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None  
        
            if use_ddim and (steps-i-1>0):
                t_next = timesteps_array[steps-i-2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # ------ Eventually decode from latent space into image space--------
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)
        
        return x_t # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, num_samples, img_size, condition=None, **kwargs):
        template = torch.zeros((num_samples, *img_size), device=self.device)
        x_T = self.noise_scheduler.x_final(template)
        x_0 = self.denoise(x_T, condition=condition, **kwargs)
        return x_0 


    @torch.no_grad()
    def interpolate(self, img1, img2, i = None, condition=None, lam = 0.5, **kwargs):
        assert img1.shape == img2.shape, "Image 1 and 2 must have equal shape"

        t = self.noise_scheduler.T-1 if i is None else i
        t = torch.full(img1.shape[:1], i, device=img1.device)

        img1_t = self.noise_scheduler.estimate_x_t(img1, t=t, clip_x0=self.clip_x0)
        img2_t = self.noise_scheduler.estimate_x_t(img2, t=t, clip_x0=self.clip_x0)

        img = (1 - lam) * img1_t + lam * img2_t
        img = self.denoise(img, i, condition, **kwargs)
        return img

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.noise_estimator.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]