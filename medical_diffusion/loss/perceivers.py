

import lpips
import torch

class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""
    def __init__(self, linear_calibration=False, normalize=False):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net='vgg', lpips=linear_calibration) # Note: only 'vgg' valid as loss  
        self.normalize = normalize # If true, normalize [0, 1] to [-1, 1]
        

    def forward(self, pred, target):
        # No need to do that because ScalingLayer was introduced in version 0.1 which does this indirectly  
        # if pred.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB
        #     pred = torch.concat([pred, pred, pred], dim=1)
        # if target.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB 
        #     target = torch.concat([target, target, target], dim=1)

        if pred.ndim == 5: # 3D Image: Just use 2D model and compute average over slices 
            depth = pred.shape[2] 
            losses = torch.stack([self.loss_fn(pred[:,:,d], target[:,:,d], normalize=self.normalize) for d in range(depth)], dim=2)
            return torch.mean(losses, dim=2, keepdim=True)
        else:
            return self.loss_fn(pred, target, normalize=self.normalize)
 