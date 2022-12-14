
import torch 
import numpy as np 

class ToTensor16bit(object):
    """PyTorch can not handle uint16 only int16. First transform to int32. Note, this function also adds a channel-dim"""
    def __call__(self, image):
        # return torch.as_tensor(np.array(image, dtype=np.int32)[None]) 
        # return torch.from_numpy(np.array(image, np.int32, copy=True)[None])
        image = np.array(image, np.int32, copy=True) # [H,W,C] or [H,W]
        image = np.expand_dims(image, axis=-1) if image.ndim ==2 else image 
        return torch.from_numpy(np.moveaxis(image, -1, 0)) #[C, H, W]

class Normalize(object):
    """Rescale the image to [0,1] and ensure float32 dtype """

    def __call__(self, image):
        image = image.type(torch.FloatTensor)
        return (image-image.min())/(image.max()-image.min())


class RandomBackground(object):
    """Fill Background (intensity ==0) with random values"""

    def __call__(self, image):
        image[image==0] = torch.rand(*image[image==0].shape) #(image.max()-image.min())
        return image 