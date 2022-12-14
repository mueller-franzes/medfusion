import torchio as tio 
from typing import Union, Optional, Sequence
from torchio.typing import TypeTripletInt
from torchio import Subject, Image
from torchio.utils import to_tuple

class CropOrPad_None(tio.CropOrPad):
    def __init__(
            self,
            target_shape: Union[int, TypeTripletInt, None] = None,
            padding_mode: Union[str, float] = 0,
            mask_name: Optional[str] = None,
            labels: Optional[Sequence[int]] = None,
            **kwargs
            ):

            # WARNING: Ugly workaround to allow None values
            if target_shape is not None:
                self.original_target_shape = to_tuple(target_shape, length=3)
                target_shape = [1 if t_s is None else t_s for t_s in target_shape]
            super().__init__(target_shape, padding_mode, mask_name, labels, **kwargs)

    def apply_transform(self, subject: Subject):
        # WARNING: This makes the transformation subject dependent - reverse transformation must be adapted 
        if self.target_shape is not None:
            self.target_shape = [s_s if t_s is None else t_s for t_s, s_s in zip(self.original_target_shape, subject.spatial_shape)]
        return super().apply_transform(subject=subject)


class SubjectToTensor(object):
    """Transforms TorchIO Subjects into a Python dict and changes axes order from TorchIO to Torch"""
    def __call__(self, subject: Subject):
        return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in subject.items()}

class ImageToTensor(object):
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1,-1)