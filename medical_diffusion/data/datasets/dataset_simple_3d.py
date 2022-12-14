
import torch.utils.data as data 
from pathlib import Path 
from torchvision import transforms as T


import torchio as tio 

from medical_diffusion.data.augmentation.augmentations_3d import ImageToTensor


class SimpleDataset3D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = ['nii'], # other options are ['nii.gz'],
        transform = None,
        image_resize = None,
        flip = False,
        image_crop = None,
        use_znorm=True, # Use z-Norm for MRI as scale is arbitrary, otherwise scale intensity to [-1, 1]
    ):
        super().__init__()
        self.path_root = path_root
        self.crawler_ext = crawler_ext

        if transform is None: 
            self.transform = T.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
                tio.ZNormalization() if use_znorm else tio.RescaleIntensity((-1,1)),
                ImageToTensor() # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform
        
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return tio.ScalarImage(path_item) # Consider to use this or tio.ScalarLabel over SimpleITK (sitk.ReadImage(str(path_item)))
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]