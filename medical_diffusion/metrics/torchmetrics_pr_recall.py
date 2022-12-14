from typing import Optional, List

import torch 
from torch import Tensor
from torchmetrics import Metric
import torchvision.models as models
from torchvision import transforms



from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
else:
    class FeatureExtractorInceptionV3(Module):  # type: ignore
        pass
    __doctest_skip__ = ["ImprovedPrecessionRecall", "IPR"]

class NoTrainInceptionV3(FeatureExtractorInceptionV3):
    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "NoTrainInceptionV3":
        """the inception network should not be able to be switched away from evaluation mode."""
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out[0].reshape(x.shape[0], -1)


# -------------------------- VGG Trans ---------------------------
# class Normalize(object):
#     """Rescale the image from 0-255 (uint8) to [0,1] (float32). 
#        Note, this doesn't ensure that min=0 and max=1 as a min-max scale would do!"""

#     def __call__(self, image):
#         return image/255

# # see https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html 
# VGG_Trans = transforms.Compose([
#     transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
#     # transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
#     # transforms.CenterCrop(224),
#     Normalize(), # scale to [0, 1]
#     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
# ])        



class ImprovedPrecessionRecall(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False


    def __init__(self, feature=2048, knn=3, splits_real=1, splits_fake=5):
        super().__init__()


        # ------------------------- Init Feature Extractor (VGG or Inception) ------------------------------
        # Original VGG: https://github.com/kynkaat/improved-precision-and-recall-metric/blob/b0247eafdead494a5d243bd2efb1b0b124379ae9/utils.py#L40 
        # Compare Inception: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L574 
        # TODO: Add option to switch between Inception and VGG feature extractor 
        # self.vgg_model = models.vgg16(weights='IMAGENET1K_V1').eval()
        # self.feature_extractor = transforms.Compose([
        #     VGG_Trans, 
        #     self.vgg_model.features,
        #     transforms.Lambda(lambda x: torch.flatten(x, 1)),
        #     self.vgg_model.classifier[:4] # [:4] corresponds to 4096 features 
        # ])

        if isinstance(feature, int):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "FrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = [64, 192, 768, 2048]
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.feature_extractor = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        elif isinstance(feature, torch.nn.Module):
            self.feature_extractor = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        # --------------------------- End Feature Extractor ---------------------------------------------------------------

        self.knn = knn 
        self.splits_real = splits_real
        self.splits_fake = splits_fake
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

        

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if ``imgs`` belong to the real or the fake distribution
        """
        assert torch.is_tensor(imgs) and imgs.dtype == torch.uint8, 'Expecting image as torch.Tensor with dtype=torch.uint8'

        features = self.feature_extractor(imgs).view(imgs.shape[0], -1)  

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self):
        real_features = torch.concat(self.real_features)
        fake_features = torch.concat(self.fake_features)

        real_distances = _compute_pairwise_distances(real_features, self.splits_real)
        real_radii = _distances2radii(real_distances, self.knn)

        fake_distances = _compute_pairwise_distances(fake_features, self.splits_fake)
        fake_radii = _distances2radii(fake_distances, self.knn)

        precision = _compute_metric(real_features, real_radii, self.splits_real, fake_features, self.splits_fake)
        recall = _compute_metric(fake_features, fake_radii, self.splits_fake, real_features, self.splits_real)

        return precision, recall
    
def _compute_metric(ref_features, ref_radii, ref_splits, pred_features, pred_splits):
    dist = _compute_pairwise_distances(ref_features, ref_splits, pred_features, pred_splits)
    num_feat = pred_features.shape[0] 
    count = 0
    for i in range(num_feat):
        count += (dist[:, i] < ref_radii).any()
    return count / num_feat

def _distances2radii(distances, knn):
    return torch.topk(distances, knn+1, dim=1, largest=False)[0].max(dim=1)[0]

def _compute_pairwise_distances(X, splits_x, Y=None, splits_y=None):
    # X = [B, features]
    # Y = [B', features]
    Y = X if Y is None else Y
    # X = X.double()
    # Y = Y.double()
    splits_y = splits_x if splits_y is None else splits_y
    dist = torch.concat([
        torch.concat([
            (torch.sum(X_batch**2, dim=1, keepdim=True) + 
             torch.sum(Y_batch**2, dim=1, keepdim=True).t() - 
             2 * torch.einsum("bd,dn->bn", X_batch, Y_batch.t())) 
        for Y_batch in Y.chunk(splits_y, dim=0)], dim=1)
        for X_batch in X.chunk(splits_x, dim=0)])

    # dist = torch.maximum(dist, torch.zeros_like(dist))
    dist[dist<0] = 0
    return torch.sqrt(dist)

    