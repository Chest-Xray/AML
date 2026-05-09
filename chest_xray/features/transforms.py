import torch
import random
from torchvision.transforms import v2


class RandomGamma(v2.Transform):
    """v2-compatible Random Gamma adjustment."""
    def __init__(self, gamma_range=(0.8, 1.2)):
        super().__init__()
        self.gamma_range = gamma_range


    def _transform(self, inpt, params):
        gamma = random.uniform(*self.gamma_range)
        return v2.functional.adjust_gamma(inpt, gamma)



class RandomNormalize(v2.Transform):
    """v2-compatible Normalization with jittered mean/std."""
    def __init__(self, mean, std, jitter=0.05):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.jitter = jitter


    def _transform(self, inpt, params):
        # Apply slight random variation to the normalization stats
        j_mean = self.mean + (torch.randn(3) * self.jitter)
        j_std = self.std + (torch.randn(3) * self.jitter)
        # Ensure we don't divide by zero/negative with std
        j_std = torch.clamp(j_std, min=0.01)
        return v2.functional.normalize(inpt, j_mean.tolist(), j_std.tolist())



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transforms = v2.Compose([
    v2.ToImage(),                          # Convert to tensor-based image
    v2.Resize(512, antialias=True),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    RandomGamma(gamma_range=(0.8, 1.2)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True)
    v2.GaussianNoise(mean=0.0, sigma=0.02),
    RandomNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, jitter=0.02)
])

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(512, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
