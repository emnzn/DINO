import random
from typing import Tuple

from torchvision import transforms
from datasets import load_dataset, Dataset
from PIL import Image, ImageFilter, ImageOps


def get_dataset(
    cache_dir: str, 
    split: str,
    global_scale: Tuple[float] = (0.4, 1.0),
    local_scale: Tuple[float] = (0.05, 0.4),
    num_local_crops: int = 2
    ) -> Dataset:
    """
    Constructs the ImageNet dataset object for pre-training.
    """

    augment = Augment(global_scale, local_scale, num_local_crops)
    
    dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=cache_dir, split=split)
    dataset.set_transform(lambda x: transform(x, augment))

    return dataset

def transform(batch, augment):
    batch["image"] = [augment(img) for img in batch["image"]]
    return batch


class Augment:
    """
    Takes in an image and creates two global crops 
    with a configurable number of local crops.
    """

    def __init__(
        self, 
        global_scale: Tuple[float],
        local_scale: Tuple[float],
        num_local_crops: int,
        ):

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transform1 = transforms.Compose([
            RGB(),
            transforms.RandomResizedCrop(224, scale=global_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            *self._color_distortion(),
            GaussianBlur(p=1.0),
            normalize
        ])

        self.global_transform2 = transforms.Compose([
            RGB(),
            transforms.RandomResizedCrop(224, scale=global_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            *self._color_distortion(),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            normalize
        ])

        self.local_transform = transforms.Compose([
            RGB(),
            transforms.RandomResizedCrop(96, scale=local_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            *self._color_distortion(),
            GaussianBlur(p=0.5),
            normalize
        ])

        self.num_local_crops = num_local_crops

    def _color_distortion(self):
        """
        Functions to apply color distortion to the images.
        """

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        
        return [rnd_color_jitter, rnd_gray]
    
    def __call__(self, img: Image):
        global_crops = [self.global_transform1(img), self.global_transform2(img)]
        local_crops = [self.local_transform(img) for _ in range(self.num_local_crops)]

        crops = {
            "global_crops": global_crops,
            "local_crops": local_crops
        }

        return crops


class RGB(object):
    def __call__(self, img: Image):
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    Taken from https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    Taken from https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img