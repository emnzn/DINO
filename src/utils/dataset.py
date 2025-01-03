import os
import random
from typing import Tuple, Dict

import torch
from torchvision import transforms
from datasets import load_dataset, Dataset
from PIL import Image, ImageFilter, ImageOps
from torchvision.datasets import CIFAR10, CIFAR100

from .class_map import CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES

# Pre-training Functions
# ---------------------

def get_dataset(
    cache_dir: str, 
    split: str,
    global_scale_min: float = 0.4,
    global_scale_max: float = 1.0,
    local_scale_min: float = 0.05,
    local_scale_max: float = 0.4,
    num_local_crops: int = 2,
    apply_augmentation: bool = True
    ) -> Dataset:
    """
    Constructs the ImageNet dataset object for pre-training.

    Parameters
    ----------
    cache_dir: str
        The dataset dir.

    split: str
        The ImageNet split.
    
    global_scale_min: float
        The minimum global crop scale.

    global_scale_max: float
        The maximum global crop scale.

    local_scale_min: float
        The minimum local crop scale.

    local_scale_max: float
        The maximum local crop scale.

    num_local_crops: int
        The number of local crops for the teacher.

    apply_augmentation: bool
        Whether to apply DINO pre-training augmentations.

    Returns
    -------
    dataset: Dataset
        The initialized Dataset. 
    """

    global_scale = (global_scale_min, global_scale_max)
    local_scale = (local_scale_min, local_scale_max)

    augment = Augment(global_scale, local_scale, num_local_crops)
    
    if apply_augmentation:
        transform = BatchTransform(augment)

    else:
        transform = StandardImageNetTransform()

    dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=cache_dir, split=split)
    dataset.set_transform(transform)

    return dataset


class BatchTransform:
    def __init__(self, augment):
        self.augment = augment

    def __call__(self, batch):
        batch["image"] = [self.augment(img) for img in batch["image"]]
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
        

# Fine-tuning Functions
# ---------------------

def get_transforms():
    transform = transforms.Compose([
            RGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    return transform


class StandardImageNetTransform:
    def __init__(self):        
        self.transform = transforms.Compose([
            RGB(),
            get_transforms()
        ])

    def __call__(self, batch):
        img = [self.transform(i) for i in batch["image"]]
        target = torch.tensor(batch["label"])

        return {"image": img, "target": target}

    
def get_finetune_datasets(
        dataset: str, 
        data_dir: str
        ):
    
    valid_datasets = ["cifar-10", "cifar-100", "imagenet-1k"]

    assert dataset in valid_datasets, f"dataset must be one of {valid_datasets}"

    if dataset == "imagenet-1k":
        train_dataset = get_dataset(cache_dir=data_dir, split="train", apply_augmentation=False)
        val_dataset = get_dataset(data_dir, split="validation", apply_augmentation=False)
        
    else:
        if dataset == "cifar-10":
            train_dataset = CIFAR10(
                data_dir, 
                train=True, 
                transform=get_transforms(), 
                download=True
                )
            
            val_dataset = CIFAR10(
                data_dir, 
                train=False, 
                transform=get_transforms(), 
                download=True
                )
    
    if dataset == "cifar-100":
        train_dataset = CIFAR100(
            data_dir, 
            train=True, 
            transform=get_transforms(), 
            download=True
            )
        
        val_dataset = CIFAR100(
            data_dir, 
            train=False, 
            transform=get_transforms(), 
            download=True
            )
        
    return train_dataset, val_dataset

# Misc Functions
# ---------------------

def get_label_map(dataset: str) -> Dict[int, str]:
    if dataset == "cifar-10":
        label_map = CIFAR10_CLASSES

    if dataset == "cifar-100":
        label_map = CIFAR100_CLASSES

    if dataset == "imagenet-1k":
        label_map = IMAGENET_CLASSES

    return label_map