from pathlib import Path
from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10, MNIST, VisionDataset


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: str,
                   root: Path,
                   transform: torchvision.transforms,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataset = get_dataset(name=dataset, root=root, transforms=transform)
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

@register_dataset(name='mnist')
class MNISTcustom(MNIST):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transform=transforms, download=True)

@register_dataset(name='cifar10')
class CIFAR10custom(CIFAR10):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transform=transforms, download=True)

@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='imagenet1k')
class ImageNet1kDataset(FFHQDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)
        
        self.fpaths = self.fpaths[:1000]  # only takes the first 1k images
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index: int):
        return super().__getitem__(index)

@register_dataset(name='afhq')
class AFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.jpg', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img

