import os
import funcy
import math
import itertools

from torch.utils.data import Dataset
from torchvision.transforms import (Compose, ToTensor, Normalize,
                                    RandomHorizontalFlip, RandomVerticalFlip)
from PIL import Image

from transform import ToLabel, ReLabel, Scale


class SegANDataset(Dataset):
    def __init__(self, fpath, augmentation=None, with_targets=True):
        if not os.path.isfile(fpath):
            raise FileNotFoundError("Could not find dataset file: '{}'".format(fpath))

        if not augmentation:
            augmentation = []
        n_augmentation = math.factorial(len(augmentation)) if len(augmentation) > 0 else 0
        augmentation_combinations = list(itertools.product([0, 1], repeat=n_augmentation))
        print(augmentation_combinations)

        self.with_targets = with_targets
        self.size = (180, 135)

        self.input_resize = Scale(self.size, Image.BILINEAR)
        self.target_resize = Scale(self.size, Image.NEAREST)
        self.input_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = Compose([
            ToLabel(),
            ReLabel(255, 1),
        ])

        self.augmentation = augmentation

        with open(fpath, "r") as f:
            lines = filter(lambda l: bool(l), f.read().split("\n"))
            if self.with_targets:
                data = [(input.strip(), target.strip())
                        for input, target in funcy.walk(lambda l: l.split(" "), lines)]
            else:
                data = [(input.strip(), None) for input in lines]

        self.data = [(d, transform_list) for transform_list in augmentation_combinations for d in data]

    @staticmethod
    def _load_input_image(fpath):
        img = Image.open(fpath).convert("RGB")
        return img

    @staticmethod
    def _load_target_image(fpath):
        img = Image.open(fpath).convert("P")
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        (input_fpath, target_fpath), aug_bins = self.data[item]
        augmentation = [aug for aug, valid in zip(self.augmentation, aug_bins) if bool(valid)]
        aug_compose = Compose(augmentation)

        input_img = self._load_input_image(input_fpath)
        input_img = aug_compose(input_img)
        input_img = self.input_resize(input_img)
        input_img = self.input_transform(input_img)

        target_img = None
        if target_fpath is not None:
            target_img = self._load_target_image(target_fpath)
            target_img = aug_compose(target_img)
            target_img = self.target_resize(target_img)
            target_img = self.target_transform(target_img)

        fname = os.path.basename(input_fpath).split(".")[0]

        return input_img, target_img, fname


if __name__ == "__main__":
    fpath = "/Users/vribeiro/Documents/isic/train.txt"
    augmetation = [RandomHorizontalFlip(p=1.0),
                   RandomVerticalFlip(p=1.0)]

    dataset = SegANDataset(fpath, augmetation)
    for input_img, target_img, fname in dataset:
        print(input_img.size(), target_img.size())
