import argparse
import funcy
import itertools
import math
import numpy as np
import random
import os
import torch

from skimage.io import imread
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


class AugmentedSegmentationDataset(Dataset):
    def __init__(self, fpath, augmentation=None, input_preprocess=None, target_preprocess=None, with_targets=True):
        self.with_targets = with_targets

        if not augmentation:
            augmentation = []
        if not input_preprocess:
            input_preprocess = []
        if not target_preprocess:
            target_preprocess = input_preprocess

        assert os.path.exists(fpath), "File path doesn't exists"
        with open(fpath, "r") as f:
            lines = filter(lambda l: bool(l), f.read().split("\n"))

            if self.with_targets:
                data = [(input.strip(), target.strip())
                        for input, target in funcy.walk(lambda l: l.split(" "), lines)]
            else:
                data = [(input.strip(), None) for input in lines]

        n_augmentation = math.factorial(len(augmentation))
        augmentation_combinations = list(itertools.product([0, 1], repeat=n_augmentation))

        self.augmentation = augmentation
        self.input_preprocess = input_preprocess
        self.target_preprocess = target_preprocess
        self.data = [(d, transform_list) for transform_list in augmentation_combinations for d in data]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_input_image(fpath):
        img = imread(fpath)
        return Image.fromarray(img)

    @staticmethod
    def _load_target_image(fpath):
        img = imread(fpath, as_gray=True)
        return Image.fromarray(img)

    def __getitem__(self, item):
        (input_path, target_path), aug_bins = self.data[item]
        augmentation = [aug for aug, valid in zip(self.augmentation, aug_bins) if bool(valid)]
        aug_compose = transforms.Compose(augmentation)

        input_image = self._load_input_image(input_path)
        input_shape = input_image.size
        input_image = aug_compose(input_image)
        input_preproc_compose = transforms.Compose(self.input_preprocess)
        input_image = input_preproc_compose(input_image)

        if self.with_targets:
            target_image = self._load_target_image(target_path)
            target_image = aug_compose(target_image)
            target_preproc_compose = transforms.Compose(self.target_preprocess)
            target_image = target_preproc_compose(target_image)
        else:
            target_image = None

        fname = os.path.basename(input_path).split(".")[0]

        return input_image, target_image, fname, input_shape


def main(model_path, data_path, save_to):
    assert torch.cuda.is_available(), "CUDA is not available"

    with open(model_path, "rb") as f:
        model = torch.load(f).cuda()

    input_preprocess = [transforms.Resize(size=(256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.713, 0.550, 0.509], [0.104, 0.112, 0.121])]

    dataset = AugmentedSegmentationDataset(data_path, input_preprocess=input_preprocess, with_targets=False)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=8, num_workers=6, worker_init_fn=set_seeds)
    progress_bar = tqdm(dataloader, desc="Generating masks")
    to_pil_image = transforms.ToPILImage()

    for i, (inputs, targets, fnames, inputs_shape) in progress_bar:
        inputs = Variable(inputs).cuda()
        outputs = model(inputs)

        for output, fname, shape in zip(outputs, fnames, inputs_shape):
            pil_image = to_pil_image(output.cpu() > 0.5)
            pil_image.save(os.path.join(save_to, fname + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest="model_path", type=str)
    parser.add_argument("--data-path", dest="data_path", type=str)
    parser.add_argument("--save-to", dest="save_to", type=str)
    args = parser.parse_args()

    assert os.path.isfile(args.model_path), "Model path does not exist"
    assert os.path.isdir(args.data_path), "Data path does not exist"

    if not os.path.isdir(args.save_to):
        os.makedirs(args.save_to)

    main(args.model_path, args.data_path, args.save_to)
