import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class SegmentationTransform:
    def __init__(self, img_size=256, train=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.train = train
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray((np.squeeze(mask) * 255).astype(np.uint8))
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        mask = (mask >= 0.5).float()
        return image, mask


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self).__init__()
        split = "train" if train else "val"
        images_dir = os.path.join(path_Data, split, "images")
        masks_dir = os.path.join(path_Data, split, "masks")
        images_list = sorted([name for name in os.listdir(images_dir) if not name.startswith(".")])
        masks_list = sorted([name for name in os.listdir(masks_dir) if not name.startswith(".")])
        if len(images_list) != len(masks_list):
            raise ValueError(f"Image and mask counts differ in {split}")
        self.data = []
        for i in range(len(images_list)):
            img_path = os.path.join(images_dir, images_list[i])
            mask_path = os.path.join(masks_dir, masks_list[i])
            self.data.append([img_path, mask_path])
        self.transformer = config.train_transformer if train else config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert("L"), dtype=np.float32), axis=2) / 255.0
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
