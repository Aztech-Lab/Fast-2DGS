# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:02:54 2025

@author: MaxGr
"""

import os
from PIL import Image

import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader

class Gaussian_Dataset(Dataset):
    def __init__(self, root_dir, image_size=512, crop=False):
        super().__init__()
        self.root_dir = root_dir
        self.image_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', 'tif'))
        ]
        if crop:
            self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop((512, 512)),
                transforms.ConvertImageDtype(torch.float32)  # scale to [0,1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ConvertImageDtype(torch.float32)  # scale to [0,1]
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = read_image(path)
        if img.shape[0] == 4:  # RGBA
            img = to_tensor(to_pil_image(img).convert("RGB"))

        img = self.transform(img)
        return img, os.path.basename(path)
        # return img
        
        