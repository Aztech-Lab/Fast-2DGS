# -*- coding: utf-8 -*-
"""Image dataset loader for training and evaluation."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, to_tensor


def short_edge_size(height: int, width: int, target: int) -> tuple[int, int]:
    """Target (H, W) after resizing the shorter side to `target` (aspect ratio preserved)."""
    if height <= width:
        new_h = target
        new_w = max(1, int(round(width * target / height)))
    else:
        new_w = target
        new_h = max(1, int(round(height * target / width)))
    return new_h, new_w


def resize_short_edge(img: torch.Tensor, size: int) -> torch.Tensor:
    """Resize CHW image; shorter side == size, aspect ratio unchanged."""
    h, w = img.shape[-2], img.shape[-1]
    new_h, new_w = short_edge_size(h, w, size)
    x = img.unsqueeze(0).float()
    if x.max() > 1.0:
        x = x / 255.0
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def center_crop_chw(img: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop CHW tensor to size x size."""
    h, w = img.shape[-2], img.shape[-1]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return img[:, top : top + size, left : left + size]


def preprocess_image_chw(
    img: torch.Tensor,
    image_size: int = 512,
    crop: bool = True,
) -> torch.Tensor:
    """
    Preprocess CHW image for train/infer.

    crop=True  — resize shorter side to image_size, then center crop (no stretch).
    crop=False — stretch to image_size x image_size (legacy, distorts aspect ratio).
    """
    if img.dtype != torch.float32:
        img = img.float()
    if img.max() > 1.0:
        img = img / 255.0

    if crop:
        img = resize_short_edge(img, image_size)
        img = center_crop_chw(img, image_size)
    else:
        img = F.interpolate(
            img.unsqueeze(0), size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return img.clamp(0, 1)


def build_image_transform(image_size: int = 512, crop: bool = True) -> transforms.Compose:
    if crop:
        return transforms.Compose([
            transforms.Resize(image_size),  # shorter side -> image_size
            transforms.CenterCrop((image_size, image_size)),
            transforms.ConvertImageDtype(torch.float32),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
    ])


class Gaussian_Dataset(Dataset):
    def __init__(self, root_dir, image_size=512, crop=True):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.crop = crop
        self.image_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", "tif"))
        ]
        self.transform = build_image_transform(image_size=image_size, crop=crop)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = read_image(path)
        if img.shape[0] == 4:  # RGBA
            img = to_tensor(to_pil_image(img).convert("RGB"))

        img = self.transform(img)
        return img, os.path.basename(path)


# Training alias — same loader, returns (image, filename).
Heatmap_Dataset = Gaussian_Dataset