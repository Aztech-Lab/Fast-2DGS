# -*- coding: utf-8 -*-
"""Shared utilities: logging, heatmap generation, training helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from datetime import datetime
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from engine import GaussianParams, multinomial_sampling, rasterize_gaussians

SamplingMode = Literal["multinomial", "topk"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_json(obj, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


class CSVLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.header_written = False
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def log(self, **kwargs) -> None:
        if not self.header_written:
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time"] + list(kwargs.keys()))
            self.header_written = True

        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                + list(kwargs.values())
            )


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def check_range(x: torch.Tensor) -> None:
    print(torch.min(x), torch.max(x))


def create_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    radius = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1),
        torch.arange(-radius, radius + 1),
        indexing="ij",
    )
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def gaussian_blur(
    input_tensor: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    b, c, h, w = input_tensor.shape
    kernel = create_gaussian_kernel(kernel_size, sigma).to(device)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(input_tensor, kernel, stride=1, padding=padding, groups=c)


def generate_xy_heatmap(
    xy_gt: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 1.0,
    kernel_size: int = 51,
) -> torch.Tensor:
    """Single-channel placement heatmap from fitted xy (normalized by max)."""
    b, k, _ = xy_gt.shape
    device = xy_gt.device
    xy = xy_gt.clamp(0, 1)
    xs = (xy[..., 0] * (width - 1)).long()
    ys = (xy[..., 1] * (height - 1)).long()
    idx = ys * width + xs
    b_idx = torch.arange(b, device=device)[:, None].expand(b, k)

    impulse = torch.zeros(b, 1, height * width, device=device)
    impulse[b_idx, 0, idx] = 1.0
    impulse = impulse.view(b, 1, height, width)

    heatmap = gaussian_blur(impulse, kernel_size=kernel_size, sigma=sigma, device=device)
    return heatmap / (heatmap.amax(dim=(2, 3), keepdim=True) + 1e-6)


def generate_heatmap(imgs, xy_gt, scale_gt, color_gt):
    """Legacy multi-channel heatmap (xy + scale)."""
    b, c, h, w = imgs.shape
    _, k, _ = xy_gt.shape
    device = xy_gt.device

    xy = xy_gt.clamp(0, 1)
    xs = (xy[..., 0] * (h - 1)).long()
    ys = (xy[..., 1] * (w - 1)).long()
    idx = ys * w + xs
    b_idx = torch.arange(b, device=device)[:, None].expand(b, k)

    impulse = torch.zeros(b, 1, h * w, device=device)
    impulse[b_idx, 0, idx] = 1.0
    scale_imp = torch.zeros(b, 2, h * w, device=device)
    scale_imp[b_idx, :, idx] = scale_gt

    impulse = impulse.view(b, 1, h, w)
    scale_imp = scale_imp.view(b, 2, h, w)
    color_imp = imgs

    xy_heatmap = gaussian_blur(impulse, kernel_size=51, sigma=5.0, device=device)
    scale_heatmap = gaussian_blur(scale_imp, kernel_size=51, sigma=5.0, device=device)
    color_heatmap = gaussian_blur(color_imp, kernel_size=51, sigma=5.0, device=device)
    heatmap_gt = torch.cat([xy_heatmap, scale_heatmap], dim=1)
    return heatmap_gt, color_heatmap


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def random_k(k_min: int, k_max: int) -> int:
    return max(int(torch.rand(1).item() * k_max), k_min)


def sample_xy_from_heatmap(
    heatmap: torch.Tensor,
    k: int,
    sampling: SamplingMode = "multinomial",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flat indices and normalized xy coordinates [B, K, 2]."""
    b, _, h, w = heatmap.shape
    if sampling == "topk":
        idx = heatmap.reshape(b, -1).topk(k, dim=1).indices
    else:
        idx = multinomial_sampling(heatmap, k)

    ys = idx // w
    xs = idx % w
    xy = torch.stack([xs / (w - 1), ys / (h - 1)], dim=-1).float()
    return idx, xy


def fit_gt_gaussians(
    imgs: torch.Tensor,
    xy_init: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 5e-3,
    xy_retain: bool = True,
) -> tuple[GaussianParams, float, torch.Tensor, torch.Tensor]:
    """Fit Gaussians to the target image; return params, PSNR, fitted render, init render."""
    b, _, h, w = imgs.shape
    k = xy_init.shape[1]
    device = imgs.device

    if xy_retain:
        xy_p = nn.Parameter(xy_init.clone().detach())
    else:
        xy_p = nn.Parameter(torch.rand(b, k, 2, device=device))
    scale_p = nn.Parameter(torch.rand(b, k, 2, device=device))
    color_p = nn.Parameter(torch.rand(b, k, 3, device=device))
    rot_p = nn.Parameter(torch.rand(b, k, 1, device=device))

    init_params = GaussianParams(
        xy=xy_p.detach(), scale=scale_p.detach(),
        color=color_p.detach(), rot=rot_p.detach(),
    )
    pred_init = rasterize_gaussians(init_params, h, w)

    optimizer = torch.optim.AdamW([xy_p, scale_p, color_p, rot_p], lr=lr)
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()

    for _ in range(steps):
        optimizer.zero_grad()
        params = GaussianParams(xy=xy_p, scale=scale_p, color=color_p, rot=rot_p)
        pred = rasterize_gaussians(params, h, w)
        loss = criterion_l1(pred, imgs) + criterion_l2(pred, imgs)
        loss.backward()
        optimizer.step()

    final = GaussianParams(
        xy=xy_p.detach(), scale=scale_p.detach(),
        color=color_p.detach(), rot=rot_p.detach(),
    )
    pred = rasterize_gaussians(final, h, w)
    mse = criterion_l2(pred, imgs).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    return final, psnr, pred, pred_init


def tensor2pil(x: torch.Tensor):
    """Per-tensor min-max norm, CHW -> HWC uint8 (training visualization)."""
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x.detach().cpu() * 255).byte().permute(1, 2, 0).numpy()


def make_xy_impulse(idx: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
    """Binary impulse map from flat sample indices [B, K]."""
    device = idx.device
    impulse = torch.zeros(batch_size, 1, height * width, device=device)
    b_idx = torch.arange(batch_size, device=device)[:, None]
    impulse[b_idx, 0, idx] = 1.0
    return impulse.view(batch_size, 1, height, width)


def sample_params_from_maps(
    heatmap: torch.Tensor,
    maps: list,
    k: int,
    sampling: SamplingMode = "multinomial",
    feat_plus: bool = True,
) -> tuple[torch.Tensor, GaussianParams]:
    """Sample K Gaussians from heatmap and gather attributes from dense maps."""
    idx, raw_xy = sample_xy_from_heatmap(heatmap, k, sampling=sampling)
    b, _, h, w = heatmap.shape
    b_idx = torch.arange(b, device=heatmap.device)[:, None]
    ys = idx // w
    xs = idx % w

    if feat_plus:
        offset_map, scale_map, color_map, rot_map = maps
        xy = raw_xy + offset_map[b_idx, :, ys, xs].contiguous()
    else:
        scale_map, color_map, rot_map = maps
        xy = raw_xy

    params = GaussianParams(
        xy=xy,
        scale=scale_map[b_idx, :, ys, xs].contiguous(),
        color=color_map[b_idx, :, ys, xs].contiguous(),
        rot=rot_map[b_idx, :, ys, xs].contiguous(),
    )
    return idx, params


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    criterion_l1: nn.Module,
    criterion_l2: nn.Module,
    mode: str = "l1+l2",
) -> tuple[torch.Tensor, float]:
    """Reconstruction loss and batch PSNR (from MSE)."""
    loss_l1 = criterion_l1(pred, target)
    loss_l2 = criterion_l2(pred, target)
    if mode == "l1":
        loss = loss_l1
    elif mode == "l2":
        loss = loss_l2
    else:
        loss = loss_l1 + loss_l2
    psnr = 10.0 * math.log10(1.0 / max(loss_l2.item(), 1e-12))
    return loss, psnr


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def copy_all_images(root_dir: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    counter = 0
    for subdir, _, files in os.walk(root_dir):
        pbar = tqdm(files)
        for _, filename in enumerate(pbar):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_ext:
                src = os.path.join(subdir, filename)
                new_name = f"img_{counter:06d}{ext}"
                dst = os.path.join(target_dir, new_name)
                shutil.copy2(src, dst)
                counter += 1
                pbar.set_postfix(file=filename)
    print(f"\nDone. Copied {counter} images to {target_dir}")