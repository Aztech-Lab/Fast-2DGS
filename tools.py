# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:31:46 2025

@author: MaxGr
"""

import json
import os

def save_json(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.header_written = False

        # create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def log(self, **kwargs):
        # write header once
        if not self.header_written:
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time"] + list(kwargs.keys()))
            self.header_written = True

        # append data
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                + list(kwargs.values())
            )



import torch
import torch.nn.functional as F

def check_range(x):
    print(torch.min(x), torch.max(x))
    
def create_gaussian_kernel(kernel_size, sigma):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    radius = kernel_size // 2
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing='ij')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_blur(input_tensor, kernel_size=5, sigma=1.0, device='cpu'):
    B, C, H, W = input_tensor.shape
    kernel = create_gaussian_kernel(kernel_size, sigma).to(device)
    kernel = kernel.repeat(C, 1, 1, 1)  # [C,1,ks,ks]
    padding = kernel_size // 2
    return F.conv2d(input_tensor, kernel, stride=1, padding=padding, groups=C)

def generate_heatmap(imgs, xy_gt, scale_gt, color_gt):
    """
    imgs:     [B,3,H,W]
    xy_gt:    [B,K,2] normalized (0~1)
    scale_gt: [B,K,2]
    color_gt: [B,K,3]
    """
    B, C, H, W = imgs.shape
    _, K, _ = xy_gt.shape
    device = xy_gt.device

    # clamp to prevent out-of-bound
    xy = xy_gt.clamp(0,1)
    xs = (xy[...,0] * (H-1)).long()   # [B,K]
    ys = (xy[...,1] * (W-1)).long()   # [B,K]

    idx = ys * W + xs                 # [B,K]
    b_idx = torch.arange(B, device=device)[:,None].expand(B,K)

    # 1-channel impulse
    impulse = torch.zeros(B, 1, H*W, device=device)
    impulse[b_idx, 0, idx] = 1.0

    # scale impulse (2 channels)
    scale_imp = torch.zeros(B, 2, H*W, device=device)
    scale_imp[b_idx, :, idx] = scale_gt   # broadcast OK

    # color impulse (3 channels)
    # color_imp = torch.zeros(B, 3, H*W, device=device)
    # color_imp[b_idx, :, idx] = color_gt

    # reshape to (B,C,H,W)
    impulse   = impulse.view(B,1,H,W)
    scale_imp = scale_imp.view(B,2,H,W)
    # color_imp = color_imp.view(B,3,H,W)
    color_imp = imgs

    # heatmap
    xy_heatmap = gaussian_blur(impulse, kernel_size=51, sigma=5.0, device=device)
    scale_heatmap = gaussian_blur(scale_imp, kernel_size=51, sigma=5.0, device=device)
    color_heatmap = gaussian_blur(color_imp, kernel_size=51, sigma=5.0, device=device)
    
    # heatmap_gt = torch.cat([xy_heatmap, scale_heatmap, color_heatmap], dim=1)
    heatmap_gt = torch.cat([xy_heatmap, scale_heatmap], dim=1)
    
    # save_image(color_heatmap[0].clamp(0, 1), f"{output_dir}/epoch_{epoch}.png")
    return heatmap_gt, color_heatmap

# ============================================================================
# Utils
# ============================================================================
import os
import shutil
from tqdm import tqdm

def copy_all_images(root_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    counter = 0
    for subdir, _, files in os.walk(root_dir):
        pbar = tqdm(files)
        for _, filename in enumerate(pbar):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_ext:
                src = os.path.join(subdir, filename)

                # avoid overwrite: rename if exists
                new_name = f"img_{counter:06d}{ext}"
                dst = os.path.join(target_dir, new_name)

                shutil.copy2(src, dst)
                counter += 1
                pbar.set_postfix(file=filename)

    print(f"\nDone. Copied {counter} images to {target_dir}")

# root_dir = 'E:/Data/SR/imageGS/textures_all/'
# target_dir = 'E:/Data/SR/imageGS/textures/'
# copy_all_images(root_dir, target_dir)




# ============================================================================
# Vis
# ============================================================================
def vis_colorcet_cv2(x, cmap="cet_fire"):
    # # normalize to 0~1
    a = x.astype(np.float32)
    normalized_img = (a - a.min()) / (a.max() - a.min() + 1e-8)

    cmap = cm.get_cmap(cmap)
    colored_img_mpl = cmap(normalized_img)

    # Matplotlib's output is typically RGBA (0-1 float), convert to BGR (0-255 uint8) for OpenCV
    colored_img_opencv = (colored_img_mpl[:, :, :3] * 255).astype(np.uint8)
    bgr = cv2.cvtColor(colored_img_opencv, cv2.COLOR_RGB2BGR) # Convert to BGR
    return bgr









