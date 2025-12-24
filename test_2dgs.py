# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:40:43 2025

@author: MaxGr
"""

import os
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
from gmod.gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gmod.gsplat.rasterize_sum import rasterize_gaussians_sum

# ==== Env ====
image_path = "assets/anime-1_2k.png"
# image_path = "media/textures/alarm-clock_2k/alarm-clock-01_2k.png"
output_dir = "test/fit_outputs"
os.makedirs(output_dir, exist_ok=True)

image_size = 512
H, W = image_size, image_size  # downsample scale
N = 5000         # num of Gaussian Node
steps = 10000      # 
log_interval = 100
learning_rate = 5e-4

device = torch.device("cuda")

# ==== Load img ====
target = read_image(image_path).float() / 255.0  # [3, H, W]
target = F.interpolate(target.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).to(device)

# ==== Init ====
xy = torch.rand(N, 2, device=device, requires_grad=True)                  # center (0~1)
scale = torch.full((N, 2), 1.0, device=device, requires_grad=True)       # init node size
rot = torch.zeros(N, 1, device=device, requires_grad=True)                # rotation invariant
feat = torch.rand(N, 3, device=device, requires_grad=True)                # RGB value

# optimizer = torch.optim.Adam([xy, scale, feat], lr=1e-1)
# optimizer = torch.optim.Adam([xy, scale, feat, rot], lr=learning_rate)
optimizer = torch.optim.Adam([xy, scale, feat], lr=learning_rate)


tile_bounds = (W // 16, H // 16, 1)
import time 

# ==== Training ====
for step in range(1, steps + 1):
    optimizer.zero_grad()
    
    time_cost = time.time()
    
    # project GS Node
    xy_pix, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xy, scale, rot, H, W, tile_bounds)

    # Render
    out = rasterize_gaussians_sum(
        xy_pix, radii, conics, num_tiles_hit,
        feat, H, W,
        BLOCK_H=16, BLOCK_W=16,
        topk_norm=True
    )
    
    # reshape: [H, W, C] → [C, H, W]
    pred = out.view(H, W, 3).permute(2, 0, 1)

    # loss
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    if step % log_interval == 0 or step == steps:
        print(f"Step {step}/{steps} - Loss: {loss.item():.6f}")
        save_image(pred.clamp(0, 1), f"{output_dir}/step_{step:04d}.png")

# save
save_image(target, f"{output_dir}/gt.png")
print("✅ Done. All outputs saved in:", output_dir)



import cv2
from tqdm import tqdm
# ==== 2. Save GIF ====
frames = []
image_files = sorted([f for f in os.listdir(output_dir) if f.startswith("step_") and f.endswith(".png")])
# image_files = image_files[::10]
for fname in tqdm(image_files):
    frame_i = imageio.imread(os.path.join(output_dir, fname))
    # frame_i = cv2.resize(frame_i, (256,256))
    frames.append(frame_i)
imageio.mimsave(os.path.join(output_dir, "reconstruction.gif"), frames, duration=0.5, loop=0)
print("✅ GIF saved: reconstruction.gif")


# import imageio
# # ==== 3. Save diff ====
# gt = read_image(os.path.join(output_dir, "gt.png")).float() / 255.0
# final = read_image(os.path.join(output_dir, image_files[-1])).float() / 255.0
# error = (gt - final).abs()
# save_image(error, os.path.join(output_dir, "error_map.png"))
# print("✅ Final error map saved: error_map.png")



