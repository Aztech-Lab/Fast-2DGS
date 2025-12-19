# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:40:43 2025

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

# =========================
# Env
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.utils import save_image

print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print(torch.cuda.get_arch_list())
device = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))

# =========================
# Utils
# =========================
import cv2
import numpy as np

def torch_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def tensor2pil(x):
    x = torch_norm(x)
    pil = (x.detach().cpu() * 255).byte().permute(1,2,0).numpy()
    return pil

def np_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def np2pil(x):
    x = np_norm(x)
    pil = (x * 255).astype(np.uint8)
    return pil


def multinomial_sampling(x, K):
    B, C, H, W = x.shape
    # ---- 1. Flatten heatmap into [B, H*W] probability ----
    prob = x.reshape(B, -1)        # [B, H*W]
    prob = prob.clamp(min=1e-8)         # avoid zeros
    prob = prob / prob.sum(dim=1, keepdim=True)   # normalize to sum=1
    # ---- 2. Multinomial sampling (with replacement) ----
    idx = torch.multinomial(prob, num_samples=K, replacement=True)   # [B,K]
    return idx

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

def generate_heatmap(imgs, xy_gt, scale_gt, color_gt, rot_gt):
    B, C, H, W = imgs.shape
    _, K, _ = xy_gt.shape
    device = xy_gt.device

    # clamp to prevent out-of-bound
    xy = xy_gt.clamp(0,1)
    xs = (xy[...,0] * (W-1)).long()   # [B,K]
    ys = (xy[...,1] * (H-1)).long()   # [B,K]

    idx = ys * W + xs                 # [B,K]
    b_idx = torch.arange(B, device=device)[:,None].expand(B,K)

    # 1-channel impulse
    impulse = torch.zeros(B, 1, H*W, device=device)
    impulse[b_idx, 0, idx] = 1.0

    # reshape to (B,C,H,W)
    impulse   = impulse.view(B,1,H,W)
    # impulse_new = update_heatmap(impulse, names)

    # heatmap
    xy_heatmap = gaussian_blur(impulse, kernel_size=51, sigma=1.0, device=device)
    max_val = xy_heatmap.amax(dim=(2, 3), keepdim=True)
    xy_heatmap = xy_heatmap / (max_val + 1e-6)
    # save_image(xy_heatmap[0].clamp(0, 1), f"{output_dir}/heatmap_{epoch}.png")
    return xy_heatmap

# =========================
# Vis
# =========================

def visualize_offset_map(offset_map):
    """
    offset_map: [2,H,W] torch tensor
    return: HxWx3 np.uint8 (BGR)
    """
    flow = offset_map.detach().cpu().numpy()
    dx = flow[0]
    dy = flow[1]

    H, W = dx.shape
    mag, ang = cv2.cartToPolar(dx, dy)
    # ang = (ang * 180 / np.pi / 2).astype(np.uint8)
    ang = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # gamma = 0.3   # <== 0.3 ~ 0.7 
    # mag = np_norm(mag) ** gamma
    k=4
    mag = (np.exp(k * np_norm(mag)) - 1) / (np.exp(k) - 1)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = mag
    hsv[..., 1] = 128   # saturation
    hsv[..., 2] = mag

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # bgr = cv2.applyColorMap(mag, cv2.COLORMAP_VIRIDIS)
    bgr = cv2.applyColorMap(mag, cv2.COLORMAP_PLASMA)
    # rgb = fz.convert_from_flow(flow.transpose(1,2,0))   # Middlebury color wheel
    # rgb = cc.cyclical_1[mag]  # cyclical palette
    # rgb = vis_colorcet_cv2(mag, cmap="cet_cyclic_isoluminant_r")
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def tensor_to_heatmap(tensor):
    """
    tensor: [H,W] torch
    return: HxWx3 np.uint8 (BGR heatmap)
    """
    arr = tensor.detach().cpu().numpy()
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr_norm = arr_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(arr_norm, cv2.COLORMAP_JET)
    return heatmap

def visualize_scale_map(scale_map):
    """
    scale_map: [2,H,W] torch
    return: (sx_map, sy_map, smag_map) np.uint8 BGR
    """
    sx = scale_map[0]
    sy = scale_map[1]
    smag = torch.sqrt(sx * sx + sy * sy)

    return (
        tensor_to_heatmap(sx),
        tensor_to_heatmap(sy),
        tensor_to_heatmap(smag)
    )

def visualize_color_map(color_map):
    """
    color_map: [3,H,W] torch, range assumed 0..1
    return: HxWx3 uint8 BGR
    """
    img = color_map.detach().cpu().permute(1,2,0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr

def visualize_rot_map(rot_map):
    """
    rot_map: [1,H,W] torch (angle, assumed -pi..pi)
    return: HxWx3 uint8 BGR
    """
    rot_map = torch_norm(rot_map)
    rot = rot_map[0].detach().cpu().numpy()

    H, W = rot.shape[-2:]
    hue = rot * 255
    # normalize angle → [0,180]
    # hue = ((rot + np.pi) / (2 * np.pi)) * 180
    # hue = np.clip(hue, 0, 180).astype(np.uint8)
    hue = hue.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def vis_maps(offset_map, scale_map, color_map, rot_map):
    offset_vis = visualize_offset_map(offset_map[0])           # HxWx3 BGR
    sx_vis, sy_vis, smag_vis = visualize_scale_map(scale_map[0])
    color_vis = visualize_color_map(color_map[0])
    rot_vis = visualize_rot_map(rot_map[0])

    return {
        "offset": offset_vis,
        # "scale_sx": sx_vis,
        # "scale_sy": sy_vis,
        "scale_mag": smag_vis,
        "color": color_vis,
        "rot": rot_vis,
    }

def visualize_gaussian_field_v2(scale_map, rot_map, step=8, scale=3, alpha=0.9):
    """
    Gaussian Field visualize
    scale_map: [2,H,W] torch
    rot_map:   [1,H,W] radian
    output: HxW BGR uint8
    """
    sx = scale_map[0].detach().cpu().numpy()
    sy = scale_map[1].detach().cpu().numpy()
    rot = rot_map[0].detach().cpu().numpy()
    rot = cv2.normalize(rot, None, 0, 255, cv2.NORM_MINMAX)

    H, W = sx.shape
    base = np.zeros((H, W, 3), dtype=np.uint8)
    overlay = np.zeros_like(base)
    # normalize 
    size_norm = np.sqrt(sx * sx + sy * sy)
    size_norm = size_norm / (size_norm.max() + 1e-6)
    for y in range(0, H, step):
        for x in range(0, W, step):
            a = max(sx[y, x], 1e-4)
            b = max(sy[y, x], 1e-4)
            angle = rot[y, x]

            # scale
            ra = max(int(a * scale), 1)
            rb = max(int(b * scale), 1)

            # HSV
            # hue = ((angle % (2*np.pi)) / (2*np.pi)) * 179  
            hue = 255-angle
            val = size_norm[y, x] * 255
            # sat = size_norm[y, x] * 255
            sat = 127

            hsv_color = np.array([hue, sat, val], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color.reshape(1,1,3), cv2.COLOR_HSV2BGR)[0,0]

            cv2.ellipse(overlay, (x, y), (ra, rb),
                angle * 180/np.pi,  # rad → degree
                0, 360,
                rgb_color.tolist(),
                -1,  # filled
                cv2.LINE_AA
            )
    # α-blend
    result = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    return result

def vis_feat(feature):
    # feature: [C, H, W]
    feat = feature.detach().cpu()
    act = torch.norm(feat, dim=0)  # [H,W]

    # 2. normalize
    act = (act - act.min()) / (act.max() - act.min())
    act_np = act.numpy()

    # 3. colormap
    act_np = (act_np * 255).astype(np.uint8)
    feat = cv2.applyColorMap(act_np, cv2.COLORMAP_TURBO)
    return feat


# =========================
# Init
# =========================
from gmod.gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gmod.gsplat.rasterize_sum import rasterize_gaussians_sum

def Gaussian_Raster(xy, scale, color, rot, H=512, W=512):
    tile_bounds = (W // 16, H // 16, 1)
    outputs = []
    for b in range(xy.size(0)):
        xy_pix, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xy[b], scale[b], rot[b], H, W, tile_bounds)
        out = rasterize_gaussians_sum(xy_pix, radii, conics, num_tiles_hit, color[b], H, W, BLOCK_H=16, BLOCK_W=16, topk_norm=True)
        proj = out.view(H, W, 3).permute(2, 0, 1)
        outputs.append(proj)
    outputs = torch.stack(outputs, dim=0)
    return outputs

def heatmap_sampling(heatmap, K, sampling='multinomial'):
    global feat_plus
    xy_map = heatmap
    b_idx = torch.arange(B, device=device)[:, None]      # [B,1]
    if sampling == 'topk':
        _, idx = xy_map.reshape(B, -1).topk(K, dim=1) # topK sampling
    if sampling == 'multinomial':
        idx = multinomial_sampling(xy_map, K)
    
    ys = idx // W   # [B, K]
    xs = idx %  W   # [B, K]
    
    xy_list = torch.stack([xs/(W-1), ys/(H-1)], dim=-1).float()  # [B,K,2]
    scale_list = scale_map[b_idx, :, ys, xs]#.contiguous() # [B,K,2]
    color_list = color_map[b_idx, :, ys, xs]#.contiguous() # [B,K,3]
    rot_list = rot_map[b_idx, :, ys, xs]#.contiguous() # [B,K,3]

    if feat_plus:
        raw_xy_list = torch.stack([xs/(W-1), ys/(H-1)], dim=-1).float()  # [B,K,2]
        offset_list = offset_map[b_idx, :, ys, xs].contiguous() # [B,K,2]
        xy_list = raw_xy_list + offset_list
        
    xy_impulse = torch.zeros(B, 1, H*W, device=device)
    xy_impulse[b_idx, 0, idx] = 1.0
    xy_impulse = xy_impulse.view(B,1,H,W)

    return [xy_list, scale_list, color_list, rot_list], xy_impulse


# =========================
# Entry
# =========================
import cv2
import math, time, random
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import argparse
from datetime import datetime
from tools import CSVLogger, save_json

import params
from dataset import Gaussian_Dataset
from models.GS_UNet import HeatmapUNet, GaussianUNet, GaussianUNet_Plus
from pytorch_msssim import ms_ssim
# import flip_evaluator
from torch_featurelayer import FeatureLayer

# def main():
if __name__ == "__main__":
    total_time = time.time()
    print("====================Start training...====================")
    date = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, default="./exp/test/")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--base_channel", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--loss_rec", type=str, default='l2')
    parser.add_argument("--data_path", type=str, default='../data/DIV2K/DIV2K_train_HR/')
    parser.add_argument("--xy_retain", type=bool, default=True)
    parser.add_argument("--sampling", type=str, default='multinomial')
    parser.add_argument("--in_steps", type=int, default=0)
    parser.add_argument("--heat_weight", type=str, default='smp_last_0.0001.pth')
    parser.add_argument("--sigma", type=float, default=1.0)

    args = parser.parse_args()
    print(args)
    
    # -----------------------------------------
    image_size = args.image_size
    base_channel = args.base_channel
    batch_size = args.batch
    learning_rate = args.lr
    num_epoch = args.num_epoch
    loss_rec = args.loss_rec
    data_path = args.data_path
    xy_retain = args.xy_retain
    sampling = args.sampling
    in_steps = args.in_steps
    heat_weight = args.heat_weight
    sigma = args.sigma
    # -----------------------------------------

    # data_path = params.Kodak_path
    # data_path = params.DIV2K_valid_HR_path
    data_path = params.ImageGS_anime
    # data_path = params.ImageGS_texture

    K = 50000
    data_name = data_path.split('/')[-2]
    xy_retain = True
    crop = True

    # # run benchmark
    # feat_plus = False

    # plot feat
    feat_plus = True

    # -----------------------------------------
    heat_model = HeatmapUNet(base_ch=32).to(device)
    feat_model = GaussianUNet(base_ch=32).to(device)

    # if heat_weight:
    heat_weight = 'smp_heat_500_0.0005.pth'
    feat_weight = 'smp_feat_best_psnr_26.pth'
    if feat_plus:
        feat_model = GaussianUNet_Plus(base_ch=32).to(device)
        feat_weight = 'smp_feat_best_psnr_26_plus.pth'

    heat_model.load_state_dict(torch.load(f'./weights/{heat_weight}'))
    feat_model.load_state_dict(torch.load(f'./weights/{feat_weight}'))

    train_dataset = Gaussian_Dataset(data_path, image_size=image_size, crop=crop)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # optimizer_heat = torch.optim.AdamW(heat_model.parameters(), lr=learning_rate, weight_decay=0.05)
    # optimizer_feat = torch.optim.AdamW(feat_model.parameters(), lr=learning_rate, weight_decay=0.05)
    criterion_1 = nn.L1Loss().to(device)
    criterion_2 = nn.MSELoss().to(device)
    
    layer_path = 'conv4'
    hooked_model = FeatureLayer(feat_model, layer_path)

    # -----------------------------------------
    print(feat_model.__class__.__name__)
    save_dir = f"./test/{data_name}_K={K}_{feat_model.__class__.__name__}/"
    logger = CSVLogger(f"{save_dir}/train_log.csv")
    os.makedirs(save_dir, exist_ok=True)
    # ----------------------------------------- Test
    heat_model.eval()
    feat_model.eval()
    
    best_psnr = 0
    PSNR_list = []
    loss_list = []
    # for epoch in range(num_epoch):
    epoch_loss = 0
    epoch_psnr = []
    
    psnr_1sec_list = []
    psnr_2sec_list = []
    psnr_5sec_list = []
    tune_psnr_list = []
    tune_step_list = []
    batch_time_list =[]
    MS_SSIM = []
    step_time_all = []
    inference_list = []
    
    pbar = tqdm(train_loader)
    for i, (imgs, names) in enumerate(pbar):
        name = names[0].split('.')[0]
        imgs = imgs.to(device)
        B, C, H, W = imgs.shape
        
        # K = max(int(torch.rand([1]) * 100000), 10000)
        batch_K = torch.full((B, 1), float(K), device=device)
        
        # ----------------------------------------- Inference
        with torch.no_grad():
            inference_time = time.time()
            heatmap_pred = heat_model(imgs, batch_K) # Heatmap pred
            if feat_plus:
                # [offset_map, scale_map, color_map, rot_map] = feat_model(imgs, batch_K) # Feature pred plus
                feature_output, [offset_map, scale_map, color_map, rot_map] = hooked_model(imgs, batch_K)

            else:
                [scale_map, color_map, rot_map] = feat_model(imgs, batch_K) # Feature pred
            inference_time = time.time() - inference_time
        inference_list.append(inference_time)
        # ----------------------------------------- Sampling & Rendering
        [xy_list,scale_list,color_list,rot_list], xy_impulse = heatmap_sampling(heatmap_pred, K, sampling='multinomial')
        # pred = Gaussian_Raster(xy_list, scale_list, color_list, rot_list, H, W)
        
        # ----------------------------------------- Visualize maps
        if feat_plus:
            vis_list = vis_maps(offset_map, scale_map, color_map, rot_map)
            gs_field = visualize_gaussian_field_v2(scale_map[0], rot_map[0], step=8, scale=2, alpha=1)
            gs_field_2 = visualize_gaussian_field_v2(scale_map[0], rot_map[0], step=16, scale=5, alpha=1)
            feat_vis = vis_feat(feature_output[0])

        # ----------------------------------------- Fine tuning
        gt_time = time.time()
        if xy_retain:
            xy_gt    = nn.Parameter(xy_list.clone().detach())
            scale_gt = nn.Parameter(scale_list.clone().detach())
            color_gt = nn.Parameter(color_list.clone().detach())
            rot_gt   = nn.Parameter(rot_list.clone().detach())
        else:
            xy_gt    = torch.nn.Parameter(torch.rand(B, K, 2, device=device))
            scale_gt = torch.nn.Parameter(torch.rand(B, K, 2, device=device))
            color_gt = torch.nn.Parameter(torch.rand(B, K, 3, device=device))
            rot_gt   = torch.nn.Parameter(torch.rand(B, K, 1, device=device))

        pred = Gaussian_Raster(xy_gt, scale_gt, color_gt, rot_gt, H, W)
        
        # ----------------------------------------- 
        tune_steps = 3000
        optimizer_gt = torch.optim.AdamW([xy_gt, scale_gt, color_gt, rot_gt], lr=2e-3, weight_decay=0.05)
        scheduler_gt = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_gt,
            mode="min",  
            factor=0.7, 
            patience=100,
            min_lr=1e-5,
            # verbose=True
        )
        tune_time = time.time()
        print_times = [1, 2, 5]
        printed = set()
        psnr_sec = []
        
        # step_time_list = []
        for k in range(tune_steps):
            optimizer_gt.zero_grad()
            # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            step_time = time.time()
            pred_gt = Gaussian_Raster(xy_gt, scale_gt, color_gt, rot_gt, H, W)
            step_time = time.time() - step_time
            elapsed = time.time() - tune_time
            # step_time_list.append(step_time)

            # loss_mae = criterion_1(pred_gt, imgs)
            loss_mse = criterion_2(pred_gt, imgs)
            
            loss_gt = loss_mse #+ loss_mae
            psnr_tune = 10 * math.log10(1.0 / loss_mse.item())
            
            loss_gt.backward()
            optimizer_gt.step()
            scheduler_gt.step(loss_gt)
            current_lr = optimizer_gt.param_groups[0]['lr']
            pbar.set_postfix(loss_gt=loss_gt.item(), PSNR=psnr_tune, step=k, 
                             time=elapsed, current_lr=current_lr)
                                
            for t in print_times:
                if elapsed >= t and t not in printed:
                    # print(f"\n[{t} sec] step={k}, time={elapsed:.2f}s | Tune PSNR: {psnr_tune:.2f}")
                    printed.add(t)
                    if t == 1:
                        psnr_1sec_list.append(psnr_tune)
                    if t == 2:
                        psnr_2sec_list.append(psnr_tune)
                    if t == 5:
                        psnr_5sec_list.append(psnr_tune)
                
        # step_time = np.mean(step_time_list)
        # step_time_all.append(step_time)
        tune_time = time.time() - tune_time
        tune_step_list.append(tune_time)
        gt_loss = loss_gt.item()
        tune_psnr_list.append(psnr_tune)
        MS_ssim = ms_ssim(pred_gt, imgs, data_range=1.0, size_average=True)
        MS_SSIM.append(MS_ssim.item())
        # print(f'GT loss: {gt_loss:.6f} | Time cost: {gt_time:.4f} | PSNR: {psnr_tune:.2f} | steps: {k}')

        # -----------------------------------------
        with torch.no_grad():
            heatmap_gt = generate_heatmap(imgs, xy_gt, scale_gt, color_gt, rot_gt)
            # diff_map = imgs - pred
        # -----------------------------------------
        # loss_l1 = criterion_1(pred, imgs)
        loss_l2 = criterion_2(pred, imgs)
        loss_rec = loss_l2 #+ loss_l1 #+ loss_ssim
        loss = loss_rec
        psnr = 10 * math.log10(1.0 / loss_l2.item())
        epoch_psnr.append(psnr)

        # -----------------------------------------
        epoch_loss += loss.item()
        avg_loss = epoch_loss / (i + 1)
        avg_psnr = np.mean(epoch_psnr)
        # -----------------------------------------
        loss = epoch_loss / len(train_loader)
        loss_list.append(loss)
        PSNR_list.append(avg_psnr)
        batch_time = inference_time + tune_time
        batch_time_list.append(batch_time)
        print(f'\nInit PSNR: {psnr:.2f} | Tune PSNR: {psnr_tune:.2f} | Time cost: {batch_time:.2f}')
        print(f'1sec PSNR: {psnr_1sec_list[-1]:.2f} | 2sec PSNR: {psnr_2sec_list[-1]:.2f} | 5sec PSNR: {psnr_5sec_list[-1]:.2f}')
        logger.log(batch=i, init_psnr=psnr, psnr_tune=psnr_tune,
                   psnr_1sec=psnr_1sec_list[-1], psnr_2sec=psnr_2sec_list[-1], psnr_5sec=psnr_5sec_list[-1],
                   inference_time=inference_time, tune_time=tune_time, K=K, step=k)

        # =========================
        # Eval
        # =========================
        eval_batch = min(B, 4)
        pils = []
        for b in range(eval_batch):
            pil1 = tensor2pil(imgs[b])
            pil2 = tensor2pil(pred[b])
            # pil2 = tensor2pil(gt_init[b])
            pil3 = tensor2pil(pred_gt[b])

            pil4 = tensor2pil(xy_impulse[b])
            pil4 = cv2.cvtColor(pil4, cv2.COLOR_GRAY2RGB)
            
            pil5 = tensor2pil(heatmap_gt[b])
            pil5 = cv2.applyColorMap(pil5, cv2.COLORMAP_JET)
            pil5 = cv2.cvtColor(pil5, cv2.COLOR_BGR2RGB)
            
            pil6 = tensor2pil(heatmap_pred[b])
            pil6 = cv2.applyColorMap(pil6, cv2.COLORMAP_JET)
            pil6 = cv2.cvtColor(pil6, cv2.COLOR_BGR2RGB)
            
            pil7 = tensor2pil((imgs[b] - pred_gt[b]).abs())
            
            pil8 = cv2.applyColorMap(pil7, cv2.COLORMAP_MAGMA)
            pil8 = cv2.cvtColor(pil8, cv2.COLOR_BGR2RGB)
            
            pil = np.hstack([pil5,pil6,pil4,pil2,pil3,pil1,pil8])
            pils.append(pil)
        
        grid = np.vstack(pils)
        result_dir = f'{save_dir}/results_{H}x{W}/'
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(f"{result_dir}/{name}_grid.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_pil.png", cv2.cvtColor(pil1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_pred.png", cv2.cvtColor(pil2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_tune.png", cv2.cvtColor(pil3, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_samp.png", cv2.cvtColor(pil4, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_heatmap_gt.png", cv2.cvtColor(pil5, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_heatmap_pred.png", cv2.cvtColor(pil6, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_diff_rgb.png", cv2.cvtColor(pil7, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{result_dir}/{name}_diff_map.png", cv2.cvtColor(pil8, cv2.COLOR_RGB2BGR))

        if feat_plus:
            cv2.imwrite(f"{result_dir}/{name}_offset.png", vis_list['offset'])
            # cv2.imwrite(f"{result_dir}/{name}_scale.png", vis_list['scale_mag'])
            cv2.imwrite(f"{result_dir}/{name}_color.png", vis_list['color'])
            # cv2.imwrite(f"{result_dir}/{name}_rot.png", vis_list['rot'])
            cv2.imwrite(f"{result_dir}/{name}_gs_field.png", gs_field)
            cv2.imwrite(f"{result_dir}/{name}_gs_field_2.png", gs_field_2)
            cv2.imwrite(f"{result_dir}/{name}_feat_vis.png", feat_vis)

    # save
    total_time = time.time() - total_time
    hours = total_time/3600.0
    total_psnr = np.mean(tune_psnr_list)
    print('====================Train done...====================')
    print(f'Total time cost: {total_time:.2f}s ({hours:.2f} hours)')
    torch.save(heat_model.state_dict(), f'{save_dir}/heat_psnr_{total_psnr:.2f}.pth')
    torch.save(feat_model.state_dict(), f'{save_dir}/feat_psnr_{total_psnr:.2f}.pth')
            
    inference_list[0] = 0
    # -------------------------------------------    
    meta = {
        "data_path": data_path,
        "dataset_size": len(train_dataset),
        "output_dir": save_dir,
        "image_size": image_size,
        "batch_size": batch_size,
        # "num_epoch": num_epoch,
        # "optimizer_heat": optimizer_heat.__class__.__name__,
        # "optimizer_feat": optimizer_feat.__class__.__name__,
        # "learning_rate": learning_rate,
        # "loss_rec": loss_rec,
        "heat_model": heat_model.__class__.__name__,
        "feat_model": feat_model.__class__.__name__,
        "xy_retain": xy_retain,
        "sampling": sampling,
        # "in_steps": in_steps,
        "total_time": total_time,
        "tune_step": np.mean(tune_step_list),
        # "step_time": np.mean(step_time_all),
        "inference_time": np.mean(inference_list),
        "batch_time": np.mean(batch_time_list),
        "FPS": 1/np.mean(inference_list),
        "MS_SSIM": np.mean(MS_SSIM),
        "init_psnr": np.mean(epoch_psnr),
        "1sec PSNR": np.mean(psnr_1sec_list),
        "2sec PSNR": np.mean(psnr_2sec_list),
        "5sec PSNR": np.mean(psnr_5sec_list),
        "tune_psnr": np.mean(tune_psnr_list),
    }
    save_json(meta, f"{save_dir}/meta_{date}.json")
    print(meta)

