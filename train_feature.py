# -*- coding: utf-8 -*-
"""
Stage 2: train GaussianUNet_Plus (attribute regression).

Frozen HeatmapUNet provides placement; the feature network predicts scale / color /
rotation (+ xy offset), then we rasterize and minimize L1+L2 reconstruction loss.

Uses the same image folder as train_heatmap.py (Heatmap_Dataset).

Quick start:  python main_train_feat.py
Full script:  python train_feature.py --data_path path/to/images --heat_weight weights/smp_heat_div2k.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from dataset import Heatmap_Dataset
from engine import DEFAULT_HEAT_WEIGHT as RELEASE_HEAT_WEIGHT, ROOT_DIR, rasterize_gaussians
from models.GS_UNet import GaussianUNet, GaussianUNet_Plus, HeatmapUNet
from tools import (
    CSVLogger,
    make_xy_impulse,
    random_k,
    reconstruction_loss,
    sample_params_from_maps,
    save_json,
    tensor2pil,
)

DEFAULT_DATA_PATH = params.DIV2K_train_HR_path
DEFAULT_HEAT_WEIGHT = "weights/smp_heat_div2k.pth"
DEFAULT_SAVE_DIR = "exp/feature"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BASE_CH = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3
DEFAULT_NUM_EPOCHS = 500
DEFAULT_K_MIN = 10_000
DEFAULT_K_MAX = 100_000
DEFAULT_LOSS_REC = "l1+l2"
DEFAULT_SAMPLING = "multinomial"
DEFAULT_FEAT_PLUS = True
DEFAULT_SAVE_EVERY = 50
DEFAULT_CROP = True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GaussianUNet feature head (stage 2)")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                   help="Image folder (same as train_heatmap.py)")
    p.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--heat_weight", type=str, default=DEFAULT_HEAT_WEIGHT,
                   help="Pretrained HeatmapUNet checkpoint from stage 1")
    p.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--base_ch", type=int, default=DEFAULT_BASE_CH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--k_min", type=int, default=DEFAULT_K_MIN)
    p.add_argument("--k_max", type=int, default=DEFAULT_K_MAX)
    p.add_argument("--loss_rec", choices=["l1", "l2", "l1+l2"], default=DEFAULT_LOSS_REC)
    p.add_argument("--sampling", choices=["multinomial", "topk"], default=DEFAULT_SAMPLING)
    p.add_argument("--feat_plus", action=argparse.BooleanOptionalAction, default=DEFAULT_FEAT_PLUS)
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=DEFAULT_CROP,
                   help="Resize shorter side then center crop (default, keeps aspect ratio)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY,
                   help="Also save feat_epoch_N.pth every N epochs")
    return p


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = ROOT_DIR / p
    return candidate if candidate.exists() else p


def resolve_weight(path: str) -> Path:
    """Resolve heat checkpoint: explicit path -> weights/smp_heat_div2k.pth."""
    candidates = [
        resolve_path(path),
        ROOT_DIR / path,
        ROOT_DIR / "weights" / Path(path).name,
        ROOT_DIR / DEFAULT_HEAT_WEIGHT,
        ROOT_DIR / RELEASE_HEAT_WEIGHT,
    ]
    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c
    return candidates[0]


def save_loss_plot(save_dir: Path, loss_history: list[float], psnr_history: list[float]) -> None:
    """Dual-axis loss + PSNR — aligned with train_GSUNet_v01.py."""
    plt.figure(dpi=300)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax1.plot(loss_history, color="C0", label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(psnr_history, color="C2", label="PSNR")
    ax2.set_ylabel("PSNR (dB)", fontsize=14, color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")

    plt.title("Loss / PSNR over Epochs", fontsize=16)
    ax1.grid(True, linestyle=":", alpha=0.6)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=12, loc="right")
    plt.savefig(save_dir / "plot_train.png", bbox_inches="tight")
    plt.close(fig)


def save_epoch_vis(
    save_dir: Path,
    epoch: int,
    imgs: torch.Tensor,
    heatmap_pred: torch.Tensor,
    pred: torch.Tensor,
    xy_impulse: torch.Tensor,
) -> None:
    """4-column grid — aligned with train_GSUNet_v01.py."""
    eval_batch = min(imgs.shape[0], 4)
    panels = []
    for i in range(eval_batch):
        pil1 = tensor2pil(imgs[i])
        pil2 = tensor2pil(heatmap_pred[i])
        pil2 = cv2.applyColorMap(pil2, cv2.COLORMAP_JET)
        pil2 = cv2.cvtColor(pil2, cv2.COLOR_BGR2RGB)
        pil3 = tensor2pil(pred[i])
        pil4 = tensor2pil(xy_impulse[i])
        pil4 = cv2.cvtColor(pil4, cv2.COLOR_GRAY2RGB)
        panels.append(np.hstack([pil1, pil2, pil3, pil4]))

    out_dir = save_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.vstack(panels)
    cv2.imwrite(str(out_dir / f"epoch_{epoch}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def load_heat_model(path: Path, base_ch: int, device: torch.device) -> HeatmapUNet:
    state = torch.load(path, map_location=device, weights_only=True)
    model = HeatmapUNet(base_ch=base_ch).to(device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Heatmap weight mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    frozen = 0
    for p in model.parameters():
        p.requires_grad = False
        frozen += p.numel()
    wnorm = sum(p.detach().abs().sum().item() for p in model.parameters())
    print(f"  heat loaded: {path.resolve()}")
    print(f"  keys={len(state)} frozen_params={frozen} weight_L1={wnorm:.3e}")
    return model


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_path = resolve_path(args.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    heat_path = resolve_weight(args.heat_weight)
    if not heat_path.is_file():
        raise FileNotFoundError(f"Heatmap checkpoint not found: {heat_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    heat_model = load_heat_model(heat_path, args.base_ch, device)
    if args.feat_plus:
        feat_model = GaussianUNet_Plus(base_ch=args.base_ch).to(device)
    else:
        feat_model = GaussianUNet(base_ch=args.base_ch).to(device)

    dataset = Heatmap_Dataset(str(data_path), image_size=args.image_size, crop=args.crop)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(feat_model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion_l1 = nn.L1Loss().to(device)
    criterion_l2 = nn.MSELoss().to(device)
    logger = CSVLogger(str(save_dir / "train_log.csv"))

    meta = {
        "stage": "feature",
        "data_path": str(data_path),
        "dataset_size": len(dataset),
        "save_dir": str(save_dir.resolve()),
        "heat_weight": str(heat_path.resolve()),
        "image_size": args.image_size,
        "base_ch": args.base_ch,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "loss_rec": args.loss_rec,
        "sampling": args.sampling,
        "feat_plus": args.feat_plus,
        "crop": args.crop,
    }
    save_json(meta, str(save_dir / "meta.json"))

    loss_history: list[float] = []
    psnr_history: list[float] = []
    best_psnr = 0.0
    t_start = time.time()

    print(f"Training feature net on {len(dataset)} images -> {save_dir.resolve()}")
    print(f"Frozen heatmap: {heat_path}")
    for epoch in range(args.num_epochs):
        feat_model.train()
        epoch_loss = 0.0
        epoch_psnr: list[float] = []
        last_k = args.k_min

        vis_imgs = vis_heatmap = vis_pred = vis_impulse = None

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, (imgs, _names) in enumerate(pbar):
            imgs = imgs.to(device)
            b, _, h, w = imgs.shape
            k = random_k(args.k_min, args.k_max)
            batch_k = torch.full((b, 1), float(k), device=device)
            last_k = k

            with torch.no_grad():
                heatmap_pred = heat_model(imgs, batch_k)
            feat_out = feat_model(imgs, batch_k)
            maps = feat_out

            idx, gauss_params = sample_params_from_maps(
                heatmap_pred, maps, k, sampling=args.sampling, feat_plus=args.feat_plus,
            )
            xy_impulse = make_xy_impulse(idx, b, h, w)
            pred = rasterize_gaussians(gauss_params, h, w)

            loss, psnr = reconstruction_loss(
                pred, imgs, criterion_l1, criterion_l2, mode=args.loss_rec,
            )
            epoch_psnr.append(psnr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            avg_psnr = float(np.mean(epoch_psnr))
            pbar.set_postfix(batch_loss=loss.item(), avg_loss=avg_loss, K=k, avg_psnr=f"{avg_psnr:.2f}")

            vis_imgs = imgs
            vis_heatmap = heatmap_pred
            vis_pred = pred.detach()
            vis_impulse = xy_impulse

        epoch_loss /= max(len(loader), 1)
        avg_psnr = float(np.mean(epoch_psnr)) if epoch_psnr else 0.0
        loss_history.append(epoch_loss)
        psnr_history.append(avg_psnr)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(epoch=epoch, loss=epoch_loss, lr=lr_now, avg_psnr=avg_psnr)
        print(f"Epoch {epoch}: Loss: {epoch_loss:.6f} | PSNR: {avg_psnr:.4f} | LR: {lr_now:.9f}")

        torch.save(feat_model.state_dict(), save_dir / "feat_last.pth")
        if avg_psnr > best_psnr:
            print(f"  best PSNR -> feat_best.pth ({best_psnr:.2f} -> {avg_psnr:.2f})")
            best_psnr = avg_psnr
            torch.save(feat_model.state_dict(), save_dir / "feat_best.pth")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(feat_model.state_dict(), save_dir / f"feat_epoch_{epoch}.pth")

        save_loss_plot(save_dir, loss_history, psnr_history)

        if vis_imgs is not None:
            save_epoch_vis(save_dir, epoch, vis_imgs, vis_heatmap, vis_pred, vis_impulse)

    elapsed = time.time() - t_start
    torch.save(feat_model.state_dict(), save_dir / f"feat_final_psnr_{best_psnr:.2f}.pth")
    print(f"Done in {elapsed / 3600:.2f}h. Best PSNR={best_psnr:.2f} dB. Checkpoints in {save_dir.resolve()}")


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()