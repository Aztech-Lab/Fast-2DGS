# -*- coding: utf-8 -*-
"""
Stage 1: train HeatmapUNet (Deep Gaussian Prior).

Online GT: sample K from the current heatmap prediction, fit Gaussians to the
image, then supervise the network with MSE against the blurred xy heatmap.

Quick start:  python main_train_heat.py
Full script:  python train_heatmap.py --data_path path/to/images --save_dir exp/heatmap
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
from engine import DEFAULT_HEAT_WEIGHT, ROOT_DIR
from models.GS_UNet import HeatmapUNet
from tools import (
    CSVLogger,
    fit_gt_gaussians,
    generate_xy_heatmap,
    random_k,
    sample_xy_from_heatmap,
    save_json,
    tensor2pil,
)

DEFAULT_DATA_PATH = params.DIV2K_train_HR_path
DEFAULT_SAVE_DIR = "exp/heatmap"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BASE_CH = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 5e-4
DEFAULT_NUM_EPOCHS = 500
DEFAULT_K_MIN = 10_000
DEFAULT_K_MAX = 100_000
DEFAULT_GT_STEPS = 200
DEFAULT_GT_LR = 5e-3
DEFAULT_IN_STEPS = 1
DEFAULT_SIGMA = 1.0
DEFAULT_SAMPLING = "multinomial"
DEFAULT_XY_RETAIN = True
DEFAULT_CROP = True
DEFAULT_INIT_WEIGHT = ""  # empty = train from scratch; set path to fine-tune
DEFAULT_SAVE_EVERY = 50


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train HeatmapUNet (stage 1)")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--base_ch", type=int, default=DEFAULT_BASE_CH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--k_min", type=int, default=DEFAULT_K_MIN)
    p.add_argument("--k_max", type=int, default=DEFAULT_K_MAX)
    p.add_argument("--gt_steps", type=int, default=DEFAULT_GT_STEPS,
                   help="Inner-loop steps to fit GT Gaussians per batch")
    p.add_argument("--gt_lr", type=float, default=DEFAULT_GT_LR)
    p.add_argument("--in_steps", type=int, default=DEFAULT_IN_STEPS,
                   help="Heatmap network update steps per batch")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                   help="Gaussian blur sigma for GT heatmap")
    p.add_argument("--sampling", choices=["multinomial", "topk"], default=DEFAULT_SAMPLING)
    p.add_argument("--xy_retain", action=argparse.BooleanOptionalAction, default=DEFAULT_XY_RETAIN)
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=DEFAULT_CROP,
                   help="Resize shorter side then center crop (default, keeps aspect ratio)")
    p.add_argument("--init_weight", type=str, default=DEFAULT_INIT_WEIGHT,
                   help="Optional HeatmapUNet checkpoint to fine-tune from")
    p.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY,
                   help="Also save heat_epoch_N.pth every N epochs")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=0)
    return p


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = ROOT_DIR / p
    return candidate if candidate.exists() else p


def resolve_init_weight(path: str) -> Path | None:
    if not path:
        return None
    candidates = [
        resolve_path(path),
        ROOT_DIR / path,
        ROOT_DIR / "weights" / Path(path).name,
        ROOT_DIR / DEFAULT_HEAT_WEIGHT,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"init_weight not found: {path}")


def load_heatmap_weights(model: HeatmapUNet, weight_path: Path) -> None:
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}")
    w = next(model.parameters())
    print(f"init_weight loaded: {weight_path.resolve()}")
    print(f"  keys={len(state)} params={sum(p.numel() for p in model.parameters())} weight_L1={w.abs().mean().item():.6f}")


def save_loss_plot(save_dir: Path, loss_history: list[float]) -> None:
    """Loss curve — aligned with train_heatmap_v2.py."""
    plt.figure(dpi=300)
    fig, ax1 = plt.subplots(dpi=300)
    ax1.plot(loss_history, color="C0", label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    plt.title("Loss over Epochs", fontsize=16)
    ax1.grid(True, linestyle=":", alpha=0.6)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, fontsize=12, loc="upper right")
    plt.savefig(save_dir / "plot_train.png", bbox_inches="tight")
    plt.close(fig)


def save_epoch_vis(
    save_dir: Path,
    epoch: int,
    imgs: torch.Tensor,
    heatmap_pred: torch.Tensor,
    heatmap_gt: torch.Tensor,
    pred_init: torch.Tensor,
    pred_gt: torch.Tensor,
    xy_impulse: torch.Tensor,
) -> None:
    """6-column grid per epoch — aligned with train_heatmap_v2.py."""
    eval_batch = min(imgs.shape[0], 4)
    panels = []
    for i in range(eval_batch):
        pil1 = tensor2pil(imgs[i])
        pil2 = tensor2pil(heatmap_pred[i])
        pil2 = cv2.applyColorMap(pil2, cv2.COLORMAP_JET)
        pil2 = cv2.cvtColor(pil2, cv2.COLOR_BGR2RGB)
        pil3 = tensor2pil(heatmap_gt[i])
        pil3 = cv2.applyColorMap(pil3, cv2.COLORMAP_JET)
        pil3 = cv2.cvtColor(pil3, cv2.COLOR_BGR2RGB)
        pil5 = tensor2pil(pred_init[i])
        pil6 = tensor2pil(pred_gt[i])
        pil7 = tensor2pil(xy_impulse[i])
        pil7 = cv2.cvtColor(pil7, cv2.COLOR_GRAY2RGB)
        panels.append(np.hstack([pil1, pil3, pil2, pil5, pil6, pil7]))

    out_dir = save_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.vstack(panels)
    cv2.imwrite(str(out_dir / f"epoch_{epoch}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_path = resolve_path(args.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = HeatmapUNet(base_ch=args.base_ch).to(device)
    init_path = resolve_init_weight(args.init_weight)
    if init_path is not None:
        load_heatmap_weights(model, init_path)
    else:
        print("init_weight: none (training from scratch)")

    dataset = Heatmap_Dataset(str(data_path), image_size=args.image_size, crop=args.crop)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.MSELoss().to(device)
    logger = CSVLogger(str(save_dir / "train_log.csv"))

    meta = {
        "stage": "heatmap",
        "data_path": str(data_path),
        "dataset_size": len(dataset),
        "save_dir": str(save_dir.resolve()),
        "image_size": args.image_size,
        "base_ch": args.base_ch,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "gt_steps": args.gt_steps,
        "gt_lr": args.gt_lr,
        "in_steps": args.in_steps,
        "sigma": args.sigma,
        "sampling": args.sampling,
        "xy_retain": args.xy_retain,
        "crop": args.crop,
        "init_weight": str(init_path.resolve()) if init_path else None,
    }
    save_json(meta, str(save_dir / "meta.json"))

    loss_history: list[float] = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"Training HeatmapUNet on {len(dataset)} images -> {save_dir.resolve()}")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        last_psnr_gt = 0.0
        last_k = args.k_min

        vis_imgs = vis_heatmap_pred = vis_heatmap_gt = None
        vis_pred_init = vis_pred_gt = vis_xy_impulse = None

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, (imgs, _names) in enumerate(pbar):
            imgs = imgs.to(device)
            b, _, h, w = imgs.shape
            k = random_k(args.k_min, args.k_max)
            batch_k = torch.full((b, 1), float(k), device=device)
            last_k = k

            heatmap_pred = model(imgs, batch_k)
            idx, xy_init = sample_xy_from_heatmap(heatmap_pred, k, sampling=args.sampling)

            xy_impulse = torch.zeros(b, 1, h * w, device=device)
            b_idx = torch.arange(b, device=device)[:, None]
            xy_impulse[b_idx, 0, idx] = 1.0
            xy_impulse = xy_impulse.view(b, 1, h, w)

            gt_params, psnr_gt, pred_gt, pred_init = fit_gt_gaussians(
                imgs, xy_init, steps=args.gt_steps, lr=args.gt_lr, xy_retain=args.xy_retain,
            )
            last_psnr_gt = psnr_gt

            with torch.no_grad():
                heatmap_gt = generate_xy_heatmap(gt_params.xy, h, w, sigma=args.sigma)

            loss_main = None
            for _ in range(args.in_steps):
                heatmap_pred = model(imgs, batch_k)
                loss_main = criterion(heatmap_pred, heatmap_gt)
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            assert loss_main is not None
            epoch_loss += loss_main.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix(batch_loss=loss_main.item(), avg_loss=avg_loss, K=k)

            vis_imgs = imgs
            vis_heatmap_pred = heatmap_pred.detach()
            vis_heatmap_gt = heatmap_gt
            vis_pred_init = pred_init
            vis_pred_gt = pred_gt
            vis_xy_impulse = xy_impulse

        epoch_loss /= max(len(loader), 1)
        loss_history.append(epoch_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(epoch=epoch, loss=epoch_loss, lr=lr_now)
        print(f"Epoch {epoch + 1}: Loss: {epoch_loss:.6f} | Selected points: {last_k} | PSNR: {last_psnr_gt:.4f}")

        torch.save(model.state_dict(), save_dir / "heat_last.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_dir / "heat_best.pth")
            print(f"  new best loss -> heat_best.pth ({best_loss:.6f})")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt = save_dir / f"heat_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  checkpoint -> {ckpt.name}")

        save_loss_plot(save_dir, loss_history)

        if vis_imgs is not None:
            save_epoch_vis(
                save_dir, epoch, vis_imgs, vis_heatmap_pred, vis_heatmap_gt,
                vis_pred_init, vis_pred_gt, vis_xy_impulse,
            )

    elapsed = time.time() - t_start
    torch.save(model.state_dict(), save_dir / f"heat_final_loss_{best_loss:.4f}.pth")
    print(f"Done in {elapsed / 3600:.2f}h. Best loss={best_loss:.6f}. Checkpoints in {save_dir.resolve()}")


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()