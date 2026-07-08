# -*- coding: utf-8 -*-
"""
Stage 1 training quick start. Run: python main_train_heat.py
"""

from __future__ import annotations

from pathlib import Path

import params
from train_heatmap import build_parser, train

ROOT = Path(__file__).resolve().parent
DEMO_DATA = params.Kodak_path
DEMO_OUT = ROOT / "exp" / "demo_heatmap"

API_GUIDE = """
# Stage 1: train HeatmapUNet (Deep Gaussian Prior)

from train_heatmap import build_parser, train

# Option A: programmatic (edit paths / hyperparams)
args = build_parser().parse_args([
    "--data_path", "2DGS_dataset/dataset/DIV2K/DIV2K_train_HR",
    "--save_dir", "exp/heatmap_div2k",
    "--init_weight", "weights/smp_heat_div2k.pth",   # optional fine-tune
    "--num_epochs", "500",
    "--lr", "5e-4",
    "--batch_size", "8",
])
train(args)

# Option B: CLI
# python train_heatmap.py --data_path path/to/images --save_dir exp/heatmap
"""


def main() -> None:
    print(API_GUIDE.strip())
    print("\n>>> Running demo (1 epoch, Kodak, reduced K)...\n")

    args = build_parser().parse_args([
        "--data_path", DEMO_DATA,
        "--save_dir", str(DEMO_OUT),
        "--num_epochs", "1",
        "--batch_size", "2",
        "--k_min", "5000",
        "--k_max", "8000",
        "--gt_steps", "50",
    ])
    train(args)
    print(f"\nnext: copy exp/demo_heatmap/heat_best.pth -> weights/  or run main_train_feat.py")


if __name__ == "__main__":
    main()