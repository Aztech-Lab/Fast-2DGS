# -*- coding: utf-8 -*-
"""
Stage 2 training quick start. Run: python main_train_feat.py
"""

from __future__ import annotations

from pathlib import Path

import params
from train_feature import build_parser, train

ROOT = Path(__file__).resolve().parent
DEMO_DATA = params.Kodak_path
DEMO_OUT = ROOT / "exp" / "demo_feature"
DEMO_HEAT = "weights/smp_heat_div2k.pth"

API_GUIDE = """
# Stage 2: train GaussianUNet_Plus (frozen heatmap)

from train_feature import build_parser, train

# Option A: programmatic (same data_path as stage 1)
args = build_parser().parse_args([
    "--data_path", "2DGS_dataset/dataset/DIV2K/DIV2K_train_HR",
    "--heat_weight", "weights/smp_heat_div2k.pth",
    "--save_dir", "exp/feature_div2k",
    "--num_epochs", "500",
    "--lr", "1e-3",
    "--batch_size", "8",
])
train(args)

# Option B: CLI
# python train_feature.py --data_path path/to/images --heat_weight weights/smp_heat_div2k.pth
"""


def main() -> None:
    print(API_GUIDE.strip())
    print("\n>>> Running demo (1 epoch, Kodak, frozen heat)...\n")

    args = build_parser().parse_args([
        "--data_path", DEMO_DATA,
        "--heat_weight", DEMO_HEAT,
        "--save_dir", str(DEMO_OUT),
        "--num_epochs", "1",
        "--batch_size", "2",
        "--k_min", "5000",
        "--k_max", "8000",
    ])
    train(args)
    print(f"\nnext: copy exp/demo_feature/feat_best.pth -> weights/  or run inference.py")


if __name__ == "__main__":
    main()