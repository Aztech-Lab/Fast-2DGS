# -*- coding: utf-8 -*-
"""Dataset path helpers (relative to this repo root)."""

from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parent
# DATASET_ROOT = ROOT_DIR / "2DGS_dataset" / "dataset"
DATASET_ROOT = Path("E:/Data/")

Kodak_path = str(DATASET_ROOT / "Kodak")
DIV2K_train_HR_path = str(DATASET_ROOT / "DIV2K" / "DIV2K_train_HR")
DIV2K_valid_HR_path = str(DATASET_ROOT / "DIV2K" / "DIV2K_valid_HR")
ImageGS_anime = str(DATASET_ROOT / "ImageGS_anime")
ImageGS_texture = str(DATASET_ROOT / "ImageGS_textures")