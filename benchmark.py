# -*- coding: utf-8 -*-
"""Dataset benchmark — metrics aligned with test_GSUNet_exp.py / main_test.py."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torchvision.utils import save_image

from engine import (
    DEFAULT_BASE_CH,
    DEFAULT_CROP,
    DEFAULT_DEVICE,
    DEFAULT_FEAT_PLUS,
    DEFAULT_FEAT_WEIGHT_PLUS,
    DEFAULT_HEAT_WEIGHT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_K,
    DEFAULT_SAMPLING,
    DEFAULT_TUNE_LR,
    DEFAULT_TUNE_STEPS,
    DEFAULT_TUNE_WEIGHT_DECAY,
    DEFAULT_XY_RETAIN,
    PSNR_NOT_RECORDED,
    ROOT_DIR,
    Fast2DGEngine,
    InferResult,
)
from tools import CSVLogger, save_json, tensor2pil

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class BenchmarkConfig:
    data_path: str
    output_dir: str = "outputs/benchmark"
    K: int = DEFAULT_K
    image_size: int = DEFAULT_IMAGE_SIZE
    base_ch: int = DEFAULT_BASE_CH
    feat_plus: bool = DEFAULT_FEAT_PLUS
    sampling: str = DEFAULT_SAMPLING
    heat_weight: str = DEFAULT_HEAT_WEIGHT
    feat_weight: str = DEFAULT_FEAT_WEIGHT_PLUS
    device: str = DEFAULT_DEVICE
    crop: bool = DEFAULT_CROP
    tune_steps: int = DEFAULT_TUNE_STEPS
    tune_lr: float = DEFAULT_TUNE_LR
    tune_weight_decay: float = DEFAULT_TUNE_WEIGHT_DECAY
    xy_retain: bool = DEFAULT_XY_RETAIN
    save_grid: bool = False
    save_images: bool = True
    show_progress: bool = False
    skip_warmup: bool = True


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = ROOT_DIR / p
    return candidate if candidate.exists() else p


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def fmt_psnr(value: float) -> str:
    return "n/a" if value <= PSNR_NOT_RECORDED else f"{value:.2f}"


def save_heatmap_png(path: Path, heatmap: torch.Tensor) -> None:
    arr = heatmap[0, 0].detach().cpu().numpy()
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(arr, cv2.COLORMAP_JET))


def save_result_grid(
    save_dir: Path,
    stem: str,
    imgs: torch.Tensor,
    result: InferResult,
) -> None:
    """7-panel grid aligned with main_test: heatmap | pred_init | pred_tune | gt | diff."""
    pil_gt = tensor2pil(imgs[0])
    pil_init = tensor2pil(result.pred[0])
    pil_tune = tensor2pil(result.tune.pred[0] if result.tuned else result.pred[0])

    pil_heat = tensor2pil(result.heatmap[0])
    pil_heat = cv2.applyColorMap(pil_heat, cv2.COLORMAP_JET)
    pil_heat = cv2.cvtColor(pil_heat, cv2.COLOR_BGR2RGB)

    pil_diff = tensor2pil((imgs[0] - result.tune.pred[0]).abs())
    pil_diff_map = cv2.applyColorMap(pil_diff, cv2.COLORMAP_MAGMA)
    pil_diff_map = cv2.cvtColor(pil_diff_map, cv2.COLOR_BGR2RGB)

    grid = np.hstack([pil_heat, pil_init, pil_tune, pil_gt, pil_diff_map])
    out = save_dir / "grids"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{stem}_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def benchmark_single(
    engine: Fast2DGEngine,
    image_path: Path,
    output_dir: Path,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    imgs, result = engine.run_image(
        image_path, k=cfg.K, do_tune=cfg.tune_steps > 0, show_progress=cfg.show_progress,
    )
    stem = image_path.stem

    # Paper "inference_time" = network forward only (heat + feat), not sample/raster.
    inference_time = result.encode_time
    tune_time = result.tune.tune_time if result.tuned else 0.0
    batch_time = inference_time + tune_time

    pred_tune = result.tune.pred if result.tuned else result.pred
    ms_ssim_val = ms_ssim(pred_tune, imgs, data_range=1.0, size_average=True).item()

    if cfg.save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(imgs[0].clamp(0, 1), output_dir / f"{stem}_gt.png")
        save_image(result.pred[0].clamp(0, 1), output_dir / f"{stem}_pred_init.png")
        if result.tuned:
            save_image(result.tune.pred[0].clamp(0, 1), output_dir / f"{stem}_pred_tune.png")
        save_heatmap_png(output_dir / f"{stem}_heatmap.png", result.heatmap)

    if cfg.save_grid:
        save_result_grid(output_dir, stem, imgs, result)

    record: dict[str, Any] = {
        "image": str(image_path.resolve()),
        "name": stem,
        "K": cfg.K,
        "init_psnr": result.psnr_init,
        "inference_time": inference_time,
        "sample_time": result.sample_time,
        "raster_time": result.raster_time,
        "forward_time": result.encode_time + result.sample_time + result.raster_time,
        "tune_time": tune_time,
        "batch_time": batch_time,
        "ms_ssim": ms_ssim_val,
    }
    if result.tuned:
        record.update({
            "tune_psnr": result.tune.psnr,
            "tune_steps": result.tune.steps_run,
            "psnr_1sec": result.tune.psnr_at_1sec,
            "psnr_2sec": result.tune.psnr_at_2sec,
            "psnr_5sec": result.tune.psnr_at_5sec,
        })

    msg = (
        f"[{stem}] init={result.psnr_init:.2f} dB"
        f" | infer={inference_time:.3f}s"
    )
    if result.tuned:
        msg += (
            f" | tune={result.tune.psnr:.2f} dB ({tune_time:.1f}s)"
            f" | MS-SSIM={ms_ssim_val:.4f}"
            f" | 1s={fmt_psnr(result.tune.psnr_at_1sec)}"
            f" 2s={fmt_psnr(result.tune.psnr_at_2sec)}"
            f" 5s={fmt_psnr(result.tune.psnr_at_5sec)}"
        )
    print(msg)
    return record


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_metrics(records: list[dict[str, Any]], *, skip_warmup: bool) -> dict[str, float]:
    """Aggregate per-image records — field names match main_test meta.json."""
    if not records:
        return {}

    infer_times = [r["inference_time"] for r in records]
    if skip_warmup and len(infer_times) > 1:
        infer_times = infer_times[1:]

    tuned = [r for r in records if "tune_psnr" in r]
    summary: dict[str, float] = {
        "init_psnr": _mean([r["init_psnr"] for r in records]),
        "inference_time": _mean(infer_times),
        "tune_time": _mean([r["tune_time"] for r in tuned]) if tuned else 0.0,
        "batch_time": _mean([r["batch_time"] for r in records]),
        "forward_time": _mean([r["forward_time"] for r in records]),
    }
    if infer_times:
        summary["FPS"] = 1.0 / summary["inference_time"] if summary["inference_time"] > 0 else 0.0

    if tuned:
        def _mean_valid(key: str) -> float:
            vals = [r[key] for r in tuned if r.get(key, PSNR_NOT_RECORDED) > PSNR_NOT_RECORDED]
            return _mean(vals) if vals else PSNR_NOT_RECORDED

        summary.update({
            "tune_psnr": _mean([r["tune_psnr"] for r in tuned]),
            "ms_ssim": _mean([r["ms_ssim"] for r in tuned]),
            "1sec PSNR": _mean_valid("psnr_1sec"),
            "2sec PSNR": _mean_valid("psnr_2sec"),
            "5sec PSNR": _mean_valid("psnr_5sec"),
        })
    return summary


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    data_path = resolve_path(cfg.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    images = list_images(data_path)
    if not images:
        raise FileNotFoundError(f"No images found in {data_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Fast2DGEngine(
        K=cfg.K, image_size=cfg.image_size, base_ch=cfg.base_ch,
        feat_plus=cfg.feat_plus, sampling=cfg.sampling,
        heat_weight=cfg.heat_weight, feat_weight=cfg.feat_weight,
        device=cfg.device, crop=cfg.crop, tune_steps=cfg.tune_steps,
        tune_lr=cfg.tune_lr, tune_weight_decay=cfg.tune_weight_decay,
        xy_retain=cfg.xy_retain,
    )

    logger = CSVLogger(str(output_dir / "benchmark_log.csv"))
    t_start = time.time()
    records: list[dict[str, Any]] = []

    print(f"Benchmarking {len(images)} images from {data_path}")
    print(f"Output -> {output_dir.resolve()}")

    for idx, img_path in enumerate(images):
        record = benchmark_single(engine, img_path, output_dir, cfg)
        records.append(record)
        log_row = {k: v for k, v in record.items() if k != "image"}
        log_row["index"] = idx
        logger.log(**log_row)

    elapsed = time.time() - t_start
    summary = aggregate_metrics(records, skip_warmup=cfg.skip_warmup)
    eng_cfg = engine.cfg

    payload = {
        "data_path": str(data_path.resolve()),
        "dataset_size": len(images),
        "output_dir": str(output_dir.resolve()),
        "total_time": elapsed,
        "num_images": len(records),
        "summary": summary,
        "config": {
            "K": eng_cfg.K,
            "image_size": eng_cfg.image_size,
            "base_ch": eng_cfg.base_ch,
            "feat_plus": eng_cfg.feat_plus,
            "feat_model": engine.feat_model.__class__.__name__,
            "heat_model": engine.heat_model.__class__.__name__,
            "sampling": eng_cfg.sampling,
            "heat_weight": eng_cfg.heat_weight,
            "feat_weight": eng_cfg.feat_weight,
            "device": eng_cfg.device,
            "crop": eng_cfg.crop,
            "tune_steps": eng_cfg.tune_steps,
            "tune_lr": eng_cfg.tune_lr,
            "tune_weight_decay": eng_cfg.tune_weight_decay,
            "xy_retain": eng_cfg.xy_retain,
            "skip_warmup": cfg.skip_warmup,
        },
        "results": records,
    }
    save_json(payload, str(output_dir / "summary.json"))
    save_json(summary, str(output_dir / "meta.json"))
    return payload