# -*- coding: utf-8 -*-
"""
User-facing entry: CLI, saving outputs, batch runs.

Core algorithms live in engine.py — import from there if you embed Fast 2DGS.

  python inference.py --input assets/anime-1_2k.png --progress

  from engine import Fast2DGEngine
  eng = Fast2DGEngine(K=50000, tune_steps=3000)
  imgs, result = eng.run_image("assets/anime-1_2k.png")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
)
from tools import save_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast 2DGS inference + fine-tuning")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Path to a single image")
    src.add_argument("--input_dir", type=str, help="Directory of images for batch inference")

    p.add_argument("--output_dir", type=str, default="outputs/infer")
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--base_ch", type=int, default=DEFAULT_BASE_CH)
    p.add_argument("--feat_plus", action=argparse.BooleanOptionalAction, default=DEFAULT_FEAT_PLUS)
    p.add_argument("--sampling", choices=["multinomial", "topk"], default=DEFAULT_SAMPLING)
    p.add_argument("--heat_weight", type=str, default=DEFAULT_HEAT_WEIGHT)
    p.add_argument("--feat_weight", type=str, default=DEFAULT_FEAT_WEIGHT_PLUS)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu"])
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=DEFAULT_CROP)
    p.add_argument("--save_heatmap", action="store_true")

    p.add_argument("--tune_steps", type=int, default=DEFAULT_TUNE_STEPS)
    p.add_argument("--no_tune", action="store_true")
    p.add_argument("--tune_lr", type=float, default=DEFAULT_TUNE_LR)
    p.add_argument("--tune_weight_decay", type=float, default=DEFAULT_TUNE_WEIGHT_DECAY)
    p.add_argument("--xy_retain", action=argparse.BooleanOptionalAction, default=DEFAULT_XY_RETAIN)
    p.add_argument("--progress", action="store_true")
    return p


def list_images(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


def save_heatmap(path: Path, heatmap) -> None:
    import cv2

    arr = heatmap[0, 0].detach().cpu().numpy()
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(arr, cv2.COLORMAP_JET))


def fmt_psnr(value: float) -> str:
    return "n/a" if value <= PSNR_NOT_RECORDED else f"{value:.2f}"


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = ROOT_DIR / path
    return candidate if candidate.exists() else path


def run_single(engine: Fast2DGEngine, image_path: Path, output_dir: Path, args) -> dict:
    imgs, result = engine.run_image(
        image_path, k=args.K, do_tune=not args.no_tune, show_progress=args.progress
    )
    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(imgs[0].clamp(0, 1), output_dir / f"{stem}_gt.png")
    save_image(result.pred[0].clamp(0, 1), output_dir / f"{stem}_pred_init.png")
    if args.save_heatmap:
        save_heatmap(output_dir / f"{stem}_heatmap.png", result.heatmap)

    record = {
        "image": str(image_path),
        "K": args.K,
        "psnr_init": result.psnr_init,
        "encode_time": result.encode_time,
        "sample_time": result.sample_time,
        "raster_time": result.raster_time,
        "total_time": result.total_time,
    }
    msg = (
        f"[{stem}] init={result.psnr_init:.2f} dB | "
        f"encode={result.encode_time:.3f}s raster={result.raster_time:.3f}s"
    )
    if result.tuned:
        save_image(result.tune.pred[0].clamp(0, 1), output_dir / f"{stem}_pred_tune.png")
        record.update({
            "psnr_tune": result.tune.psnr,
            "tune_time": result.tune.tune_time,
            "tune_steps": result.tune.steps_run,
            "psnr_1sec": result.tune.psnr_at_1sec,
            "psnr_2sec": result.tune.psnr_at_2sec,
            "psnr_5sec": result.tune.psnr_at_5sec,
        })
        msg += f" | tune={result.tune.psnr:.2f} dB ({result.tune.tune_time:.1f}s)"
        if result.tune.psnr_at_1sec > PSNR_NOT_RECORDED:
            msg += (
                f" | 1s={fmt_psnr(result.tune.psnr_at_1sec)}"
                f" 2s={fmt_psnr(result.tune.psnr_at_2sec)}"
                f" 5s={fmt_psnr(result.tune.psnr_at_5sec)}"
            )
    print(msg)
    return record


def main() -> None:
    args = build_parser().parse_args()
    if args.no_tune:
        args.tune_steps = 0

    engine = Fast2DGEngine(
        K=args.K, image_size=args.image_size, base_ch=args.base_ch,
        feat_plus=args.feat_plus, sampling=args.sampling,
        heat_weight=args.heat_weight, feat_weight=args.feat_weight,
        device=args.device, crop=args.crop, tune_steps=args.tune_steps,
        tune_lr=args.tune_lr, tune_weight_decay=args.tune_weight_decay,
        xy_retain=args.xy_retain,
    )

    if args.input:
        images = [resolve_input_path(Path(args.input))]
    else:
        input_dir = resolve_input_path(Path(args.input_dir))
        images = list_images(input_dir)
        if not images:
            raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir = Path(args.output_dir)
    records = [run_single(engine, img, output_dir, args) for img in images]

    if args.tune_steps > 0:
        mean_key, mean_psnr = "psnr_tune", sum(r["psnr_tune"] for r in records) / len(records)
    else:
        mean_key, mean_psnr = "psnr_init", sum(r["psnr_init"] for r in records) / len(records)

    cfg = engine.cfg
    save_json({
        "num_images": len(records),
        f"mean_{mean_key}": mean_psnr,
        "config": {
            "K": cfg.K, "image_size": cfg.image_size, "base_ch": cfg.base_ch,
            "feat_plus": cfg.feat_plus, "sampling": cfg.sampling,
            "heat_weight": cfg.heat_weight, "feat_weight": cfg.feat_weight,
            "device": cfg.device, "crop": cfg.crop, "tune_steps": cfg.tune_steps,
            "tune_lr": cfg.tune_lr, "tune_weight_decay": cfg.tune_weight_decay,
            "xy_retain": cfg.xy_retain,
        },
        "results": records,
    }, str(output_dir / "summary.json"))
    print(f"Done. {len(records)} image(s), mean {mean_key}={mean_psnr:.2f} dB")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()