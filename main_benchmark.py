# -*- coding: utf-8 -*-
"""
Paper-aligned dataset benchmark. Run: python main_benchmark.py

Metrics (aligned with test_GSUNet_exp.py / main_test.py):
  init PSNR, tune PSNR, 1s/2s/5s PSNR, MS-SSIM,
  inference time (network forward), tune time, batch time, FPS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import params
from benchmark import BenchmarkConfig, run_benchmark
from engine import PSNR_NOT_RECORDED

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = params.Kodak_path
DEFAULT_OUT = ROOT / "outputs" / "benchmark_kodak"

API_GUIDE = """
# Fast 2DGS benchmark (Python API)

from benchmark import BenchmarkConfig, run_benchmark

cfg = BenchmarkConfig(
    data_path="2DGS_dataset/dataset/Kodak",
    output_dir="outputs/benchmark_kodak",
    K=50000,
    tune_steps=3000,
    save_grid=True,       # per-image visualization grids
    show_progress=False,  # per-image tune tqdm
)
result = run_benchmark(cfg)
print(result["summary"])

# Other standard sets:
#   DIV2K valid  -> params.DIV2K_valid_HR_path
#   ImageGS anime -> params.ImageGS_anime
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast 2DGS paper-aligned benchmark")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA,
                   help="Image folder (default: Kodak)")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--K", type=int, default=50_000)
    p.add_argument("--tune_steps", type=int, default=3000)
    p.add_argument("--no_tune", action="store_true")
    p.add_argument("--save_grid", action="store_true",
                   help="Save per-image grid PNGs under output_dir/grids/")
    p.add_argument("--no_save_images", action="store_true",
                   help="Skip per-image gt/pred/heatmap PNGs")
    p.add_argument("--progress", action="store_true", help="Show tune progress bar")
    p.add_argument("--no_warmup_skip", action="store_true",
                   help="Include first image in FPS / inference-time average")
    return p


def main() -> None:
    print(API_GUIDE.strip())
    print("\n>>> Running benchmark...\n")

    args = build_parser().parse_args()
    cfg = BenchmarkConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        K=args.K,
        tune_steps=0 if args.no_tune else args.tune_steps,
        save_grid=args.save_grid,
        save_images=not args.no_save_images,
        show_progress=args.progress,
        skip_warmup=not args.no_warmup_skip,
    )
    payload = run_benchmark(cfg)
    s = payload["summary"]

    print("\n=== Summary (paper metrics) ===")
    for key in (
        "init_psnr", "tune_psnr", "1sec PSNR", "2sec PSNR", "5sec PSNR",
        "ms_ssim", "inference_time", "tune_time", "batch_time", "FPS",
    ):
        if key in s:
            val = s[key]
            if key == "FPS":
                print(f"  {key:16s}: {val:.1f}")
            elif key == "ms_ssim":
                print(f"  {key:16s}: {val:.4f}")
            elif key.endswith("time"):
                print(f"  {key:16s}: {val:.4f}s")
            elif val <= PSNR_NOT_RECORDED:
                print(f"  {key:16s}: n/a")
            else:
                print(f"  {key:16s}: {val:.2f} dB")

    print(f"\nSaved: {Path(args.output_dir).resolve()}")
    print("  summary.json  — full results + config")
    print("  meta.json     — aggregated metrics only")
    print("  benchmark_log.csv — per-image log")


if __name__ == "__main__":
    main()