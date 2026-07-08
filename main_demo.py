# -*- coding: utf-8 -*-
"""
Fast 2DGS — Python API quick start. Run: python main_demo.py
"""

from __future__ import annotations

from pathlib import Path

from torchvision.utils import save_image

from engine import Fast2DGEngine, compute_psnr, load_image

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "assets" / "anime-1_2k.png"
OUT = ROOT / "outputs" / "demo"

API_GUIDE = """
# Fast 2DGS inference (Python API)

from engine import Fast2DGEngine, load_image

engine = Fast2DGEngine(K=50000, tune_steps=3000)
imgs = load_image("assets/anime-1_2k.png", image_size=512, crop=True, device=engine.device)

# Option A: step-by-step
pred, heatmap, params = engine.forward(imgs)       # encode -> sample -> rasterize
tune = engine.tune(imgs, params, tune_steps=3000)  # fine-tune

# Option B: one call
imgs, result = engine.run_image("assets/anime-1_2k.png")
print(result.psnr_init, result.psnr_tuned)
"""


def main() -> None:
    print(API_GUIDE.strip())
    print("\n>>> Running demo...\n")

    engine = Fast2DGEngine(K=50000, tune_steps=3000)
    imgs = load_image(IMAGE, image_size=engine.cfg.image_size,
                      crop=engine.cfg.crop, device=engine.device)

    pred, heatmap, params = engine.forward(imgs)
    psnr_init = compute_psnr(pred, imgs)
    print(f"forward:  PSNR={psnr_init:.2f} dB")

    tune = engine.tune(imgs, params, tune_steps=engine.cfg.tune_steps)
    print(f"tune:     PSNR={tune.psnr:.2f} dB  ({tune.tune_time:.1f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    save_image(imgs[0].clamp(0, 1), OUT / "gt.png")
    save_image(pred[0].clamp(0, 1), OUT / "pred_init.png")
    save_image(tune.pred[0].clamp(0, 1), OUT / "pred_tune.png")
    print(f"saved:    {OUT.resolve()}")


if __name__ == "__main__":
    main()