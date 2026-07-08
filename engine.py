# -*- coding: utf-8 -*-
"""Fast 2DGS core: encode, sample, rasterize, fine-tune."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, to_tensor

from models.GS_UNet import GaussianUNet, GaussianUNet_Plus, HeatmapUNet

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_K = 50_000
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BASE_CH = 32
DEFAULT_FEAT_PLUS = True
DEFAULT_SAMPLING: Literal["multinomial", "topk"] = "multinomial"
DEFAULT_HEAT_WEIGHT = "weights/smp_heat_div2k.pth"
DEFAULT_FEAT_WEIGHT_PLUS = "weights/smp_feat_best_psnr_26_plus.pth"
DEFAULT_FEAT_WEIGHT_BASE = "weights/smp_feat_best_psnr_26.pth"
DEFAULT_DEVICE = "cuda"
DEFAULT_CROP = True
DEFAULT_TUNE_STEPS = 3000
DEFAULT_TUNE_LR = 2e-3
DEFAULT_TUNE_WEIGHT_DECAY = 0.05
DEFAULT_XY_RETAIN = True
DEFAULT_SCHEDULER_FACTOR = 0.7
DEFAULT_SCHEDULER_PATIENCE = 100
DEFAULT_SCHEDULER_MIN_LR = 1e-5
DEFAULT_TUNE_MILESTONE_SECS = (1, 2, 5)
PSNR_NOT_RECORDED = -1.0

USE_CFG = -1
USE_CFG_SAMPLING = ""

SamplingMode = Literal["multinomial", "topk"]


def _ensure_gmod_importable() -> None:
    try:
        import gmod.gsplat  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    for base in (ROOT_DIR, ROOT_DIR.parent):
        if (base / "gmod").is_dir() and str(base) not in sys.path:
            sys.path.append(str(base))
            try:
                import gmod.gsplat  # noqa: F401
                return
            except ModuleNotFoundError:
                continue

    raise ModuleNotFoundError(
        "Cannot import gmod. Install the renderer first:\n"
        "  git clone https://github.com/Aztech-Lab/gmod.git\n"
        "  cd gmod && pip install -e . --no-build-isolation"
    )


_ensure_gmod_importable()
from gmod.gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gmod.gsplat.rasterize_sum import rasterize_gaussians_sum


@dataclass
class EngineConfig:
    K: int = DEFAULT_K
    image_size: int = DEFAULT_IMAGE_SIZE
    base_ch: int = DEFAULT_BASE_CH
    feat_plus: bool = DEFAULT_FEAT_PLUS
    sampling: SamplingMode = DEFAULT_SAMPLING
    heat_weight: str = DEFAULT_HEAT_WEIGHT
    feat_weight: str = DEFAULT_FEAT_WEIGHT_PLUS
    device: str = DEFAULT_DEVICE
    crop: bool = DEFAULT_CROP
    tune_steps: int = DEFAULT_TUNE_STEPS
    tune_lr: float = DEFAULT_TUNE_LR
    tune_weight_decay: float = DEFAULT_TUNE_WEIGHT_DECAY
    xy_retain: bool = DEFAULT_XY_RETAIN
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE
    scheduler_min_lr: float = DEFAULT_SCHEDULER_MIN_LR

    def resolve_paths(self) -> "EngineConfig":
        self.heat_weight = str(_resolve_path(self.heat_weight))
        if not self.feat_plus and self.feat_weight == DEFAULT_FEAT_WEIGHT_PLUS:
            self.feat_weight = DEFAULT_FEAT_WEIGHT_BASE
        elif self.feat_plus and self.feat_weight == DEFAULT_FEAT_WEIGHT_BASE:
            self.feat_weight = DEFAULT_FEAT_WEIGHT_PLUS
        self.feat_weight = str(_resolve_path(self.feat_weight))
        return self


# backward-compatible alias
InferConfig = EngineConfig


@dataclass
class GaussianParams:
    xy: torch.Tensor
    scale: torch.Tensor
    color: torch.Tensor
    rot: torch.Tensor


@dataclass
class TuneResult:
    params: GaussianParams
    pred: torch.Tensor
    psnr: float
    tune_time: float
    steps_run: int
    psnr_at_1sec: float = PSNR_NOT_RECORDED
    psnr_at_2sec: float = PSNR_NOT_RECORDED
    psnr_at_5sec: float = PSNR_NOT_RECORDED


@dataclass
class InferResult:
    pred: torch.Tensor
    heatmap: torch.Tensor
    params: GaussianParams
    psnr: float
    encode_time: float
    sample_time: float
    raster_time: float
    total_time: float
    tune: TuneResult

    @property
    def psnr_init(self) -> float:
        return self.psnr

    @property
    def pred_init(self) -> torch.Tensor:
        return self.pred

    @property
    def tuned(self) -> bool:
        return self.tune.steps_run > 0

    @property
    def psnr_tuned(self) -> float:
        return self.tune.psnr if self.tuned else PSNR_NOT_RECORDED

    @property
    def pred_tuned(self) -> torch.Tensor:
        return self.tune.pred if self.tuned else self.pred


def _resolve_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def _resolve_k(k: int, cfg: EngineConfig) -> int:
    return cfg.K if k == USE_CFG else k


def _resolve_image_size(image_size: int, cfg: EngineConfig) -> int:
    return cfg.image_size if image_size == USE_CFG else image_size


def load_image(
    image_path: Union[str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop: bool = DEFAULT_CROP,
    device: str = DEFAULT_DEVICE,
) -> torch.Tensor:
    from dataset import preprocess_image_chw

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = read_image(str(image_path))
    if img.shape[0] == 4:
        img = to_tensor(to_pil_image(img).convert("RGB"))

    img = preprocess_image_chw(img, image_size=image_size, crop=crop)
    return img.unsqueeze(0).to(device)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def multinomial_sampling(heatmap: torch.Tensor, k: int) -> torch.Tensor:
    b, _, _, w = heatmap.shape
    prob = heatmap.reshape(b, -1).clamp(min=1e-8)
    prob = prob / prob.sum(dim=1, keepdim=True)
    return torch.multinomial(prob, num_samples=k, replacement=True)


class Fast2DGEngine:
    """Core pipeline: encode -> sample -> rasterize -> optional fine-tune."""

    def __init__(self, **kwargs):
        self.cfg = EngineConfig(**kwargs).resolve_paths()
        self.device = self.cfg.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.cfg.device = "cpu"

        self.heat_model = HeatmapUNet(base_ch=self.cfg.base_ch).to(self.device).eval()
        self.heat_model.load_state_dict(
            torch.load(self.cfg.heat_weight, map_location=self.device, weights_only=True)
        )

        if self.cfg.feat_plus:
            self.feat_model = GaussianUNet_Plus(base_ch=self.cfg.base_ch).to(self.device).eval()
        else:
            self.feat_model = GaussianUNet(base_ch=self.cfg.base_ch).to(self.device).eval()
        self.feat_model.load_state_dict(
            torch.load(self.cfg.feat_weight, map_location=self.device, weights_only=True)
        )

    @torch.inference_mode()
    def encode(self, imgs: torch.Tensor, k: int = USE_CFG) -> tuple[torch.Tensor, list]:
        k = _resolve_k(k, self.cfg)
        batch_k = torch.full((imgs.shape[0], 1), float(k), device=self.device)
        heatmap = self.heat_model(imgs, batch_k)
        feat_out = self.feat_model(imgs, batch_k)
        if self.cfg.feat_plus:
            return heatmap, feat_out
        return heatmap, [None, *feat_out]

    def sample(
        self,
        heatmap: torch.Tensor,
        maps: list,
        k: int = USE_CFG,
        sampling: str = USE_CFG_SAMPLING,
    ) -> GaussianParams:
        k = _resolve_k(k, self.cfg)
        if sampling not in ("multinomial", "topk"):
            sampling = self.cfg.sampling
        offset_map, scale_map, color_map, rot_map = maps

        b, _, h, w = heatmap.shape
        b_idx = torch.arange(b, device=self.device)[:, None]

        if sampling == "topk":
            idx = heatmap.reshape(b, -1).topk(k, dim=1).indices
        else:
            idx = multinomial_sampling(heatmap, k)

        ys, xs = idx // w, idx % w
        xy = torch.stack([xs / (w - 1), ys / (h - 1)], dim=-1).float()
        if self.cfg.feat_plus and offset_map is not None:
            xy = xy + offset_map[b_idx, :, ys, xs].contiguous()

        return GaussianParams(
            xy=xy,
            scale=scale_map[b_idx, :, ys, xs],
            color=color_map[b_idx, :, ys, xs],
            rot=rot_map[b_idx, :, ys, xs],
        )

    def rasterize(self, params: GaussianParams, height: int, width: int) -> torch.Tensor:
        return rasterize_gaussians(params, height, width)

    def forward(
        self, imgs: torch.Tensor, k: int = USE_CFG, sampling: str = USE_CFG_SAMPLING,
    ) -> tuple[torch.Tensor, torch.Tensor, GaussianParams]:
        heatmap, maps = self.encode(imgs, k=k)
        params = self.sample(heatmap, maps, k=k, sampling=sampling)
        pred = self.rasterize(params, imgs.shape[-2], imgs.shape[-1])
        return pred, heatmap, params

    def reconstruct(
        self, imgs: torch.Tensor, k: int = USE_CFG, sampling: str = USE_CFG_SAMPLING,
    ) -> torch.Tensor:
        pred, _, _ = self.forward(imgs, k=k, sampling=sampling)
        return pred

    def tune(
        self,
        imgs: torch.Tensor,
        params: GaussianParams,
        tune_steps: int = DEFAULT_TUNE_STEPS,
        lr: float = DEFAULT_TUNE_LR,
        weight_decay: float = DEFAULT_TUNE_WEIGHT_DECAY,
        xy_retain: bool = DEFAULT_XY_RETAIN,
        show_progress: bool = False,
    ) -> TuneResult:
        if tune_steps <= 0:
            raise ValueError("tune_steps must be > 0")

        b, _, h, w = imgs.shape
        if xy_retain:
            xy_p = nn.Parameter(params.xy.clone().detach())
            scale_p = nn.Parameter(params.scale.clone().detach())
            color_p = nn.Parameter(params.color.clone().detach())
            rot_p = nn.Parameter(params.rot.clone().detach())
        else:
            xy_p = nn.Parameter(torch.rand(b, params.xy.shape[1], 2, device=self.device))
            scale_p = nn.Parameter(torch.rand(b, params.scale.shape[1], 2, device=self.device))
            color_p = nn.Parameter(torch.rand(b, params.color.shape[1], 3, device=self.device))
            rot_p = nn.Parameter(torch.rand(b, params.rot.shape[1], 1, device=self.device))

        optimizer = torch.optim.AdamW(
            [xy_p, scale_p, color_p, rot_p], lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=self.cfg.scheduler_factor,
            patience=self.cfg.scheduler_patience,
            min_lr=self.cfg.scheduler_min_lr,
        )
        criterion = nn.MSELoss()

        psnr_at = {s: PSNR_NOT_RECORDED for s in DEFAULT_TUNE_MILESTONE_SECS}
        printed: set[float] = set()
        tune_time_start = time.time()
        step_iter = range(tune_steps)
        if show_progress:
            from tqdm import tqdm
            step_iter = tqdm(step_iter, desc="fine-tune")

        for _step in step_iter:
            optimizer.zero_grad()
            tuned_params = GaussianParams(xy=xy_p, scale=scale_p, color=color_p, rot=rot_p)
            pred = self.rasterize(tuned_params, h, w)
            loss = criterion(pred, imgs)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            last_psnr = compute_psnr(pred, imgs)
            elapsed = time.time() - tune_time_start
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(PSNR=f"{last_psnr:.2f}", time=f"{elapsed:.1f}s")
            for sec in DEFAULT_TUNE_MILESTONE_SECS:
                if elapsed >= sec and sec not in printed:
                    psnr_at[sec] = last_psnr
                    printed.add(sec)

        tune_time = time.time() - tune_time_start
        final_params = GaussianParams(
            xy=xy_p.detach(), scale=scale_p.detach(),
            color=color_p.detach(), rot=rot_p.detach(),
        )
        pred_tuned = self.rasterize(final_params, h, w)
        return TuneResult(
            params=final_params, pred=pred_tuned,
            psnr=compute_psnr(pred_tuned, imgs),
            tune_time=tune_time, steps_run=tune_steps,
            psnr_at_1sec=psnr_at[1], psnr_at_2sec=psnr_at[2], psnr_at_5sec=psnr_at[5],
        )

    def run(
        self,
        imgs: torch.Tensor,
        k: int = USE_CFG,
        do_tune: bool = True,
        show_progress: bool = False,
    ) -> InferResult:
        t_enc = time.time()
        heatmap, maps = self.encode(imgs, k=k)
        encode_time = time.time() - t_enc

        t_samp = time.time()
        params = self.sample(heatmap, maps, k=k)
        sample_time = time.time() - t_samp

        t_ras = time.time()
        pred = self.rasterize(params, imgs.shape[-2], imgs.shape[-1])
        raster_time = time.time() - t_ras

        tune_result = TuneResult(
            params=GaussianParams(
                xy=torch.empty(0), scale=torch.empty(0),
                color=torch.empty(0), rot=torch.empty(0),
            ),
            pred=torch.empty(0), psnr=PSNR_NOT_RECORDED, tune_time=0.0, steps_run=0,
        )
        if do_tune and self.cfg.tune_steps > 0:
            tune_result = self.tune(
                imgs, params,
                tune_steps=self.cfg.tune_steps, lr=self.cfg.tune_lr,
                weight_decay=self.cfg.tune_weight_decay,
                xy_retain=self.cfg.xy_retain, show_progress=show_progress,
            )

        return InferResult(
            pred=pred, heatmap=heatmap, params=params,
            psnr=compute_psnr(pred, imgs),
            encode_time=encode_time, sample_time=sample_time,
            raster_time=raster_time,
            total_time=encode_time + sample_time + raster_time + tune_result.tune_time,
            tune=tune_result,
        )

    def run_image(
        self,
        image_path: Union[str, Path],
        k: int = USE_CFG,
        image_size: int = USE_CFG,
        crop: int = USE_CFG,
        do_tune: bool = True,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, InferResult]:
        image_size = _resolve_image_size(image_size, self.cfg)
        use_crop = self.cfg.crop if crop == USE_CFG else bool(crop)
        imgs = load_image(image_path, image_size=image_size, crop=use_crop, device=self.device)
        return imgs, self.run(imgs, k=k, do_tune=do_tune, show_progress=show_progress)


def rasterize_gaussians(params: GaussianParams, height: int, width: int) -> torch.Tensor:
    tile_bounds = (width // 16, height // 16, 1)
    outputs = []
    for b in range(params.xy.shape[0]):
        xy_pix, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            params.xy[b], params.scale[b], params.rot[b], height, width, tile_bounds
        )
        out = rasterize_gaussians_sum(
            xy_pix, radii, conics, num_tiles_hit, params.color[b],
            height, width, BLOCK_H=16, BLOCK_W=16, topk_norm=True,
        )
        outputs.append(out.view(height, width, 3).permute(2, 0, 1))
    return torch.stack(outputs, dim=0)


# backward-compatible aliases
Fast2DGSInference = Fast2DGEngine
Fast2DGEngine.infer_tensor = Fast2DGEngine.run
Fast2DGEngine.infer_image = Fast2DGEngine.run_image