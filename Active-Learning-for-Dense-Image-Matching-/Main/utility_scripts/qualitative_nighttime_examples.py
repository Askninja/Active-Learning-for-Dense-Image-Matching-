#!/usr/bin/env python3
"""
RoMa-style warp evidence figure:
Warp Image A -> Image B using the predicted dense correspondence field,
masking by high certainty. Background is white (paper-friendly).

Assumes:
  dense_matches: HxWx4 in normalized coords [-1,1]
    either [ax, ay, bx, by] OR [bx, by, ax, ay]
  dense_certainty: HxW

This script auto-detects which 2 channels correspond to the SOURCE grid (A).
Then uses the other 2 channels as TARGET coords (B) and forward-splats A into B.

Outputs:
  <out_png> : warped A->B with certainty masking and white background.

Run example:
  python3 warp_evidence.py \
    --image_a path/to/pair17_1.jpg \
    --image_b path/to/pair17_2.jpg \
    --ckpt /projects/_hdd/roma/Nighttime/Nighttime_random/Nighttime_random_cycle1_best.pth \
    --resolution medium --device cuda \
    --cert_thresh 0.85 \
    --out_png /tmp/warp_pair17_random.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Adjust these imports to your repo layout
from experiments.train_roma_outdoor import get_model


# -------------------------
# Coord helpers
# -------------------------
def _pixel_to_norm(px: np.ndarray, h: int, w: int) -> np.ndarray:
    px = np.asarray(px, dtype=np.float32)
    out = np.empty_like(px, dtype=np.float32)
    out[..., 0] = (px[..., 0] / max(w - 1, 1e-6)) * 2.0 - 1.0
    out[..., 1] = (px[..., 1] / max(h - 1, 1e-6)) * 2.0 - 1.0
    return out


def _norm_to_pixel(norm: np.ndarray, h: int, w: int) -> np.ndarray:
    norm = np.asarray(norm, dtype=np.float32)
    out = np.empty_like(norm, dtype=np.float32)
    out[..., 0] = (norm[..., 0] + 1.0) * 0.5 * (w - 1)
    out[..., 1] = (norm[..., 1] + 1.0) * 0.5 * (h - 1)
    return out


def _make_grid_norm(h: int, w: int) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid_px = np.stack([xx, yy], axis=-1)  # (h,w,2)
    return _pixel_to_norm(grid_px, h, w)


def _channel_pair_grid_error(matches_norm: np.ndarray, grid_norm: np.ndarray, start: int) -> float:
    d = matches_norm[..., start : start + 2] - grid_norm
    return float(np.mean(np.sqrt(np.sum(d * d, axis=-1))))


def _detect_src_tgt_channels(dense_matches: torch.Tensor) -> tuple[int, int]:
    """
    Detect whether channels 0:2 or 2:4 correspond to the *source* grid.
    Returns src_start, tgt_start (either (0,2) or (2,0)).
    """
    if dense_matches.shape[-1] != 4:
        raise ValueError(f"Expected dense_matches[...,4], got {dense_matches.shape}")

    h, w = dense_matches.shape[:2]
    m = dense_matches.detach().cpu().numpy().astype(np.float32)
    grid = _make_grid_norm(h, w)

    err01 = _channel_pair_grid_error(m, grid, start=0)
    err23 = _channel_pair_grid_error(m, grid, start=2)

    if err01 <= err23:
        return 0, 2
    return 2, 0


# -------------------------
# Warp evidence (forward splat)
# -------------------------
def warp_evidence_A_to_B(
    im_a: Image.Image,
    dense_matches: torch.Tensor,
    dense_certainty: torch.Tensor,
    cert_thresh: float,
    bg_white: bool = True,
) -> Image.Image:
    """
    Forward-splat warp: for each pixel in A (source grid), write its color into
    predicted location in B.

    Uses certainty threshold to mask low-confidence pixels.
    """
    h, w = dense_matches.shape[:2]
    im_a = im_a.convert("RGB").resize((w, h))  # align with match field resolution
    src_rgb = np.asarray(im_a, dtype=np.uint8)

    m = dense_matches.detach().cpu().numpy().astype(np.float32)
    c = dense_certainty.detach().cpu().numpy().astype(np.float32)

    src_start, tgt_start = _detect_src_tgt_channels(dense_matches)

    # src coords should be near grid; we only need TARGET coords for splat
    tgt_norm = m[..., tgt_start : tgt_start + 2]  # (h,w,2) in [-1,1]
    tgt_px = _norm_to_pixel(tgt_norm, h, w)

    xt = np.clip(np.round(tgt_px[..., 0]).astype(np.int32), 0, w - 1)
    yt = np.clip(np.round(tgt_px[..., 1]).astype(np.int32), 0, h - 1)

    # source pixel coordinates are simply (x,y) on the grid
    xs = np.arange(w, dtype=np.int32)[None, :].repeat(h, axis=0)
    ys = np.arange(h, dtype=np.int32)[:, None].repeat(w, axis=1)

    keep = c >= float(cert_thresh)

    if bg_white:
        out = np.full((h, w, 3), 255, dtype=np.uint8)
    else:
        out = np.zeros((h, w, 3), dtype=np.uint8)

    out[yt[keep], xt[keep]] = src_rgb[ys[keep], xs[keep]]
    return Image.fromarray(out, mode="RGB")


# -------------------------
# Model loading
# -------------------------
def load_model(
    ckpt_path: str, resolution: str, symmetric: bool, device: str
) -> tuple[torch.nn.Module, torch.device]:
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")

    model = get_model(
        pretrained_backbone=True,
        resolution=resolution,
        attenuate_cert=True,
        symmetric=symmetric,
    ).to(target_device)

    ckpt = torch.load(ckpt_path, map_location=target_device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, target_device


# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="RoMa-style warp-evidence qualitative example (A->B).")
    p.add_argument("--image_a", required=True, help="Path to image A (e.g., pair*_1.jpg)")
    p.add_argument("--image_b", required=True, help="Path to image B (e.g., pair*_2.jpg)")
    p.add_argument("--ckpt", required=True, help="Checkpoint .pth path")
    p.add_argument("--resolution", default="medium", choices=["tiny", "small", "medium", "large"])
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cert_thresh", type=float, default=0.85)
    p.add_argument("--out_png", required=True)
    args = p.parse_args()

    model, device = load_model(args.ckpt, args.resolution, args.symmetric, args.device)

    im_a = Image.open(args.image_a)
    im_b = Image.open(args.image_b)

    with torch.no_grad():
        dense_matches, dense_certainty = model.match(
            args.image_a, args.image_b, device=device
        )

    warped = warp_evidence_A_to_B(
        im_a=im_a,
        dense_matches=dense_matches,
        dense_certainty=dense_certainty,
        cert_thresh=args.cert_thresh,
        bg_white=True,
    )

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warped.save(out_path)
    print(f"Saved warp evidence to: {out_path}")


if __name__ == "__main__":
    main()
