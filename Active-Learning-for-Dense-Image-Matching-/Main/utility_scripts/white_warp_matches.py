#!/usr/bin/env python3
"""
Dense-match warp to white background:
1) Match image A to image B.
2) Keep matches with certainty >= threshold (default 0.95).
3) Create a white canvas (image B space) and paste matched pixels from A.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.train_roma_outdoor import get_model


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
    grid_px = np.stack([xx, yy], axis=-1)
    return _pixel_to_norm(grid_px, h, w)


def _channel_pair_grid_error(matches_norm: np.ndarray, grid_norm: np.ndarray, start: int) -> float:
    d = matches_norm[..., start : start + 2] - grid_norm
    return float(np.mean(np.sqrt(np.sum(d * d, axis=-1))))


def _detect_src_tgt_channels(dense_matches: torch.Tensor) -> tuple[int, int]:
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


def _fill_white_holes(img: np.ndarray, max_iters: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(max_iters):
        white = np.all(out == 255, axis=-1)
        if not np.any(white):
            break
        padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        updated = out.copy()
        ys, xs = np.where(white)
        for y, x in zip(ys, xs):
            patch = padded[y : y + 3, x : x + 3].reshape(-1, 3)
            non_white = patch[~np.all(patch == 255, axis=1)]
            if non_white.size:
                updated[y, x] = np.mean(non_white, axis=0).astype(np.uint8)
        out = updated
    return out


def warp_white(
    im_a: Image.Image,
    dense_matches: torch.Tensor,
    dense_certainty: torch.Tensor,
    cert_thresh: float,
    fill_holes: bool,
    fill_iters: int,
) -> Image.Image:
    h, w = dense_matches.shape[:2]
    im_a = im_a.convert("RGB").resize((w, h))
    src_rgb = np.asarray(im_a, dtype=np.uint8)

    m = dense_matches.detach().cpu().numpy().astype(np.float32)
    c = dense_certainty.detach().cpu().numpy().astype(np.float32)

    src_start, tgt_start = _detect_src_tgt_channels(dense_matches)
    tgt_norm = m[..., tgt_start : tgt_start + 2]
    tgt_px = _norm_to_pixel(tgt_norm, h, w)

    xt = np.clip(np.round(tgt_px[..., 0]).astype(np.int32), 0, w - 1)
    yt = np.clip(np.round(tgt_px[..., 1]).astype(np.int32), 0, h - 1)

    xs = np.arange(w, dtype=np.int32)[None, :].repeat(h, axis=0)
    ys = np.arange(h, dtype=np.int32)[:, None].repeat(w, axis=1)

    keep = c >= float(cert_thresh)
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[yt[keep], xt[keep]] = src_rgb[ys[keep], xs[keep]]
    if fill_holes:
        out = _fill_white_holes(out, max_iters=max(1, int(fill_iters)))
    return Image.fromarray(out, mode="RGB")


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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Warp A->B with dense matches, keep >=99.5% certainty, white background."
    )
    p.add_argument("--image_a", required=True, help="Path to image A (pair*_1.jpg)")
    p.add_argument("--image_b", required=True, help="Path to image B (pair*_2.jpg)")
    p.add_argument("--ckpt", required=True, help="Checkpoint .pth path")
    p.add_argument("--resolution", default="medium", choices=["tiny", "small", "medium", "large"])
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cert_thresh", type=float, default=0.95)
    p.add_argument("--out_png", required=True)
    p.add_argument("--fill_holes", action="store_true", default=True)
    p.add_argument("--no_fill_holes", action="store_false", dest="fill_holes")
    p.add_argument("--fill_iters", type=int, default=3)
    args = p.parse_args()

    model, device = load_model(args.ckpt, args.resolution, args.symmetric, args.device)

    im_a = Image.open(args.image_a)

    with torch.no_grad():
        dense_matches, dense_certainty = model.match(
            args.image_a, args.image_b, device=device
        )

    out = warp_white(
        im_a=im_a,
        dense_matches=dense_matches,
        dense_certainty=dense_certainty,
        cert_thresh=args.cert_thresh,
        fill_holes=args.fill_holes,
        fill_iters=args.fill_iters,
    )

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Saved white-warp output to: {out_path}")


if __name__ == "__main__":
    main()
