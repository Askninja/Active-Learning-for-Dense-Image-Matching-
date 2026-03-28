#!/usr/bin/env python3
"""Visualize top-N correspondences for a single Optical-Infrared pair."""

from __future__ import annotations

import argparse
import os.path as osp
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.train_AL_cycle import DATASET_DIRS, RESOLUTIONS, get_dataset_root
from experiments.train_roma_outdoor import get_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datasets"
BOX_RATIO = 0.5


def _load_model(ckpt_path: Path, resolution: str, symmetric: bool, device: str) -> torch.nn.Module:
    target_device = device
    if "cuda" in device and not torch.cuda.is_available():
        target_device = "cpu"
    model = get_model(
        pretrained_backbone=True,
        resolution=resolution,
        attenuate_cert=True,
        symmetric=symmetric,
    ).to(target_device)
    ckpt = torch.load(str(ckpt_path), map_location=target_device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[VIS] Missing keys while loading checkpoint ({len(missing)}).")
    if unexpected:
        print(f"[VIS] Unexpected keys while loading checkpoint ({len(unexpected)}).")
    model.eval()
    return model


def _resolve_pair_paths(data_root: str, pair_idx: int) -> tuple[str, str]:
    a_path = osp.join(data_root, f"pair{pair_idx}_1.jpg")
    b_path = osp.join(data_root, f"pair{pair_idx}_2.jpg")
    if not osp.isfile(a_path) or not osp.isfile(b_path):
        raise FileNotFoundError(f"Missing pair files for idx={pair_idx}: {a_path}, {b_path}")
    return a_path, b_path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _palette(n: int) -> list[tuple[int, int, int]]:
    colors = [
        (239, 71, 111),
        (255, 209, 102),
        (6, 214, 160),
        (17, 138, 178),
        (7, 59, 76),
        (255, 127, 80),
        (147, 112, 219),
        (0, 128, 128),
        (255, 165, 0),
        (46, 139, 87),
        (0, 191, 255),
        (220, 20, 60),
        (154, 205, 50),
        (70, 130, 180),
        (199, 21, 133),
        (210, 105, 30),
        (0, 0, 0),
        (128, 0, 0),
        (0, 0, 128),
        (0, 128, 0),
    ]
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


def _draw_poly(draw: ImageDraw.ImageDraw, pts: np.ndarray, offset_x: float, color: tuple[int, int, int]) -> None:
    if pts is None or pts.size != 8:
        return
    poly = [(float(x) + offset_x, float(y)) for x, y in pts.reshape(-1, 2)]
    draw.line(poly + [poly[0]], fill=color, width=3)


def _draw_matches(
    im_a: Image.Image,
    im_b: Image.Image,
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    out_path: Path,
    title: str,
    gt_box: np.ndarray | None = None,
    strat_box: np.ndarray | None = None,
    gt_color: tuple[int, int, int] = (220, 20, 60),
    strat_color: tuple[int, int, int] = (0, 191, 255),
) -> None:
    w1, h1 = im_a.size
    w2, h2 = im_b.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), color=(0, 0, 0))
    canvas.paste(im_a, (0, 0))
    canvas.paste(im_b, (w1, 0))
    draw = ImageDraw.Draw(canvas)
    colors = _palette(len(kpts_a))
    for i, (p1, p2) in enumerate(zip(kpts_a, kpts_b)):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]) + w1, float(p2[1])
        color = colors[i]
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        r = 3
        draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], outline=color, width=2)
        draw.ellipse([(x2 - r, y2 - r), (x2 + r, y2 + r)], outline=color, width=2)
    _draw_poly(draw, gt_box, w1, gt_color)
    _draw_poly(draw, strat_box, w1, strat_color)
    draw.text((8, 8), title, fill=(255, 255, 255))
    canvas.save(out_path)


def _top_k_matches(matches: torch.Tensor, certainty: torch.Tensor, k: int) -> np.ndarray:
    if matches.numel() == 0:
        return np.empty((0, 4), dtype=np.float32)
    cert = certainty.detach().float().cpu().numpy().reshape(-1)
    order = np.argsort(-cert)[:k]
    sel = matches.detach().cpu().numpy().reshape(-1, 4)[order]
    return sel


def _norm_to_pixel(coords: np.ndarray, h: int, w: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    out = np.empty_like(coords, dtype=np.float32)
    out[..., 0] = (coords[..., 0] + 1.0) * 0.5 * w
    out[..., 1] = (coords[..., 1] + 1.0) * 0.5 * h
    return out


def _warp_points(warp: torch.Tensor, points_px: np.ndarray) -> np.ndarray:
    h, w = warp.shape[:2]
    pts = np.asarray(points_px, dtype=np.float32)
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    warp_np = warp.detach().cpu().numpy()
    b_norm = warp_np[y, x, 2:4]
    return _norm_to_pixel(b_norm, h, w)


def _center_box_pixels(w: int, h: int, ratio: float) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.05, 0.95))
    bw = (w - 1) * ratio
    bh = (h - 1) * ratio
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    x0 = cx - bw * 0.5
    x1 = cx + bw * 0.5
    y0 = cy - bh * 0.5
    y1 = cy + bh * 0.5
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _load_gt_homography(gt_path: str) -> np.ndarray:
    H = np.loadtxt(gt_path).astype(np.float32)
    if H.shape == (2, 3):
        H = np.vstack([H, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)])
    if H.shape != (3, 3):
        raise ValueError(f"Unexpected gt shape at {gt_path}: {H.shape}")
    return H


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)
    return proj.astype(np.float32)


def _gt_box_poly_from_H(image_a: str, image_b: str, h: int, w: int, H: np.ndarray) -> np.ndarray:
    im_a = Image.open(image_a)
    im_b = Image.open(image_b)
    wa, ha = im_a.size
    wb, hb = im_b.size
    box_A_orig = _center_box_pixels(wa, ha, BOX_RATIO)
    box_B_orig = _apply_homography(H, box_A_orig)
    sx = w / float(wb)
    sy = h / float(hb)
    box_B_resized = box_B_orig.copy()
    box_B_resized[:, 0] *= sx
    box_B_resized[:, 1] *= sy
    return box_B_resized


def _infer_pair_id(path: str) -> int | None:
    name = osp.basename(path)
    m = re.match(r"pair(\d+)_\d+\.jpg$", name)
    if not m:
        return None
    return int(m.group(1))


def _run_for_checkpoint(
    ckpt_path: Path,
    label: str,
    image_a: str,
    image_b: str,
    resolution: str,
    symmetric: bool,
    device: str,
    num_matches: int,
    sample_pool: int,
    thresh_score: float,
    out_dir: Path,
    gt_path: str | None,
    strat_color: tuple[int, int, int],
) -> None:
    model = _load_model(ckpt_path, resolution, symmetric, device)
    warp, certainty = model.match(image_a, image_b)
    sparse_matches, sparse_certainty = model.sample(
        warp,
        certainty,
        num=max(sample_pool, num_matches),
        thresh_score=thresh_score,
        sample_seed=sum(ord(ch) for ch in f"{image_a}|{image_b}"),
    )
    h, w = warp.shape[:2]
    im_a = Image.open(image_a).convert("RGB").resize((w, h))
    im_b = Image.open(image_b).convert("RGB").resize((w, h))
    box_A_resized = _center_box_pixels(w, h, BOX_RATIO)
    strat_box = _warp_points(warp, box_A_resized)
    gt_box = None
    if gt_path:
        H_gt = _load_gt_homography(gt_path)
        gt_box = _gt_box_poly_from_H(image_a, image_b, h, w, H_gt)
    matches_np = _top_k_matches(sparse_matches, sparse_certainty, num_matches)
    if matches_np.size == 0:
        raise RuntimeError("No matches available for visualization.")
    kpts_a_t, kpts_b_t = model.to_pixel_coordinates(
        torch.from_numpy(matches_np),
        h,
        w,
        h,
        w,
    )
    kpts_a_np = kpts_a_t.detach().cpu().numpy()
    kpts_b_np = kpts_b_t.detach().cpu().numpy()
    out_path = out_dir / f"{label}_top{num_matches}.png"
    _draw_matches(
        im_a,
        im_b,
        kpts_a_np,
        kpts_b_np,
        out_path,
        title=label,
        gt_box=gt_box,
        strat_box=strat_box,
        strat_color=strat_color,
    )
    print(f"[VIS] Saved {out_path}")


def _resolve_ckpt(ckpt_arg: str | None, root: Path, dataset_name: str, job_name: str, cycle: int) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg).expanduser()
    stem = f"{job_name}_cycle{cycle}_best.pth"
    return root / dataset_name / job_name / stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize top-N correspondences for a single pair using two checkpoints."
    )
    parser.add_argument("--dataset_name", default="Optical-Infrared")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--pair_idx", type=int, default=None)
    parser.add_argument("--image_a", default=None)
    parser.add_argument("--image_b", default=None)
    parser.add_argument("--resolution", default="medium", choices=sorted(RESOLUTIONS.keys()))
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_matches", type=int, default=20)
    parser.add_argument("--sample_pool", type=int, default=2000)
    parser.add_argument("--thresh_score", type=float, default=0.05)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--checkpoint_root", default="/projects/_hdd/roma")
    parser.add_argument("--random_job", default="Optical-Infrared_random_1")
    parser.add_argument("--tuwe_job", default="Optical-Infrared_TUWE_1")
    parser.add_argument("--random_ckpt", default=None)
    parser.add_argument("--tuwe_ckpt", default=None)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "workspace" / "pics" / "pair_vis"))
    parser.add_argument("--gt_path", default=None, help="Path to gt_*.txt homography for the pair.")
    parser.add_argument("--random_color", default="cyan", help="Strategy box color for random.")
    parser.add_argument("--tuwe_color", default="orange", help="Strategy box color for TUWE.")
    args = parser.parse_args()

    if args.image_a and args.image_b:
        image_a, image_b = args.image_a, args.image_b
        if args.gt_path is None:
            a_id = _infer_pair_id(image_a)
            b_id = _infer_pair_id(image_b)
            if a_id is not None and a_id == b_id:
                data_root = get_dataset_root(args.data_root, args.dataset_name)
                args.gt_path = osp.join(data_root, f"gt_{a_id}.txt")
    else:
        if args.pair_idx is None:
            raise ValueError("Provide --pair_idx or --image_a/--image_b.")
        if args.dataset_name not in DATASET_DIRS:
            raise ValueError(f"Unknown dataset mapping for {args.dataset_name}")
        data_root = get_dataset_root(args.data_root, args.dataset_name)
        image_a, image_b = _resolve_pair_paths(data_root, args.pair_idx)
        if args.gt_path is None:
            args.gt_path = osp.join(data_root, f"gt_{args.pair_idx}.txt")

    out_dir = Path(args.out_dir).expanduser()
    _ensure_dir(out_dir)

    ckpt_root = Path(args.checkpoint_root).expanduser()
    ckpt_random = _resolve_ckpt(args.random_ckpt, ckpt_root, args.dataset_name, args.random_job, args.cycle)
    ckpt_tuwe = _resolve_ckpt(args.tuwe_ckpt, ckpt_root, args.dataset_name, args.tuwe_job, args.cycle)
    if not ckpt_random.is_file():
        raise FileNotFoundError(f"Random checkpoint not found: {ckpt_random}")
    if not ckpt_tuwe.is_file():
        raise FileNotFoundError(f"TUWE checkpoint not found: {ckpt_tuwe}")

    def parse_color(value: str) -> tuple[int, int, int]:
        color_map = {
            "cyan": (0, 191, 255),
            "orange": (255, 165, 0),
            "magenta": (255, 0, 255),
            "green": (0, 200, 0),
            "yellow": (255, 215, 0),
            "red": (220, 20, 60),
            "blue": (65, 105, 225),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        key = value.strip().lower()
        if key in color_map:
            return color_map[key]
        parts = [p.strip() for p in key.split(",")]
        if len(parts) == 3:
            return tuple(int(float(p)) for p in parts)  # type: ignore[return-value]
        raise ValueError(f"Unsupported color '{value}'. Use a name or 'R,G,B'.")

    if args.gt_path and not osp.isfile(args.gt_path):
        raise FileNotFoundError(f"GT file not found: {args.gt_path}")

    _run_for_checkpoint(
        ckpt_random,
        label=f"{args.random_job}_cycle{args.cycle}",
        image_a=image_a,
        image_b=image_b,
        resolution=args.resolution,
        symmetric=args.symmetric,
        device=args.device,
        num_matches=args.num_matches,
        sample_pool=args.sample_pool,
        thresh_score=args.thresh_score,
        out_dir=out_dir,
        gt_path=args.gt_path,
        strat_color=parse_color(args.random_color),
    )
    _run_for_checkpoint(
        ckpt_tuwe,
        label=f"{args.tuwe_job}_cycle{args.cycle}",
        image_a=image_a,
        image_b=image_b,
        resolution=args.resolution,
        symmetric=args.symmetric,
        device=args.device,
        num_matches=args.num_matches,
        sample_pool=args.sample_pool,
        thresh_score=args.thresh_score,
        out_dir=out_dir,
        gt_path=args.gt_path,
        strat_color=parse_color(args.tuwe_color),
    )


if __name__ == "__main__":
    main()
