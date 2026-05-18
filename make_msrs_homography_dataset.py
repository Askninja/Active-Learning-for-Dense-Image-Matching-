#!/usr/bin/env python3
"""
MSRS (Multi-Spectral Road Scenarios) homography dataset generator.

- Scans MSRS split dirs for matching vi/<name>.png / ir/<name>.png pairs
- Applies 4-corner perturbation homography to the IR image
- pairN_1.jpg = visible (RGB) image
- pairN_2.jpg = warped infrared image
- gt_N.txt    = full 3x3 homography matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# =====================================================
# CONFIG
# =====================================================
DEFAULT_ROOT    = Path("/projects/ALData/MSRS")
DEFAULT_OUT_DIR = Path("/projects/ALData/Active_Datasets/MSRS")


# =====================================================
# CLI
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msrs_root",   type=Path, default=DEFAULT_ROOT,
                        help="Root of MSRS dataset (contains train/ and test/ subdirs)")
    parser.add_argument("--out_dir",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split",       type=str,  default="train",
                        choices=["train", "test", "all"],
                        help="Which split(s) to process")
    parser.add_argument("--max_samples", type=int,  default=None)
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--seed",        type=int,  default=42)
    return parser.parse_args()


# =====================================================
# PAIR COLLECTION
# =====================================================
def collect_pairs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    """Return sorted list of (vi_path, ir_path) matched by filename."""
    splits = ["train", "test"] if split == "all" else [split]
    pairs: List[Tuple[Path, Path]] = []

    for sp in splits:
        vi_dir = root / sp / "vi"
        ir_dir = root / sp / "ir"

        if not vi_dir.is_dir():
            raise FileNotFoundError(vi_dir)
        if not ir_dir.is_dir():
            raise FileNotFoundError(ir_dir)

        vi_files = {p.name: p for p in vi_dir.iterdir() if p.suffix.lower() == ".png"}
        ir_files = {p.name: p for p in ir_dir.iterdir() if p.suffix.lower() == ".png"}

        common = sorted(set(vi_files) & set(ir_files))
        pairs.extend((vi_files[name], ir_files[name]) for name in common)

    return pairs


# =====================================================
# IMAGE LOAD
# =====================================================
def load_image(path: Path, grayscale: bool = False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img


# =====================================================
# HOMOGRAPHY GENERATION
# =====================================================
def random_homography(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    src = np.float32([
        [0,     0    ],
        [w - 1, 0    ],
        [w - 1, h - 1],
        [0,     h - 1],
    ])

    rho = max(8, int(0.125 * min(h, w)))
    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.integers(-rho, rho + 1)
        dst[i, 1] += rng.integers(-rho, rho + 1)

    return cv2.getPerspectiveTransform(src, dst)


# =====================================================
# SAVE MATRIX
# =====================================================
def save_matrix(path: Path, H: np.ndarray) -> None:
    np.savetxt(str(path), H, fmt="%.8f")


# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.msrs_root, args.split)
    total = len(pairs)

    if args.max_samples is not None:
        total = min(total, args.max_samples)

    written = 0
    skipped = 0

    for i in range(total):
        idx  = i + 1
        out1 = args.out_dir / f"pair{idx}_1.jpg"
        out2 = args.out_dir / f"pair{idx}_2.jpg"
        gt   = args.out_dir / f"gt_{idx}.txt"

        if (
            not args.overwrite
            and out1.exists()
            and out2.exists()
            and gt.exists()
        ):
            skipped += 1
            continue

        vi_path, ir_path = pairs[i]

        try:
            vi = load_image(vi_path, grayscale=False)
            ir = load_image(ir_path, grayscale=True)

            h, w = ir.shape[:2]
            H = random_homography(rng, h, w)

            warped_ir = cv2.warpPerspective(
                ir,
                H,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            warped_ir_bgr = cv2.cvtColor(warped_ir, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(str(out1), vi,          [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(out2), warped_ir_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            save_matrix(gt, H)

            written += 1

        except Exception as e:
            print(f"skip {idx}: {e}")
            skipped += 1

        if idx % 100 == 0:
            print(f"{idx}/{total} processed | written={written} skipped={skipped}")

    print("----- SUMMARY -----")
    print("total pairs found :", len(pairs))
    print("pairs processed   :", total)
    print("written           :", written)
    print("skipped           :", skipped)


if __name__ == "__main__":
    main()
