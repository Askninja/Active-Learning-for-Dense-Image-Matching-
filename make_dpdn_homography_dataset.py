#!/usr/bin/env python3
"""
DPDN (Depth-RGB) homography dataset generator.

- Scans flat dir under --dpdn_root for RGB_N.png / Depth_N.bmp pairs
- Applies DPDN-style 4-corner perturbation homography
- pairN_1.jpg = RGB image
- pairN_2.jpg = warped Depth image
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
DEFAULT_ROOT    = Path("/projects/ALData/DPDN")
DEFAULT_OUT_DIR = Path("/projects/ALData/Active_Datasets/DPDN")

RHO = 32


# =====================================================
# CLI
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpdn_root",    type=Path, default=DEFAULT_ROOT,
                        help="Flat dir containing RGB_N.png and Depth_N.bmp files")
    parser.add_argument("--out_dir",      type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_samples",  type=int,  default=None)
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--seed",         type=int,  default=42)
    return parser.parse_args()


# =====================================================
# PAIR COLLECTION
# =====================================================
def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Return sorted list of (rgb_path, depth_path) matched by index N."""
    if not root.is_dir():
        raise FileNotFoundError(root)

    rgb_files = {
        int(p.stem.split("_")[1]): p
        for p in root.iterdir()
        if p.is_file() and p.stem.startswith("RGB_") and p.suffix.lower() == ".png"
    }

    depth_files = {
        int(p.stem.split("_")[1]): p
        for p in root.iterdir()
        if p.is_file() and p.stem.startswith("Depth_") and p.suffix.lower() == ".bmp"
    }

    common = sorted(set(rgb_files) & set(depth_files))
    return [(rgb_files[k], depth_files[k]) for k in common]


# =====================================================
# IMAGE LOAD
# =====================================================
def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img


# =====================================================
# HOMOGRAPHY GENERATION
# =====================================================
def random_homography(rng, h, w):
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
def save_matrix(path: Path, H):
    np.savetxt(str(path), H, fmt="%.8f")


# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.dpdn_root)
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

        rgb_path, depth_path = pairs[i]

        try:
            rgb   = load_image(rgb_path)
            depth = load_image(depth_path)

            h, w = depth.shape[:2]
            H = random_homography(rng, h, w)

            warped_depth = cv2.warpPerspective(
                depth,
                H,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            if warped_depth.ndim == 2:
                warped_depth = cv2.cvtColor(warped_depth, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(str(out1), rgb,         [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(out2), warped_depth, [cv2.IMWRITE_JPEG_QUALITY, 100])
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
