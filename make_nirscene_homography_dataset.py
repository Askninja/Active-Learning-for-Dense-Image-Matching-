#!/usr/bin/env python3
"""
NIR-RGB homography dataset generator for nirscene1.

Mirrors the SAR-Optical generator:
- Scans all category subdirs under --nir_rgb_root
- Pairs *_rgb.tiff with matching *_nir.tiff by stem ID
- Merges all categories into one flat output
- Applies DPDN-style 4-corner perturbation homography
- pairN_1.jpg = RGB image
- pairN_2.jpg = warped NIR image
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
ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_ROOT    = Path("/projects/ALData/nirscene1/nirscene1")
DEFAULT_OUT_DIR = Path("/projects/ALData/Active_Datasets/NIR-RGB")

RHO = 32


# =====================================================
# CLI
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nir_rgb_root", type=Path, default=DEFAULT_ROOT,
                        help="Root dir containing one subdir per scene category")
    parser.add_argument("--out_dir",      type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_samples",  type=int,  default=None)
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--seed",         type=int,  default=42)
    return parser.parse_args()


# =====================================================
# PAIR COLLECTION
# =====================================================
def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Return sorted list of (rgb_path, nir_path) across all category dirs."""
    if not root.is_dir():
        raise FileNotFoundError(root)

    pairs: List[Tuple[Path, Path]] = []

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue

        rgb_files = {
            p.stem.replace("_rgb", ""): p
            for p in sorted(cat_dir.iterdir())
            if p.is_file()
            and p.suffix.lower() in ACCEPTED_EXTS
            and p.stem.endswith("_rgb")
        }

        nir_files = {
            p.stem.replace("_nir", ""): p
            for p in sorted(cat_dir.iterdir())
            if p.is_file()
            and p.suffix.lower() in ACCEPTED_EXTS
            and p.stem.endswith("_nir")
        }

        common = sorted(set(rgb_files) & set(nir_files))
        for key in common:
            pairs.append((rgb_files[key], nir_files[key]))

    return pairs


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

    pairs = collect_pairs(args.nir_rgb_root)
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

        rgb_path, nir_path = pairs[i]

        try:
            rgb = load_image(rgb_path)
            nir = load_image(nir_path)

            h, w = nir.shape[:2]
            H = random_homography(rng, h, w)

            warped_nir = cv2.warpPerspective(
                nir,
                H,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            cv2.imwrite(str(out1), rgb,        [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(out2), warped_nir, [cv2.IMWRITE_JPEG_QUALITY, 100])
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
