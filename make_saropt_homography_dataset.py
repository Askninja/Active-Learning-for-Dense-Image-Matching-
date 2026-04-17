#!/usr/bin/env python3
"""
Create paired Optical-SAR dataset with visible affine transforms.

Requirements implemented:
- pairN_1.jpg = Nth image in sorted optical folder
- pairN_2.jpg = Nth image in sorted SAR folder after transform
- Indexes match by sorted order position (NOT basename intersection)
- No forced grayscale conversion (preserve original color/channels)
- Do NOT fill empty rotated areas:
    borderMode = BORDER_CONSTANT (black empty regions)
- Save gt_N.txt = 2x3 affine matrix
- Output images resized to 256x256 if needed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

# =====================================================
# CONFIG
# =====================================================
ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_OPT_DIR = Path("/projects/ALData/SAR/QXSLAB_SAROPT/opt_256_oc_0.2")
DEFAULT_SAR_DIR = Path("/projects/ALData/SAR/QXSLAB_SAROPT/sar_256_oc_0.2")
DEFAULT_OUT_DIR = Path(
    "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-SAR"
)

TARGET_SIZE = (256, 256)  # (width, height)


# =====================================================
# CLI
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_dir", type=Path, default=DEFAULT_OPT_DIR)
    parser.add_argument("--sar_dir", type=Path, default=DEFAULT_SAR_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# =====================================================
# FILE COLLECTION (INDEX MATCHING BY ORDER)
# =====================================================
def collect_sorted(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS
    )
    return files


# =====================================================
# LOAD IMAGE (COLOR, NO GRAYSCALE)
# =====================================================
def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Failed to read: {path}")

    h, w = img.shape[:2]

    if (w, h) != TARGET_SIZE:
        interp = cv2.INTER_AREA if (w > TARGET_SIZE[0] or h > TARGET_SIZE[1]) else cv2.INTER_LINEAR
        img = cv2.resize(img, TARGET_SIZE, interpolation=interp)

    return img


# =====================================================
# RANDOM AFFINE
# =====================================================
def random_affine_matrix(rng, width=256, height=256):
    rotation_deg = rng.uniform(-75, 75)
    tx = rng.uniform(-60, 60)
    ty = rng.uniform(-60, 60)
    scale = rng.uniform(0.80, 1.30)
    shear_deg = rng.uniform(-18, 18)

    theta = np.deg2rad(rotation_deg)
    shear = np.deg2rad(shear_deg)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float64)

    back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    Sh = np.array([
        [1, np.tan(shear), 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    M = T @ back @ Sh @ R @ S @ to_origin

    return M[:2, :]


# =====================================================
# SAVE MATRIX
# =====================================================
def save_matrix(path: Path, matrix):
    np.savetxt(str(path), matrix, fmt="%.8f")


# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    opt_files = collect_sorted(args.opt_dir)
    sar_files = collect_sorted(args.sar_dir)

    total = min(len(opt_files), len(sar_files))

    if args.max_samples is not None:
        total = min(total, args.max_samples)

    rng = np.random.default_rng(args.seed)

    written = 0
    skipped = 0

    for i in range(total):
        idx = i + 1

        opt_path = opt_files[i]
        sar_path = sar_files[i]

        out1 = args.out_dir / f"pair{idx}_1.jpg"
        out2 = args.out_dir / f"pair{idx}_2.jpg"
        gt = args.out_dir / f"gt_{idx}.txt"

        if (
            not args.overwrite
            and out1.exists()
            and out2.exists()
            and gt.exists()
        ):
            skipped += 1
            continue

        try:
            optical = load_image(opt_path)
            sar = load_image(sar_path)

            M = random_affine_matrix(rng, 256, 256)

            warped = cv2.warpAffine(
                sar,
                M,
                TARGET_SIZE,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            cv2.imwrite(str(out1), optical, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(out2), warped, [cv2.IMWRITE_JPEG_QUALITY, 100])

            save_matrix(gt, M)

            written += 1

        except Exception as e:
            print(f"skip {idx}: {e}")
            skipped += 1

        if idx % 100 == 0:
            print(f"{idx}/{total} processed | written={written} skipped={skipped}")

    print("----- SUMMARY -----")
    print("optical files :", len(opt_files))
    print("sar files     :", len(sar_files))
    print("pairs made    :", total)
    print("written       :", written)
    print("skipped       :", skipped)


if __name__ == "__main__":
    main()