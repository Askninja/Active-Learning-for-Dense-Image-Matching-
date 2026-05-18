#!/usr/bin/env python3
"""
Optical-SAR dataset generator for OSEval.

Mirrors the NIR-RGB generator:
- Scans train / test / val splits under --sar_opt_root/OSdataset/256/
- Pairs optN.png with sarN.png within each split, then merges all splits

Default (homography) mode:
- Applies DPDN-style 4-corner perturbation homography
- pairN_1.jpg = Optical image
- pairN_2.jpg = warped SAR image
- gt_N.txt    = full 3x3 homography matrix

--no_homography mode (1-to-1):
- pairN_1.jpg = Optical image (unchanged)
- pairN_2.jpg = SAR image (unchanged, no warp)
- gt_N.txt    = 3x3 identity matrix

Always writes:
- Idx_files/train.npy  = 1-based indices of pairs from the train split
- Idx_files/test.npy   = 1-based indices of pairs from the test  split
- Idx_files/val.npy    = 1-based indices of pairs from the val   split
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
DEFAULT_ROOT    = Path("/projects/ALData/OSEval/OSdataset/256")
DEFAULT_OUT_DIR = Path("/projects/ALData/Active_Datasets/Optical-SAR")

SPLITS = ["train", "test", "val"]


# =====================================================
# CLI
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar_opt_root", type=Path, default=DEFAULT_ROOT,
                        help="Root dir containing train/test/val subdirs")
    parser.add_argument("--out_dir",      type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_samples",  type=int,  default=None)
    parser.add_argument("--overwrite",       action="store_true")
    parser.add_argument("--seed",            type=int,  default=42)
    parser.add_argument("--no_homography",   action="store_true",
                        help="Save images 1-to-1 (no warp); gt = identity matrix")
    return parser.parse_args()


# =====================================================
# PAIR COLLECTION
# =====================================================
def collect_pairs(root: Path) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    """Return (pairs, split_labels) across all splits.

    pairs        – sorted list of (opt_path, sar_path)
    split_labels – parallel list; split_labels[i] is the split name for pairs[i]
    """
    if not root.is_dir():
        raise FileNotFoundError(root)

    pairs: List[Tuple[Path, Path]] = []
    split_labels: List[str] = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"Warning: split dir not found, skipping: {split_dir}")
            continue

        opt_files = {
            int(p.stem.replace("opt", "")): p
            for p in sorted(split_dir.iterdir())
            if p.is_file() and p.stem.startswith("opt")
        }

        sar_files = {
            int(p.stem.replace("sar", "")): p
            for p in sorted(split_dir.iterdir())
            if p.is_file() and p.stem.startswith("sar")
        }

        common = sorted(set(opt_files) & set(sar_files))
        for key in common:
            pairs.append((opt_files[key], sar_files[key]))
            split_labels.append(split)

        print(f"  {split}: {len(common)} pairs")

    return pairs, split_labels


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

    print("Collecting pairs from splits:", SPLITS)
    pairs, split_labels = collect_pairs(args.sar_opt_root)
    total = len(pairs)
    print(f"Total pairs collected: {total}")

    if args.max_samples is not None:
        total = min(total, args.max_samples)

    written = 0
    skipped = 0

    # 1-based indices per split (populated as we write)
    split_indices: dict[str, list[int]] = {s: [] for s in SPLITS}

    for i in range(total):
        idx   = i + 1
        out1  = args.out_dir / f"pair{idx}_1.jpg"
        out2  = args.out_dir / f"pair{idx}_2.jpg"
        gt    = args.out_dir / f"gt_{idx}.txt"
        split = split_labels[i]

        if (
            not args.overwrite
            and out1.exists()
            and out2.exists()
            and gt.exists()
        ):
            split_indices[split].append(idx)
            skipped += 1
            continue

        opt_path, sar_path = pairs[i]

        try:
            opt = load_image(opt_path)
            sar = load_image(sar_path)

            if args.no_homography:
                # 1-to-1: save images as-is, gt = identity
                H = np.eye(3, dtype=np.float64)
                cv2.imwrite(str(out1), opt, [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imwrite(str(out2), sar, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                h, w = sar.shape[:2]
                H = random_homography(rng, h, w)
                warped_sar = cv2.warpPerspective(
                    sar,
                    H,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                cv2.imwrite(str(out1), opt,        [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imwrite(str(out2), warped_sar, [cv2.IMWRITE_JPEG_QUALITY, 100])

            save_matrix(gt, H)
            split_indices[split].append(idx)
            written += 1

        except Exception as e:
            print(f"skip {idx}: {e}")
            skipped += 1

        if idx % 100 == 0:
            print(f"{idx}/{total} processed | written={written} skipped={skipped}")

    # ------------------------------------------------------------------
    # Write Idx_files/train.npy, test.npy, val.npy
    # ------------------------------------------------------------------
    idx_dir = args.out_dir / "Idx_files"
    idx_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        arr = np.array(split_indices[split], dtype=np.int64)
        np.save(str(idx_dir / f"{split}.npy"), arr)
        print(f"Idx_files/{split}.npy  →  {len(arr)} indices")

    print("----- SUMMARY -----")
    print("total pairs found :", len(pairs))
    print("pairs processed   :", total)
    print("written           :", written)
    print("skipped           :", skipped)


if __name__ == "__main__":
    main()
