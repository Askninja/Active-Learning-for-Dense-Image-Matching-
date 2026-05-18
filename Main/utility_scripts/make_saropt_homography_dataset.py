#!/usr/bin/env python3
"""
Generate paired Optical-SAR dataset with ground-truth 3x3 homography matrices.

Outputs per pair:
  pairN_1.jpg  -> optical image
  pairN_2.jpg  -> warped SAR image
  gt_N.txt     -> 3x3 homography matrix H_AB
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Hardcoded paths ────────────────────────────────────────────────────
OPT_DIR = Path("/projects/ALData/SAR/QXSLAB_SAROPT/opt_256_oc_0.2")
SAR_DIR = Path("/projects/ALData/SAR/QXSLAB_SAROPT/sar_256_oc_0.2")
OUT_DIR = Path("/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-SAR")

RHO         = 64    # max corner perturbation in pixels (rho = image_size / 4)
SEED        = 42
MAX_SAMPLES = None  # set to an int to limit, e.g. 5000

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_sorted(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS
    )


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    return img


def generate_pair(
    image_A: np.ndarray,
    image_B: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (image_A, warped_image_B, H_AB_3x3) or None if degenerate.
    """
    h, w = image_A.shape[:2]

    if image_B.shape[:2] != (h, w):
        image_B = cv2.resize(image_B, (w, h))

    # 4 corners of the full image: TL, TR, BR, BL
    corners_A = np.array([
        [0,     0    ],
        [w - 1, 0    ],
        [w - 1, h - 1],
        [0,     h - 1],
    ], dtype=np.float32)

    # Random perturbation in [-rho, rho] per corner per axis
    deltas    = rng.integers(-RHO, RHO + 1, size=(4, 2)).astype(np.float32)
    corners_B = corners_A + deltas

    # H_AB from exact 4-point correspondences
    H_AB = cv2.getPerspectiveTransform(corners_A, corners_B)
    if H_AB is None:
        return None
    H_AB = H_AB / H_AB[2, 2]

    # Warp SAR with H_BA = inv(H_AB)
    H_BA = np.linalg.inv(H_AB)
    H_BA = H_BA / H_BA[2, 2]

    warped_B = cv2.warpPerspective(
        image_B, H_BA, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return image_A, warped_B, H_AB.astype(np.float64)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    opt_files = collect_sorted(OPT_DIR)
    sar_files = collect_sorted(SAR_DIR)

    total = min(len(opt_files), len(sar_files))
    if MAX_SAMPLES:
        total = min(total, MAX_SAMPLES)

    rng     = np.random.default_rng(SEED)
    written = 0

    print(f"Generating {total} pairs | rho={RHO}px | out={OUT_DIR}")

    for i in range(total):
        idx   = i + 1
        out_A = OUT_DIR / f"pair{idx}_1.jpg"
        out_B = OUT_DIR / f"pair{idx}_2.jpg"
        out_H = OUT_DIR / f"gt_{idx}.txt"

        try:
            result = generate_pair(
                load_image(opt_files[i]),
                load_image(sar_files[i]),
                rng=rng,
            )
        except Exception as e:
            print(f"  [error {idx}] {e}")
            continue

        if result is None:
            print(f"  [skip  {idx}] degenerate homography")
            continue

        img_A, img_B, H_AB = result

        cv2.imwrite(str(out_A), img_A, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(out_B), img_B, [cv2.IMWRITE_JPEG_QUALITY, 100])
        np.savetxt(str(out_H), H_AB, fmt="%.10f")

        written += 1
        if idx % 200 == 0:
            print(f"  {idx}/{total} done")

    print(f"\nDone. written={written}  out={OUT_DIR}")


if __name__ == "__main__":
    main()