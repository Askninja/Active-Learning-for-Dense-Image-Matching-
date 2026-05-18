#!/usr/bin/env python3
"""
Flatten the METU VisTIR dataset into the standard active-learning format:

    pair{N}_1.jpg   visible image (camera A)
    pair{N}_2.jpg   thermal image (camera B)
    gt_{N}.txt      3×3 fundamental matrix F  (so that x_B^T F x_A = 0)

Split index files (1-based pair indices):
    train.npy       val-list pairs  (full pool for AL)
    preseed.npy     random subset of train  (initial labelled seed)
    test.npy        test-list pairs  (held-out evaluation)
    val.npy         random subset of test-list (hyper-param tuning)

Usage:
    python make_metu_vistir_dataset.py [--overwrite]
"""

from __future__ import annotations

import argparse

from pathlib import Path

import cv2
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/projects/ALData/METU_VisTIR")
OUT_DIR   = Path("/projects/ALData/Active_Datasets/METU_VisTIR")

MODALITY   = "vis_tir"   # "vis_tir" | "vis_vis" | "tir_tir"
SEED       = 42
VAL_SIZE   = 217         # drawn from test-list pairs
PRESEED    = 75          # drawn from train-list pairs

JPEG_Q = [cv2.IMWRITE_JPEG_QUALITY, 95]
# ─────────────────────────────────────────────────────────────────────────────


def skew(t: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([
        [    0, -t[2],  t[1]],
        [ t[2],     0, -t[0]],
        [-t[1],  t[0],     0],
    ])


def fundamental_matrix(K_A: np.ndarray, K_B: np.ndarray,
                       T_i: np.ndarray, T_j: np.ndarray) -> np.ndarray:
    """Return F such that  x_B^T F x_A = 0.

    Poses are cam-to-world (4×4).  T_rel = inv(T_j) @ T_i maps a point
    expressed in cam_i into cam_j.
    """
    T_rel = np.linalg.inv(T_j) @ T_i
    R     = T_rel[:3, :3]
    t     = T_rel[:3,  3]
    E = skew(t) @ R
    F = np.linalg.inv(K_B).T @ E @ np.linalg.inv(K_A)
    if abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]
    return F


def load_pairs_from_list(split: str) -> list[dict]:
    """Read all NPZ scene files for *split* and return a flat list of pair dicts."""
    list_path = DATA_ROOT / "index" / "val_test_list" / f"{split}_list.txt"
    scene_names = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]

    pairs: list[dict] = []
    for name in scene_names:
        npz_path = DATA_ROOT / "index" / f"scene_info_{split}" / name
        d = np.load(str(npz_path), allow_pickle=True)
        image_paths = d["image_paths"]   # (N, 2)  [vis, tir]
        intrinsics  = d["intrinsics"]    # (N, 2, 3, 3)
        poses       = d["poses"]         # (N, 4, 4)  cam-to-world
        pair_infos  = d["pair_infos"]    # (M, 2)

        for pi in pair_infos:
            i, j = int(pi[0]), int(pi[1])

            if MODALITY == "vis_tir":
                path_A = DATA_ROOT / image_paths[i][0]
                path_B = DATA_ROOT / image_paths[j][1]
                K_A = intrinsics[i][0].astype(np.float64)
                K_B = intrinsics[j][1].astype(np.float64)
            elif MODALITY == "vis_vis":
                path_A = DATA_ROOT / image_paths[i][0]
                path_B = DATA_ROOT / image_paths[j][0]
                K_A = intrinsics[i][0].astype(np.float64)
                K_B = intrinsics[j][0].astype(np.float64)
            else:   # tir_tir
                path_A = DATA_ROOT / image_paths[i][1]
                path_B = DATA_ROOT / image_paths[j][1]
                K_A = intrinsics[i][1].astype(np.float64)
                K_B = intrinsics[j][1].astype(np.float64)

            T_i_mat = poses[i].astype(np.float64)
            T_j_mat = poses[j].astype(np.float64)

            pairs.append(dict(
                path_A=path_A,
                path_B=path_B,
                K_A=K_A,
                K_B=K_B,
                T_i=T_i_mat,
                T_j=T_j_mat,
            ))

    return pairs


def write_pair(idx: int, pair: dict, out_dir: Path, overwrite: bool) -> bool:
    """Write pair{idx}_1.jpg, pair{idx}_2.jpg, gt_{idx}.txt, ka_{idx}.txt, kb_{idx}.txt.

    Returns True if anything was written.
    Images are skipped when they already exist (unless overwrite).
    K and F files are always written if missing.
    """
    out_A  = out_dir / f"pair{idx}_1.jpg"
    out_B  = out_dir / f"pair{idx}_2.jpg"
    out_F  = out_dir / f"gt_{idx}.txt"
    out_KA = out_dir / f"ka_{idx}.txt"
    out_KB = out_dir / f"kb_{idx}.txt"

    wrote = False

    if overwrite or not (out_A.exists() and out_B.exists()):
        img_A = cv2.imread(str(pair["path_A"]))
        img_B = cv2.imread(str(pair["path_B"]))
        if img_A is None or img_B is None:
            raise RuntimeError(f"Cannot read images:\n  {pair['path_A']}\n  {pair['path_B']}")
        cv2.imwrite(str(out_A), img_A, JPEG_Q)
        cv2.imwrite(str(out_B), img_B, JPEG_Q)
        wrote = True

    if overwrite or not out_F.exists():
        F = fundamental_matrix(pair["K_A"], pair["K_B"], pair["T_i"], pair["T_j"])
        np.savetxt(str(out_F), F, fmt="%.10f")
        wrote = True

    if overwrite or not (out_KA.exists() and out_KB.exists()):
        np.savetxt(str(out_KA), pair["K_A"], fmt="%.10f")
        np.savetxt(str(out_KB), pair["K_B"], fmt="%.10f")
        wrote = True

    return wrote


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true",
                   help="Re-write files that already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load pairs ────────────────────────────────────────────────────────────
    print("Loading val-list pairs  …", end=" ", flush=True)
    val_pairs  = load_pairs_from_list("val")
    print(len(val_pairs))

    print("Loading test-list pairs …", end=" ", flush=True)
    test_pairs = load_pairs_from_list("test")
    print(len(test_pairs))

    all_pairs = val_pairs + test_pairs
    n_val_list  = len(val_pairs)
    n_total     = len(all_pairs)
    print(f"Total pairs: {n_total}  (val-list={n_val_list}, test-list={len(test_pairs)})")

    # ── write images + F matrices ─────────────────────────────────────────────
    written = skipped = errors = 0
    for i, pair in enumerate(all_pairs):
        idx = i + 1
        try:
            did_write = write_pair(idx, pair, OUT_DIR, args.overwrite)
            if did_write:
                written += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [error pair {idx}] {e}")
            errors += 1

        if idx % 200 == 0:
            print(f"  {idx}/{n_total}  written={written}  skipped={skipped}  errors={errors}")

    print(f"\nDone writing: written={written}  skipped={skipped}  errors={errors}")

    # ── create split index arrays (70 / 15 / 15, 1-based) ───────────────────
    all_idx = np.arange(1, n_total + 1)
    rng.shuffle(all_idx)

    n_test  = round(n_total * 0.15)
    n_val   = n_total - round(n_total * 0.70) - n_test

    test_idx    = np.sort(all_idx[:n_test])
    val_idx     = np.sort(all_idx[n_test:n_test + n_val])
    train_idx   = np.sort(all_idx[n_test + n_val:])
    n_preseed   = round(len(train_idx) * 0.07)
    preseed_idx = np.sort(rng.choice(train_idx, size=n_preseed, replace=False))

    idx_dir = OUT_DIR / "Idx_files"
    idx_dir.mkdir(exist_ok=True)

    np.save(idx_dir / "train.npy",   train_idx)
    np.save(idx_dir / "preseed.npy", preseed_idx)
    np.save(idx_dir / "test.npy",    test_idx)
    np.save(idx_dir / "val.npy",     val_idx)

    print(f"\nSplit sizes:")
    print(f"  train   = {train_idx.size}  (full pool)")
    print(f"  preseed = {preseed_idx.size}  (initial labelled seed ⊂ train)")
    print(f"  test    = {test_idx.size}")
    print(f"  val     = {val_idx.size}")
    print(f"\nIdx files → {idx_dir}")
    print(f"Images    → {OUT_DIR}")


if __name__ == "__main__":
    main()
