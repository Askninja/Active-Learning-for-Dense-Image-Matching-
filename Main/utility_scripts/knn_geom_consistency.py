#!/usr/bin/env python3
"""KNN geometric consistency: g(x) vs f(x) neighbour quality.

For each pair in the pool:
  g(x)  8D geometric descriptor (RANSAC-based, IQR-normalised) — same as in
         strategy_geometry_diversity / strategy_pairwise_hscert_df_dg.
  f(x)  RoMa backbone at finest scale, avg-pooled A+B, L2-normalised —
         same as strategy_coreset_appearance / geom_feature_correlation.

For each pair i find its K nearest neighbours under g(x) and under f(x).
Geometric quality of a neighbourhood is mean |diff| in each scalar GT-
homography property between query pair and its neighbours — lower means
the neighbours are more geometrically homogeneous.

GT-homography scalar properties (from gt_{idx}.txt):
  translation_mag   projected-centre displacement / image_size
  scale             sqrt(|det(H[:2,:2])|)
  rotation_deg      |rotation angle| extracted via SVD, degrees
  perspective_mag   ||H[2, :2]||_2

Output: bar chart + CSV.

Usage
-----
python utility_scripts/knn_geom_consistency.py \\
    --checkpoint /projects/roma/Optical-Infrared/Optical-Infrared_geometry_diversity/Optical-Infrared_geometry_diversity_cycle3_best.pth \\
    --data-root  ../datasets/cross_modality/Optical-Infrared \\
    --pool-npy   ../datasets/cross_modality/Optical-Infrared/Idx_files/train_idx.npy \\
    --out-dir    workspace/knn_consistency \\
    --cycle-label pretrained \\
    --k 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores  # noqa: E402
from roma.strategies.strategy_geometry_diversity import (  # noqa: E402
    compute_geometric_diversity_from_homographies,
    normalize_geometric_descriptors,
)
from roma.utils.utils import get_tuple_transform_ops  # noqa: E402
from utility_scripts.hs_cert_geom_correlation import load_model  # noqa: E402

LOGGER = logging.getLogger(__name__)

GEO_PROPS = ["translation_mag", "scale", "rotation_deg", "perspective_mag"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True, help="Path to model checkpoint.")
    p.add_argument("--data-root",    required=True, help="Dataset root directory.")
    p.add_argument("--pool-npy",     required=True, help=".npy of pool pair indices.")
    p.add_argument("--out-dir",      default="workspace/knn_consistency")
    p.add_argument("--resolution",   choices=("low", "medium", "high"), default="medium")
    p.add_argument("--image-size",   type=int, default=560,
                   help="Image side length for geometric descriptor (default 560).")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--k",            type=int, default=10,
                   help="Number of nearest neighbours (default 10).")
    p.add_argument("--cycle-label",  default=None,
                   help="Display label; defaults to checkpoint stem.")
    p.add_argument("--log-level",    default="INFO")
    return p.parse_args(argv)


def configure_logging(level_str: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_str.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# GT homography → scalar properties
# ---------------------------------------------------------------------------

def load_gt_homography(data_root: str, pair_id: int) -> np.ndarray | None:
    """Load gt_{pair_id}.txt → (3, 3) float64, normalised so H[2,2]=1."""
    path = Path(data_root) / f"gt_{pair_id}.txt"
    if not path.exists():
        return None
    H = np.loadtxt(str(path), dtype=np.float64)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        return None
    if abs(H[2, 2]) < 1e-10:
        return None
    return H / H[2, 2]


def homography_scalar_properties(H: np.ndarray, image_size: int = 560) -> dict[str, float]:
    """Four scalar geometric properties from a (3,3) normalised homography.

    translation_mag: how far the image centre moves under H, divided by image_size.
    scale:           area scaling factor = sqrt(|det(affine part)|).
    rotation_deg:    absolute rotation angle (degrees) from SVD decomposition.
    perspective_mag: L2 norm of the perspective row H[2, :2].
    """
    s = float(image_size)

    # Translation: project image centre and measure displacement
    ctr = np.array([s / 2.0, s / 2.0, 1.0])
    proj = H @ ctr
    denom = proj[2] if abs(proj[2]) > 1e-10 else 1e-10
    proj_cart = proj[:2] / denom
    translation_mag = float(np.linalg.norm(proj_cart - ctr[:2]) / s)

    # Scale: geometric mean of singular values of affine part
    A = H[:2, :2]
    scale = float(np.sqrt(max(abs(np.linalg.det(A)), 0.0)))

    # Rotation: from SVD of affine part
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    rotation_deg = float(abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))

    # Perspective: magnitude of last-row off-diagonal elements
    perspective_mag = float(np.linalg.norm(H[2, :2]))

    return {
        "translation_mag": translation_mag,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "perspective_mag": perspective_mag,
    }


# ---------------------------------------------------------------------------
# Feature extraction  (mirrors geom_feature_correlation.extract_feature_vector)
# ---------------------------------------------------------------------------

def _resolve_pair_paths(data_root: str, idx: int) -> tuple[str, str]:
    root = Path(data_root)
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        a, b = root / f"pair{idx}_1{ext}", root / f"pair{idx}_2{ext}"
        if a.exists() and b.exists():
            return str(a), str(b)
    raise FileNotFoundError(f"Images for pair {idx} not found under {data_root}")


@torch.no_grad()
def extract_feature_vector(model, a_path: str, b_path: str, device: str) -> np.ndarray:
    """Penultimate-scale backbone features, avg-pooled A+B, L2-normalised.

    Matches the corrected _pair_embedding_fine in strategies.py:
    uses sorted_scales[-2] (penultimate = second-coarsest) with global avg-pool
    instead of spatial flatten.
    """
    transform = get_tuple_transform_ops(
        resize=(model.h_resized, model.w_resized), normalize=True, clahe=False
    )
    im_a, im_b = transform((Image.open(a_path).convert("RGB"), Image.open(b_path).convert("RGB")))
    batch = {"im_A": im_a[None].to(device), "im_B": im_b[None].to(device)}
    fp = model.extract_backbone_features(batch, batched=True, upsample=False)
    penultimate_scale = sorted(fp.keys())[-2]
    feats = fp[penultimate_scale]
    feat_a, feat_b = feats.chunk(2, dim=0)
    f = torch.cat([feat_a.mean(dim=(2, 3)), feat_b.mean(dim=(2, 3))], dim=1)[0].float().cpu().numpy()
    norm = float(np.linalg.norm(f))
    return f / norm if norm > 1e-8 else f


# ---------------------------------------------------------------------------
# Per-pair computation
# ---------------------------------------------------------------------------

def compute_all_descriptors(
    model,
    device: str,
    data_root: str,
    pool_ids: np.ndarray,
    image_size: int,
) -> pd.DataFrame:
    """For every pair: compute g(x) (raw 8D), f(x) (L2-norm backbone), and GT scalar properties."""
    strategy = types.SimpleNamespace()
    strategy.data_root = data_root
    strategy.homography_sets = {}

    LOGGER.info("Running _hs_cert_scores for %d pairs …", len(pool_ids))
    _hs_cert_scores(strategy, model, pool_ids)

    rows: list[dict] = []
    feat_dim: int | None = None

    for step, pair_id in enumerate(pool_ids.tolist()):
        pair_id = int(pair_id)

        g = compute_geometric_diversity_from_homographies(
            strategy.homography_sets.get(pair_id, []),
            image_size=image_size,
        )

        try:
            a_path, b_path = _resolve_pair_paths(data_root, pair_id)
            f = extract_feature_vector(model, a_path, b_path, device)
        except Exception as exc:
            LOGGER.warning("Pair %d: f(x) extraction failed (%s) — skipping.", pair_id, exc)
            continue

        H_gt = load_gt_homography(data_root, pair_id)
        if H_gt is None:
            LOGGER.warning("Pair %d: GT homography missing — skipping.", pair_id)
            continue

        if feat_dim is None:
            feat_dim = len(f)
            LOGGER.info("f(x) dim=%d (finest backbone scale, L2-norm)", feat_dim)

        row: dict = {"idx": pair_id}
        row.update({f"g{j}": float(g[j]) for j in range(8)})
        row.update({f"f{j}": float(f[j]) for j in range(len(f))})
        row.update(homography_scalar_properties(H_gt, image_size=image_size))
        rows.append(row)

        if (step + 1) % 50 == 0:
            LOGGER.info("  processed %d / %d pairs", step + 1, len(pool_ids))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KNN consistency metric
# ---------------------------------------------------------------------------

def knn_consistency(embeddings: np.ndarray, geo_props: np.ndarray, k: int) -> np.ndarray:
    """Mean |diff| in each geometric property between each pair and its K nearest neighbours.

    Args:
        embeddings: (N, D) embedding matrix used for neighbour search.
        geo_props:  (N, P) scalar geometric properties.
        k:          neighbourhood size.

    Returns:
        (P,) mean absolute difference per property, averaged over all N pairs.
    """
    N = embeddings.shape[0]
    k = min(k, N - 1)

    dist = cdist(embeddings, embeddings, metric="euclidean")   # (N, N), avoids (N,N,D)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argsort(dist, axis=1)[:, :k]                  # (N, k)

    mean_abs_diff = np.zeros(geo_props.shape[1], dtype=np.float64)
    for i in range(N):
        nbr = geo_props[nn_idx[i]]                             # (k, P)
        mean_abs_diff += np.mean(np.abs(nbr - geo_props[i]), axis=0)
    return mean_abs_diff / N


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_bar_chart(
    consistency_g: np.ndarray,
    consistency_f: np.ndarray,
    prop_names: list[str],
    cycle_label: str,
    k: int,
    out_path: Path,
) -> None:
    n_props = len(prop_names)
    x = np.arange(n_props)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_props * 2.2), 5))
    bars_g = ax.bar(x - width / 2, consistency_g, width, label="g(x) neighbours", color="#4477aa")
    bars_f = ax.bar(x + width / 2, consistency_f, width, label="f(x) neighbours", color="#ee6677")

    for bars in (bars_g, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + h * 0.01 + 1e-9,
                f"{h:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(prop_names, fontsize=11)
    ax.set_ylabel("Mean |diff|  (lower = more consistent)", fontsize=11)
    ax.set_title(
        f"{cycle_label}  —  K={k} nearest neighbours\n"
        "Geometric consistency of g(x) vs f(x) neighbourhoods",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Bar chart -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle_label = args.cycle_label or Path(args.checkpoint).stem
    safe_label = cycle_label.replace("/", "_").replace(" ", "_")

    pool_npy = Path(args.pool_npy).expanduser()
    if not pool_npy.is_file():
        LOGGER.error("Pool npy not found: %s", pool_npy)
        return 1
    pool_ids = np.load(str(pool_npy)).astype(int).ravel()
    LOGGER.info("Pool: %d pairs from %s", len(pool_ids), pool_npy)

    # Cache CSV skips the expensive model forward-pass
    cache_csv = out_dir / f"{safe_label}_descriptors.csv"
    df: pd.DataFrame | None = None
    if cache_csv.is_file():
        LOGGER.info("Loading cache: %s", cache_csv)
        df = pd.read_csv(str(cache_csv))
        df = df[df["idx"].isin(set(pool_ids.tolist()))].reset_index(drop=True)
        if len(df) == 0:
            LOGGER.warning("Cache found but no matching indices — recomputing.")
            df = None

    if df is None:
        model, device = load_model(args.checkpoint, args.resolution, args.device)
        LOGGER.info("Device: %s", device)
        with torch.no_grad():
            df = compute_all_descriptors(
                model=model,
                device=device,
                data_root=args.data_root,
                pool_ids=pool_ids,
                image_size=args.image_size,
            )
        df.to_csv(str(cache_csv), index=False)
        LOGGER.info("Cached descriptors -> %s  (%d rows)", cache_csv, len(df))

    LOGGER.info("Valid pairs: %d", len(df))
    if len(df) < args.k + 1:
        LOGGER.error("Not enough valid pairs (%d) for K=%d — aborting.", len(df), args.k)
        return 1

    g_cols = sorted([c for c in df.columns if c.startswith("g") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    f_cols = sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()], key=lambda c: int(c[1:]))

    G_norm = normalize_geometric_descriptors(df[g_cols].to_numpy(dtype=np.float64))
    F      = df[f_cols].to_numpy(dtype=np.float64)
    props  = df[GEO_PROPS].to_numpy(dtype=np.float64)

    LOGGER.info("Computing KNN consistency (K=%d) …", args.k)
    consistency_g = knn_consistency(G_norm, props, args.k)
    consistency_f = knn_consistency(F,      props, args.k)

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"Cycle: {cycle_label}   K={args.k}   N={len(df)}")
    print(f"{'Property':<22} {'g(x) nbrs':>12} {'f(x) nbrs':>12} {'ratio g/f':>10}")
    print("-" * 62)
    for prop, vg, vf in zip(GEO_PROPS, consistency_g, consistency_f):
        ratio = vg / vf if vf > 1e-12 else float("nan")
        print(f"{prop:<22} {vg:>12.6f} {vf:>12.6f} {ratio:>10.4f}")
    print(f"{'='*62}\n")

    # ── CSV ───────────────────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "property": GEO_PROPS,
        "g_neighbors_mean_abs_diff": consistency_g,
        "f_neighbors_mean_abs_diff": consistency_f,
        "ratio_g_over_f": consistency_g / np.maximum(consistency_f, 1e-12),
    })
    result_csv = out_dir / f"{safe_label}_knn_k{args.k}.csv"
    result_df.to_csv(str(result_csv), index=False)
    LOGGER.info("Results CSV -> %s", result_csv)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    bar_path = out_dir / f"{safe_label}_knn_k{args.k}.png"
    plot_bar_chart(consistency_g, consistency_f, GEO_PROPS, cycle_label, args.k, bar_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
