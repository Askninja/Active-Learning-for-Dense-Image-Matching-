#!/usr/bin/env python3
"""Spearman correlation between pairwise geometric descriptor distance and feature distance.

For each pair in the unlabeled pool:
  g(x)  8D geometric descriptor via compute_geometric_diversity_from_homographies,
         reusing homography_sets populated as a side effect of _hs_cert_scores
  f(x)  RoMa backbone features at the finest scale (min(feature_pyramid.keys())),
         channel-wise avg-pool on A and B concatenated, then L2-normalized —
         exactly as in strategy_coreset_appearance / _pair_embedding_fine

For all (i, j):
  delta_g(i,j) = ||g(i) - g(j)||_2   pairs with delta_g > --delta-g-max are excluded
  delta_f(i,j) = ||f(i) - f(j)||_2

Spearman correlation(delta_g, delta_f) measures whether the geometric descriptor
space is consistent with the learned feature representation space.

Reuses load_model, _flat_to_upper_triangle from hs_cert_geom_correlation.

Usage
-----
python utility_scripts/geom_feature_correlation.py \\
    --checkpoint workspace/checkpoints/Optical-Depth/pretrained_seed.pth \\
    --data-root  ../datasets/cross_modality/Optical-Depth \\
    --pool-npy   ../datasets/cross_modality/Optical-Depth/Idx_files/train_idx.npy \\
    --cycle-label pretrained \\
    --out-dir     workspace/corr_plots/geom_feat
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
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores  # noqa: E402
from roma.strategies.strategy_geometry_diversity import (  # noqa: E402
    compute_geometric_diversity_from_homographies,
)
from roma.utils.utils import get_tuple_transform_ops  # noqa: E402
from utility_scripts.hs_cert_geom_correlation import (  # noqa: E402
    load_model,
    _flat_to_upper_triangle,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_PAIRS = 1_000_000
DEFAULT_DELTA_G_MAX = 20.0  # exclude geometric outliers above this


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth).")
    p.add_argument("--data-root",  required=True, help="Dataset root directory.")
    p.add_argument("--pool-npy",   required=True, help=".npy of unlabeled pool indices.")
    p.add_argument("--out-dir",    default="workspace/corr_plots/geom_feat")
    p.add_argument("--resolution", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--image-size", type=int, default=560,
                   help="image_size passed to geometric descriptor (default 560).")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--max-pairs",  type=int, default=DEFAULT_MAX_PAIRS)
    p.add_argument("--delta-g-max", type=float, default=DEFAULT_DELTA_G_MAX,
                   help="Pairs with delta_g above this are excluded from the correlation.")
    p.add_argument("--cycle-label", default=None,
                   help="Display label (defaults to checkpoint filename stem).")
    p.add_argument("--log-level",  default="INFO")
    return p.parse_args(argv)


def configure_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Feature extraction at scale 16
# ---------------------------------------------------------------------------

def _resolve_pair_paths(data_root: str, idx: int):
    root = Path(data_root)
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        a = root / f"pair{idx}_1{ext}"
        b = root / f"pair{idx}_2{ext}"
        if a.exists() and b.exists():
            return str(a), str(b)
    raise FileNotFoundError(f"Images for pair {idx} not found under {data_root}")


@torch.no_grad()
def extract_feature_vector(
    model,
    a_path: str,
    b_path: str,
    device: str,
) -> np.ndarray:
    """Mirror _pair_embedding_fine from strategies.py + L2-norm from coreset_appearance.

    Uses finest scale (min(feature_pyramid.keys())), avg-pools A and B separately,
    concatenates, then L2-normalizes — identical to what strategy_coreset_appearance
    feeds into k-center greedy.
    """
    h_res, w_res = model.h_resized, model.w_resized
    transform = get_tuple_transform_ops(resize=(h_res, w_res), normalize=True, clahe=False)

    im_a = Image.open(a_path).convert("RGB")
    im_b = Image.open(b_path).convert("RGB")
    im_a, im_b = transform((im_a, im_b))

    batch = {
        "im_A": im_a[None].to(device),
        "im_B": im_b[None].to(device),
    }

    feature_pyramid = model.extract_backbone_features(batch, batched=True, upsample=False)
    finest_scale = min(feature_pyramid.keys())
    feats = feature_pyramid[finest_scale]        # (2, C, H', W')
    feat_a, feat_b = feats.chunk(2, dim=0)       # each (1, C, H', W')

    embedding = torch.cat(
        [feat_a.mean(dim=(2, 3)), feat_b.mean(dim=(2, 3))], dim=1
    )                                            # (1, 2C)  — raw, matches _pair_embedding_fine

    f = embedding[0].float().cpu().numpy()       # (2C,)

    # L2-normalize — matches coreset_appearance which normalizes before k-center
    norm = float(np.linalg.norm(f))
    if norm > 1e-8:
        f = f / norm

    return f


# ---------------------------------------------------------------------------
# Per-pair computation
# ---------------------------------------------------------------------------

def compute_descriptors_and_features(
    model,
    device: str,
    data_root: str,
    pool_ids: np.ndarray,
    image_size: int,
) -> pd.DataFrame:
    """
    For each pair: run _hs_cert_scores to get homography_sets, then extract
    g(x) from those homographies and f(x) from backbone scale 16.

    Returns DataFrame: idx | g0..g7 | f0..f_{2C-1}
    """
    strategy = types.SimpleNamespace()
    strategy.data_root = data_root
    strategy.homography_sets = {}

    # _hs_cert_scores: forward pass + RANSAC → hs_cert + populates homography_sets
    LOGGER.info("Running _hs_cert_scores for %d pairs …", len(pool_ids))
    _hs_cert_scores(strategy, model, pool_ids)

    rows = []
    feat_dim = None

    for step, pair_id in enumerate(pool_ids.tolist()):
        # ── geometric descriptor (reuses stored homographies, no second pass) ──
        g = compute_geometric_diversity_from_homographies(
            strategy.homography_sets.get(int(pair_id), []),
            image_size=image_size,
        )

        # ── feature vector (one backbone forward pass at scale 16) ────────────
        try:
            a_path, b_path = _resolve_pair_paths(data_root, int(pair_id))
            f = extract_feature_vector(model, a_path, b_path, device)
        except Exception as exc:
            LOGGER.warning("Pair %d: feature extraction failed (%s); skipping.", pair_id, exc)
            continue

        if feat_dim is None:
            feat_dim = len(f)
            LOGGER.info("Feature dimension (2×C, finest backbone scale, L2-norm): %d", feat_dim)

        row = {"idx": int(pair_id)}
        row.update({f"g{j}": float(g[j]) for j in range(8)})
        row.update({f"f{j}": float(f[j]) for j in range(len(f))})
        rows.append(row)

        if (step + 1) % 50 == 0:
            LOGGER.info("  processed %d / %d pairs", step + 1, len(pool_ids))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise correlation
# ---------------------------------------------------------------------------

def compute_pairwise_correlation(
    df: pd.DataFrame,
    delta_g_max: float,
    max_pairs: int,
    seed: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (spearman_r, p_value, delta_g, delta_f, mask_used)."""
    g_cols = [c for c in df.columns if c.startswith("g") and c[1:].isdigit()]
    f_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]

    G = df[sorted(g_cols, key=lambda c: int(c[1:]))].to_numpy(dtype=np.float64)
    F = df[sorted(f_cols, key=lambda c: int(c[1:]))].to_numpy(dtype=np.float64)
    N = len(df)

    total = N * (N - 1) // 2
    if total == 0:
        return float("nan"), float("nan"), np.empty(0), np.empty(0), np.empty(0, dtype=bool)

    if total <= max_pairs:
        ii, jj = np.triu_indices(N, k=1)
    else:
        LOGGER.info("Subsampling %d / %d pairs (seed=%d).", max_pairs, total, seed)
        rng = np.random.default_rng(seed)
        flat = np.sort(rng.choice(total, size=max_pairs, replace=False))
        ii, jj = _flat_to_upper_triangle(N, flat)

    delta_g = np.sqrt(((G[ii] - G[jj]) ** 2).sum(axis=1))
    delta_f = np.sqrt(((F[ii] - F[jj]) ** 2).sum(axis=1))

    mask = np.isfinite(delta_g) & np.isfinite(delta_f) & (delta_g <= delta_g_max)
    dg_filt = delta_g[mask]
    df_filt = delta_f[mask]

    n_excluded = int((~mask).sum())
    LOGGER.info(
        "Pairs: total=%d  after filter (delta_g<=%.1f): %d  excluded: %d",
        len(delta_g), delta_g_max, mask.sum(), n_excluded,
    )

    if len(dg_filt) < 3:
        return float("nan"), float("nan"), dg_filt, df_filt, mask

    result = spearmanr(dg_filt, df_filt)
    return float(result.statistic), float(result.pvalue), dg_filt, df_filt, mask


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_scatter(
    delta_g: np.ndarray,
    delta_f: np.ndarray,
    spearman_r: float,
    p_value: float,
    cycle_label: str,
    delta_g_max: float,
    out_path: Path,
) -> None:
    n_pts = len(delta_g)
    if n_pts > 50_000:
        rng = np.random.default_rng(0)
        sel = rng.choice(n_pts, 50_000, replace=False)
        dg_plot, df_plot = delta_g[sel], delta_f[sel]
    else:
        dg_plot, df_plot = delta_g, delta_f

    p_str = f"{p_value:.2e}" if np.isfinite(p_value) else "N/A"
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dg_plot, df_plot, s=4, alpha=0.3, rasterized=True, color="#44aa77")
    ax.set_xlabel(f"‖g(i) − g(j)‖₂  (geometric, delta_g ≤ {delta_g_max:.0f})", fontsize=12)
    ax.set_ylabel("‖f(i) − f(j)‖₂  (L2-norm feature, finest scale)", fontsize=12)
    ax.set_title(
        f"{cycle_label}\n"
        f"Spearman r = {spearman_r:.4f}   p = {p_str}   n_pairs = {n_pts:,}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Scatter plot -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle_label = args.cycle_label or Path(args.checkpoint).stem
    safe_label  = cycle_label.replace("/", "_").replace(" ", "_")

    pool_npy = Path(args.pool_npy).expanduser()
    if not pool_npy.is_file():
        LOGGER.error("Pool npy not found: %s", pool_npy)
        return 1
    pool_ids = np.load(str(pool_npy)).astype(int).ravel()
    LOGGER.info("Pool: %d pairs", len(pool_ids))

    # ── Cache CSV ─────────────────────────────────────────────────────────────
    cache_csv = out_dir / f"{safe_label}_geom_feat.csv"
    if cache_csv.is_file():
        LOGGER.info("Loading cache from %s", cache_csv)
        df = pd.read_csv(str(cache_csv))
        df = df[df["idx"].isin(set(pool_ids.tolist()))].reset_index(drop=True)
        if len(df) == 0:
            LOGGER.warning("Cache found but no matching pool indices — recomputing.")
            df = None
    else:
        df = None

    if df is None:
        LOGGER.info("Loading checkpoint: %s", args.checkpoint)
        model, device = load_model(args.checkpoint, args.resolution, args.device)
        LOGGER.info("Device: %s", device)

        with torch.no_grad():
            df = compute_descriptors_and_features(
                model=model,
                device=device,
                data_root=args.data_root,
                pool_ids=pool_ids,
                image_size=args.image_size,
            )

        df.to_csv(str(cache_csv), index=False)
        LOGGER.info("Cached -> %s  (%d rows)", cache_csv, len(df))

    f_dim = len([c for c in df.columns if c.startswith("f") and c[1:].isdigit()])
    LOGGER.info("Valid pairs: %d | g_dim=8 | f_dim=%d", len(df), f_dim)

    if len(df) < 2:
        LOGGER.error("Too few valid pairs (%d) — cannot compute correlation.", len(df))
        return 1

    # ── Pairwise Spearman ─────────────────────────────────────────────────────
    LOGGER.info(
        "Computing pairwise Spearman correlation "
        "(delta_g <= %.1f, max_pairs=%d) …",
        args.delta_g_max, args.max_pairs,
    )
    spearman_r, p_value, delta_g, delta_f, _ = compute_pairwise_correlation(
        df,
        delta_g_max=args.delta_g_max,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    n_pairs = len(delta_g)
    p_str = f"{p_value:.4e}" if np.isfinite(p_value) else "N/A"

    print(f"\n{'='*60}")
    print(f"Cycle:          {cycle_label}")
    print(f"Pool size:      {len(df)}")
    print(f"n_pairs (filt): {n_pairs:,}")
    print(f"f_dim:          {f_dim}  (finest scale, avg-pooled A+B, L2-norm)")
    print(f"Spearman r:     {spearman_r:.6f}")
    print(f"p-value:        {p_str}")
    print(f"{'='*60}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    pairs_csv = out_dir / f"{safe_label}_pairwise.csv"
    pd.DataFrame({"delta_g": delta_g, "delta_f": delta_f}).to_csv(str(pairs_csv), index=False)
    LOGGER.info("Pairwise table -> %s", pairs_csv)

    summary_csv = out_dir / "correlation_summary.csv"
    new_row = pd.DataFrame([{
        "cycle": cycle_label,
        "pool_size": len(df),
        "n_pairs_filtered": n_pairs,
        "f_dim": f_dim,
        "feature_scale": "finest",
        "delta_g_max": args.delta_g_max,
        "spearman_r": spearman_r,
        "p_value": p_value,
    }])
    if summary_csv.is_file():
        prev = pd.read_csv(str(summary_csv))
        prev = prev[prev["cycle"] != cycle_label]
        new_row = pd.concat([prev, new_row], ignore_index=True)
    new_row.to_csv(str(summary_csv), index=False)
    LOGGER.info("Summary -> %s", summary_csv)

    scatter_path = out_dir / f"{safe_label}_scatter.png"
    plot_scatter(delta_g, delta_f, spearman_r, p_value, cycle_label,
                 args.delta_g_max, scatter_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
