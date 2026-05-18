#!/usr/bin/env python3
"""Spearman correlation between pairwise hs_cert uncertainty and geometric descriptor distance.

For a given cycle (checkpoint + unlabeled pool), uses:
  - strategy_hs_cert_3._hs_cert_scores  → hs_cert per pair + homography_sets side-effect
  - strategy_geometry_diversity.compute_geometric_diversity_from_homographies
      → reuses the *same* K=50 homographies; no second forward pass

For every ordered pair (i, j) in the pool:
  delta_u(i,j) = |u(i) - u(j)|      where u(x) = 1 - hs_cert(x)
  delta_g(i,j) = ||g(i) - g(j)||_2  (L2 distance between 8D descriptors)

Reports Spearman correlation(delta_u, delta_g).

Usage
-----
python hs_cert_geom_correlation.py \\
    --checkpoint path/to/cycle_best.pth \\
    --data-root  path/to/dataset \\
    --pool-npy   path/to/pool_idx.npy \\
    --out-dir    workspace/corr_plots \\
    --cycle-label cycle2
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
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores  # noqa: E402
from roma.strategies.strategy_geometry_diversity import (  # noqa: E402
    compute_geometric_diversity_from_homographies,
)

LOGGER = logging.getLogger(__name__)

RESOLUTIONS = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

# Cap on pairwise comparisons; pairs are subsampled if N*(N-1)/2 exceeds this.
DEFAULT_MAX_PAIRS = 1_000_000


# ---------------------------------------------------------------------------
# Lightweight strategy stub — satisfies what _hs_cert_scores needs
# ---------------------------------------------------------------------------

def _make_strategy(data_root: str) -> types.SimpleNamespace:
    strategy = types.SimpleNamespace()
    strategy.data_root = data_root
    strategy.homography_sets = {}
    return strategy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth).")
    p.add_argument("--data-root",  required=True, help="Dataset root directory.")
    p.add_argument(
        "--pool-npy", required=True,
        help=".npy file of pair indices forming the unlabeled pool for this cycle.",
    )
    p.add_argument("--out-dir", default="workspace/hs_cert_geom_corr",
                   help="Output directory (CSV, scatter plot, summary).")
    p.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="medium")
    p.add_argument("--image-size", type=int, default=560,
                   help="Image side length passed to the geometric descriptor (default 560).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for pairwise subsampling when pool is large.")
    p.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                   help="Cap on pairwise comparisons used for Spearman.")
    p.add_argument("--cycle-label", default=None,
                   help="Human-readable label (used in filenames and plot title). "
                        "Defaults to the checkpoint filename stem.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def configure_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, resolution: str, device: str):
    import torch
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    model = get_model(pretrained_backbone=True, resolution=resolution,
                      attenuate_cert=False, symmetric=False).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning("Checkpoint load: missing=%d unexpected=%d", len(missing), len(unexpected))
    model.eval()
    return model, dev


# ---------------------------------------------------------------------------
# Score + descriptor computation
# ---------------------------------------------------------------------------

def compute_scores_and_descriptors(
    model,
    data_root: str,
    pool_ids: np.ndarray,
    image_size: int,
) -> pd.DataFrame:
    """Return a DataFrame with columns: idx, hs_cert, uncertainty, g0…g7."""
    strategy = _make_strategy(data_root)

    # _hs_cert_scores: computes hs_cert for every id in pool_ids AND populates
    # strategy.homography_sets[pair_id] with the K=50 RANSAC homographies.
    hs_cert = _hs_cert_scores(strategy, model, pool_ids)   # (N,) in (0, 1]

    rows = []
    for i, pair_id in enumerate(pool_ids.tolist()):
        descriptor = compute_geometric_diversity_from_homographies(
            strategy.homography_sets.get(int(pair_id), []),
            image_size=image_size,
        )
        rows.append({
            "idx": int(pair_id),
            "hs_cert": float(hs_cert[i]),
            "uncertainty": float(1.0 - hs_cert[i]),
            **{f"g{j}": float(descriptor[j]) for j in range(8)},
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise Spearman correlation
# ---------------------------------------------------------------------------

def _flat_to_upper_triangle(N: int, flat_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat upper-triangle indices (k=1) to (row, col) arrays."""
    f = flat_idx.astype(np.float64)
    b = 2.0 * N - 1.0
    i = np.floor((b - np.sqrt(b * b - 8.0 * f)) / 2.0).astype(np.int64)
    i = np.clip(i, 0, N - 2)
    j = (f - i * (2 * N - i - 1) / 2 - 1).astype(np.int64) + i + 1
    # Fix any floating-point off-by-one
    bad = (j < i + 1) | (j >= N)
    i[bad] += 1
    j[bad] = (f[bad] - i[bad] * (2 * N - i[bad] - 1) / 2).astype(np.int64) + i[bad] + 1
    return i, j


def compute_pairwise_correlation(
    df: pd.DataFrame,
    max_pairs: int,
    seed: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (spearman_r, p_value, delta_u, delta_g)."""
    u = df["uncertainty"].to_numpy(dtype=np.float64)
    G = df[[f"g{j}" for j in range(8)]].to_numpy(dtype=np.float64)
    N = len(df)
    total = N * (N - 1) // 2

    if total == 0:
        return float("nan"), float("nan"), np.empty(0), np.empty(0)

    if total <= max_pairs:
        ii, jj = np.triu_indices(N, k=1)
    else:
        LOGGER.info("Subsampling %d / %d pairwise comparisons (seed=%d).", max_pairs, total, seed)
        rng = np.random.default_rng(seed)
        flat = np.sort(rng.choice(total, size=max_pairs, replace=False))
        ii, jj = _flat_to_upper_triangle(N, flat)

    delta_u = np.abs(u[ii] - u[jj])
    delta_g = np.sqrt(((G[ii] - G[jj]) ** 2).sum(axis=1))

    mask = np.isfinite(delta_u) & np.isfinite(delta_g)
    delta_u, delta_g = delta_u[mask], delta_g[mask]

    if len(delta_u) < 3:
        return float("nan"), float("nan"), delta_u, delta_g

    result = spearmanr(delta_u, delta_g)
    return float(result.statistic), float(result.pvalue), delta_u, delta_g


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter(
    delta_u: np.ndarray,
    delta_g: np.ndarray,
    spearman_r: float,
    p_value: float,
    cycle_label: str,
    out_path: Path,
) -> None:
    n_pts = len(delta_u)
    if n_pts > 50_000:
        rng = np.random.default_rng(0)
        sel = rng.choice(n_pts, 50_000, replace=False)
        du_plot, dg_plot = delta_u[sel], delta_g[sel]
    else:
        du_plot, dg_plot = delta_u, delta_g

    p_str = f"{p_value:.2e}" if np.isfinite(p_value) else "N/A"
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(du_plot, dg_plot, s=4, alpha=0.3, rasterized=True, color="#4477aa")
    ax.set_xlabel("|u(i) − u(j)|  (uncertainty difference)", fontsize=12)
    ax.set_ylabel("‖g(i) − g(j)‖₂  (geometric distance)", fontsize=12)
    ax.set_title(
        f"{cycle_label}\nSpearman r = {spearman_r:.4f}   p = {p_str}   n_pairs = {n_pts:,}",
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

    import torch  # noqa: F401 — imported here so the module is usable without GPU at import time

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

    # ── Cached CSV skips the expensive model run ─────────────────────────────
    cache_csv = out_dir / f"{safe_label}_scores_descriptors.csv"
    if cache_csv.is_file():
        LOGGER.info("Loading cached scores/descriptors from %s", cache_csv)
        df = pd.read_csv(cache_csv)
        df = df[df["idx"].isin(set(pool_ids.tolist()))].reset_index(drop=True)
        if len(df) == 0:
            LOGGER.warning("Cache exists but contains no pool indices — recomputing.")
            df = None
    else:
        df = None

    if df is None:
        LOGGER.info("Loading checkpoint: %s", args.checkpoint)
        model, device = load_model(args.checkpoint, args.resolution, args.device)
        LOGGER.info("Device: %s", device)

        LOGGER.info("Computing hs_cert + 8D geometric descriptors for %d pairs …", len(pool_ids))
        import torch
        with torch.no_grad():
            df = compute_scores_and_descriptors(
                model=model,
                data_root=args.data_root,
                pool_ids=pool_ids,
                image_size=args.image_size,
            )

        df.to_csv(str(cache_csv), index=False)
        LOGGER.info("Cached scores/descriptors -> %s  (%d rows)", cache_csv, len(df))

    LOGGER.info("Valid pairs after scoring: %d", len(df))
    if len(df) < 2:
        LOGGER.error("Too few valid pairs (%d) — cannot compute correlation.", len(df))
        return 1

    # ── Pairwise Spearman ────────────────────────────────────────────────────
    LOGGER.info("Computing pairwise Spearman correlation (max_pairs=%d) …", args.max_pairs)
    spearman_r, p_value, delta_u, delta_g = compute_pairwise_correlation(
        df, max_pairs=args.max_pairs, seed=args.seed,
    )

    n_pairs = len(delta_u)
    p_str = f"{p_value:.4e}" if np.isfinite(p_value) else "N/A"

    print(f"\n{'='*60}")
    print(f"Cycle:      {cycle_label}")
    print(f"Pool size:  {len(df)}")
    print(f"n_pairs:    {n_pairs:,}")
    print(f"Spearman r: {spearman_r:.6f}")
    print(f"p-value:    {p_str}")
    print(f"{'='*60}\n")

    # ── Save outputs ─────────────────────────────────────────────────────────
    pairs_csv = out_dir / f"{safe_label}_pairwise.csv"
    pd.DataFrame({"delta_u": delta_u, "delta_g": delta_g}).to_csv(str(pairs_csv), index=False)
    LOGGER.info("Pairwise table -> %s", pairs_csv)

    summary_csv = out_dir / "correlation_summary.csv"
    new_row = pd.DataFrame([{
        "cycle": cycle_label,
        "pool_size": len(df),
        "n_pairs": n_pairs,
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
    plot_scatter(delta_u, delta_g, spearman_r, p_value, cycle_label, scatter_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
