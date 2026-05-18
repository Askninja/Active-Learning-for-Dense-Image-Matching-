#!/usr/bin/env python3
"""Dataset signal diagnostics: hs_cert discriminability vs geometric diversity spread.

Compares two datasets (default: Optical-Infrared and Optical-Depth) at the
pretrained checkpoint using the same scoring pipeline as hs_cert_geom_correlation.

Metrics per dataset
-------------------
  std(hs_cert)              uncertainty score spread across the pool
  mean(delta_g | <=20)      mean pairwise geometric descriptor distance,
                            outlier pairs with delta_g > 20 excluded

Outputs
-------
  {out_dir}/signal_diagnostics.png   side-by-side histograms (hs_cert | delta_g)
  {out_dir}/signal_diagnostics.csv   one row per dataset

Usage
-----
python utility_scripts/dataset_signal_diagnostics.py \\
    --checkpoint-a workspace/checkpoints/Optical-Infrared/pretrained_seed.pth \\
    --pool-npy-a   ../datasets/cross_modality/Optical-Infrared/Idx_files/train_idx.npy \\
    --data-root-a  ../datasets/cross_modality/Optical-Infrared \\
    --label-a      Optical-Infrared \\
    --checkpoint-b workspace/checkpoints/Optical-Depth/pretrained_seed.pth \\
    --pool-npy-b   ../datasets/cross_modality/Optical-Depth/Idx_files/train_idx.npy \\
    --data-root-b  ../datasets/cross_modality/Optical-Depth \\
    --label-b      Optical-Depth \\
    --out-dir      workspace/corr_plots/diagnostics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the scoring pipeline from the correlation script
from utility_scripts.hs_cert_geom_correlation import (  # noqa: E402
    load_model,
    compute_scores_and_descriptors,
)

LOGGER = logging.getLogger(__name__)

DELTA_G_OUTLIER_THRESHOLD = 20.0   # pairs with delta_g > this are excluded from mean
MAX_PAIRS_DELTA_G = 1_000_000      # cap pairwise sampling for large pools


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    for tag, default_ds in (("a", "Optical-Infrared"), ("b", "Optical-Depth")):
        p.add_argument(f"--checkpoint-{tag}", required=True,
                       help=f"Checkpoint for dataset {tag.upper()}.")
        p.add_argument(f"--data-root-{tag}", required=True,
                       help=f"Dataset root for dataset {tag.upper()}.")
        p.add_argument(f"--pool-npy-{tag}", required=True,
                       help=f".npy of unlabeled pool indices for dataset {tag.upper()}.")
        p.add_argument(f"--label-{tag}", default=default_ds,
                       help=f"Display label for dataset {tag.upper()} (default: {default_ds}).")

    p.add_argument("--out-dir", default="workspace/corr_plots/diagnostics")
    p.add_argument("--resolution", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--image-size", type=int, default=560)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delta-g-threshold", type=float, default=DELTA_G_OUTLIER_THRESHOLD,
                   help="Pairs with delta_g above this are excluded from mean(delta_g).")
    p.add_argument("--max-pairs", type=int, default=MAX_PAIRS_DELTA_G)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def configure_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Pairwise delta_g
# ---------------------------------------------------------------------------

def pairwise_delta_g(G: np.ndarray, max_pairs: int, seed: int) -> np.ndarray:
    """Return 1-D array of all pairwise L2 distances, subsampled if necessary."""
    N = len(G)
    total = N * (N - 1) // 2
    if total == 0:
        return np.empty(0, dtype=np.float64)

    if total <= max_pairs:
        ii, jj = np.triu_indices(N, k=1)
    else:
        LOGGER.info("Subsampling %d / %d pairwise delta_g (seed=%d).", max_pairs, total, seed)
        rng = np.random.default_rng(seed)
        flat = np.sort(rng.choice(total, size=max_pairs, replace=False))
        ii, jj = _flat_to_upper_triangle(N, flat)

    diff = G[ii] - G[jj]
    return np.sqrt((diff ** 2).sum(axis=1))


def _flat_to_upper_triangle(N: int, flat_idx: np.ndarray):
    f = flat_idx.astype(np.float64)
    b = 2.0 * N - 1.0
    i = np.floor((b - np.sqrt(b * b - 8.0 * f)) / 2.0).astype(np.int64)
    i = np.clip(i, 0, N - 2)
    j = (f - i * (2 * N - i - 1) / 2 - 1).astype(np.int64) + i + 1
    bad = (j < i + 1) | (j >= N)
    i[bad] += 1
    j[bad] = (f[bad] - i[bad] * (2 * N - i[bad] - 1) / 2).astype(np.int64) + i[bad] + 1
    return i, j


# ---------------------------------------------------------------------------
# Per-dataset metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    delta_g_threshold: float,
    max_pairs: int,
    seed: int,
) -> dict:
    hs = df["hs_cert"].to_numpy(dtype=np.float64)
    G = df[[f"g{j}" for j in range(8)]].to_numpy(dtype=np.float64)

    dg = pairwise_delta_g(G, max_pairs=max_pairs, seed=seed)
    dg_filtered = dg[dg <= delta_g_threshold]

    return {
        "n_pairs": len(df),
        "hs_cert_mean": float(np.mean(hs)),
        "hs_cert_std": float(np.std(hs)),
        "hs_cert_min": float(np.min(hs)),
        "hs_cert_max": float(np.max(hs)),
        "n_pairwise": len(dg),
        "n_pairwise_filtered": len(dg_filtered),
        "mean_delta_g_filtered": float(np.mean(dg_filtered)) if len(dg_filtered) > 0 else float("nan"),
        "std_delta_g_filtered": float(np.std(dg_filtered)) if len(dg_filtered) > 0 else float("nan"),
        "_hs_array": hs,
        "_dg_array": dg_filtered,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_diagnostics(
    results: list[dict],   # each dict has label, metrics
    out_path: Path,
    delta_g_threshold: float,
) -> None:
    colors = ["#4477aa", "#ee6677"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: hs_cert histograms ────────────────────────────────────────────
    ax = axes[0]
    for r, color in zip(results, colors):
        hs = r["metrics"]["_hs_array"]
        ax.hist(hs, bins=40, range=(0.0, 1.0), alpha=0.6, color=color,
                label=f"{r['label']}  (std={r['metrics']['hs_cert_std']:.4f})",
                density=True, edgecolor="none")
    ax.set_xlabel("hs_cert", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("hs_cert distribution\n(higher std → more discriminable)", fontsize=11)
    ax.legend(fontsize=10)

    # ── Right: delta_g histograms (filtered) ────────────────────────────────
    ax = axes[1]
    for r, color in zip(results, colors):
        dg = r["metrics"]["_dg_array"]
        if len(dg) == 0:
            continue
        mean_dg = r["metrics"]["mean_delta_g_filtered"]
        ax.hist(dg, bins=50, alpha=0.6, color=color,
                label=f"{r['label']}  (mean={mean_dg:.3f})",
                density=True, edgecolor="none")
    ax.set_xlabel(f"‖g(i) − g(j)‖₂  (delta_g ≤ {delta_g_threshold:.0f})", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pairwise geometric descriptor distance\n(outliers > threshold excluded)", fontsize=11)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Diagnostics figure -> %s", out_path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], delta_g_threshold: float) -> None:
    col_w = 32
    header = f"{'Dataset':<{col_w}} {'N pairs':>8} {'std(hs_cert)':>14} {'mean(hs_cert)':>14} {'mean(delta_g|<=threshold)':>26} {'pct outliers':>13}"
    print()
    print("=" * len(header))
    print("Signal Discriminability Diagnostics")
    print(f"  delta_g outlier threshold = {delta_g_threshold}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        n_total = m["n_pairwise"]
        n_filt  = m["n_pairwise_filtered"]
        pct_out = 100.0 * (1.0 - n_filt / max(n_total, 1))
        print(
            f"{r['label']:<{col_w}} "
            f"{m['n_pairs']:>8d} "
            f"{m['hs_cert_std']:>14.6f} "
            f"{m['hs_cert_mean']:>14.6f} "
            f"{m['mean_delta_g_filtered']:>26.4f} "
            f"{pct_out:>12.1f}%"
        )
    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_or_compute(
    checkpoint: str,
    data_root: str,
    pool_npy: str,
    label: str,
    resolution: str,
    image_size: int,
    device: str,
    out_dir: Path,
) -> pd.DataFrame:
    safe = label.replace("/", "_").replace(" ", "_")
    cache_csv = out_dir / f"{safe}_scores_descriptors.csv"

    pool_ids = np.load(pool_npy).astype(int).ravel()
    LOGGER.info("[%s] Pool: %d pairs", label, len(pool_ids))

    if cache_csv.is_file():
        LOGGER.info("[%s] Loading cache from %s", label, cache_csv)
        df = pd.read_csv(str(cache_csv))
        df = df[df["idx"].isin(set(pool_ids.tolist()))].reset_index(drop=True)
        if len(df) > 0:
            return df
        LOGGER.warning("[%s] Cache found but no matching pool indices — recomputing.", label)

    LOGGER.info("[%s] Loading checkpoint: %s", label, checkpoint)
    model, dev = load_model(checkpoint, resolution, device)
    LOGGER.info("[%s] Device: %s", label, dev)

    LOGGER.info("[%s] Computing hs_cert + 8D descriptors …", label)
    with torch.no_grad():
        df = compute_scores_and_descriptors(
            model=model,
            data_root=data_root,
            pool_ids=pool_ids,
            image_size=image_size,
        )

    df.to_csv(str(cache_csv), index=False)
    LOGGER.info("[%s] Cached -> %s  (%d rows)", label, cache_csv, len(df))
    return df


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        dict(
            checkpoint=args.checkpoint_a,
            data_root=args.data_root_a,
            pool_npy=args.pool_npy_a,
            label=args.label_a,
        ),
        dict(
            checkpoint=args.checkpoint_b,
            data_root=args.data_root_b,
            pool_npy=args.pool_npy_b,
            label=args.label_b,
        ),
    ]

    results = []
    for ds in datasets:
        df = _load_or_compute(
            checkpoint=ds["checkpoint"],
            data_root=ds["data_root"],
            pool_npy=ds["pool_npy"],
            label=ds["label"],
            resolution=args.resolution,
            image_size=args.image_size,
            device=args.device,
            out_dir=out_dir,
        )
        metrics = compute_metrics(
            df,
            delta_g_threshold=args.delta_g_threshold,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
        results.append({"label": ds["label"], "metrics": metrics})
        LOGGER.info(
            "[%s] std(hs_cert)=%.4f  mean(delta_g|<=%.0f)=%.4f",
            ds["label"], metrics["hs_cert_std"], args.delta_g_threshold,
            metrics["mean_delta_g_filtered"],
        )

    print_summary(results, delta_g_threshold=args.delta_g_threshold)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig_path = out_dir / "signal_diagnostics.png"
    plot_diagnostics(results, fig_path, delta_g_threshold=args.delta_g_threshold)

    # ── CSV summary ──────────────────────────────────────────────────────────
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "dataset": r["label"],
            "n_pairs": m["n_pairs"],
            "hs_cert_mean": m["hs_cert_mean"],
            "hs_cert_std": m["hs_cert_std"],
            "hs_cert_min": m["hs_cert_min"],
            "hs_cert_max": m["hs_cert_max"],
            "n_pairwise_sampled": m["n_pairwise"],
            "n_pairwise_filtered": m["n_pairwise_filtered"],
            "mean_delta_g_filtered": m["mean_delta_g_filtered"],
            "std_delta_g_filtered": m["std_delta_g_filtered"],
        })
    summary_csv = out_dir / "signal_diagnostics.csv"
    pd.DataFrame(rows).to_csv(str(summary_csv), index=False)
    LOGGER.info("Summary CSV -> %s", summary_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
