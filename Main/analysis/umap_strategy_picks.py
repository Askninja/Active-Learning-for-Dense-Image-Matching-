#!/usr/bin/env python3
"""UMAP visualisation of active-learning picks across strategies.

Computes backbone embeddings for every sample in the train split using a
chosen model checkpoint, projects them with UMAP, then overlays which
indices were picked by each strategy at a given cycle.

Typical usage
-------------
# Latest strategies at cycle 3
python analysis/umap_strategy_picks.py

# Different model / cycle
python analysis/umap_strategy_picks.py \
    --checkpoint /projects/roma/Optical-Infrared/Optical-Infrared_hs_cert/Optical-Infrared_hs_cert_cycle3_best.pth \
    --cycle 1

# Custom strategy subset
python analysis/umap_strategy_picks.py --strategies coreset,hs_cert_3,badge --cycle 2
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.datasets import OpticalMap  # noqa: E402

LOGGER = logging.getLogger(__name__)

RESOLUTIONS = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

# Keep this aligned with ActiveLearningStrategy dispatch in roma/strategies/strategies.py.
LATEST_STRATEGIES = [
    "random",
    "coreset",
    "entropy",
    "hs_cert",
    "hs_cert_new",
    "hs_cert_3",
    "combined_diversity",
    "combined_metric_diversity",
    "uncertainty_metric_diversity",
    "entropy_weighted_coreset",
    "hs_cert_weighted_coreset",
    "geometry_diversity",
    "coreset_appearance",
    "eigenvalue_diversity",
    "displacement_diversity",
    "combined_eigen_displacement",
    "hs_cert_weighted_eigenvalue_diversity",
    "entropy_weighted_geometric_diversity",
    "hs_cert_weighted_geometric_diversity",
    "hs_cert_delta4_geomdiv",
    "badge",
    "learn_loss",
]

DEFAULT_CHECKPOINT = (
    "/projects/roma/Optical-Infrared/"
    "Optical-Infrared_geometry_diversity/"
    "Optical-Infrared_geometry_diversity_cycle3_best.pth"
)
DEFAULT_DATA_ROOT = (
    "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/"
    "datasets/cross_modality/Optical-Infrared"
)
DEFAULT_OUT_DIR = str(REPO_ROOT / "analysis" / "plots" / "umap_strategy_picks")

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Model checkpoint used to compute backbone embeddings.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--idx-root", default=None,
                   help="Path to Idx_files dir. Defaults to <data-root>/Idx_files.")
    p.add_argument("--split-stem", default="train_idx",
                   help="Filename stem (no .npy) for the pool split.")
    p.add_argument("--dataset-name", default="Optical-Infrared")
    p.add_argument("--resolution", choices=list(RESOLUTIONS), default="medium")
    p.add_argument("--cycle", type=int, default=3,
                   help="Which AL cycle's picked indices to visualise.")
    p.add_argument("--strategies", default="latest",
                   help='"latest" (default), "all" (auto-discover), or comma-separated names.')
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=784)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", default="euclidean")
    p.add_argument("--cache-embeddings", action="store_true", default=True,
                   help="Save/load embeddings to .npy cache so re-runs skip the forward pass.")
    p.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def configure_logging(level_str: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_str.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_model(ckpt_path: str, resolution: str, device: torch.device) -> torch.nn.Module:
    model = get_model(
        pretrained_backbone=True,
        resolution=resolution,
        attenuate_cert=False,
        symmetric=False,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        LOGGER.warning("Missing keys (%d) loading checkpoint", len(missing))
    if unexpected:
        LOGGER.warning("Unexpected keys (%d) loading checkpoint", len(unexpected))
    model.eval()
    return model


def build_dataset(data_root: str, split_path: str, resolution: str) -> OpticalMap:
    ht, wt = RESOLUTIONS[resolution]
    return OpticalMap(
        data_root=data_root,
        ht=ht,
        wt=wt,
        use_horizontal_flip_aug=False,
        use_cropping_aug=False,
        use_color_jitter_aug=False,
        use_swap_aug=False,
        use_dual_cropping_aug=False,
        split=split_path,
    )


def compute_embeddings(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    embeddings = []
    total = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            feat_pyr = model.extract_backbone_features(batch, batched=True, upsample=False)
            scale = max(feat_pyr.keys())
            features = feat_pyr[scale]
            feat_a, feat_b = features.chunk(2, dim=0)
            emb_a = feat_a.mean(dim=(2, 3))
            emb_b = feat_b.mean(dim=(2, 3))
            emb = torch.cat((emb_a, emb_b), dim=1)
            emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
            embeddings.append(emb[0].detach().float().cpu().numpy())
            if (i + 1) % 100 == 0:
                LOGGER.info("  Embeddings: %d / %d", i + 1, total)
    return np.asarray(embeddings, dtype=np.float32)


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    n = len(embeddings)
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)

    try:
        import umap.umap_ as umap_module
    except Exception as exc:
        raise RuntimeError(
            "UMAP is not installed. Install it with: pip install umap-learn"
        ) from exc

    n_neighbors = int(np.clip(n_neighbors, 2, max(2, n - 1)))
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        init="spectral",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def discover_strategies(idx_root: str, dataset_name: str, cycle: int) -> list[str]:
    found = []
    prefix = f"{dataset_name}_"
    suffix = f"_cycle{cycle}.npy"
    for fname in sorted(os.listdir(idx_root)):
        if fname.startswith(prefix) and fname.endswith(suffix):
            strategy = fname[len(prefix):-len(suffix)]
            found.append(strategy)
    return found


def resolve_strategy_names(strategies_arg: str, idx_root: str, dataset_name: str, cycle: int) -> list[str]:
    mode = strategies_arg.strip().lower()
    if mode == "all":
        names = discover_strategies(idx_root, dataset_name, cycle)
        LOGGER.info("Auto-discovered %d strategies for cycle %d.", len(names), cycle)
        return names
    if mode == "latest":
        return list(LATEST_STRATEGIES)
    return [s.strip() for s in strategies_arg.split(",") if s.strip()]


def load_picks(idx_root: str, dataset_name: str, strategy: str, cycle: int) -> np.ndarray | None:
    path = osp.join(idx_root, f"{dataset_name}_{strategy}_cycle{cycle}.npy")
    if not osp.isfile(path):
        return None
    return np.load(path).astype(int)


def _scatter_background(ax, coords: np.ndarray, alpha: float = 0.45) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=8, c="#4a5568", alpha=alpha, linewidths=0, rasterized=True)


def plot_all_strategies_overview(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    strategy_picks: dict[str, np.ndarray],
    cycle: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    _scatter_background(ax, coords)

    id_to_pos = {int(v): i for i, v in enumerate(sample_ids.tolist())}
    legend_handles = []

    for si, (strategy, picked_ids) in enumerate(sorted(strategy_picks.items())):
        color = _PALETTE[si % len(_PALETTE)]
        positions = [id_to_pos[pid] for pid in picked_ids.tolist() if pid in id_to_pos]
        if not positions:
            continue
        positions = np.asarray(positions, dtype=int)
        ax.scatter(
            coords[positions, 0], coords[positions, 1],
            s=30, c=color, alpha=0.85, linewidths=0, rasterized=True,
        )
        legend_handles.append(mpatches.Patch(color=color, label=strategy))

    ax.legend(handles=legend_handles, loc="best", fontsize=7, ncol=2, framealpha=0.8, markerscale=1.5)
    ax.set_title(f"UMAP - all strategies, cycle {cycle}", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved overview -> %s", out_path)


def plot_per_strategy_grid(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    strategy_picks: dict[str, np.ndarray],
    cycle: int,
    out_path: Path,
) -> None:
    strategies = sorted(strategy_picks.keys())
    n = len(strategies)
    if n == 0:
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.atleast_2d(axes)
    flat = list(axes.flat)
    for ax in flat:
        ax.axis("off")

    id_to_pos = {int(v): i for i, v in enumerate(sample_ids.tolist())}

    for si, strategy in enumerate(strategies):
        ax = flat[si]
        ax.axis("on")
        color = _PALETTE[si % len(_PALETTE)]
        picked_ids = strategy_picks[strategy]
        positions = np.asarray([id_to_pos[pid] for pid in picked_ids.tolist() if pid in id_to_pos], dtype=int)

        ax.scatter(coords[:, 0], coords[:, 1], s=5, c="#4a5568", alpha=0.45, linewidths=0, rasterized=True)
        if positions.size > 0:
            ax.scatter(coords[positions, 0], coords[positions, 1], s=20, c=color, alpha=0.9, linewidths=0, rasterized=True)

        ax.set_title(f"{strategy}\\n({len(positions)} picks, cycle {cycle})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"UMAP strategy picks - cycle {cycle}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved per-strategy grid -> %s", out_path)


def plot_single_strategy(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    strategy: str,
    picked_ids: np.ndarray,
    cycle: int,
    color: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    id_to_pos = {int(v): i for i, v in enumerate(sample_ids.tolist())}
    positions = np.asarray([id_to_pos[pid] for pid in picked_ids.tolist() if pid in id_to_pos], dtype=int)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=8, c="#4a5568", alpha=0.45, linewidths=0, rasterized=True,
        label="pool (unselected)",
    )
    if positions.size > 0:
        ax.scatter(
            coords[positions, 0], coords[positions, 1],
            s=35, c=color, alpha=0.9, linewidths=0, rasterized=True,
            label=f"{strategy} picks ({len(positions)})",
        )

    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{strategy} - cycle {cycle}", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = osp.expanduser(args.data_root)
    idx_root = osp.expanduser(args.idx_root) if args.idx_root else osp.join(data_root, "Idx_files")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split_npy = osp.join(idx_root, f"{args.split_stem}.npy")
    if not osp.isfile(split_npy):
        LOGGER.error("Split npy not found: %s", split_npy)
        return 1
    split_path = f"Idx_files/{args.split_stem}"

    dataset = build_dataset(data_root, split_path, args.resolution)
    sample_ids = np.asarray(dataset.train_idx, dtype=int)
    LOGGER.info("Pool size: %d samples", len(sample_ids))

    ckpt_stem = Path(args.checkpoint).stem
    cache_path = out_dir / f"embeddings_{ckpt_stem}_{args.split_stem}.npy"

    if args.cache_embeddings and cache_path.is_file():
        LOGGER.info("Loading cached embeddings from %s", cache_path)
        embeddings = np.load(str(cache_path))
        if len(embeddings) != len(sample_ids):
            LOGGER.warning("Cached embeddings size mismatch (%d vs %d). Recomputing.", len(embeddings), len(sample_ids))
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        LOGGER.info("Loading model: %s", args.checkpoint)
        model = load_model(args.checkpoint, args.resolution, device)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        LOGGER.info("Computing embeddings for %d samples", len(sample_ids))
        embeddings = compute_embeddings(model, dataloader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.cache_embeddings:
            np.save(str(cache_path), embeddings)
            LOGGER.info("Cached embeddings -> %s", cache_path)

    umap_cache = out_dir / (
        f"umap_{ckpt_stem}_{args.split_stem}_n{args.umap_neighbors}_d{args.umap_min_dist:.2f}_{args.umap_metric}.npy"
    )
    if umap_cache.is_file():
        LOGGER.info("Loading cached UMAP from %s", umap_cache)
        coords = np.load(str(umap_cache))
        if len(coords) != len(sample_ids):
            coords = None
    else:
        coords = None

    if coords is None:
        LOGGER.info(
            "Running UMAP (neighbors=%d, min_dist=%.3f, metric=%s)",
            args.umap_neighbors,
            args.umap_min_dist,
            args.umap_metric,
        )
        coords = run_umap(
            embeddings,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.seed,
        )
        np.save(str(umap_cache), coords)
        LOGGER.info("Cached UMAP -> %s", umap_cache)

    strategy_names = resolve_strategy_names(args.strategies, idx_root, args.dataset_name, args.cycle)
    LOGGER.info("Requested %d strategies.", len(strategy_names))

    strategy_picks: dict[str, np.ndarray] = {}
    for sname in strategy_names:
        picks = load_picks(idx_root, args.dataset_name, sname, args.cycle)
        if picks is None:
            LOGGER.warning("No index file for strategy '%s' cycle %d - skipping.", sname, args.cycle)
            continue
        strategy_picks[sname] = picks
        LOGGER.info("  %-45s  %d picks", sname, len(picks))

    if not strategy_picks:
        LOGGER.error("No strategy pick files found. Nothing to plot.")
        return 1

    cycle = args.cycle
    stem = f"{args.dataset_name}_cycle{cycle}_{ckpt_stem}"

    plot_all_strategies_overview(coords, sample_ids, strategy_picks, cycle, out_dir / f"{stem}_overview.png")
    plot_per_strategy_grid(coords, sample_ids, strategy_picks, cycle, out_dir / f"{stem}_grid.png")

    ind_dir = out_dir / f"individual_cycle{cycle}"
    ind_dir.mkdir(exist_ok=True)
    for si, (sname, picked_ids) in enumerate(sorted(strategy_picks.items())):
        color = _PALETTE[si % len(_PALETTE)]
        plot_single_strategy(
            coords,
            sample_ids,
            sname,
            picked_ids,
            cycle,
            color,
            ind_dir / f"{args.dataset_name}_{sname}_cycle{cycle}.png",
        )
    LOGGER.info("Saved %d individual plots -> %s", len(strategy_picks), ind_dir)

    LOGGER.info("Done. All outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
