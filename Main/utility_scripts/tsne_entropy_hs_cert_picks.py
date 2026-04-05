#!/usr/bin/env python3
"""Create a t-SNE view for Optical-Depth and plot top picks by entropy and HS Cert."""

from __future__ import annotations

import argparse
import logging
import math
import os.path as osp
import sys
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.datasets import OpticalMap  # noqa: E402
from roma.strategies.strategies import mean_entropy_score  # noqa: E402


LOGGER = logging.getLogger(__name__)
RESOLUTIONS = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="Optical-Depth")
    parser.add_argument(
        "--data-root",
        default="/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-Depth",
        help="Dataset root directory.",
    )
    parser.add_argument("--split-stem", default="train_idx", help="Split stem under Idx_files or dataset root.")
    parser.add_argument(
        "--checkpoint",
        default="/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/checkpoints/Optical-Depth/pretrained_seed.pth",
        help="Checkpoint used for scoring and embeddings.",
    )
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="medium")
    parser.add_argument("--temperature", type=float, default=0.5, help="Entropy softmax temperature.")
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 to keep HS Cert scoring exact.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=784)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--K", type=int, default=10, help="HS Cert homography subsets.")
    parser.add_argument("--P", type=int, default=50, help="HS Cert sampled points.")
    parser.add_argument("--num-matches", type=int, default=5000, help="HS Cert sampled matches.")
    parser.add_argument(
        "--out-dir",
        default="/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/tsne_pick_plots/Optical-Depth",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def configure_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level '{level_str}'")
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def resolve_device(device_str: str) -> torch.device:
    if "cuda" in device_str and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def resolve_split_path(data_root: str, split_stem: str) -> str:
    idx_path = osp.join(data_root, "Idx_files", f"{split_stem}.npy")
    if osp.isfile(idx_path):
        return f"Idx_files/{split_stem}"
    root_path = osp.join(data_root, f"{split_stem}.npy")
    if osp.isfile(root_path):
        return split_stem
    raise FileNotFoundError(f"Could not find split npy for stem '{split_stem}' under {data_root}")


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


def load_model(ckpt_path: str, resolution: str, device: torch.device) -> torch.nn.Module:
    model = get_model(
        pretrained_backbone=True,
        resolution=resolution,
        attenuate_cert=False,
        symmetric=False,
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning(
            "load_state_dict issues for %s: missing=%d unexpected=%d",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
    model.eval()
    return model


def select_gm_cls_scale(corresps: dict) -> int:
    gm_scales = [scale for scale, payload in corresps.items() if payload.get("gm_cls") is not None]
    if not gm_scales:
        raise ValueError("RoMa forward pass did not return gm_cls at any coarse scale")
    return max(gm_scales)


def _get_matches(flow: torch.Tensor, certainty: torch.Tensor, height: int, width: int, num_matches: int) -> np.ndarray:
    x_coords = torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=flow.device)
    y_coords = torch.linspace(-1 + 1 / height, 1 - 1 / height, height, device=flow.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords_a = torch.stack((xx, yy), dim=-1)[None]
    coords_b = coords_a + flow.permute(0, 2, 3, 1)
    matches = torch.cat((coords_a, coords_b), dim=-1).reshape(1, -1, 4)[0]
    cert = certainty.sigmoid().reshape(-1).float()
    cert = torch.nan_to_num(cert, nan=0.0, posinf=0.0, neginf=0.0)
    cert = torch.clamp(cert, min=0.0)
    if cert.sum() <= 0:
        cert = torch.ones_like(cert)
    num_matches = min(int(num_matches), len(cert))
    if num_matches == 0:
        return matches.new_zeros((0, 4)).cpu().numpy()
    chosen = torch.multinomial(cert, num_matches, replacement=False)
    return matches[chosen].cpu().numpy()


def _compute_hs_uncertainty(matches: np.ndarray, height: int, width: int, K: int, P: int, rng: np.random.Generator) -> float:
    if len(matches) < 4:
        return 1.0
    center = np.array([width / 2.0, height / 2.0, width / 2.0, height / 2.0], dtype=np.float64)
    scale = np.array([width / 2.0, height / 2.0, width / 2.0, height / 2.0], dtype=np.float64)
    points_a = rng.random((P, 2)) * np.array([width, height], dtype=np.float64)
    points_a_norm = (points_a - np.array([width / 2.0, height / 2.0], dtype=np.float64)) / np.array(
        [width / 2.0, height / 2.0], dtype=np.float64
    )

    projections = []
    for _ in range(K):
        subset_size = min(1000, len(matches))
        subset_idx = rng.choice(len(matches), subset_size, replace=False)
        subset = matches[subset_idx]
        subset_px = subset * scale + center
        kpts_a = subset_px[:, :2]
        kpts_b = subset_px[:, 2:]
        H_mat, _ = cv2.findHomography(kpts_a, kpts_b, cv2.RANSAC, 5.0)
        if H_mat is None:
            H_mat = np.eye(3, dtype=np.float64)
        points_h = np.hstack((points_a_norm, np.ones((P, 1), dtype=np.float64)))
        proj_h = points_h @ H_mat.T
        denom = np.where(np.abs(proj_h[:, 2:3]) < 1e-12, 1e-12, proj_h[:, 2:3])
        projections.append(proj_h[:, :2] / denom)

    projections = np.asarray(projections, dtype=np.float64)
    stds = []
    for i in range(P):
        pts = projections[:, i, :]
        stds.append(float(np.sqrt(np.std(pts[:, 0]) ** 2 + np.std(pts[:, 1]) ** 2)))
    spread = float(np.mean(stds))
    cert = 1.0 / (1.0 + max(spread, 0.0))
    return 1.0 - cert


def compute_pair_embedding(model: torch.nn.Module, batch: dict) -> np.ndarray:
    with torch.no_grad():
        feature_pyramid = model.extract_backbone_features(batch, batched=True, upsample=False)
    scale = max(feature_pyramid.keys())
    features = feature_pyramid[scale]
    feat_a, feat_b = features.chunk(2, dim=0)
    emb_a = feat_a.mean(dim=(2, 3))
    emb_b = feat_b.mean(dim=(2, 3))
    embedding = torch.cat((emb_a, emb_b), dim=1)
    embedding = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)
    return embedding[0].detach().float().cpu().numpy()


def score_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    sample_ids: np.ndarray,
    temperature: float,
    device: torch.device,
    K: int,
    P: int,
    num_matches: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    embeddings = []
    gm_scale = None
    finest_scale = None
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for offset, batch in enumerate(dataloader):
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            corresps = model(batch)
            if gm_scale is None:
                gm_scale = select_gm_cls_scale(corresps)
                finest_scale = max(corresps.keys())
                LOGGER.info("Using gm_cls scale=%s and finest scale=%s", gm_scale, finest_scale)

            gm_cls = corresps[gm_scale]["gm_cls"].detach().float().cpu().numpy()
            entropy = float(mean_entropy_score(gm_cls, temperature=temperature))

            flow = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"]
            height, width = flow.shape[-2:]
            matches = _get_matches(flow, certainty, height, width, num_matches=num_matches)
            hs_cert = _compute_hs_uncertainty(matches, height, width, K=K, P=P, rng=rng)

            sample_id = int(sample_ids[offset])
            rows.append({"idx": sample_id, "entropy": entropy, "hs_cert": hs_cert})
            embeddings.append(compute_pair_embedding(model, batch))

            if (offset + 1) % 50 == 0:
                LOGGER.info("Processed %d / %d samples", offset + 1, len(sample_ids))

    return pd.DataFrame(rows), np.asarray(embeddings, dtype=np.float32)


def compute_tsne(embeddings: np.ndarray, perplexity: float, seed: int, iterations: int) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 2), dtype=np.float32)
    perplexity = min(float(perplexity), float(len(embeddings) - 1))
    perplexity = max(perplexity, 1.0)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=iterations,
    )
    return tsne.fit_transform(embeddings).astype(np.float32)


def kcenter_greedy(points: np.ndarray, k: int, initial_idx: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    if n == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), n)

    selected = []
    if initial_idx is not None:
        initial_idx = np.asarray(initial_idx, dtype=int)
        selected = [int(i) for i in initial_idx.tolist() if 0 <= int(i) < n]
    if not selected:
        norms = np.sum(points * points, axis=1)
        selected = [int(np.argmax(norms))]

    min_dist = np.full(n, np.inf, dtype=np.float32)
    selected_mask = np.zeros(n, dtype=bool)
    for idx in selected:
        selected_mask[idx] = True
        dist = np.linalg.norm(points - points[idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    min_dist[selected_mask] = 0.0

    while len(selected) < k:
        next_idx = int(np.argmax(min_dist))
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True
        dist = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected_mask] = 0.0
    return np.asarray(selected, dtype=int)


def compute_coreset_indices(sample_ids: np.ndarray, embeddings: np.ndarray, top_k: int, seed: int) -> np.ndarray:
    if len(sample_ids) == 0 or top_k <= 0:
        return np.empty(0, dtype=int)
    top_k = min(int(top_k), len(sample_ids))
    if len(sample_ids) <= top_k:
        return sample_ids.astype(int)

    cluster_count = min(len(sample_ids), max(top_k + 1, min(4 * top_k, len(sample_ids))))
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    kmeans.fit(embeddings)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    chosen_centroid_idx = kcenter_greedy(centroids, k=top_k)
    chosen = []
    used_sample_pos = set()
    for centroid_idx in chosen_centroid_idx.tolist():
        centroid = centroids[centroid_idx]
        distances = np.linalg.norm(embeddings - centroid[None, :], axis=1)
        for pos in np.argsort(distances):
            pos = int(pos)
            if pos not in used_sample_pos:
                used_sample_pos.add(pos)
                chosen.append(int(sample_ids[pos]))
                break

    if len(chosen) < top_k:
        chosen_pos = np.asarray(sorted(used_sample_pos), dtype=int) if used_sample_pos else np.empty(0, dtype=int)
        extra_pos = kcenter_greedy(embeddings, k=top_k, initial_idx=chosen_pos)
        for pos in extra_pos.tolist():
            pos = int(pos)
            if pos not in used_sample_pos:
                used_sample_pos.add(pos)
                chosen.append(int(sample_ids[pos]))
            if len(chosen) >= top_k:
                break

    return np.asarray(chosen[:top_k], dtype=int)


def normalize_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.ones_like(values, dtype=np.float32)
    normalized = (values - vmin) / (vmax - vmin)
    return 1.0 + normalized


def build_pair_montage(data_root: str, sample_id: int, target_height: int = 180) -> np.ndarray:
    paths = [
        Path(data_root) / f"pair{sample_id}_1.jpg",
        Path(data_root) / f"pair{sample_id}_2.jpg",
    ]
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scaled_width = max(1, int(round(width * target_height / max(height, 1))))
        image = image.resize((scaled_width, target_height), Image.Resampling.BILINEAR)
        images.append(np.asarray(image))

    gap = 8
    total_width = images[0].shape[1] + images[1].shape[1] + gap
    canvas = np.full((target_height, total_width, 3), 255, dtype=np.uint8)
    cursor = 0
    for image in images:
        canvas[:, cursor:cursor + image.shape[1]] = image
        cursor += image.shape[1] + gap
    return canvas


def plot_tsne_overview(df: pd.DataFrame, out_path: Path, top_k: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(2, 3)
    ring_colors = {
        "entropy": "#ffcc00",
        "hs_cert": "#00e5ff",
        "coreset": "#ff4d6d",
        "entropy_weighted_coreset": "#7ae582",
        "hs_cert_weighted_coreset": "#9b5de5",
    }
    plot_specs = (
        ("entropy", "viridis"),
        ("hs_cert", "magma"),
        ("coreset", "cividis"),
        ("entropy_weighted_coreset", "Greens"),
        ("hs_cert_weighted_coreset", "Purples"),
    )
    flat_axes = list(axes.flat)
    for ax in flat_axes:
        ax.axis("off")
    for ax, (score_col, cmap) in zip(flat_axes, plot_specs):
        ax.axis("on")
        scatter = ax.scatter(df["tsne_x"], df["tsne_y"], c=df[score_col], s=18, cmap=cmap, alpha=0.85)
        rank_col = f"{score_col}_rank"
        if score_col.endswith("coreset"):
            top_df = df[df[rank_col] > 0].sort_values(rank_col).reset_index(drop=True)
        else:
            top_df = df.nlargest(top_k, score_col).reset_index(drop=True)
        ax.scatter(
            top_df["tsne_x"],
            top_df["tsne_y"],
            s=340,
            facecolors="none",
            edgecolors="black",
            linewidths=4.0,
            marker="o",
            zorder=5,
        )
        ax.scatter(
            top_df["tsne_x"],
            top_df["tsne_y"],
            s=280,
            facecolors="none",
            edgecolors=ring_colors[score_col],
            linewidths=2.6,
            marker="o",
            zorder=6,
        )
        for rank, row in top_df.iterrows():
            ax.annotate(
                str(int(row[rank_col])) if score_col.endswith("coreset") else str(rank + 1),
                (float(row["tsne_x"]), float(row["tsne_y"])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color="black",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ring_colors[score_col], lw=1.2),
                zorder=7,
            )
        ax.set_title(f"{score_col} t-SNE (top {top_k} circled)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(score_col)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_picks(df: pd.DataFrame, score_col: str, data_root: str, out_path: Path, top_k: int) -> None:
    rank_col = f"{score_col}_rank"
    if score_col.endswith("coreset"):
        top_df = df[df[rank_col] > 0].sort_values(rank_col).reset_index(drop=True)
    else:
        top_df = df.nlargest(top_k, score_col).reset_index(drop=True)
    cols = 3
    rows = int(math.ceil(len(top_df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for rank, row in top_df.iterrows():
        ax = axes.flat[rank]
        montage = build_pair_montage(data_root, int(row["idx"]))
        ax.imshow(montage)
        if score_col.endswith("coreset"):
            title = f"#{int(row[rank_col])}  idx={int(row['idx'])}  {score_col}"
        else:
            title = f"#{rank + 1}  idx={int(row['idx'])}  {score_col}={float(row[score_col]):.4f}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Top {len(top_df)} {score_col} picks", fontsize=16)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if int(args.batch_size) != 1:
        raise ValueError("--batch-size must be 1 for this script because HS Cert is computed per sample")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = osp.expanduser(args.data_root)
    checkpoint = osp.expanduser(args.checkpoint)
    split_path = resolve_split_path(data_root, args.split_stem)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset from %s (%s)", data_root, split_path)
    dataset = build_dataset(data_root, split_path, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    sample_ids = np.asarray(dataset.train_idx, dtype=int)

    LOGGER.info("Loading checkpoint %s", checkpoint)
    model = load_model(checkpoint, args.resolution, device)

    LOGGER.info("Scoring %d samples for entropy, hs_cert, and embeddings", len(sample_ids))
    df, embeddings = score_dataset(
        model=model,
        dataloader=dataloader,
        sample_ids=sample_ids,
        temperature=args.temperature,
        device=device,
        K=args.K,
        P=args.P,
        num_matches=args.num_matches,
        seed=args.seed,
    )

    LOGGER.info("Running t-SNE on %d embeddings", len(embeddings))
    coords = compute_tsne(embeddings, perplexity=args.tsne_perplexity, seed=args.seed, iterations=args.tsne_iterations)
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]
    df["coreset"] = 0.0
    df["coreset_rank"] = 0
    df["entropy_weighted_coreset"] = 0.0
    df["entropy_weighted_coreset_rank"] = 0
    df["hs_cert_weighted_coreset"] = 0.0
    df["hs_cert_weighted_coreset_rank"] = 0

    coreset_idx = compute_coreset_indices(sample_ids, embeddings, top_k=args.top_k, seed=args.seed)
    coreset_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(coreset_idx.tolist())}
    df["coreset_rank"] = df["idx"].map(lambda idx: coreset_rank_map.get(int(idx), 0)).astype(int)
    df["coreset"] = (df["coreset_rank"] > 0).astype(float)

    entropy_weighted_embeddings = embeddings * normalize_weights(df["entropy"].to_numpy(dtype=np.float32))[:, None]
    entropy_weighted_coreset_idx = compute_coreset_indices(
        sample_ids,
        entropy_weighted_embeddings,
        top_k=args.top_k,
        seed=args.seed,
    )
    entropy_weighted_rank_map = {
        int(idx): rank + 1 for rank, idx in enumerate(entropy_weighted_coreset_idx.tolist())
    }
    df["entropy_weighted_coreset_rank"] = (
        df["idx"].map(lambda idx: entropy_weighted_rank_map.get(int(idx), 0)).astype(int)
    )
    df["entropy_weighted_coreset"] = (df["entropy_weighted_coreset_rank"] > 0).astype(float)

    hs_cert_weighted_embeddings = embeddings * normalize_weights(df["hs_cert"].to_numpy(dtype=np.float32))[:, None]
    hs_cert_weighted_coreset_idx = compute_coreset_indices(
        sample_ids,
        hs_cert_weighted_embeddings,
        top_k=args.top_k,
        seed=args.seed,
    )
    hs_cert_weighted_rank_map = {
        int(idx): rank + 1 for rank, idx in enumerate(hs_cert_weighted_coreset_idx.tolist())
    }
    df["hs_cert_weighted_coreset_rank"] = (
        df["idx"].map(lambda idx: hs_cert_weighted_rank_map.get(int(idx), 0)).astype(int)
    )
    df["hs_cert_weighted_coreset"] = (df["hs_cert_weighted_coreset_rank"] > 0).astype(float)

    stem = f"{args.dataset_name}_{args.split_stem}"
    csv_path = out_dir / f"{stem}_tsne_scores.csv"
    tsne_path = out_dir / f"{stem}_tsne_overview.png"
    entropy_path = out_dir / f"{stem}_top{args.top_k}_entropy.png"
    hs_cert_path = out_dir / f"{stem}_top{args.top_k}_hs_cert.png"
    coreset_path = out_dir / f"{stem}_top{args.top_k}_coreset.png"
    entropy_weighted_coreset_path = out_dir / f"{stem}_top{args.top_k}_entropy_weighted_coreset.png"
    hs_cert_weighted_coreset_path = out_dir / f"{stem}_top{args.top_k}_hs_cert_weighted_coreset.png"
    entropy_idx_path = out_dir / f"{stem}_top{args.top_k}_entropy_idx.npy"
    hs_cert_idx_path = out_dir / f"{stem}_top{args.top_k}_hs_cert_idx.npy"
    coreset_idx_path = out_dir / f"{stem}_top{args.top_k}_coreset_idx.npy"
    entropy_weighted_coreset_idx_path = out_dir / f"{stem}_top{args.top_k}_entropy_weighted_coreset_idx.npy"
    hs_cert_weighted_coreset_idx_path = out_dir / f"{stem}_top{args.top_k}_hs_cert_weighted_coreset_idx.npy"

    df.to_csv(csv_path, index=False)
    plot_tsne_overview(df, tsne_path, top_k=args.top_k)
    plot_top_picks(df, "entropy", data_root, entropy_path, top_k=args.top_k)
    plot_top_picks(df, "hs_cert", data_root, hs_cert_path, top_k=args.top_k)
    plot_top_picks(df, "coreset", data_root, coreset_path, top_k=args.top_k)
    plot_top_picks(
        df,
        "entropy_weighted_coreset",
        data_root,
        entropy_weighted_coreset_path,
        top_k=args.top_k,
    )
    plot_top_picks(
        df,
        "hs_cert_weighted_coreset",
        data_root,
        hs_cert_weighted_coreset_path,
        top_k=args.top_k,
    )
    np.save(entropy_idx_path, df.nlargest(args.top_k, "entropy")["idx"].to_numpy(dtype=int))
    np.save(hs_cert_idx_path, df.nlargest(args.top_k, "hs_cert")["idx"].to_numpy(dtype=int))
    np.save(coreset_idx_path, coreset_idx.astype(int))
    np.save(entropy_weighted_coreset_idx_path, entropy_weighted_coreset_idx.astype(int))
    np.save(hs_cert_weighted_coreset_idx_path, hs_cert_weighted_coreset_idx.astype(int))

    LOGGER.info("Wrote scores CSV -> %s", csv_path)
    LOGGER.info("Wrote t-SNE figure -> %s", tsne_path)
    LOGGER.info("Wrote entropy picks -> %s", entropy_path)
    LOGGER.info("Wrote hs_cert picks -> %s", hs_cert_path)
    LOGGER.info("Wrote coreset picks -> %s", coreset_path)
    LOGGER.info("Wrote entropy weighted coreset picks -> %s", entropy_weighted_coreset_path)
    LOGGER.info("Wrote hs_cert weighted coreset picks -> %s", hs_cert_weighted_coreset_path)
    LOGGER.info("Wrote entropy indices -> %s", entropy_idx_path)
    LOGGER.info("Wrote hs_cert indices -> %s", hs_cert_idx_path)
    LOGGER.info("Wrote coreset indices -> %s", coreset_idx_path)
    LOGGER.info("Wrote entropy weighted coreset indices -> %s", entropy_weighted_coreset_idx_path)
    LOGGER.info("Wrote hs_cert weighted coreset indices -> %s", hs_cert_weighted_coreset_idx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
