#!/usr/bin/env python3
"""Plot a histogram of RoMa coarse entropy scores for a dataset split."""

from __future__ import annotations

import argparse
import logging
import os.path as osp
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import roma  # noqa: E402
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
    parser.add_argument("--dataset-name", default="Optical-Depth", help="Dataset label for output naming.")
    parser.add_argument(
        "--data-root",
        default="/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-Depth",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--split-stem",
        default="train_idx",
        help="Split stem under Idx_files or dataset root, without .npy.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/checkpoints/Optical-Depth/pretrained_seed.pth",
        help="RoMa checkpoint path.",
    )
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="medium")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda", help="Torch device string.")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "entropy_histograms"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
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


def select_gm_cls_scale(corresps: dict) -> int:
    gm_scales = [scale for scale, payload in corresps.items() if payload.get("gm_cls") is not None]
    if not gm_scales:
        raise ValueError("RoMa forward pass did not return gm_cls at any coarse scale")
    return max(gm_scales)


def compute_entropy_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    sample_ids: np.ndarray,
    temperature: float,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    gm_scale = None
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = int(batch["im_A"].shape[0])
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            corresps = model(batch)
            if gm_scale is None:
                gm_scale = select_gm_cls_scale(corresps)
                LOGGER.info("Using coarse gm_cls scale=%s", gm_scale)
            gm_cls = corresps[gm_scale]["gm_cls"].detach().float().cpu().numpy()
            scores = np.asarray(mean_entropy_score(gm_cls, temperature=temperature), dtype=np.float64)
            if scores.ndim == 0:
                scores = scores[None]
            batch_ids = sample_ids[offset:offset + batch_size]
            for sample_id, score in zip(batch_ids, scores):
                rows.append({"idx": int(sample_id), "entropy": float(score)})
            offset += batch_size
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, out_dir: Path, dataset_name: str, split_stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset_name.replace("/", "_")
    safe_split = split_stem.replace("/", "_")
    csv_path = out_dir / f"{safe_dataset}_{safe_split}_entropy.csv"
    png_path = out_dir / f"{safe_dataset}_{safe_split}_entropy_hist.png"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["entropy"].to_numpy(dtype=float), bins=40, range=(0.0, 1.0))
    ax.set_xlabel("image mean entropy")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_name} {split_stem} entropy histogram")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return csv_path, png_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    data_root = osp.expanduser(args.data_root)
    checkpoint = osp.expanduser(args.checkpoint)
    split_path = resolve_split_path(data_root, args.split_stem)
    device = resolve_device(args.device)

    LOGGER.info("Loading dataset from %s (%s)", data_root, split_path)
    dataset = build_dataset(data_root, split_path, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    LOGGER.info("Loading checkpoint %s", checkpoint)
    model = load_model(checkpoint, args.resolution, device)

    LOGGER.info("Computing entropy scores for %d samples", len(dataset))
    df = compute_entropy_scores(
        model=model,
        dataloader=dataloader,
        sample_ids=np.asarray(dataset.train_idx, dtype=int),
        temperature=args.temperature,
        device=device,
    )
    csv_path, png_path = save_outputs(
        df=df,
        out_dir=Path(args.out_dir).expanduser(),
        dataset_name=args.dataset_name,
        split_stem=args.split_stem,
    )

    LOGGER.info(
        "Entropy stats: count=%d mean=%.6f std=%.6f min=%.6f max=%.6f",
        len(df),
        float(df["entropy"].mean()),
        float(df["entropy"].std(ddof=0)),
        float(df["entropy"].min()),
        float(df["entropy"].max()),
    )
    LOGGER.info("Wrote CSV -> %s", csv_path)
    LOGGER.info("Wrote histogram -> %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
