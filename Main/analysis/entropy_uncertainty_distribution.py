#!/usr/bin/env python3
"""Compute the entropy uncertainty distribution used by strategy_entropy.py."""

from __future__ import annotations

import csv
import logging
import os.path as osp
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.strategies.strategy_utils import mean_entropy_score  # noqa: E402
from roma.strategies.strategies import ActiveLearningStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# Edit this block when you want to change the model / dataset / outputs.
# Mirrors the entropy signal from Main/roma/strategies/strategy_entropy.py.
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-Infrared"
IDX_ROOT = None
SPLIT_STEM = "train_idx"
CHECKPOINT_PATH = "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/checkpoints/Optical-Infrared/pretrained_seed.pth"
RESOLUTION = "medium"
TEMPERATURE = 0.5
BATCH_SIZE = 4
NUM_WORKERS = 0
DEVICE = "cuda"
MAX_SAMPLES = None
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
OUTPUT_STEM = "entropy_distribution"


LOGGER = logging.getLogger("entropy_distribution")


def resolve_device(device_str: str) -> torch.device:
    if "cuda" in device_str and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


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
            "load_state_dict issues: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    model.eval()
    return model


def build_selector(data_root: str, idx_root: str | None, split_stem: str, resolution: str) -> ActiveLearningStrategy:
    args = SimpleNamespace(
        job_name="analysis_entropy_distribution",
        strategy="entropy",
        train_resolution=resolution,
        entropy_temperature=TEMPERATURE,
    )
    return ActiveLearningStrategy(
        args=args,
        cycle=0,
        data_root=data_root,
        split=split_stem,
        idx_root=idx_root,
    )


def select_gm_cls_scale(corresps: dict) -> int:
    gm_scales = [scale for scale, payload in corresps.items() if payload.get("gm_cls") is not None]
    if not gm_scales:
        raise ValueError("RoMa forward pass did not return gm_cls at any coarse scale")
    return max(gm_scales)


def compute_scores(
    selector: ActiveLearningStrategy,
    model: torch.nn.Module,
    sample_ids: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[tuple[int, float]]:
    dataset = selector._entropy_dataset(sample_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: list[tuple[int, float]] = []
    gm_scale = None
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            current_batch = int(batch["im_A"].shape[0])
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            corresps = model(batch)
            if gm_scale is None:
                gm_scale = select_gm_cls_scale(corresps)
                LOGGER.info("Using coarse gm_cls scale=%s", gm_scale)
            gm_cls = corresps[gm_scale]["gm_cls"].detach().float().cpu().numpy()
            scores = np.asarray(mean_entropy_score(gm_cls, temperature=selector.temperature), dtype=np.float64)
            if scores.ndim == 0:
                scores = scores[None]
            batch_ids = sample_ids[offset:offset + current_batch]
            rows.extend((int(sample_id), float(score)) for sample_id, score in zip(batch_ids, scores))
            offset += current_batch
    return rows


def write_csv(rows: list[tuple[int, float]], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "entropy"])
        writer.writerows(rows)


def save_histogram(rows: list[tuple[int, float]], png_path: Path) -> None:
    values = np.asarray([score for _, score in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=40, range=(0.0, 1.0))
    ax.set_xlabel("mean entropy")
    ax.set_ylabel("count")
    ax.set_title("Entropy uncertainty distribution")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = resolve_device(DEVICE)
    selector = build_selector(DATA_ROOT, IDX_ROOT, SPLIT_STEM, RESOLUTION)
    sample_ids = selector.train_pool_idx.astype(int)
    if MAX_SAMPLES is not None:
        sample_ids = sample_ids[: int(MAX_SAMPLES)]

    LOGGER.info("Loading model from %s", CHECKPOINT_PATH)
    model = load_model(CHECKPOINT_PATH, RESOLUTION, device)

    LOGGER.info("Scoring %d samples", sample_ids.size)
    rows = compute_scores(
        selector=selector,
        model=model,
        sample_ids=sample_ids,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{OUTPUT_STEM}.csv"
    png_path = out_dir / f"{OUTPUT_STEM}.png"
    write_csv(rows, csv_path)
    save_histogram(rows, png_path)

    values = np.asarray([score for _, score in rows], dtype=np.float64)
    LOGGER.info(
        "count=%d mean=%.6f std=%.6f min=%.6f max=%.6f",
        values.size,
        float(values.mean()),
        float(values.std(ddof=0)),
        float(values.min()),
        float(values.max()),
    )
    LOGGER.info("CSV: %s", csv_path)
    LOGGER.info("Plot: %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
