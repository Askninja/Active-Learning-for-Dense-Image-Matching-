#!/usr/bin/env python3
"""Compute HS-Cert uncertainty distribution with selectable analysis modes."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.strategies.strategies import ActiveLearningStrategy  # noqa: E402
from roma.strategies.uncertainty_estimation import (  # noqa: E402
    compute_uncertainty_and_homographies,
    compute_uncertainty_and_homographies_grid,
)


# ---------------------------------------------------------------------------
# Edit this block when you want to change the model / dataset / outputs.
# MODE="hs_cert"      -> Main/roma/strategies/strategy_hs_cert.py behavior
# MODE="hs_cert_new"  -> Main/roma/strategies/strategy_hs_cert_new.py behavior
# MODE="hs_cert_3"    -> custom local homography-stability certainty
# ---------------------------------------------------------------------------
MODE = "hs_cert_3"
DATA_ROOT = "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-Infrared"
IDX_ROOT = None
SPLIT_STEM = "train_idx"
CHECKPOINT_PATH = "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/checkpoints/Optical-Infrared/pretrained_seed.pth"
RESOLUTION = "medium"
DEVICE = "cuda"
MAX_SAMPLES = None
GRID_SIZE = 5
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
MODE_TO_OUTPUT_STEM = {
    "hs_cert": "hs_cert_distribution",
    "hs_cert_new": "hs_cert_new_distribution",
    "hs_cert_3": "hs_cert_3_distribution",
}
OUTPUT_STEM = MODE_TO_OUTPUT_STEM[MODE]


LOGGER = logging.getLogger("hs_cert_distribution")


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
    if MODE not in {"hs_cert", "hs_cert_new", "hs_cert_3"}:
        raise ValueError(f"Unsupported MODE={MODE!r}. Expected one of: hs_cert, hs_cert_new, hs_cert_3")

    strategy_name = "hs_cert_new" if MODE == "hs_cert_new" else "hs_cert"
    args = SimpleNamespace(
        job_name="analysis_hs_cert_distribution",
        strategy=strategy_name,
        train_resolution=resolution,
    )
    return ActiveLearningStrategy(
        args=args,
        cycle=0,
        data_root=data_root,
        split=split_stem,
        idx_root=idx_root,
    )


def _resolve_pair_paths(data_root: str, pair_id: int) -> tuple[str, str]:
    """Resolve pair image paths from dataset naming used by OpticalMap."""
    root = Path(data_root)

    def pick(base: str) -> Path:
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
            candidate = root / f"{base}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find image for {base} with known extensions under {root}")

    a_path = pick(f"pair{pair_id}_1")
    b_path = pick(f"pair{pair_id}_2")
    return str(a_path), str(b_path)


def _hs_cert_scores(self, model_for_uncertainty, avail: np.ndarray) -> np.ndarray:
    """Compute homography-stability based certainty for each candidate pair."""
    hs_vals = []
    for i in avail:
        a_path, b_path = _resolve_pair_paths(self.data_root, int(i))
        dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
        sparse_matches, _ = model_for_uncertainty.sample(
            dense_matches, dense_certainty, 5000, thresh_score=0.05
        )
        sm = sparse_matches.detach().cpu().numpy()
        if sm.shape[0] < 8:
            hs_vals.append(0.0)
            continue
        with Image.open(a_path) as imA:
            w1, h1 = imA.size
        with Image.open(b_path) as imB:
            w2, h2 = imB.size
        A_px = np.stack((w1 * (sm[:, 0] + 1) / 2 - 0.5, h1 * (sm[:, 1] + 1) / 2 - 0.5), axis=1)
        B_px = np.stack((w2 * (sm[:, 2] + 1) / 2 - 0.5, h2 * (sm[:, 3] + 1) / 2 - 0.5), axis=1)
        g = np.random.default_rng(1234)
        Hs = []
        subset = min(2000, A_px.shape[0])
        thresh = 3 * min(w2, h2) / 480
        for _ in range(50):
            sel = (
                g.choice(A_px.shape[0], size=subset, replace=False)
                if A_px.shape[0] > subset
                else np.arange(A_px.shape[0])
            )
            pA, pB = A_px[sel], B_px[sel]
            H, _ = cv2.findHomography(
                pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999
            )
            if H is not None and abs(H[2, 2]) > 1e-12:
                Hs.append(H / (H[2, 2] + 1e-12))
        if len(Hs) < 2:
            hs_vals.append(0.0)
            continue
        Hs = np.stack(Hs, axis=0)
        c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
        warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs], axis=0)
        s = float(warped.std(axis=0).mean())
        hs_vals.append(s)
    hs_vals = np.asarray(hs_vals, dtype=float)
    hs_cert = 1.0 / (1.0 + np.maximum(hs_vals, 0.0))
    return hs_cert


def compute_scores(
    selector: ActiveLearningStrategy,
    model: torch.nn.Module,
    sample_ids: np.ndarray,
) -> list[tuple[int, float, float]]:
    if MODE == "hs_cert_new":
        uncertainties, certainties, _homographies, valid_ids = compute_uncertainty_and_homographies_grid(
            selector,
            model,
            sample_ids,
            grid_size=GRID_SIZE,
        )
    elif MODE == "hs_cert":
        uncertainties, certainties, _homographies, valid_ids = compute_uncertainty_and_homographies(
            selector,
            model,
            sample_ids,
        )
    elif MODE == "hs_cert_3":
        certainties = _hs_cert_scores(selector, model, sample_ids)
        uncertainties = 1.0 - certainties
        valid_ids = sample_ids
    else:
        raise ValueError(f"Unsupported MODE={MODE!r}. Expected one of: hs_cert, hs_cert_new, hs_cert_3")
    return [
        (int(sample_id), float(uncertainty), float(certainty))
        for sample_id, uncertainty, certainty in zip(valid_ids, uncertainties, certainties)
    ]


def write_csv(rows: list[tuple[int, float, float]], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "uncertainty", "certainty"])
        writer.writerows(rows)


def save_histogram(rows: list[tuple[int, float, float]], png_path: Path) -> None:
    values = np.asarray([uncertainty for _, uncertainty, _ in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=40, range=(0.0, 1.0))
    ax.set_xlabel("HS-Cert uncertainty")
    ax.set_ylabel("count")
    title_map = {
        "hs_cert": "HS-Cert uncertainty distribution",
        "hs_cert_new": "HS-Cert-New uncertainty distribution",
        "hs_cert_3": "HS-Cert-3 uncertainty distribution",
    }
    ax.set_title(title_map[MODE])
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

    strategy_msg = {
        "hs_cert": "strategy_hs_cert.py",
        "hs_cert_new": "strategy_hs_cert_new.py",
        "hs_cert_3": "local hs_cert_3 scoring path",
    }
    LOGGER.info("Using %s", strategy_msg[MODE])
    LOGGER.info("Loading model from %s", CHECKPOINT_PATH)
    model = load_model(CHECKPOINT_PATH, RESOLUTION, device)

    LOGGER.info("Scoring %d samples", sample_ids.size)
    rows = compute_scores(selector=selector, model=model, sample_ids=sample_ids)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{OUTPUT_STEM}.csv"
    png_path = out_dir / f"{OUTPUT_STEM}.png"
    write_csv(rows, csv_path)
    save_histogram(rows, png_path)

    values = np.asarray([uncertainty for _, uncertainty, _ in rows], dtype=np.float64)
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
