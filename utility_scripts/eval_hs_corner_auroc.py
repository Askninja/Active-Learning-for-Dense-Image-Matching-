#!/usr/bin/env python3
"""Evaluate HS certainty vs. corner EPE and AUROC for a single checkpoint."""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalConfig:
    """Runtime configuration for evaluation."""

    ckpt_path: str
    data_root: str
    split_stem: str
    resolution: str = "low"
    symmetric: bool = False
    device: str = "cuda"
    n_sample: int = 5000
    thresh_score: float = 0.05
    hs_iters: int = 50
    hs_ransac_confidence: float = 0.999
    epe_threshold: float = 3.0  # pixels (normalized by 480 baseline)
    max_pairs: int | None = None
    out_csv: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--split-stem", required=True, help="Split stem (e.g., Nighttime_uncertainty_cycle0).")
    parser.add_argument("--resolution", default="low", help="RoMa resolution.")
    parser.add_argument("--symmetric", action="store_true", help="Enable symmetric matching.")
    parser.add_argument("--device", default="cuda", help="Device string for torch.")
    parser.add_argument("--n-sample", type=int, default=5000, help="Sparse samples drawn from dense matches.")
    parser.add_argument("--thresh-score", type=float, default=0.05, help="Certainty threshold used for sampling.")
    parser.add_argument("--hs-iters", type=int, default=50, help="Homography fits used for HS certainty.")
    parser.add_argument("--hs-ransac-confidence", type=float, default=0.999, help="RANSAC confidence for HS.")
    parser.add_argument("--epe-threshold", type=float, default=3.0, help="Corner EPE threshold for AUROC labels.")
    parser.add_argument("--max-pairs", type=int, help="Optional cap on number of evaluated pairs.")
    parser.add_argument("--out-csv", type=str, help="Optional CSV path to store per-pair metrics.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid log level '{level}'")
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")


def _normalize_path(value: str | None) -> str:
    if value is None:
        raise ValueError("Path value may not be None.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Path value may not be empty.")
    return osp.expanduser(cleaned)


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        ckpt_path=_normalize_path(args.ckpt),
        data_root=_normalize_path(args.data_root),
        split_stem=args.split_stem.strip(),
        resolution=args.resolution,
        symmetric=bool(args.symmetric),
        device=args.device,
        n_sample=int(args.n_sample),
        thresh_score=float(args.thresh_score),
        hs_iters=int(args.hs_iters),
        hs_ransac_confidence=float(args.hs_ransac_confidence),
        epe_threshold=float(args.epe_threshold),
        max_pairs=args.max_pairs if args.max_pairs is None else int(args.max_pairs),
        out_csv=Path(args.out_csv).expanduser() if args.out_csv else None,
    )


# Enable repo-local imports
import sys

sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from experiments.train_roma_outdoor import get_model  # noqa: E402


def load_model(ckpt_path: str, resolution: str, symmetric: bool, device: str) -> torch.nn.Module:
    model = get_model(
        pretrained_backbone=True,
        resolution=resolution,
        attenuate_cert=True,
        symmetric=symmetric,
    ).to(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    ckpt = torch.load(ckpt_path, map_location=(device if torch.cuda.is_available() else "cpu"))
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.warning(
            "load_state_dict issues for %s: missing=%d unexpected=%d",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            logging.debug("missing keys (first few): %s", missing[:5])
        if unexpected:
            logging.debug("unexpected keys (first few): %s", unexpected[:5])
    model.eval()
    return model


def _convert_coords_norm_to_px(coordsA: np.ndarray, coordsB: np.ndarray, w1: int, h1: int, w2: int, h2: int) -> tuple[np.ndarray, np.ndarray]:
    offset = 0.5
    A = np.stack((w1 * (coordsA[..., 0] + 1) / 2, h1 * (coordsA[..., 1] + 1) / 2), axis=-1) - offset
    B = np.stack((w2 * (coordsB[..., 0] + 1) / 2, h2 * (coordsB[..., 1] + 1) / 2), axis=-1) - offset
    return A, B


def _estimate_homography_and_epe(pos_a: np.ndarray, pos_b: np.ndarray, w1: int, h1: int, w2: int, h2: int, H_gt: np.ndarray) -> float:
    if pos_a.shape[0] < 4:
        return float("nan")
    try:
        H_pred, _ = cv2.findHomography(
            pos_a,
            pos_b,
            method=cv2.RANSAC,
            confidence=0.99999,
            ransacReprojThreshold=3 * min(w2, h2) / 480,
        )
    except Exception:
        H_pred = None
    if H_pred is None:
        H_pred = np.zeros((3, 3), dtype=float)
        H_pred[2, 2] = 1.0
    corners = np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]], dtype=float)
    real_warped = corners @ H_gt.T
    real_warped = real_warped[:, :2] / real_warped[:, 2:]
    warped = corners @ H_pred.T
    warped = warped[:, :2] / warped[:, 2:]
    mean_dist = np.mean(np.linalg.norm(real_warped - warped, axis=1)) / (min(w2, h2) / 480.0)
    return float(mean_dist)


def _hs_certainty(pos_a: np.ndarray, pos_b: np.ndarray, w1: int, h1: int, w2: int, h2: int, iters: int, ransac_conf: float) -> float:
    if pos_a.shape[0] < 8:
        return 0.0
    Hs = []
    rng = np.random.default_rng(1234)
    subset = min(2000, pos_a.shape[0])
    reproj_thresh = 3 * min(w2, h2) / 480
    for _ in range(iters):
        if pos_a.shape[0] > subset:
            sel = rng.choice(pos_a.shape[0], size=subset, replace=False)
            pA = pos_a[sel]
            pB = pos_b[sel]
        else:
            pA = pos_a
            pB = pos_b
        H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=reproj_thresh, confidence=ransac_conf)
        if H is not None and abs(H[2, 2]) > 1e-12:
            Hs.append(H / (H[2, 2] + 1e-12))
    if len(Hs) < 2:
        return 0.0
    Hs = np.stack(Hs, axis=0)
    corners = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
    warped = [cv2.perspectiveTransform(corners, H).reshape(4, 2) for H in Hs]
    warped = np.stack(warped, axis=0)
    std_xy = warped.std(axis=0)
    score = float(std_xy.mean())
    return 1.0 / (1.0 + max(score, 0.0))


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    scores = scores.astype(float)
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tps = labels_sorted.cumsum()
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / pos
    fpr = fps / neg
    return float(np.trapz(tpr, fpr))


def iter_indices(split_path: str, limit: int | None) -> Iterable[int]:
    idxs = np.load(split_path)
    if limit is not None:
        idxs = idxs[:limit]
    for idx in idxs:
        yield int(idx)


def evaluate(config: EvalConfig) -> dict[str, float]:
    if not osp.isdir(config.data_root):
        raise FileNotFoundError(f"data_root not found: {config.data_root}")
    split_path = osp.join(config.data_root, f"{config.split_stem}.npy")
    if not osp.isfile(split_path):
        raise FileNotFoundError(f"split npy not found: {split_path}")

    device = config.device if (torch.cuda.is_available() and "cuda" in config.device) else "cpu"
    logging.info("Loading model on device '%s'.", device)
    model = load_model(config.ckpt_path, config.resolution, config.symmetric, device)
    logging.info("Loaded checkpoint %s", config.ckpt_path)

    per_pair = []
    csv_rows = []
    with torch.no_grad():
        for idx in iter_indices(split_path, config.max_pairs):
            optical = osp.join(config.data_root, f"pair{idx}_1.jpg")
            depth = osp.join(config.data_root, f"pair{idx}_2.jpg")
            homo_path = osp.join(config.data_root, f"gt_{idx}.txt")
            if not (osp.isfile(optical) and osp.isfile(depth) and osp.isfile(homo_path)):
                logging.warning("Skipping idx=%s due to missing files.", idx)
                continue
            homo = np.loadtxt(homo_path)
            if homo.shape[0] == 2:
                homo = np.vstack([homo, np.array([0, 0, 1])])
            H_gt = homo.astype(float)

            dense_matches, dense_certainty = model.match(optical, depth)
            sparse_matches, _ = model.sample(dense_matches, dense_certainty, config.n_sample, thresh_score=config.thresh_score)
            sm = sparse_matches.detach().cpu().numpy()
            if sm.shape[0] < 4:
                logging.warning("Too few matches for idx=%s (got %d); skipping.", idx, sm.shape[0])
                continue

            with Image.open(optical) as imA_pil:
                w1, h1 = imA_pil.size
            with Image.open(depth) as imB_pil:
                w2, h2 = imB_pil.size

            pos_a, pos_b = _convert_coords_norm_to_px(sm[:, :2], sm[:, 2:], w1, h1, w2, h2)
            corner_epe = _estimate_homography_and_epe(pos_a, pos_b, w1, h1, w2, h2, H_gt)
            hs_cert = _hs_certainty(pos_a, pos_b, w1, h1, w2, h2, config.hs_iters, config.hs_ransac_confidence)
            per_pair.append((hs_cert, corner_epe))
            csv_rows.append({"idx": idx, "hs_cert": hs_cert, "corner_epe": corner_epe})
            logging.debug("idx=%s hs_cert=%.4f corner_epe=%.4f", idx, hs_cert, corner_epe)

    if not per_pair:
        raise RuntimeError("No pairs were evaluated. Check data paths and split size.")

    hs_vals = np.asarray([p[0] for p in per_pair], dtype=float)
    epe_vals = np.asarray([p[1] for p in per_pair], dtype=float)
    labels = (epe_vals <= config.epe_threshold).astype(int)
    auroc = _binary_auroc(labels, hs_vals)
    metrics = {
        "mean_hs_cert": float(np.nanmean(hs_vals)),
        "mean_corner_epe": float(np.nanmean(epe_vals)),
        "auroc_hs_vs_corner_epe": auroc,
        "evaluated_pairs": len(per_pair),
    }

    if config.out_csv:
        config.out_csv.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd  # imported lazily to avoid dependency unless needed

        pd.DataFrame(csv_rows).to_csv(config.out_csv, index=False)
        logging.info("Wrote per-pair metrics to %s", config.out_csv)

    logging.info(
        "Evaluation done on %d pairs | mean_hs=%.4f mean_corner_epe=%.4f AUROC=%.4f",
        len(per_pair),
        metrics["mean_hs_cert"],
        metrics["mean_corner_epe"],
        metrics["auroc_hs_vs_corner_epe"],
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    config = build_config(args)
    evaluate(config)


if __name__ == "__main__":
    main()
