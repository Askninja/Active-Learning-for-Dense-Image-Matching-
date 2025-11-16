#!/usr/bin/env python3
"""Certainty and homography-certainty dumper for multiple checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

OPTICALMAP_DATASETS = {
    "opticalmap",
    "Optical-Map",
    "Optical-Infrared",
    "Optical-Depth",
    "Optical-Optical",
    "Nighttime",
}


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for a single checkpoint evaluation."""

    name: str
    path: str


@dataclass(frozen=True)
class HomographySettings:
    """Parameters for the homography stability computation."""

    iterations: int = 50
    n_sample: int = 5000
    thresh_score: float = 0.05
    ransac_confidence: float = 0.999


@dataclass
class CertDumpConfig:
    """Runtime configuration for the certainty dumper."""

    dataset_name: str = "opticalmap"
    cand_npy: str | None = "/home/abhiram001/active_learning/abhiram/AMD_ab/datasets/cross_modality/Optical-Infrared/train_idx.npy"
    data_root: str | None = None
    split_stem: str | None = None
    resolution: str = "medium"
    symmetric: bool = False
    device: str = "cuda"
    out_dir: Path = REPO_ROOT / "certainty_ckpts"
    write_json: bool = False
    save_vis: bool = True
    vis_thresh_score: float = 0.05
    plot_dirname: str = "adaptive_homog_uwe_plots1"
    hs_only: bool = False
    homography: HomographySettings = field(default_factory=HomographySettings)
    ckpts: tuple[CheckpointConfig, ...] = field(
        default_factory=lambda: (
            CheckpointConfig(
                name="cycle0",
                path="/home/abhiram001/active_learning/abhiram/AMD_ab/RoMa-main/workspace/checkpoints/Optical-Depth/pretrained_seed.pth",
            ),
            CheckpointConfig(
                name="cycle1",
                path="/projects/_hdd/roma/Optical-Depth/Depth_UWE/Depth_UWE_cycle0_strategy_adaptive_homog_uwe_best.pth",
            ),
            CheckpointConfig(
                name="cycle2",
                path="/projects/_hdd/roma/Optical-Depth/Depth_UWE/Depth_UWE_cycle1_strategy_adaptive_homog_uwe_best.pth",
            ),
        )
    )

    def resolve_split(self) -> tuple[str, str]:
        """Resolve data_root and split stem from cand_npy or explicit paths."""
        if self.cand_npy:
            if not osp.isfile(self.cand_npy):
                raise FileNotFoundError(f"Candidate npy not found: {self.cand_npy}")
            return stem_from_npy(self.cand_npy)
        if not (self.data_root and self.split_stem):
            raise ValueError("Provide either cand_npy or both data_root and split_stem.")
        return self.data_root, self.split_stem

    def iter_checkpoints(self) -> Iterable[CheckpointConfig]:
        yield from self.ckpts


DEFAULT_CONFIG = CertDumpConfig()
logger = logging.getLogger(__name__)


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_optional_path(value: str | None) -> str | None:
    cleaned = _normalize_optional_str(value)
    return osp.expanduser(cleaned) if cleaned else None


def _parse_checkpoint_args(values: Sequence[str] | None,
                           fallback: Sequence[CheckpointConfig]) -> tuple[CheckpointConfig, ...]:
    """Parse NAME=PATH pairs from CLI."""
    if not values:
        return tuple(fallback)
    parsed = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid checkpoint specification '{raw}'. Use NAME=PATH.")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = osp.expanduser(path.strip())
        if not name or not path:
            raise ValueError(f"Invalid checkpoint specification '{raw}'.")
        parsed.append(CheckpointConfig(name=name, path=path))
    return tuple(parsed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    defaults = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=tuple(sorted(OPTICALMAP_DATASETS)),
                        default=defaults.dataset_name, help="Dataset identifier.")
    parser.add_argument("--cand-npy", default=defaults.cand_npy,
                        help="Path to candidate npy file used to infer data_root and split_stem.")
    parser.add_argument("--data-root", default=defaults.data_root,
                        help="Dataset root; overrides cand-npy inference when paired with split-stem.")
    parser.add_argument("--split-stem", default=defaults.split_stem,
                        help="Split name; overrides cand-npy inference when paired with data-root.")
    parser.add_argument("--resolution", default=defaults.resolution,
                        help="RoMa model resolution.")
    parser.add_argument("--symmetric", action="store_true", default=defaults.symmetric,
                        help="Enable symmetric RoMa matching.")
    parser.add_argument("--device", default=defaults.device,
                        help="Device string handed to torch (e.g. cuda, cuda:1, cpu).")
    parser.add_argument("--out-dir", default=str(defaults.out_dir),
                        help="Directory where CSV/JSON outputs are stored.")
    parser.add_argument("--write-json", action="store_true", dest="write_json",
                        default=defaults.write_json, help="Emit JSON alongside CSV.")
    parser.add_argument("--no-write-json", action="store_false", dest="write_json",
                        help="Disable JSON output even if enabled by default.")
    parser.add_argument("--save-vis", action="store_true", dest="save_vis",
                        default=defaults.save_vis, help="Run expensive visualization benchmark.")
    parser.add_argument("--no-save-vis", action="store_false", dest="save_vis",
                        help="Skip visualization benchmark for speed.")
    parser.add_argument("--vis-thresh-score", type=float, default=defaults.vis_thresh_score,
                        help="Certainty threshold used for visualization benchmark.")
    parser.add_argument("--plot-dirname", default=defaults.plot_dirname,
                        help="Relative directory (under data-root) to store tau histograms.")
    parser.add_argument("--hs-only", action="store_true", default=defaults.hs_only,
                        help="Only retain hs_cert-related columns in the CSV.")
    parser.add_argument("--hs-iters", type=int, default=defaults.homography.iterations,
                        help="Iterations for homography stability estimation.")
    parser.add_argument("--hs-n-sample", type=int, default=defaults.homography.n_sample,
                        help="Sparse matches sampled per image.")
    parser.add_argument("--hs-thresh-score", type=float, default=defaults.homography.thresh_score,
                        help="Certainty threshold for sampling matches.")
    parser.add_argument("--hs-ransac-confidence", type=float,
                        default=defaults.homography.ransac_confidence,
                        help="RANSAC confidence for cv2.findHomography.")
    parser.add_argument("--ckpt", action="append", metavar="NAME=PATH",
                        help="Checkpoint specification; repeatable. Defaults to built-ins.")
    parser.add_argument("--log-level", default="INFO",
                        help="Python logging level (INFO, DEBUG, ...).")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> CertDumpConfig:
    """Create CertDumpConfig from argparse args."""
    ckpts = _parse_checkpoint_args(args.ckpt, DEFAULT_CONFIG.ckpts)
    homography = HomographySettings(
        iterations=args.hs_iters,
        n_sample=args.hs_n_sample,
        thresh_score=args.hs_thresh_score,
        ransac_confidence=args.hs_ransac_confidence,
    )
    return CertDumpConfig(
        dataset_name=args.dataset_name,
        cand_npy=_normalize_optional_path(args.cand_npy),
        data_root=_normalize_optional_path(args.data_root),
        split_stem=_normalize_optional_str(args.split_stem),
        resolution=args.resolution,
        symmetric=args.symmetric,
        device=args.device,
        out_dir=Path(args.out_dir).expanduser(),
        write_json=args.write_json,
        save_vis=args.save_vis,
        vis_thresh_score=args.vis_thresh_score,
        plot_dirname=args.plot_dirname,
        homography=homography,
        ckpts=ckpts,
        hs_only=args.hs_only,
    )


def configure_logging(level_str: str) -> None:
    """Configure root logger from CLI input."""
    level = getattr(logging, level_str.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level '{level_str}'")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Make sure imports from repo work
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
import roma  # noqa: E402
from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.benchmarks import OpticalmapHomogBenchmark  # noqa: E402

# --- plotting & kde tools (for tau computation) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
# --------------------------------------------------

def load_model(ckpt_path: str, resolution: str = "low", symmetric: bool = False, device: str = "cuda"):
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
        logger.warning(
            "load_state_dict issues for %s: missing=%d unexpected=%d",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.debug("missing (first few): %s", missing[:5])
        if unexpected:
            logger.debug("unexpected (first few): %s", unexpected[:5])
    model.eval()
    return model

def stem_from_npy(npy_path: str) -> tuple[str, str]:
    data_root = osp.dirname(npy_path)
    stem = osp.splitext(osp.basename(npy_path))[0]
    return data_root, stem

def run_benchmark_uncertainty(dataset_name: str, data_root: str, split_stem: str,
                              model: torch.nn.Module, model_name: str,
                              save_vis: bool, vis_thresh_score: float):
    if dataset_name not in OPTICALMAP_DATASETS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    bench = OpticalmapHomogBenchmark(data_root, split_stem)

    with torch.no_grad():
        pairs = bench.benchmark_uncertainty(model)

        if save_vis:
            try:
                _ = bench.benchmark(
                    model,
                    model_name=model_name,
                    vis=True,
                    thresh_score=vis_thresh_score,
                )
            except Exception as exc:  # pragma: no cover - visualization is optional
                logger.warning("Visualization benchmark failed for %s: %s", model_name, exc)

    return pairs

def normalize_minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def _idx_to_paths(data_root: str, idx: int) -> tuple[str, str]:
    a = osp.join(data_root, f"pair{int(idx)}_1.jpg")
    b = osp.join(data_root, f"pair{int(idx)}_2.jpg")
    if not (osp.isfile(a) and osp.isfile(b)):
        raise FileNotFoundError(f"missing pair files for idx={idx}: {a} or {b}")
    return a, b

def _convert_coords_norm_to_px(coordsA, coordsB, w1, h1, w2, h2):
    offset = 0.5
    A = np.stack((w1 * (coordsA[..., 0] + 1) / 2,
                  h1 * (coordsA[..., 1] + 1) / 2), axis=-1) - offset
    B = np.stack((w2 * (coordsB[..., 0] + 1) / 2,
                  h2 * (coordsB[..., 1] + 1) / 2), axis=-1) - offset
    return A, B

@torch.no_grad()
def _roma_homography_std(model, a_path: str, b_path: str, settings: HomographySettings) -> float:
    dense_matches, dense_certainty = model.match(a_path, b_path)
    sparse_matches, _ = model.sample(
        dense_matches,
        dense_certainty,
        settings.n_sample,
        thresh_score=settings.thresh_score,
    )
    sm = sparse_matches.detach().cpu().numpy()
    if sm.shape[0] < 8:
        return 0.0

    with Image.open(a_path) as imA_pil:
        w1, h1 = imA_pil.size
    with Image.open(b_path) as imB_pil:
        w2, h2 = imB_pil.size

    A_px, B_px = _convert_coords_norm_to_px(sm[:, :2], sm[:, 2:], w1, h1, w2, h2)
    if A_px.shape[0] < 8:
        return 0.0

    Hs = []
    g = np.random.default_rng(1234)
    subset = min(2000, A_px.shape[0])
    reproj_thresh = 3 * min(w2, h2) / 480

    for _ in range(settings.iterations):
        if A_px.shape[0] > subset:
            sel = g.choice(A_px.shape[0], size=subset, replace=False)
            pA = A_px[sel]
            pB = B_px[sel]
        else:
            pA = A_px
            pB = B_px
        H, _ = cv2.findHomography(
            pA,
            pB,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_thresh,
            confidence=settings.ransac_confidence,
        )
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
    return score

def _extract_cycle_number(model_name: str) -> int:
    try:
        if model_name.lower().startswith("cycle"):
            return int(model_name.lower().replace("cycle", "").strip())
    except Exception:
        pass
    return -1  # unknown

def compute_tau_and_plot(stability_values, cycle_label: str, plot_dir: Path):
    s = np.asarray(stability_values, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0

    s_min, s_max = float(s.min()), float(s.max())
    s_norm = (s - s_min) / (s_max - s_min + 1e-8)

    x = np.linspace(0.0, 1.0, 512)
    kde = gaussian_kde(s_norm, bw_method=0.1)
    y = kde(x)
    peaks, _ = find_peaks(y, prominence=0.01)

    if peaks.size < 2:
        D = 0.0
        R_sim = 0.0
    else:
        peak_x = x[peaks]
        peak_y = y[peaks]
        top_idx = np.argsort(peak_y)[-2:]
        peak_x = peak_x[top_idx]
        peak_y = peak_y[top_idx]
        order = np.argsort(peak_x)
        peak_x = peak_x[order]
        peak_y = peak_y[order]
        D = float(abs(peak_x[1] - peak_x[0]))
        R_sim = float(1.0 - abs(peak_y[1] - peak_y[0]) / (peak_y[1] + peak_y[0] + 1e-12))

    S = D * R_sim
    m = float(s_norm.mean())
    tau = 2.0 * S * m

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(s_norm, bins=32, range=(0.0, 1.0), density=True)
    ax.set_xlabel("stability (normalized)")
    ax.set_ylabel("density")
    ax.set_title(f"{cycle_label} tau={tau:.4f}")
    fig.tight_layout()
    # choose filename like your snippet (cycleN) if possible
    cyc_num = _extract_cycle_number(cycle_label)
    if cyc_num >= 0:
        out_path = out_dir / f"cycle{cyc_num}_stability_hist.png"
    else:
        safe_label = cycle_label.replace("/", "_")
        out_path = out_dir / f"{safe_label}_stability_hist.png"
    fig.savefig(str(out_path))
    plt.close(fig)
    return float(tau)

def process_single_ckpt(config: CertDumpConfig, ckpt: CheckpointConfig,
                        data_root: str, split_stem: str, device: str):
    logger.info("Processing checkpoint '%s' from %s", ckpt.name, ckpt.path)
    model = load_model(
        ckpt.path,
        resolution=config.resolution,
        symmetric=config.symmetric,
        device=device,
    )

    pairs = run_benchmark_uncertainty(
        config.dataset_name,
        data_root,
        split_stem,
        model,
        model_name=ckpt.name,
        save_vis=config.save_vis,
        vis_thresh_score=config.vis_thresh_score,
    )
    if not pairs:
        raise RuntimeError(f"benchmark_uncertainty returned no pairs for model={ckpt.name}")

    idxs, uncerts = [], []
    for p in pairs:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            idxs.append(int(p[0]))
            uncerts.append(float(p[1]))
        elif isinstance(p, dict) and "idx" in p and ("uncertainty" in p or "u" in p):
            idxs.append(int(p["idx"]))
            uncerts.append(float(p.get("uncertainty", p.get("u"))))
        else:
            raise ValueError(f"Unrecognized pair format: {p}")

    idxs = np.asarray(idxs, dtype=int)
    uncerts = np.asarray(uncerts, dtype=float)

    hs_std_list = []
    hs_cert_list = []
    for idx in idxs:
        a_path, b_path = _idx_to_paths(data_root, int(idx))
        s = _roma_homography_std(
            model,
            a_path,
            b_path,
            settings=config.homography,
        )
        hs_std_list.append(float(s))
        hs_cert_list.append(float(1.0 / (1.0 + max(s, 0.0))))

    plot_dir = Path(data_root) / config.plot_dirname
    tau = compute_tau_and_plot(
        hs_cert_list,
        cycle_label=ckpt.name,
        plot_dir=plot_dir,
    )
    logger.info("[tau] %s: tau=%.6f", ckpt.name, tau)

    base_data = {
        "idx": idxs,
        "hs_std": np.asarray(hs_std_list, dtype=float),
        "hs_cert": np.asarray(hs_cert_list, dtype=float),
        "tau": tau,
    }
    if not config.hs_only:
        base_data.update({
            "uncertainty": uncerts,
            "certainty_neg": -uncerts,
            "certainty_inv": 1.0 / (1.0 + np.maximum(uncerts, 0.0)),
            "certainty_minmax": 1.0 - normalize_minmax(uncerts),
        })
    df = pd.DataFrame(base_data)
    if not config.hs_only:
        df = df.sort_values(["uncertainty", "hs_std"], ascending=[True, True])
    else:
        df = df.sort_values(["hs_cert", "hs_std"], ascending=[False, True])
    df = df.reset_index(drop=True)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = split_stem.replace("/", "_")
    csv_path = config.out_dir / f"certainty_{ckpt.name}_{safe_stem}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows -> %s", len(df), csv_path)

    if config.write_json:
        json_path = config.out_dir / f"certainty_{ckpt.name}_{safe_stem}.json"
        with open(json_path, "w") as f:
            json.dump(df.to_dict(orient="records"), f)
        logger.info("Wrote JSON -> %s", json_path)

    if not config.hs_only and "uncertainty" in df.columns:
        logger.info("Most certain by uncertainty (lowest):\n%s",
                    df.sort_values("uncertainty", ascending=True).head(10).to_string(index=False))
    logger.info("Most certain by hs_cert (highest):\n%s",
                df.sort_values("hs_cert", ascending=False).head(10).to_string(index=False))

def run_cert_dump(config: CertDumpConfig) -> None:
    data_root, split_stem = config.resolve_split()
    logger.info("Using data_root='%s', split_stem='%s'", data_root, split_stem)
    device = config.device if (torch.cuda.is_available() and "cuda" in config.device) else "cpu"
    logger.info("Running on device '%s'", device)

    for ckpt in config.iter_checkpoints():
        process_single_ckpt(
            config=config,
            ckpt=ckpt,
            data_root=data_root,
            split_stem=split_stem,
            device=device,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    config = build_config(args)
    run_cert_dump(config)


if __name__ == "__main__":
    main()
