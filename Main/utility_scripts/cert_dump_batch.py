#!/usr/bin/env python3
"""Hard-coded batch runner for cert_dump (hs-cert only, night/optical datasets)."""

from __future__ import annotations

import logging
from pathlib import Path

from cert_dump import (
    CertDumpConfig,
    CheckpointConfig,
    HomographySettings,
    configure_logging,
    run_cert_dump,
    REPO_ROOT,
)

# ---------------------------------------------------------------------------
# Hard-coded knobs – tweak here if paths/checkpoints change.
# ---------------------------------------------------------------------------
TARGET_DATASETS = ("Nighttime", "Optical-Infrared", "Optical-Depth", "Optical-Optical")
JOB_SUFFIX = "random"
INCLUDE_PRESEED = True
CYCLES = (0, 1)
DEVICE = "cuda"
RESOLUTION = "medium"
SYMMETRIC = False
SAVE_VIS = False
WRITE_JSON = False
HS_ONLY = True
LOG_LEVEL = "INFO"
HOMOGRAPHY_SETTINGS = HomographySettings(
    iterations=50,
    n_sample=5000,
    thresh_score=0.05,
    ransac_confidence=0.999,
)

PROJECT_ROOT = REPO_ROOT.parent
DATASET_BASE = PROJECT_ROOT / "datasets" / "cross_modality"
CKPT_ROOT = Path("/projects/_hdd/roma")
OUT_ROOT = REPO_ROOT / "certainty_ckpts" / JOB_SUFFIX
PLOT_DIRNAME = f"{JOB_SUFFIX}_hs_cert_plots"

logger = logging.getLogger(__name__)


def _ckpt_entries(dataset: str) -> list[CheckpointConfig]:
    entries: list[CheckpointConfig] = []
    if INCLUDE_PRESEED:
        preseed_dir = CKPT_ROOT / dataset / f"{dataset}_Preseed"
        preseed_path = preseed_dir / f"{dataset}_Preseed_cycle0_best.pth"
        if preseed_path.is_file():
            entries.append(CheckpointConfig(name="preseed", path=str(preseed_path)))
        else:
            logger.warning("Missing preseed checkpoint for %s at %s", dataset, preseed_path)
    job_dir = CKPT_ROOT / dataset / f"{dataset}_{JOB_SUFFIX}"
    for cycle in CYCLES:
        ckpt_path = job_dir / f"{dataset}_{JOB_SUFFIX}_cycle{cycle}_best.pth"
        if ckpt_path.is_file():
            entries.append(CheckpointConfig(name=f"cycle{cycle}", path=str(ckpt_path)))
        else:
            logger.warning("Missing checkpoint for %s cycle %d at %s", dataset, cycle, ckpt_path)
    return entries


def _build_configs() -> list[CertDumpConfig]:
    configs: list[CertDumpConfig] = []
    for dataset in TARGET_DATASETS:
        dataset_root = DATASET_BASE / dataset
        split_path = dataset_root / "Idx_files" / "train_idx.npy"
        if not split_path.is_file():
            logger.warning("Skipping %s: missing train_idx npy at %s", dataset, split_path)
            continue
        ckpts = _ckpt_entries(dataset)
        if not ckpts:
            logger.warning("Skipping %s: no checkpoints available.", dataset)
            continue
        out_dir = (OUT_ROOT / dataset)
        out_dir.mkdir(parents=True, exist_ok=True)
        config = CertDumpConfig(
            dataset_name=dataset,
            cand_npy=None,
            data_root=str(dataset_root),
            split_stem="Idx_files/train_idx",
            resolution=RESOLUTION,
            symmetric=SYMMETRIC,
            device=DEVICE,
            out_dir=out_dir,
            write_json=WRITE_JSON,
            save_vis=SAVE_VIS,
            vis_thresh_score=0.05,
            plot_dirname=PLOT_DIRNAME,
            homography=HOMOGRAPHY_SETTINGS,
            ckpts=tuple(ckpts),
            hs_only=HS_ONLY,
        )
        configs.append(config)
    return configs


def main() -> None:
    configure_logging(LOG_LEVEL)
    configs = _build_configs()
    if not configs:
        logger.error("No valid dataset configurations to process.")
        return
    for config in configs:
        logger.info("Running cert_dump for dataset '%s' (%d checkpoints).",
                    config.dataset_name, len(tuple(config.iter_checkpoints())))
        run_cert_dump(config)


if __name__ == "__main__":
    main()
