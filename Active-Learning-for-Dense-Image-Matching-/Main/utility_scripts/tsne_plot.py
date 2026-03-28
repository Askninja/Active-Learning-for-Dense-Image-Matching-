#!/usr/bin/env python3
"""Hard-coded t-SNE visualizations for Optical-Optical selections."""

from __future__ import annotations

import os.path as osp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.train_AL_cycle import RESOLUTIONS, DATASET_DIRS, get_dataset_root
from experiments.train_roma_outdoor import get_model
from roma.strategies.strategies import ActiveLearningStrategy
from roma.utils import get_tuple_transform_ops

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_ROOT = REPO_ROOT.parent / "datasets"

# --- static configuration ---
DATASET_NAME = "Optical-Optical"
JOB_NAME = "Optical-Optical_coreset"
SPLIT_NAME = "train_idx"
TSNE_CYCLES = (0, 1, 2)
CHECKPOINT_ROOT = Path("/projects/_hdd/roma") / DATASET_NAME / JOB_NAME
RESOLUTION = "medium"
STEM_TEMPLATE = "{job}_cycle{cycle}_standalone"
OUT_DIR = REPO_ROOT / "workspace" / "pics"
MAX_POINTS = 2000
RNG_SEED = 784
DEVICE = "cuda"
SYMMETRIC = False
STRATEGY_NAME = "Coreset"
# --------------------------------


def load_model(ckpt_path: Path, resolution: str, symmetric: bool, device: str) -> torch.nn.Module:
    target_device = device
    if "cuda" in device and not torch.cuda.is_available():
        target_device = "cpu"
    model = get_model(
        pretrained_backbone=True,
        resolution=resolution,
        attenuate_cert=True,
        symmetric=symmetric,
    ).to(target_device)
    ckpt = torch.load(str(ckpt_path), map_location=target_device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[TSNE] Missing keys while loading checkpoint ({len(missing)}).")
    if unexpected:
        print(f"[TSNE] Unexpected keys while loading checkpoint ({len(unexpected)}).")
    model.eval()
    return model


def get_data_root() -> str:
    base = str(BASE_DATA_ROOT)
    if DATASET_NAME not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset mapping for {DATASET_NAME}")
    return get_dataset_root(base, DATASET_NAME)


def load_indices(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(str(path)).astype(int)


def index_lookup(avail: np.ndarray) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(avail.tolist())}


def build_selector(cycle: int, data_root: str, idx_root: str) -> ActiveLearningStrategy:
    stub_args = SimpleNamespace(
        job_name=JOB_NAME,
        strategy=STRATEGY_NAME,
        tsne_plot_dir=str(OUT_DIR),
    )
    return ActiveLearningStrategy(
        stub_args,
        cycle=cycle,
        data_root=data_root,
        split=SPLIT_NAME,
        idx_root=idx_root,
        rng_seed=RNG_SEED,
    )


def compute_embeddings(selector: ActiveLearningStrategy,
                       encoder,
                       device,
                       tform,
                       indices: np.ndarray) -> torch.Tensor:
    emb_list = [
        selector._pair_embedding_scale1(encoder, device, tform, *selector._idx_to_paths(int(idx)))
        for idx in indices
    ]
    cand_embs = torch.stack(emb_list, dim=0).float()
    cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
    return cand_embs


def main() -> None:
    data_root = get_data_root()
    idx_root = osp.join(data_root, "Idx_files")
    candidates_path = Path(idx_root) / f"{SPLIT_NAME}.npy"
    avail = load_indices(candidates_path)
    lookup = index_lookup(avail)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cycle in TSNE_CYCLES:
        ckpt_path = CHECKPOINT_ROOT / f"{JOB_NAME}_cycle{cycle}_best.pth"
        if not ckpt_path.is_file():
            print(f"[TSNE] Skipping cycle {cycle}: checkpoint not found at {ckpt_path}")
            continue
        model = load_model(ckpt_path, RESOLUTION, SYMMETRIC, DEVICE)
        encoder = getattr(model, "encoder", model)
        device = next(encoder.parameters()).device
        h_res, w_res = RESOLUTIONS[RESOLUTION]
        Ht = getattr(model, "h_resized", h_res)
        Wt = getattr(model, "w_resized", w_res)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        selector = build_selector(cycle, data_root, idx_root)
        cand_embs = compute_embeddings(selector, encoder, device, tform, avail)
        hs_cert = selector._hs_cert_scores(model, avail)
        selected_path = Path(idx_root) / f"{JOB_NAME}_cycle{cycle}.npy"
        picked_local = None
        if selected_path.is_file():
            selected_idx = load_indices(selected_path)
            picked_positions = [lookup[k] for k in selected_idx.tolist() if k in lookup]
            if picked_positions:
                picked_local = np.array(sorted(set(picked_positions)), dtype=int)
        stem = STEM_TEMPLATE.format(job=JOB_NAME, cycle=cycle)
        selector._save_tsne_plot(
            cand_embs,
            values=hs_cert,
            picked_local=picked_local,
            stem=stem,
            value_label="hs_cert",
            out_dir=str(OUT_DIR),
            max_points=MAX_POINTS,
        )
        print(f"[TSNE] Cycle {cycle} plot saved to {OUT_DIR / (stem + '_tsne.png')}")


if __name__ == "__main__":
    main()
