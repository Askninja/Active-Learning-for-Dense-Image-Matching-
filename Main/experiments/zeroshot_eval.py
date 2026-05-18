import argparse
import os
import os.path as osp
import numpy as np
import torch
import cv2
from experiments.train_roma_outdoor import get_model
from roma.benchmarks import OpticalmapHomogBenchmark

# ── defaults ──────────────────────────────────────────────────────────────────
CHECKPOINT   = "/projects/ALData/weights/matchanything_roma.ckpt"
DATASET_NAME = "Optical-SAR-512"
DATA_BASE    = "/projects/ALData/Active_Datasets/"
SPLIT        = "test"
DEVICE       = 0
EVAL_SEED    = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_weights(path, device):
    ckpt = torch.load(path, map_location=f"cuda:{device}")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # PyTorch Lightning checkpoint — strip "matcher.model." prefix
        sd = ckpt["state_dict"]
        return {k.replace("matcher.model.", "", 1): v for k, v in sd.items()
                if k.startswith("matcher.model.")}
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")


def main():
    torch.cuda.set_device(DEVICE)

    print(f"Loading checkpoint: {CHECKPOINT}")
    weights = load_weights(CHECKPOINT, DEVICE)

    model = get_model(pretrained_backbone=False, resolution="low")
    model.load_state_dict(weights)
    model = model.cuda(DEVICE).eval()
    print("Model loaded.")

    data_root = osp.join(DATA_BASE, DATASET_NAME)

    # benchmark loads <data_root>/<split>.npy and images from <data_root>
    bench = OpticalmapHomogBenchmark(data_root, osp.join("Idx_files", SPLIT))
    print(f"Dataset : {DATASET_NAME}  |  Split: {SPLIT}  |  Pairs: {bench.test_idx.size}")

    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    cv2.setRNGSeed(EVAL_SEED)

    with torch.no_grad():
        results = bench.benchmark(model)

    print("\n── Zero-shot results ──────────────────────────")
    for k, v in results.items():
        print(f"  {k:<10}: {v:.4f}")
    print("───────────────────────────────────────────────")


if __name__ == "__main__":
    main()
