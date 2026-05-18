"""Quick smoke-test: run the METU VisTIR benchmark on 10 random test pairs."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from experiments.train_roma_outdoor import get_model
from roma.benchmarks.metu_vistir_flat_benchmark import METUVisTIRFlatBenchmark

DATA_ROOT   = "/projects/ALData/Active_Datasets/cross_modality/METU_VisTIR"
SPLIT       = "Idx_files/test"
N_PAIRS     = 10
DEVICE      = 0
CHECKPOINT  = "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/checkpoints/roma_outdoor.pth"

# --- load pretrained model (no fine-tuning) --------------------------------
model = get_model(
    pretrained_backbone=True,
    resolution="low",
    attenuate_cert=False,
    symmetric=False,
).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=f"cuda:{DEVICE}")
weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(weights, strict=True)
model.eval()
print(f"Model loaded from {CHECKPOINT} (zero-shot).")

# --- build benchmark and restrict to N_PAIRS random pairs ------------------
bench = METUVisTIRFlatBenchmark(DATA_ROOT, SPLIT)
rng = np.random.default_rng(42)
bench.test_idx = rng.choice(bench.test_idx, size=N_PAIRS, replace=False)
print(f"Running on pairs: {bench.test_idx}")

# --- run -------------------------------------------------------------------
with torch.no_grad():
    results = bench.benchmark(model)

print("\n=== Zero-shot results (10 pairs) ===")
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")
