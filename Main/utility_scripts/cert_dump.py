#!/usr/bin/env python3
# Hard-coded certainty + homography certainty dumper (multi-ckpt, no argparse)

import os
import os.path as osp
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
import sys

# ----------------- HARD-CODED SETTINGS -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../RoMa-main
# If your layout differs, set: REPO_ROOT = Path("/home/abhiram001/active_learning/abhiram/AMD_ab/RoMa-main")

DATASET_NAME = "opticalmap"  # "opticalmap" | "amd"

# Single split: use this npy to infer data_root + split_stem
CAND_NPY = "/home/abhiram001/active_learning/abhiram/AMD_ab/datasets/cross_modality/Optical-Infrared/train_idx.npy"
DATA_ROOT = None
SPLIT_STEM = None

RESOLUTION = "medium"       # "low" | "medium" | "high"
SYMMETRIC = False           # True | False
DEVICE = "cuda"             # "cuda" | "cpu"

# List of checkpoints to evaluate on the SAME split
CKPT_CONFIGS = [
    {
        "name": "cycle0",
        "path": "/home/abhiram001/active_learning/abhiram/AMD_ab/RoMa-main/workspace/checkpoints/Optical-Depth/pretrained_seed.pth",
    },
    {
        "name": "cycle1",
        "path": '/projects/_hdd/roma/Optical-Depth/Depth_UWE/Depth_UWE_cycle0_strategy_adaptive_homog_uwe_best.pth',
    },
    {
        "name": "cycle2",
        "path": "/projects/_hdd/roma/Optical-Depth/Depth_UWE/Depth_UWE_cycle1_strategy_adaptive_homog_uwe_best.pth",
    },
]

# Output directory; one CSV (and optional JSON) per checkpoint
OUT_DIR = REPO_ROOT / "certainty_ckpts"
WRITE_JSON = False          # set True if you also want JSON per ckpt

SAVE_VIS = True
VIS_THRESH_SCORE = 0.05

# RoMa-homography stability params (tune if too slow)
HS_ITERS = 50
HS_N_SAMPLE = 5000
HS_THRESH_SCORE = 0.05
RANSAC_CONFIDENCE = 0.999
# -------------------------------------------------------

# Make sure imports from repo work
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
import roma  # noqa: E402
from experiments.train_roma_outdoor import get_model  # noqa: E402
from roma.benchmarks import OpticalmapHomogBenchmark, AmdHomogBenchmark  # noqa: E402

# --- plotting & kde tools (for tau computation) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
# --------------------------------------------------

RESOLUTIONS = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

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
        print(f"[warn] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing (first few):", missing[:5])
        if unexpected:
            print("  unexpected (first few):", unexpected[:5])
    model.eval()
    return model

def stem_from_npy(npy_path: str) -> tuple[str, str]:
    data_root = osp.dirname(npy_path)
    stem = osp.splitext(osp.basename(npy_path))[0]
    return data_root, stem

def run_benchmark_uncertainty(dataset_name: str, data_root: str, split_stem: str,
                              model: torch.nn.Module, model_name: str):
    if dataset_name == "opticalmap":
        bench = OpticalmapHomogBenchmark(data_root, split_stem)
    elif dataset_name == "amd":
        bench = AmdHomogBenchmark(data_root, split_stem)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    with torch.no_grad():
        pairs = bench.benchmark_uncertainty(model)

        if SAVE_VIS:
            try:
                _ = bench.benchmark(
                    model,
                    model_name=model_name,
                    vis=True,
                    thresh_score=VIS_THRESH_SCORE,
                )
            except Exception as e:
                print(f"[warn] visualization benchmark failed for {model_name}: {e}")

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
def _roma_homography_std(model, a_path: str, b_path: str,
                         iters: int = HS_ITERS,
                         n_sample: int = HS_N_SAMPLE,
                         thresh_score: float = HS_THRESH_SCORE,
                         ransac_confidence: float = RANSAC_CONFIDENCE) -> float:
    dense_matches, dense_certainty = model.match(a_path, b_path)
    sparse_matches, _ = model.sample(dense_matches, dense_certainty, n_sample, thresh_score=thresh_score)
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

    for _ in range(iters):
        if A_px.shape[0] > subset:
            sel = g.choice(A_px.shape[0], size=subset, replace=False)
            pA = A_px[sel]
            pB = B_px[sel]
        else:
            pA = A_px
            pB = B_px
        H, _ = cv2.findHomography(
            pA, pB,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_thresh,
            confidence=ransac_confidence
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

def compute_tau_and_plot(stability_values, cycle_label: str, data_root: str):
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

    out_dir = osp.join(data_root, "adaptive_homog_uwe_plots1")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(s_norm, bins=32, range=(0.0, 1.0), density=True)
    ax.set_xlabel("stability (normalized)")
    ax.set_ylabel("density")
    ax.set_title(f"{cycle_label} tau={tau:.4f}")
    fig.tight_layout()
    # choose filename like your snippet (cycleN) if possible
    cyc_num = _extract_cycle_number(cycle_label)
    if cyc_num >= 0:
        out_path = osp.join(out_dir, f"cycle{cyc_num}_stability_hist.png")
    else:
        safe_label = cycle_label.replace("/", "_")
        out_path = osp.join(out_dir, f"{safe_label}_stability_hist.png")
    fig.savefig(out_path)
    plt.close(fig)
    return float(tau)

def process_single_ckpt(model_name: str, ckpt_path: str,
                        data_root: str, split_stem: str, device: str):
    print(f"\n[info] Processing checkpoint '{model_name}' from {ckpt_path}")
    model = load_model(ckpt_path, resolution=RESOLUTION, symmetric=SYMMETRIC, device=device)

    pairs = run_benchmark_uncertainty(DATASET_NAME, data_root, split_stem, model, model_name=model_name)
    if not pairs:
        raise RuntimeError(f"benchmark_uncertainty returned no pairs for model={model_name}")

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

    cert_neg = -uncerts
    cert_inv = 1.0 / (1.0 + np.maximum(uncerts, 0.0))
    cert_minmax = 1.0 - normalize_minmax(uncerts)

    hs_std_list = []
    hs_cert_list = []
    for idx in idxs:
        a_path, b_path = _idx_to_paths(data_root, int(idx))
        s = _roma_homography_std(
            model, a_path, b_path,
            iters=HS_ITERS,
            n_sample=HS_N_SAMPLE,
            thresh_score=HS_THRESH_SCORE,
            ransac_confidence=RANSAC_CONFIDENCE,
        )
        hs_std_list.append(float(s))
        hs_cert_list.append(float(1.0 / (1.0 + max(s, 0.0))))

    # ---- new: compute tau from "stability" and plot exactly like your snippet ----
    tau = compute_tau_and_plot(hs_cert_list, cycle_label=model_name, data_root=data_root)
    print(f"[tau] {model_name}: tau={tau:.6f}")

    df = pd.DataFrame({
        "idx": idxs,
        "uncertainty": uncerts,
        "certainty_neg": cert_neg,
        "certainty_inv": cert_inv,
        "certainty_minmax": cert_minmax,
        "hs_std": np.asarray(hs_std_list, dtype=float),
        "hs_cert": np.asarray(hs_cert_list, dtype=float),
        "tau": tau,
    }).sort_values(["uncertainty", "hs_std"], ascending=[True, True]).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = OUT_DIR / f"certainty_{model_name}_{split_stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ok] wrote {len(df)} rows -> {csv_path}")

    if WRITE_JSON:
        json_path = OUT_DIR / f"certainty_{model_name}_{split_stem}.json"
        with open(json_path, "w") as f:
            json.dump(df.to_dict(orient="records"), f)
        print(f"[ok] wrote JSON -> {json_path}")

    print("\nMost certain by uncertainty (lowest):")
    print(df.sort_values('uncertainty', ascending=True).head(10).to_string(index=False))
    print("\nMost certain by hs_cert (highest):")
    print(df.sort_values('hs_cert', ascending=False).head(10).to_string(index=False))

def main():
    if CAND_NPY:
        if not osp.isfile(CAND_NPY):
            raise FileNotFoundError(f"cand_npy not found: {CAND_NPY}")
        data_root, split_stem = stem_from_npy(CAND_NPY)
    else:
        if not (DATA_ROOT and SPLIT_STEM):
            raise ValueError("Provide either CAND_NPY or both DATA_ROOT and SPLIT_STEM.")
        data_root, split_stem = DATA_ROOT, SPLIT_STEM

    print(f"[info] Using data_root='{data_root}', split_stem='{split_stem}'")

    device = DEVICE if (torch.cuda.is_available() and "cuda" in DEVICE) else "cpu"

    for cfg in CKPT_CONFIGS:
        process_single_ckpt(
            model_name=cfg["name"],
            ckpt_path=cfg["path"],
            data_root=data_root,
            split_stem=split_stem,
            device=device,
        )

if __name__ == "__main__":
    main()
