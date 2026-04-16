"""HS-Cert-3 strategy: homography-spread uncertainty computed inline without shared utilities."""

import cv2
import numpy as np
from PIL import Image

from roma.strategies.strategy_utils import log_strategy_action


def _resolve_pair_paths(data_root, idx):
    """Resolve image paths using the OpticalMap naming convention: pair{idx}_1, pair{idx}_2."""
    from pathlib import Path
    root = Path(data_root)

    def pick(base):
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
            candidate = root / f"{base}{ext}"
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not find image for {base} with known extensions under {root}"
        )

    return pick(f"pair{idx}_1"), pick(f"pair{idx}_2")


def _hs_cert_scores(strategy, model_for_uncertainty, avail: np.ndarray) -> np.ndarray:
    """Compute homography-stability based certainty for each candidate pair.

    Returns:
        hs_cert: (N,) certainty scores in (0, 1].  Low = uncertain.
                 Uncertainty = 1 - hs_cert.

    Side effect:
        Populates strategy.homography_sets[pair_id] with the list of
        valid (3, 3) homography matrices so downstream geometry-diversity
        descriptors can reuse them without a second forward pass.
    """
    if not hasattr(strategy, "homography_sets"):
        strategy.homography_sets = {}

    hs_vals = []
    for i in avail:
        a_path, b_path = _resolve_pair_paths(strategy.data_root, int(i))
        dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
        sparse_matches, _ = model_for_uncertainty.sample(
            dense_matches, dense_certainty, 5000, thresh_score=0.05
        )
        sm = sparse_matches.detach().cpu().numpy()
        if sm.shape[0] < 8:
            strategy.homography_sets[int(i)] = []
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
        strategy.homography_sets[int(i)] = Hs
        if len(Hs) < 2:
            hs_vals.append(0.0)
            continue
        Hs_stack = np.stack(Hs, axis=0)
        c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
        warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs_stack], axis=0)
        s = float(warped.std(axis=0).mean())
        hs_vals.append(s)
    hs_vals = np.asarray(hs_vals, dtype=float)
    hs_cert = 1.0 / (1.0 + np.maximum(hs_vals, 0.0))
    return hs_cert


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs with the lowest HS-Cert score (highest uncertainty).

    Computes homography-stability certainty inline: for each candidate pair,
    runs 50 RANSAC homographies on random subsets of sparse matches, projects
    the 4 corners of image A through each homography, and measures the spread
    (std of corner positions).  Low certainty (high spread) = high uncertainty.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices sorted by ascending certainty (descending uncertainty).
    """
    if model is None:
        raise ValueError("model is required for hs_cert_3 strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    hs_cert = _hs_cert_scores(strategy, model, avail)

    # Select pairs with lowest certainty (= highest uncertainty)
    order = np.argsort(hs_cert)
    chosen = avail[order[:k]].astype(int)
    log_strategy_action(
        f"HS-Cert-3: scored {avail.size} samples, "
        f"mean_cert={float(hs_cert.mean()):.4f}, selected {chosen.size}."
    )
    return chosen
