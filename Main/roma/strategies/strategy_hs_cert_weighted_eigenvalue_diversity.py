"""hs_cert weighted eigenvalue diversity: hs_cert-weighted k-center greedy on eigenvalue features."""

import numpy as np
from roma.strategies.strategy_utils import (
    compute_mean_homography,
    compute_eigenvalue_features,
    normalize_features,
    k_center_greedy,
)
from roma.strategies.strategy_utils import log_strategy_action


def hs_cert_weighted_eigenvalue_diversity(
    homographies: np.ndarray, uncertainties: np.ndarray, b: int
) -> np.ndarray:
    """Select b pairs by hs_cert-weighted k-center greedy on eigenvalue features.

    Scales each pair's normalized eigenvalue feature vector by its uncertainty weight
    before running k-center greedy. Uncertain pairs appear farther apart in the
    weighted feature space, biasing selection toward pairs that are both geometrically
    novel and hard for the current model.

    Weight mapping: w_i = 1 + (u_i - u_min) / (u_max - u_min) ∈ [1, 2], so the
    most uncertain pair has its distances doubled relative to the least uncertain.

    Args:
        homographies:  (N, K, 3, 3) — K RANSAC homographies per pair.
        uncertainties: (N,) — hs_cert score per pair (higher = more uncertain).
        b:             number of samples to select.

    Returns:
        (b,) indices selected by hs_cert-weighted k-center greedy on eigenvalue features.
    """
    mean_H = compute_mean_homography(homographies)
    e_norm = normalize_features(compute_eigenvalue_features(mean_H)).astype(np.float32)

    uncertainties = np.asarray(uncertainties, dtype=np.float32)
    u_min, u_max = float(uncertainties.min()), float(uncertainties.max())
    if u_max - u_min < 1e-8:
        weights = np.ones(len(uncertainties), dtype=np.float32)
    else:
        weights = 1.0 + (uncertainties - u_min) / (u_max - u_min)

    weighted = e_norm * weights[:, None]
    return k_center_greedy(weighted, b)


def run(strategy, k: int, model) -> np.ndarray:
    """Run the hs_cert_weighted_eigenvalue_diversity strategy.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices.
    """
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    sample_ids, homographies = strategy._compute_ransac_homographies(model, avail)
    if sample_ids.size == 0:
        return np.empty(0, dtype=int)

    score_ids, score_values = strategy._score_avail(model, avail, score_name="hs_cert")
    if score_values.size == 0:
        return np.empty(0, dtype=int)

    score_map = {int(idx): float(s) for idx, s in zip(score_ids.tolist(), score_values.tolist())}
    uncertainties = np.asarray([score_map[int(idx)] for idx in sample_ids.tolist()], dtype=np.float32)

    selected_pos = hs_cert_weighted_eigenvalue_diversity(homographies, uncertainties, k)
    log_strategy_action(
        f"HS-Cert weighted eigenvalue diversity: selected {len(selected_pos)} from {len(sample_ids)} candidates."
    )
    return sample_ids[selected_pos].astype(int)
