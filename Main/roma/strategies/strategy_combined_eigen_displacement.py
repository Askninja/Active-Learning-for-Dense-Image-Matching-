"""Combined eigen+displacement strategy: k-center greedy on concatenated eigenvalue and displacement features."""

import numpy as np
from roma.strategies.strategy_utils import (
    compute_mean_homography,
    compute_eigenvalue_features,
    compute_displacement_features,
    normalize_features,
    k_center_greedy,
)
from roma.strategies.strategy_utils import log_strategy_action


def combined_eigen_displacement(homographies: np.ndarray, b: int) -> np.ndarray:
    """Select b pairs by k-center greedy on concatenated eigenvalue + displacement features.

    Combines complementary geometric signals:
    - Eigenvalue features (6-d): capture the transformation class (what type of warp)
    - Displacement features (8-d): capture the spatial extent and direction (how much warp)

    Together the 14-d vector g(x) = [e_norm(x); d_norm(x)] provides a richer geometric
    signature than either alone, at no additional computational cost.

    Args:
        homographies: (N, K, 3, 3).
        b:            number of samples to select.

    Returns:
        (b,) indices selected by k-center greedy on the 14-d combined feature.
    """
    mean_H = compute_mean_homography(homographies)
    e_norm = normalize_features(compute_eigenvalue_features(mean_H))
    d_norm = normalize_features(compute_displacement_features(mean_H))
    g = np.concatenate([e_norm, d_norm], axis=1).astype(np.float32)  # (N, 14)
    return k_center_greedy(g, b)


def run(strategy, k: int, model) -> np.ndarray:
    """Run the combined_eigen_displacement strategy.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model used to compute matches for RANSAC.

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
    selected_pos = combined_eigen_displacement(homographies, k)
    log_strategy_action(f"Combined eigen+displacement: selected {len(selected_pos)} from {len(sample_ids)} candidates.")
    return sample_ids[selected_pos].astype(int)
