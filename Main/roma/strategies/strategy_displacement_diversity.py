"""Displacement diversity strategy: k-center greedy on corner displacement vectors."""

import numpy as np
from roma.strategies.strategy_utils import (
    compute_mean_homography,
    compute_displacement_features,
    normalize_features,
    k_center_greedy,
)
from roma.strategies.strategy_utils import log_strategy_action


def displacement_diversity(homographies: np.ndarray, b: int) -> np.ndarray:
    """Select b pairs by k-center greedy on corner displacement vectors.

    The displacement of the four image corners through the mean homography encodes
    the spatial warp pattern directly: translation, rotation, scale, and perspective
    all produce characteristic corner displacement signatures.

    Args:
        homographies: (N, K, 3, 3).
        b:            number of samples to select.

    Returns:
        (b,) indices selected by k-center greedy on normalized displacement features.
    """
    mean_H = compute_mean_homography(homographies)
    d_norm = normalize_features(compute_displacement_features(mean_H)).astype(np.float32)
    return k_center_greedy(d_norm, b)


def run(strategy, k: int, model) -> np.ndarray:
    """Run the displacement_diversity strategy.

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
    selected_pos = displacement_diversity(homographies, k)
    log_strategy_action(f"Displacement diversity: selected {len(selected_pos)} from {len(sample_ids)} candidates.")
    return sample_ids[selected_pos].astype(int)
