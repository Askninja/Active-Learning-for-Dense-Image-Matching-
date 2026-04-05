"""Eigenvalue diversity strategy: k-center greedy on the eigenvalue spectrum of the mean homography."""

import numpy as np
from roma.strategies.strategy_utils import (
    compute_mean_homography,
    compute_eigenvalue_features,
    normalize_features,
    k_center_greedy,
)
from roma.strategies.strategy_utils import log_strategy_action


def eigenvalue_diversity(homographies: np.ndarray, b: int) -> np.ndarray:
    """Select b pairs by k-center greedy on the eigenvalue spectrum of the mean homography.

    The eigenvalue spectrum captures the transformation class of the image pair:
    - Scale + rotation: complex conjugate eigenvalues near the unit circle
    - Pure perspective: one dominant real eigenvalue with two smaller ones
    - Degenerate scenes: poorly conditioned homography, small eigenvalues

    Args:
        homographies: (N, K, 3, 3).
        b:            number of samples to select.

    Returns:
        (b,) indices selected by k-center greedy on normalized eigenvalue features.
    """
    mean_H = compute_mean_homography(homographies)
    e_norm = normalize_features(compute_eigenvalue_features(mean_H)).astype(np.float32)
    return k_center_greedy(e_norm, b)


def run(strategy, k: int, model) -> np.ndarray:
    """Run the eigenvalue_diversity strategy.

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
    selected_pos = eigenvalue_diversity(homographies, k)
    log_strategy_action(f"Eigenvalue diversity: selected {len(selected_pos)} from {len(sample_ids)} candidates.")
    return sample_ids[selected_pos].astype(int)
