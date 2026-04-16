"""Displacement diversity strategy: k-center greedy on corner displacement vectors."""

import cv2
import numpy as np
from roma.strategies.strategy_utils import (
    compute_displacement_features,
    normalize_features,
    k_center_greedy,
)
from roma.strategies.strategy_utils import log_strategy_action


def _compute_single_homography(
    matches: np.ndarray,
    confidences: np.ndarray,
    image_size: int = 560,
    top_k: int = 5000,
) -> np.ndarray:
    """Fit one benchmark-style homography from the highest-confidence matches."""
    if matches is None or confidences is None or len(matches) < 4:
        return np.eye(3, dtype=np.float64)

    matches = np.asarray(matches, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if len(matches) > top_k:
        top_idx = np.argpartition(confidences, -top_k)[-top_k:]
        matches = matches[top_idx]

    if len(matches) < 4:
        return np.eye(3, dtype=np.float64)

    try:
        H_mat, _ = cv2.findHomography(
            matches[:, :2],
            matches[:, 2:],
            method=cv2.RANSAC,
            confidence=0.99999,
            ransacReprojThreshold=3.0 * float(image_size) / 480.0,
        )
    except Exception:
        H_mat = None

    if H_mat is None:
        return np.eye(3, dtype=np.float64)
    return np.asarray(H_mat, dtype=np.float64)


def displacement_diversity(homographies: np.ndarray, b: int) -> np.ndarray:
    """Select b pairs by k-center greedy on corner displacement vectors.

    The displacement of the four image corners through the fitted homography encodes
    the spatial warp pattern directly: translation, rotation, scale, and perspective
    all produce characteristic corner displacement signatures.

    Args:
        homographies: (N, 3, 3).
        b:            number of samples to select.

    Returns:
        (b,) indices selected by k-center greedy on normalized displacement features.
    """
    d_norm = normalize_features(compute_displacement_features(homographies)).astype(np.float32)
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
    image_size = int(getattr(strategy, "_image_size", 560))
    sample_ids = []
    homographies = []

    for pair_id in avail.astype(int).tolist():
        try:
            matches, confs = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"Displacement diversity: skipping pair {pair_id} because match extraction failed: {exc}"
            )
            continue
        if matches is None or len(matches) == 0:
            continue
        sample_ids.append(int(pair_id))
        homographies.append(_compute_single_homography(matches, confs, image_size=image_size))

    if not sample_ids:
        return np.empty(0, dtype=int)
    sample_ids = np.asarray(sample_ids, dtype=int)
    homographies = np.stack(homographies, axis=0)
    selected_pos = displacement_diversity(homographies, k)
    log_strategy_action(f"Displacement diversity: selected {len(selected_pos)} from {len(sample_ids)} candidates.")
    return sample_ids[selected_pos].astype(int)
