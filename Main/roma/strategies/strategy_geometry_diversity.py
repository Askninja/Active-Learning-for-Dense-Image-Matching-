"""Geometry diversity strategy: k-center greedy on corner displacement descriptors.

Each image pair is represented by an 8-dimensional descriptor g(x) computed by
averaging displacement vectors across K=50 RANSAC homographies (mathematically
valid: averages in R^8 displacement space, not in homography matrix space).
Pairs are selected to cover the geometric diversity of the unlabeled pool using
k-center greedy, no labeled-set seeding.
"""

import numpy as np
from typing import Optional
from roma.strategies.strategy_utils import (
    k_center_greedy,
    log_strategy_action,
)


# ---------------------------------------------------------------------------
# Per-pair descriptor (averaged displacement vectors, K=50 RANSAC runs)
# ---------------------------------------------------------------------------

def homography_to_geom_descriptor(H: np.ndarray, image_size: int = 560) -> np.ndarray:
    """Map image corners through H and return the 8-D corner-displacement vector."""
    s = float(image_size)
    corners = np.array([[0.0, 0.0], [s - 1, 0.0], [s - 1, s - 1], [0.0, s - 1]], dtype=np.float64)
    corners_h = np.concatenate([corners, np.ones((4, 1), dtype=np.float64)], axis=1)
    warped_h = (H @ corners_h.T).T
    denom = warped_h[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    warped = warped_h[:, :2] / denom
    delta = (warped - corners) / s
    delta = np.clip(delta, -3.0, 3.0)
    return delta.reshape(-1)


def compute_geometric_diversity_from_homographies(
    homographies: list,
    image_size: int = 560,
) -> np.ndarray:
    """Average displacement vectors across K valid homographies.

    Mathematically valid: averages in R^8 displacement space,
    not in homography matrix space.
    """
    displacements = []
    for H in homographies:
        if H is None:
            continue
        d = homography_to_geom_descriptor(H, image_size=image_size)
        if np.any(d != 0):
            displacements.append(d)
    if len(displacements) < 3:
        return np.zeros(8, dtype=np.float64)
    return np.mean(np.stack(displacements, axis=0), axis=0)


def compute_geometric_diversity(
    homographies: list[Optional[np.ndarray]],
    image_size: int = 560,
) -> np.ndarray:
    """Compute an 8-dimensional geometric diversity descriptor for one image pair.

    Args:
        homographies: K homographies as a list of (3, 3) arrays or None for failed
                      RANSAC runs.  Some or all entries may be None.
        image_size:   Side length of the square image in pixels (default 560).

    Returns:
        (8,) descriptor g_raw(x). Returns np.zeros(8) when fewer than 3 valid
        homographies are present.
    """
    return compute_geometric_diversity_from_homographies(homographies, image_size=image_size)


# ---------------------------------------------------------------------------
# Pool-level normalization
# ---------------------------------------------------------------------------

def normalize_geometric_descriptors(G: np.ndarray) -> np.ndarray:
    """Robustly normalize N geometric descriptors to have zero median and unit IQR.

    Each dimension is normalized independently using the median and
    interquartile range (IQR) across the N samples, making the normalization
    resistant to outliers.  Constant dimensions (IQR < 1e-8) are zeroed out.
    Final values are clipped to [-5, 5].

    Args:
        G: (N, 8) raw geometric descriptors.

    Returns:
        (N, 8) normalized descriptors G_norm.
    """
    G = np.asarray(G, dtype=np.float64)
    N, D = G.shape
    G_norm = np.zeros_like(G)
    for j in range(D):
        col = G[:, j]
        med = np.median(col)
        iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
        if iqr < 1e-8:
            G_norm[:, j] = 0.0
        else:
            G_norm[:, j] = (col - med) / iqr
    return np.clip(G_norm, -5.0, 5.0)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_geometric_diversity(
    G_norm: np.ndarray,
    b: int,
) -> np.ndarray:
    """Select b indices from the unlabeled pool via k-center greedy.

    Args:
        G_norm: (N, 8) normalized descriptors for the unlabeled pool.
        b:      number of samples to select.

    Returns:
        (b,) selected indices into G_norm.
    """
    return k_center_greedy(G_norm.astype(np.float32), b)


# ---------------------------------------------------------------------------
# Strategy entry point
# ---------------------------------------------------------------------------

def run(strategy, k: int, model) -> np.ndarray:
    """Run the geometry_diversity strategy.

    K=50 RANSAC homographies are computed per pair via _hs_cert_scores (which
    populates strategy.homography_sets as a side effect).  An 8-dimensional
    descriptor is built by averaging displacement vectors across those K valid
    homographies.  k-center greedy selects the most geometrically diverse subset.
    No labeled-set seeding. Normalization over unlabeled pool only.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model used to compute matches.

    Returns:
        (k,) selected pool indices.
    """
    from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores  # noqa: PLC0415

    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    image_size = int(getattr(strategy, "_image_size", 560))

    # Populate strategy.homography_sets with K=50 RANSAC homographies per pair
    _hs_cert_scores(strategy, model, avail)

    G_raw_list = []
    valid_unlabeled_ids = []
    for pair_id in avail.tolist():
        d = compute_geometric_diversity_from_homographies(
            strategy.homography_sets.get(int(pair_id), []),
            image_size=image_size,
        )
        if np.any(d != 0):
            G_raw_list.append(d)
            valid_unlabeled_ids.append(int(pair_id))
        else:
            log_strategy_action(
                f"Geometry diversity: skipping pair {pair_id} — zero/degenerate descriptor."
            )

    if not valid_unlabeled_ids:
        return np.empty(0, dtype=int)

    valid_unlabeled_ids = np.asarray(valid_unlabeled_ids, dtype=int)
    G_raw_unlabeled = np.asarray(G_raw_list, dtype=np.float64)
    N_u = valid_unlabeled_ids.size
    k = min(k, N_u)

    # Normalize over unlabeled pool only
    G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled)

    # k-center greedy on unlabeled pool, no seeding
    unlabeled_pos = select_geometric_diversity(G_norm_unlabeled, k)

    log_strategy_action(
        f"Geometry diversity: {N_u} unlabeled, descriptor_dim=8, selected {unlabeled_pos.size} samples."
    )
    return valid_unlabeled_ids[unlabeled_pos].astype(int)