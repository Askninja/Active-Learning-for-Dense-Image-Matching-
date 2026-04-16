"""Geometry diversity strategy: k-center greedy on corner displacement descriptors.

Each image pair is represented by an 8-dimensional descriptor g(x) containing
the normalized displacement of the four image corners. Pairs are then
selected to cover the geometric diversity of the unlabeled pool using k-center
greedy, seeded from already-labeled pairs when available.

Homographies are obtained via ``uncertainty_estimation.compute_uncertainty_and_homographies``
and cached on the strategy object, so a prior hs_cert pass in a combined strategy
incurs no extra forward passes here.
"""

import numpy as np
from typing import List, Optional
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.uncertainty_estimation import compute_uncertainty_and_homographies


# ---------------------------------------------------------------------------
# Per-pair descriptor
# ---------------------------------------------------------------------------

def compute_geometric_diversity(
    homographies: List[Optional[np.ndarray]],
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
    # Step 1: filter out failed RANSAC runs
    valid = [H for H in homographies if H is not None]

    # Step 2: degenerate case — not enough homographies to be meaningful
    if len(valid) < 3:
        return np.zeros(8, dtype=np.float64)

    # Step 3: mean homography in R^{3x3}
    mu = np.mean(np.stack(valid, axis=0), axis=0)  # (3, 3)

    s = float(image_size)

    # Step 4: four corners of I_A in homogeneous coordinates
    #   [0,0,1], [W,0,1], [0,H,1], [W,H,1]
    corners_hom = np.array([
        [0., 0., 1.],
        [s,  0., 1.],
        [0., s,  1.],
        [s,  s,  1.],
    ], dtype=np.float64)  # (4, 3)

    # Step 5 & 6: project each corner through mu, compute normalized displacement
    deltas = []
    for o in corners_hom:
        p = mu @ o                                    # (3,)
        denom = p[2] if abs(p[2]) >= 1e-10 else 1e-10
        p_cart = p[:2] / denom                        # (2,) Cartesian
        delta = (p_cart - o[:2]) / s                  # normalized displacement
        # Clip to avoid extreme outliers from near-singular homographies
        delta = np.clip(delta, -3.0, 3.0)
        deltas.append(delta)

    # Step 7: d(x) = [delta_1, delta_2, delta_3, delta_4] ∈ R^8
    g_raw = np.concatenate(deltas)  # (8,)

    # Step 8: replace any NaN / Inf with 0
    g_raw = np.where(np.isfinite(g_raw), g_raw, 0.0)

    return g_raw


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
            # Constant dimension — zero it out to avoid numerical blow-up
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
    already_selected: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select b indices from the unlabeled pool via k-center greedy.

    If already_selected is provided (indices into a combined array where the
    first len(already_selected) rows correspond to labeled pairs), the greedy
    search is seeded from those rows so that selected points are also diverse
    relative to the labeled set.

    Args:
        G_norm:           (N, 8) normalized descriptors for the unlabeled pool.
        b:                number of samples to select.
        already_selected: optional (M,) indices into the embedding matrix that
                          should be treated as already chosen (used for seeding).

    Returns:
        (b,) selected indices into G_norm (i.e., into the unlabeled pool).
    """
    return k_center_greedy(G_norm.astype(np.float32), b, initial_idx=already_selected)


# ---------------------------------------------------------------------------
# Strategy entry point
# ---------------------------------------------------------------------------

def run(strategy, k: int, model) -> np.ndarray:
    """Run the geometry_diversity strategy.

    For each unlabeled image pair, K=50 RANSAC homographies are obtained via
    ``compute_uncertainty_and_homographies`` (which also populates
    ``strategy.homography_sets`` for combined-strategy reuse). An 8-dimensional
    displacement descriptor is built from the mean homography, then k-center greedy
    selects the most geometrically diverse subset.  If labeled pairs are available
    their descriptors are included in the normalization and seed the greedy search.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model used to compute matches for RANSAC homographies.

    Returns:
        (k,) selected pool indices.
    """
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    # --- Populate strategy.homography_sets for the unlabeled pool ---
    # If a prior hs_cert pass already filled the cache for these pairs,
    # compute_uncertainty_and_homographies is still called but the model
    # forward passes are cheap relative to the RANSAC work.
    _u, _c, _h, valid_unlabeled_ids = compute_uncertainty_and_homographies(
        strategy, model, avail
    )
    if valid_unlabeled_ids.size == 0:
        return np.empty(0, dtype=int)

    N_u = valid_unlabeled_ids.size

    # --- Build g_raw for each unlabeled pair from the cached homographies ---
    G_raw_unlabeled = np.zeros((N_u, 8), dtype=np.float64)
    for i, pid in enumerate(valid_unlabeled_ids.tolist()):
        h_list = strategy.homography_sets.get(int(pid), [])
        G_raw_unlabeled[i] = compute_geometric_diversity(h_list, image_size=560)

    # --- Optionally include labeled pairs in normalization ---
    labeled_idx = strategy.train_current_idx
    G_raw_labeled = None
    N_l = 0

    if labeled_idx.size > 0:
        try:
            _u_l, _c_l, _h_l, valid_labeled_ids = compute_uncertainty_and_homographies(
                strategy, model, labeled_idx
            )
            if valid_labeled_ids.size > 0:
                N_l = valid_labeled_ids.size
                G_raw_labeled = np.zeros((N_l, 8), dtype=np.float64)
                for i, pid in enumerate(valid_labeled_ids.tolist()):
                    h_list = strategy.homography_sets.get(int(pid), [])
                    G_raw_labeled[i] = compute_geometric_diversity(h_list, image_size=560)
        except Exception as exc:
            log_strategy_action(
                f"Geometry diversity: could not compute labeled descriptors ({exc}); "
                "normalizing on unlabeled pool only."
            )
            G_raw_labeled = None
            N_l = 0

    # --- Robust normalization across the combined pool ---
    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_unlabeled], axis=0)  # (N_l + N_u, 8)
        G_norm_all = normalize_geometric_descriptors(G_raw_all)
        G_norm_unlabeled = G_norm_all[N_l:]   # (N_u, 8)
        G_norm_labeled   = G_norm_all[:N_l]   # (N_l, 8)
    else:
        G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled)
        G_norm_labeled = None

    # --- k-center greedy selection ---
    # When labeled descriptors exist, combine them so that k_center_greedy
    # computes initial min-distances relative to the already-labeled set.
    if G_norm_labeled is not None and N_l > 0:
        G_combined = np.concatenate([G_norm_labeled, G_norm_unlabeled], axis=0)
        initial_idx = np.arange(N_l, dtype=int)
        total_needed = min(N_l + k, N_l + N_u)
        all_selected = k_center_greedy(
            G_combined.astype(np.float32), total_needed, initial_idx=initial_idx
        )
        # Keep only indices that fall in the unlabeled block [N_l, N_l + N_u)
        unlabeled_pos = np.array(
            [idx - N_l for idx in all_selected if idx >= N_l], dtype=int
        )[:k]
        if unlabeled_pos.size < k:
            # Top up from remaining uncovered pool positions
            covered = set(unlabeled_pos.tolist())
            extras = [i for i in range(N_u) if i not in covered]
            unlabeled_pos = np.concatenate([
                unlabeled_pos,
                np.asarray(extras[:k - unlabeled_pos.size], dtype=int),
            ])
    else:
        log_strategy_action("Geometry diversity: no labeled descriptors; using k-center greedy.")
        unlabeled_pos = select_geometric_diversity(G_norm_unlabeled, k)

    log_strategy_action(
        f"Geometry diversity: {N_l} labeled, {N_u} unlabeled, "
        f"descriptor_dim=8, selected {unlabeled_pos.size} samples."
    )
    return valid_unlabeled_ids[unlabeled_pos].astype(int)
