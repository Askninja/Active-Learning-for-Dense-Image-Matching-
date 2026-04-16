"""Task-aware uncertainty estimation via RANSAC stability.

RoMa outputs 313,600 dense matches per image pair. We filter to the top
2000 by confidence, then run K=50 RANSAC homography estimations on random
subsets of 500 points each. The variance of corner projections across K
runs measures how unstable the geometric estimation is — high variance
means the model is uncertain about the transformation for this pair.

The same K homographies are reused by geometric_diversity.py to compute
the geometric diversity descriptor g(x), so both signals come from a
single computation.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core signal: uncertainty + homographies from K RANSAC runs
# ---------------------------------------------------------------------------

def compute_ransac_signals(
    matches: np.ndarray,
    confidences: np.ndarray,
    image_size: int = 560,
    K: int = 50,
    subset_size: int = 500,
    top_k: int = 2000,
) -> Tuple[float, float, List[Optional[np.ndarray]]]:
    """Compute per-pair uncertainty and the K RANSAC homographies from dense matches.

    Filters to the top `top_k` most-confident matches, then runs K independent
    RANSAC homography fits on random subsets of `subset_size` points each.
    Uncertainty is measured as the variance of the 4 corner projections across
    the K runs: a stable scene produces tightly clustered projections (low
    uncertainty), while an ambiguous or texture-less scene produces widely
    scattered ones (high uncertainty).

    Args:
        matches:     (N, 4) pixel-space correspondences [xA, yA, xB, yB].
        confidences: (N,) confidence scores in [0, 1] from RoMa.
        image_size:  side length of the square image in pixels (default 560).
        K:           number of RANSAC subsets (default 50).
        subset_size: number of points per RANSAC subset (default 500).
        top_k:       number of top-confidence matches to retain (default 2000).

    Returns:
        u(x):         float uncertainty score in [0, 1].
        c(x):         float certainty score in [0, 1]  (= 1 - u(x)).
        homographies: List[Optional[np.ndarray]] of length K; each entry is a
                      (3, 3) homography matrix or None if RANSAC failed or the
                      homography was degenerate.
    """
    _degenerate = (0.0, 1.0, [None] * K)

    # Step 1: filter to top_k matches by confidence
    if matches is None or len(matches) < 8:
        return _degenerate
    matches = np.asarray(matches, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)

    n_total = len(matches)
    if n_total > top_k:
        top_idx = np.argpartition(confidences, -top_k)[-top_k:]
        matches = matches[top_idx]
        confidences = confidences[top_idx]

    if len(matches) < 8:
        return _degenerate

    kpts_A = matches[:, :2]   # (M, 2) pixel coords in image A
    kpts_B = matches[:, 2:]   # (M, 2) pixel coords in image B
    n_pts = len(matches)

    # Define the 4 image corners in homogeneous coordinates.
    # This function measures instability via corner spread; use
    # compute_ransac_signals_grid() for a denser probe lattice.
    s = float(image_size)
    corners_hom = np.array([
        [0.0, 0.0, 1.0],
        [s, 0.0, 1.0],
        [0.0, s, 1.0],
        [s, s, 1.0],
    ], dtype=np.float64)  # (4, 3)

    # Step 2: K independent RANSAC runs
    homographies: List[Optional[np.ndarray]] = []
    corner_projections: List[np.ndarray] = []  # list of (4, 2) arrays for valid runs

    for _ in range(K):
        # Sample subset_size points; use replacement if fewer are available
        replace = n_pts < subset_size
        ss = subset_size if not replace else n_pts
        idx = np.random.choice(n_pts, ss, replace=replace)

        H_mat, _ = cv2.findHomography(
            kpts_A[idx], kpts_B[idx],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if H_mat is None:
            homographies.append(None)
            continue

        # Step 3: project corners, check for degenerate result
        projs = np.zeros((4, 2), dtype=np.float64)
        degenerate = False
        for j, c in enumerate(corners_hom):
            p = H_mat @ c                    # (3,)
            # Handle corner behind camera: preserve sign to avoid flipping
            denom = p[2]
            if abs(denom) < 1e-10:
                denom = 1e-10 * (1.0 if denom >= 0 else -1.0)
            p_cart = p[:2] / denom           # (2,) Cartesian

            # Reject homographies that project corners to unreasonable positions
            if np.any(np.abs(p_cart) > 10.0 * s):
                degenerate = True
                break
            projs[j] = p_cart

        if degenerate:
            homographies.append(None)
            continue

        homographies.append(H_mat)
        corner_projections.append(projs)  # (4, 2)

    # Step 4: compute variance across valid runs
    n_valid = len(corner_projections)
    if n_valid < 3:
        # Cannot estimate variance — treat as maximally certain since we have
        # no evidence of inconsistency (all runs failed → scene may be too easy
        # or degenerate, not uncertain in the model-uncertainty sense)
        return _degenerate

    proj_stack = np.stack(corner_projections, axis=0)  # (V, 4, 2)
    stds_per_corner = []
    for i in range(4):
        u_coords = proj_stack[:, i, 0]
        v_coords = proj_stack[:, i, 1]
        # If any corner has fewer than 3 valid projections, signal max uncertainty
        if len(u_coords) < 3:
            return (1.0 - 1.0 / (1.0 + 10.0), 1.0 / (1.0 + 10.0), homographies)
        std_i = 0.5 * (float(np.std(u_coords)) + float(np.std(v_coords)))
        stds_per_corner.append(std_i)

    # Step 4 (cont.): mean spread across 4 corners
    s_x = float(np.mean(stds_per_corner))

    # Step 5: map to certainty / uncertainty in [0, 1]
    c_x = 1.0 / (1.0 + s_x)
    u_x = 1.0 - c_x

    return u_x, c_x, homographies


def compute_ransac_signals_grid(
    matches: np.ndarray,
    confidences: np.ndarray,
    image_size: int = 560,
    K: int = 50,
    subset_size: int = 500,
    top_k: int = 2000,
    grid_size: int = 5,
) -> Tuple[float, float, List[Optional[np.ndarray]]]:
    """Like compute_ransac_signals but probes a uniform grid instead of 4 corners.

    Samples a ``grid_size x grid_size`` lattice of points evenly covering image A
    (including boundary), projects each point through every valid RANSAC homography,
    and measures uncertainty as the mean per-point projection std across K runs.

    Args:
        matches:    (N, 4) pixel-space correspondences [xA, yA, xB, yB].
        confidences:(N,) confidence scores in [0, 1] from RoMa.
        image_size: side length of the square image in pixels (default 560).
        K:          number of RANSAC subsets (default 50).
        subset_size:number of points per RANSAC subset (default 500).
        top_k:      number of top-confidence matches to retain (default 2000).
        grid_size:  number of grid divisions per axis (default 5 → 25 probe points).

    Returns:
        u(x):         float uncertainty score in [0, 1].
        c(x):         float certainty score in [0, 1]  (= 1 - u(x)).
        homographies: List[Optional[np.ndarray]] of length K.
    """
    _degenerate = (0.0, 0.0, [None] * K)

    if matches is None or len(matches) < 8:
        return _degenerate
    matches = np.asarray(matches, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)

    n_total = len(matches)
    if n_total > top_k:
        top_idx = np.argpartition(confidences, -top_k)[-top_k:]
        matches = matches[top_idx]
        confidences = confidences[top_idx]

    if len(matches) < 8:
        return _degenerate

    kpts_A = matches[:, :2]
    kpts_B = matches[:, 2:]
    n_pts = len(matches)

    # Build a grid_size x grid_size uniform grid over image A
    s = float(image_size)
    xs = np.linspace(0., s, grid_size)
    ys = np.linspace(0., s, grid_size)
    gx, gy = np.meshgrid(xs, ys)
    pts_xy = np.column_stack([gx.ravel(), gy.ravel()])          # (G, 2)
    pts_hom = np.hstack([pts_xy, np.ones((len(pts_xy), 1))])    # (G, 3)
    n_probes = len(pts_hom)

    homographies: List[Optional[np.ndarray]] = []
    probe_projections: List[np.ndarray] = []

    for _ in range(K):
        replace = n_pts < subset_size
        ss = subset_size if not replace else n_pts
        idx = np.random.choice(n_pts, ss, replace=replace)

        H_mat, _ = cv2.findHomography(
            kpts_A[idx], kpts_B[idx],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if H_mat is None:
            homographies.append(None)
            continue

        projs = np.zeros((n_probes, 2), dtype=np.float64)
        degenerate = False
        for j, c in enumerate(pts_hom):
            p = H_mat @ c
            denom = p[2]
            if abs(denom) < 1e-10:
                denom = 1e-10 * (1.0 if denom >= 0 else -1.0)
            p_cart = p[:2] / denom
            if np.any(np.abs(p_cart) > 10.0 * s):
                degenerate = True
                break
            projs[j] = p_cart

        if degenerate:
            homographies.append(None)
            continue

        homographies.append(H_mat)
        probe_projections.append(projs)

    n_valid = len(probe_projections)
    if n_valid < 3:
        return _degenerate

    proj_stack = np.stack(probe_projections, axis=0)  # (V, G, 2)
    stds_per_probe = []
    for i in range(n_probes):
        u_coords = proj_stack[:, i, 0]
        v_coords = proj_stack[:, i, 1]
        if len(u_coords) < 3:
            return (1.0 - 1.0 / (1.0 + 10.0), 1.0 / (1.0 + 10.0), homographies)
        std_i = 0.5 * (float(np.std(u_coords)) + float(np.std(v_coords)))
        stds_per_probe.append(std_i)

    s_x = float(np.mean(stds_per_probe))
    c_x = 1.0 / (1.0 + s_x)
    u_x = 1.0 - c_x
    return u_x, c_x, homographies


# ---------------------------------------------------------------------------
# Wrapper: run over a batch of pairs and cache results on the strategy object
# ---------------------------------------------------------------------------

def compute_uncertainty_and_homographies(
    strategy,
    model,
    pair_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[List[Optional[np.ndarray]]], np.ndarray]:
    """Compute HS-uncertainty and K homographies for a batch of image pairs.

    Runs `compute_ransac_signals` for every pair in `pair_ids`, caches the
    K homographies in ``strategy.homography_sets[pair_id]`` for downstream
    reuse by the geometric diversity descriptor, and returns the per-pair
    signals as arrays.

    Args:
        strategy: ActiveLearningStrategy instance.  Must have a
                  ``_get_matches_and_confidences(model, pair_id)`` method
                  and a ``homography_sets`` dict attribute.
        model:    RoMa model used to produce dense matches.
        pair_ids: (M,) pool indices to process.

    Returns:
        uncertainties:    (V,) uncertainty scores u(x) ∈ [0, 1].
        certainties:      (V,) certainty scores c(x) ∈ [0, 1].
        all_homographies: List[List[Optional[np.ndarray]]] of shape (V, K).
        valid_ids:        (V,) subset of pair_ids that returned matches.
    """
    from roma.strategies.strategy_utils import log_strategy_action

    pair_ids = np.asarray(pair_ids, dtype=int)
    if pair_ids.size == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            np.empty(0, dtype=int),
        )

    # Ensure the homography cache exists on the strategy object
    if not hasattr(strategy, "homography_sets"):
        strategy.homography_sets = {}

    uncertainties = []
    certainties = []
    all_homographies = []
    valid_ids = []

    for pair_id in pair_ids.tolist():
        try:
            matches, confs = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"Uncertainty estimation: skipping pair {pair_id} — "
                f"_get_matches_and_confidences failed: {exc}"
            )
            continue

        if matches is None or len(matches) == 0:
            # No matches returned — store empty homography set and skip scoring
            strategy.homography_sets[int(pair_id)] = [None] * 50
            continue

        u, c, homogs = compute_ransac_signals(
            matches, confs,
            image_size=getattr(strategy, "_image_size", 560),
        )

        # Cache homographies for reuse by geometric_diversity.py
        strategy.homography_sets[int(pair_id)] = homogs

        uncertainties.append(float(u))
        certainties.append(float(c))
        all_homographies.append(homogs)
        valid_ids.append(int(pair_id))

    if not valid_ids:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            np.empty(0, dtype=int),
        )

    log_strategy_action(
        f"Uncertainty estimation: computed signals for {len(valid_ids)}/{len(pair_ids)} pairs."
    )
    return (
        np.asarray(uncertainties, dtype=np.float32),
        np.asarray(certainties, dtype=np.float32),
        all_homographies,
        np.asarray(valid_ids, dtype=int),
    )


def compute_uncertainty_and_homographies_grid(
    strategy,
    model,
    pair_ids: np.ndarray,
    grid_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[List[Optional[np.ndarray]]], np.ndarray]:
    """Like compute_uncertainty_and_homographies but uses a grid of probe points.

    Calls ``compute_ransac_signals_grid`` (grid_size × grid_size probe points)
    instead of the 4-corner variant.  Everything else — caching, return shape —
    is identical so the result can be dropped into any strategy that consumes
    ``compute_uncertainty_and_homographies``.

    Args:
        strategy:  ActiveLearningStrategy instance.
        model:     RoMa model.
        pair_ids:  (M,) pool indices to process.
        grid_size: grid divisions per axis (default 5 → 25 probe points).

    Returns:
        uncertainties, certainties, all_homographies, valid_ids  (same as above).
    """
    from roma.strategies.strategy_utils import log_strategy_action

    pair_ids = np.asarray(pair_ids, dtype=int)
    if pair_ids.size == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            np.empty(0, dtype=int),
        )

    if not hasattr(strategy, "homography_sets"):
        strategy.homography_sets = {}

    uncertainties = []
    certainties = []
    all_homographies = []
    valid_ids = []

    for pair_id in pair_ids.tolist():
        try:
            matches, confs = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"Grid uncertainty: skipping pair {pair_id} — "
                f"_get_matches_and_confidences failed: {exc}"
            )
            continue

        if matches is None or len(matches) == 0:
            strategy.homography_sets[int(pair_id)] = [None] * 50
            continue

        u, c, homogs = compute_ransac_signals_grid(
            matches, confs,
            image_size=getattr(strategy, "_image_size", 560),
            grid_size=grid_size,
        )

        strategy.homography_sets[int(pair_id)] = homogs
        uncertainties.append(float(u))
        certainties.append(float(c))
        all_homographies.append(homogs)
        valid_ids.append(int(pair_id))

    if not valid_ids:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            np.empty(0, dtype=int),
        )

    log_strategy_action(
        f"Grid uncertainty: computed signals for {len(valid_ids)}/{len(pair_ids)} pairs "
        f"({grid_size}x{grid_size} probe grid)."
    )
    return (
        np.asarray(uncertainties, dtype=np.float32),
        np.asarray(certainties, dtype=np.float32),
        all_homographies,
        np.asarray(valid_ids, dtype=int),
    )


# ---------------------------------------------------------------------------
# Pool-level certainty and EMA adaptive weight
# ---------------------------------------------------------------------------

def compute_pool_certainty(certainties: np.ndarray) -> float:
    """Compute mean certainty across the pool.

    C_t = mean(certainties)

    Used as input to the EMA adaptive weighting mechanism that balances
    uncertainty and diversity signals across AL cycles.

    Args:
        certainties: (N,) per-pair certainty scores in [0, 1].

    Returns:
        C_t: float mean certainty, or 1.0 if the array is empty.
    """
    certainties = np.asarray(certainties, dtype=np.float64)
    if certainties.size == 0:
        return 1.0
    return float(np.mean(certainties))


def compute_ema_tau(tau_prev: float, C_t: float, beta: float = 0.9) -> float:
    """Update the EMA adaptive weighting parameter τ.

    τ_t = β · τ_{t-1} + (1 - β) · C_t

    τ trades off uncertainty vs. diversity in combined AL strategies.
    Initialize with τ_0 = C_0 (the mean certainty at cycle 0).

    Args:
        tau_prev: τ_{t-1}, the EMA weight from the previous cycle.
        C_t:      current pool mean certainty.
        beta:     EMA smoothing coefficient (default 0.9).

    Returns:
        tau_t: updated EMA weight.
    """
    return float(beta * tau_prev + (1.0 - beta) * C_t)
