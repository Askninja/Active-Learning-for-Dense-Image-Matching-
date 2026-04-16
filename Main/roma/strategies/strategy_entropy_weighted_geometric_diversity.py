"""Entropy-weighted geometric diversity: entropy-scaled geometric descriptors + k-center greedy."""

import numpy as np
from roma.strategies.strategy_utils import normalize_weights
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.uncertainty_estimation import compute_uncertainty_and_homographies
from roma.strategies.strategy_geometry_diversity import (
    compute_geometric_diversity,
    normalize_geometric_descriptors,
)


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by entropy-weighted geometric diversity.

    Entropy scores each pair by the mean GM classifier entropy (same as the
    standalone entropy strategy).  Geometric descriptors are built from K=50
    RANSAC homographies (same as the standalone geometry_diversity strategy).
    The normalized 8-dim descriptors are then scaled by entropy weights before
    k-center greedy, biasing selection toward pairs that are both geometrically
    novel and uncertain to the model.  When labeled pairs exist the greedy
    search is seeded from their descriptors.

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

    # Entropy scores — identical to standalone entropy strategy
    score_ids, score_values = strategy._score_avail(model, avail, score_name="entropy")
    if score_values.shape[0] == 0:
        return np.empty(0, dtype=int)

    # Homographies — identical to standalone geometry_diversity strategy
    _u, _c, _h, valid_unlabeled_ids = compute_uncertainty_and_homographies(strategy, model, avail)
    if valid_unlabeled_ids.size == 0:
        return np.empty(0, dtype=int)

    # Intersect: keep only pairs that have both a valid entropy score and homographies
    score_map = {int(idx): float(s) for idx, s in zip(score_ids.tolist(), score_values.tolist())}
    valid_ids = np.array(
        [idx for idx in valid_unlabeled_ids.tolist() if int(idx) in score_map], dtype=int
    )
    if valid_ids.size == 0:
        return np.empty(0, dtype=int)
    k = min(k, valid_ids.size)
    N_u = valid_ids.size

    # Build geometric descriptors from cached homographies
    G_raw_unlabeled = np.zeros((N_u, 8), dtype=np.float64)
    for i, pid in enumerate(valid_ids.tolist()):
        h_list = strategy.homography_sets.get(int(pid), [])
        G_raw_unlabeled[i] = compute_geometric_diversity(h_list, image_size=560)

    # Labeled pairs for normalization + seeding
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
                f"Entropy-weighted geometric diversity: could not compute labeled descriptors ({exc}); "
                "normalizing on unlabeled pool only."
            )
            G_raw_labeled = None
            N_l = 0

    # Robust normalization across the combined pool
    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_unlabeled], axis=0)
        G_norm_all = normalize_geometric_descriptors(G_raw_all)
        G_norm_unlabeled = G_norm_all[N_l:]
        G_norm_labeled = G_norm_all[:N_l]
    else:
        G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled)
        G_norm_labeled = None

    # Scale unlabeled descriptors by entropy weights
    aligned = np.asarray([score_map[int(idx)] for idx in valid_ids.tolist()], dtype=np.float32)
    G_weighted = (G_norm_unlabeled * normalize_weights(aligned)[:, None]).astype(np.float32)
    log_strategy_action(
        f"Entropy-weighted geometric diversity: weighting {N_u} descriptors by entropy."
    )

    # k-center greedy with labeled seeding (unweighted labeled descriptors as seeds)
    if G_norm_labeled is not None and N_l > 0:
        log_strategy_action(
            f"Entropy-weighted geometric diversity: seeding k-center from {N_l} labeled pairs."
        )
        G_combined = np.concatenate([G_norm_labeled.astype(np.float32), G_weighted], axis=0)
        initial_idx = np.arange(N_l, dtype=int)
        total_needed = min(N_l + k, N_l + N_u)
        all_selected = k_center_greedy(G_combined, total_needed, initial_idx=initial_idx)
        unlabeled_pos = np.array(
            [idx - N_l for idx in all_selected if idx >= N_l], dtype=int
        )[:k]
    else:
        unlabeled_pos = k_center_greedy(G_weighted, k)

    log_strategy_action(
        f"Entropy-weighted geometric diversity: {N_l} labeled, {N_u} unlabeled, "
        f"selected {unlabeled_pos.size} samples."
    )
    return valid_ids[unlabeled_pos].astype(int)
