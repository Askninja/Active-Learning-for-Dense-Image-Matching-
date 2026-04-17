"""Entropy-weighted geometric diversity: entropy-scaled single-homography descriptors + k-center greedy."""

import numpy as np
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.hs_cert_delta4_geomdiv import (
    _compute_single_homography,
    homography_to_geom_descriptor,
)
from roma.strategies.strategy_geometry_diversity import normalize_geometric_descriptors


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by entropy-weighted geometric diversity.

    Entropy scores each pair by the mean GM classifier entropy (same as the
    standalone entropy strategy).  A single benchmark-style homography is fitted
    per pair to build the 8D corner-displacement descriptor.  Both signals are
    computed together in a per-pair loop so entropy scores and descriptors stay
    aligned without a separate intersection step.  The normalized descriptors are
    scaled by raw entropy weights ∈ [0, 1] before k-center greedy.  When labeled
    pairs exist the greedy search is seeded from their unweighted descriptors.

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

    # Entropy scores for the full available pool (batch call)
    score_ids, score_values = strategy._score_avail(model, avail, score_name="entropy")
    if score_values.shape[0] == 0:
        return np.empty(0, dtype=int)
    score_map = {int(idx): float(s) for idx, s in zip(score_ids.tolist(), score_values.tolist())}

    image_size = int(getattr(strategy, "_image_size", 560))

    # Per-pair loop: compute descriptor and align entropy score together.
    # Pairs missing from score_map or failing match extraction are skipped.
    G_raw_list = []
    entropy_aligned = []
    valid_ids = []
    for pair_id in avail.tolist():
        if int(pair_id) not in score_map:
            continue
        try:
            matches, confs = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"Entropy-weighted geometric diversity: skipping pair {pair_id} — "
                f"match extraction failed: {exc}"
            )
            continue
        H = _compute_single_homography(matches, confs, image_size=image_size)
        G_raw_list.append(homography_to_geom_descriptor(H, image_size=image_size))
        entropy_aligned.append(score_map[int(pair_id)])
        valid_ids.append(int(pair_id))

    if not valid_ids:
        return np.empty(0, dtype=int)

    valid_ids = np.asarray(valid_ids, dtype=int)
    G_raw_unlabeled = np.asarray(G_raw_list, dtype=np.float64)
    entropy_aligned = np.asarray(entropy_aligned, dtype=np.float32)
    N_u = valid_ids.size
    k = min(k, N_u)

    G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled)

    log_strategy_action(
        f"Entropy-weighted geometric diversity: weighting {N_u} descriptors by entropy."
    )

    G_weighted = (G_norm_unlabeled * entropy_aligned[:, None]).astype(np.float32)
    unlabeled_pos = k_center_greedy(G_weighted, k)

    log_strategy_action(
        f"Entropy-weighted geometric diversity: {N_u} unlabeled, selected {unlabeled_pos.size} samples."
    )
    return valid_ids[unlabeled_pos].astype(int)
