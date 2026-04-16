"""HS-Cert-3-weighted geometric diversity: hs_cert_3-scaled geometric descriptors + k-center greedy."""

import numpy as np
from roma.strategies.strategy_utils import normalize_weights
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores
from roma.strategies.strategy_geometry_diversity import (
    compute_geometric_diversity,
    normalize_geometric_descriptors,
)


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by hs_cert_3-weighted geometric diversity.

    _hs_cert_scores provides certainty scores (low = uncertain) via
    model.match / model.sample and also populates strategy.homography_sets so
    geometric descriptors can be built without a second forward pass.
    Uncertainty weights = 1 - certainty are used to scale the descriptors,
    biasing k-center greedy toward pairs that are both geometrically novel
    and uncertain to the model.  When labeled pairs exist the greedy search
    is seeded from their (unweighted) descriptors.

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

    # hs_cert_3 certainty scores + populates strategy.homography_sets
    hs_cert = _hs_cert_scores(strategy, model, avail)
    uncertainties = 1.0 - hs_cert          # high = uncertain
    valid_ids = avail
    N_u = valid_ids.size
    k = min(k, N_u)

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
            _hs_cert_scores(strategy, model, labeled_idx)  # populates homography_sets for labeled
            N_l = labeled_idx.size
            G_raw_labeled = np.zeros((N_l, 8), dtype=np.float64)
            for i, pid in enumerate(labeled_idx.tolist()):
                h_list = strategy.homography_sets.get(int(pid), [])
                G_raw_labeled[i] = compute_geometric_diversity(h_list, image_size=560)
        except Exception as exc:
            log_strategy_action(
                f"HS-Cert-3-weighted geometric diversity: could not compute labeled descriptors ({exc}); "
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

    # Scale unlabeled descriptors by hs_cert_3 uncertainty weights
    G_weighted = (G_norm_unlabeled * normalize_weights(uncertainties)[:, None]).astype(np.float32)
    log_strategy_action(
        f"HS-Cert-3-weighted geometric diversity: weighting {N_u} descriptors by hs_cert_3."
    )

    # k-center greedy with labeled seeding (unweighted labeled descriptors as seeds)
    if G_norm_labeled is not None and N_l > 0:
        log_strategy_action(
            f"HS-Cert-weighted geometric diversity: seeding k-center from {N_l} labeled pairs."
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
        f"HS-Cert-3-weighted geometric diversity: {N_l} labeled, {N_u} unlabeled, "
        f"selected {unlabeled_pos.size} samples."
    )
    return valid_ids[unlabeled_pos].astype(int)
