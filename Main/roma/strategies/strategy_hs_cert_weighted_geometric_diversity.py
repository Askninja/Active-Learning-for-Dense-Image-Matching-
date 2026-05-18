"""HS-Cert-3-weighted geometric diversity: hs_cert_3-scaled averaged-displacement descriptors + k-center greedy."""

import numpy as np
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.strategy_utils import normalize_weights
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores, hs_cert_from_homography_sets
from roma.strategies.strategy_geometry_diversity import (
    compute_geometric_diversity_from_homographies,
    normalize_geometric_descriptors,
)


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by hs_cert_3-weighted geometric diversity.

    _hs_cert_scores runs K=50 RANSAC per pair and populates
    strategy.homography_sets as a side effect — no second forward pass needed.
    The geometric descriptor is built by averaging displacement vectors across
    those K valid homographies.  Raw uncertainty weights = 1 - certainty ∈ [0, 1]
    scale the normalized descriptors, biasing k-center greedy toward pairs that
    are both geometrically novel and uncertain to the model.

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

    # _hs_cert_scores: certainty scores + populates strategy.homography_sets (K=50 RANSAC)
    hs_cert = _hs_cert_scores(strategy, model, avail)
    uncertainties_all = 1.0 - hs_cert   # raw weights in [0, 1]

    image_size = int(getattr(strategy, "_image_size", 560))

    # Build averaged-displacement descriptors using stored homography sets (no second forward pass)
    G_raw_list = []
    valid_positions = []   # indices into avail / uncertainties_all
    valid_ids = []
    for i, pair_id in enumerate(avail.tolist()):
        d = compute_geometric_diversity_from_homographies(
            strategy.homography_sets.get(int(pair_id), []),
            image_size=image_size,
        )
        if np.any(d != 0):
            G_raw_list.append(d)
            valid_positions.append(i)
            valid_ids.append(int(pair_id))
        else:
            log_strategy_action(
                f"HS-Cert-3-weighted geometric diversity: skipping pair {pair_id} — "
                f"zero/degenerate descriptor."
            )

    if not valid_ids:
        return np.empty(0, dtype=int)

    valid_ids = np.asarray(valid_ids, dtype=int)
    G_raw_unlabeled = np.asarray(G_raw_list, dtype=np.float64)
    uncertainties = uncertainties_all[np.asarray(valid_positions, dtype=int)]
    N_u = valid_ids.size
    k = min(k, N_u)

    # Build labeled descriptors for combined-pool normalization and k-center seeding
    labeled_idx = strategy.train_current_idx
    G_raw_labeled = None
    N_l = 0
    if labeled_idx.size > 0:
        try:
            # Only run the expensive _hs_cert_scores forward pass for labeled pairs
            # whose homography sets were not already populated (e.g. from a prior cycle).
            missing = np.asarray(
                [pid for pid in labeled_idx.tolist() if int(pid) not in strategy.homography_sets],
                dtype=int,
            )
            if missing.size > 0:
                _hs_cert_scores(strategy, model, missing)
            labeled_descs = []
            for pid in labeled_idx.tolist():
                d = compute_geometric_diversity_from_homographies(
                    strategy.homography_sets.get(int(pid), []),
                    image_size=image_size,
                )
                labeled_descs.append(d)
            G_raw_labeled = np.asarray(labeled_descs, dtype=np.float64)
            N_l = G_raw_labeled.shape[0]
        except Exception as exc:
            log_strategy_action(
                f"HS-Cert-3-weighted geometric diversity: labeled descriptor build failed ({exc}); "
                "normalizing on unlabeled pool only."
            )
            G_raw_labeled = None
            N_l = 0

    # Normalize over combined pool for calibrated scale
    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_unlabeled], axis=0)
        G_norm_all = normalize_geometric_descriptors(G_raw_all)
        G_norm_labeled = G_norm_all[:N_l]
        G_norm_unlabeled = G_norm_all[N_l:]
    else:
        G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled)
        G_norm_labeled = None

    log_strategy_action(
        f"HS-Cert-3-weighted geometric diversity: weighting {N_u} descriptors by hs_cert_3."
    )

    # Weight unlabeled descriptors by uncertainty [0,1]: certain pairs collapse toward origin
    G_weighted = (G_norm_unlabeled * normalize_weights(uncertainties)[:, None]).astype(np.float32)

    # k-center seeded from labeled descriptors — also weighted for a consistent embedding space
    if G_norm_labeled is not None and N_l > 0:
        # Compute hs_cert for labeled pairs from cached homography_sets (no new forward pass)
        labeled_ids = np.asarray(labeled_idx, dtype=int)
        lab_hs_cert = hs_cert_from_homography_sets(strategy, labeled_ids)
        lab_uncertainties = 1.0 - lab_hs_cert
        G_lab_weighted = (G_norm_labeled * normalize_weights(lab_uncertainties)[:, None]).astype(np.float32)

        log_strategy_action(
            f"HS-Cert-3-weighted geometric diversity: seeding k-center from {N_l} weighted labeled pairs."
        )
        G_combined = np.concatenate([G_lab_weighted, G_weighted], axis=0)
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
