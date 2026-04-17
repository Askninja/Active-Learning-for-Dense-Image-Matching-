"""HS-Cert-3-weighted geometric diversity: hs_cert_3-scaled single-homography descriptors + k-center greedy."""

import numpy as np
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores
from roma.strategies.hs_cert_delta4_geomdiv import (
    _compute_single_homography,
    homography_to_geom_descriptor,
)
from roma.strategies.strategy_geometry_diversity import normalize_geometric_descriptors


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by hs_cert_3-weighted geometric diversity.

    _hs_cert_scores provides certainty scores (low = uncertain) via
    model.match / model.sample (one forward pass per pair).  The geometric
    descriptor is built from a single benchmark-style homography fitted via
    _get_matches_and_confidences (a second forward pass per pair — unavoidable
    without refactoring _hs_cert_scores).  Raw uncertainty weights = 1 - certainty
    ∈ [0, 1] scale the descriptors, biasing k-center greedy toward pairs that are
    both geometrically novel and uncertain to the model.  When labeled pairs exist
    the greedy search is seeded from their unweighted descriptors.

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

    # _hs_cert_scores uses model.match + model.sample (forward pass 1 per pair)
    hs_cert = _hs_cert_scores(strategy, model, avail)
    uncertainties_all = 1.0 - hs_cert   # raw weights in [0, 1]

    image_size = int(getattr(strategy, "_image_size", 560))

    # Per-pair descriptor loop uses _get_matches_and_confidences (forward pass 2 per pair)
    G_raw_list = []
    valid_positions = []   # indices into avail / uncertainties_all
    valid_ids = []
    for i, pair_id in enumerate(avail.tolist()):
        try:
            matches, confs = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"HS-Cert-3-weighted geometric diversity: skipping pair {pair_id} — "
                f"match extraction failed: {exc}"
            )
            continue
        H = _compute_single_homography(matches, confs, image_size=image_size)
        G_raw_list.append(homography_to_geom_descriptor(H, image_size=image_size))
        valid_positions.append(i)
        valid_ids.append(int(pair_id))

    if not valid_ids:
        return np.empty(0, dtype=int)

    valid_ids = np.asarray(valid_ids, dtype=int)
    G_raw_unlabeled = np.asarray(G_raw_list, dtype=np.float64)
    # Align uncertainty weights with the pairs that produced valid descriptors
    uncertainties = uncertainties_all[np.asarray(valid_positions, dtype=int)].astype(np.float32)
    N_u = valid_ids.size
    k = min(k, N_u)

    G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled)

    log_strategy_action(
        f"HS-Cert-3-weighted geometric diversity: weighting {N_u} descriptors by hs_cert_3."
    )

    G_weighted = (G_norm_unlabeled * uncertainties[:, None]).astype(np.float32)
    unlabeled_pos = k_center_greedy(G_weighted, k)

    log_strategy_action(
        f"HS-Cert-3-weighted geometric diversity: {N_u} unlabeled, selected {unlabeled_pos.size} samples."
    )
    return valid_ids[unlabeled_pos].astype(int)
