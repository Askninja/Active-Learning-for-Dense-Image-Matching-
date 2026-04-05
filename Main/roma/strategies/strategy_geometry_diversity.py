"""Geometry diversity strategy: select pairs most novel relative to the labeled set."""

import numpy as np
import torch
from roma.strategies.strategy_utils import geometry_novelty_scores, k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs whose flow/geometry descriptors are most novel vs the labeled set.

    Computes flow magnitude+direction histograms for both labeled and unlabeled pairs.
    Ranks unlabeled pairs by minimum distance to any labeled descriptor (novelty).
    Falls back to k-center greedy initialization if no labeled descriptors exist.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices.
    """
    if model is None:
        raise ValueError("model is required for geometry_diversity strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    labeled_ids, labeled_desc = strategy._compute_geometry_descriptors(model, strategy.train_current_idx)
    unlabeled_ids, unlabeled_desc = strategy._compute_geometry_descriptors(model, avail)
    if unlabeled_desc.shape[0] == 0:
        return np.empty(0, dtype=int)

    if labeled_desc.shape[0] == 0:
        log_strategy_action("Geometry diversity: no labeled descriptors; using k-center greedy.")
        selected_pos = k_center_greedy(unlabeled_desc.numpy(), k)
        return unlabeled_ids[selected_pos].astype(int)

    scores = geometry_novelty_scores(
        unlabeled_desc,
        labeled_desc,
        chunk_size=strategy.geometry_chunk_size,
    )
    order = torch.argsort(scores, descending=True)
    chosen = unlabeled_ids[order[:k].cpu().numpy()].astype(int)
    log_strategy_action(
        f"Geometry diversity: {labeled_ids.size} labeled, {unlabeled_ids.size} unlabeled, "
        f"dim={unlabeled_desc.shape[1]}, selected {chosen.size} samples."
    )
    return chosen
