"""Coreset appearance strategy: k-center greedy on L2-normalized RoMa appearance features."""

import numpy as np
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action


def coreset_appearance(features: np.ndarray, b: int) -> np.ndarray:
    """Select b pairs by k-center greedy on L2-normalized RoMa appearance features.

    Purely diversity: no task signal. Covers the appearance space of the unlabeled
    pool uniformly, agnostic to how well RoMa matches each pair.

    Args:
        features: (N, D) appearance feature vectors (e.g. mean-pooled backbone).
        b:        number of samples to select.

    Returns:
        (b,) indices selected by k-center greedy.
    """
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return k_center_greedy(features / norms, b)


def run(strategy, k: int, model) -> np.ndarray:
    """Run the coreset_appearance strategy.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model used to compute appearance embeddings.

    Returns:
        (k,) selected pool indices.
    """
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)
    sample_ids, embeddings = strategy._compute_fine_feature_embeddings(model, avail)
    if embeddings.shape[0] == 0:
        return np.empty(0, dtype=int)
    selected_pos = coreset_appearance(embeddings, k)
    log_strategy_action(f"Coreset appearance: selected {len(selected_pos)} from {len(sample_ids)} candidates.")
    return sample_ids[selected_pos].astype(int)
