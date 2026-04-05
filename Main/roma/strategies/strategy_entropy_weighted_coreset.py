"""Entropy-weighted coreset: entropy-scaled appearance embeddings + KMeans k-center."""

import numpy as np
from roma.strategies.strategy_utils import normalize_weights
from roma.strategies.strategy_coreset import coreset_select
from roma.strategies.strategy_utils import log_strategy_action


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by entropy-weighted coreset on appearance embeddings.

    Scales each pair's embedding by its entropy weight before running KMeans + k-center
    greedy. Uncertain pairs appear farther apart in the weighted space, biasing
    selection toward pairs that are both appearance-diverse and hard for the model.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices.
    """
    if model is None:
        raise ValueError("model is required for entropy_weighted_coreset strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    sample_ids, embeddings = strategy._compute_fine_feature_embeddings(model, avail)
    score_ids, score_values = strategy._score_avail(model, avail, score_name="entropy")
    if embeddings.shape[0] == 0 or score_values.shape[0] == 0:
        return np.empty(0, dtype=int)

    score_map = {int(idx): float(s) for idx, s in zip(score_ids.tolist(), score_values.tolist())}
    aligned = np.asarray([score_map[int(idx)] for idx in sample_ids.tolist()], dtype=np.float32)
    weighted_embeddings = embeddings * normalize_weights(aligned)[:, None]
    log_strategy_action(f"Entropy-weighted coreset: weighting {len(sample_ids)} embeddings by entropy.")
    return coreset_select(sample_ids, weighted_embeddings, k, strategy.cycle)
