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

    When labeled pairs are available the k-center greedy search is seeded from
    their L2-normalized embeddings so that selected unlabeled pairs are diverse
    relative to what is already labeled.

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

    # L2-normalize unlabeled embeddings (same as coreset_appearance does internally)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    norm_embeddings = (embeddings / norms).astype(np.float32)

    labeled_idx = strategy.train_current_idx
    if labeled_idx.size > 0:
        try:
            lab_ids, lab_embeddings = strategy._compute_fine_feature_embeddings(model, labeled_idx)
            if lab_embeddings.shape[0] > 0:
                lab_norms = np.linalg.norm(lab_embeddings, axis=1, keepdims=True)
                lab_norms = np.where(lab_norms < 1e-8, 1.0, lab_norms)
                norm_lab_embeddings = (lab_embeddings / lab_norms).astype(np.float32)
                log_strategy_action(
                    f"Coreset appearance: seeding k-center from {len(lab_ids)} labeled pairs."
                )
                combined = np.concatenate([norm_lab_embeddings, norm_embeddings], axis=0)
                initial_idx = np.arange(len(lab_ids), dtype=int)
                total_needed = min(len(lab_ids) + k, len(lab_ids) + len(sample_ids))
                all_selected = k_center_greedy(
                    combined.astype(np.float32), total_needed, initial_idx=initial_idx
                )
                unlabeled_pos = np.array(
                    [idx - len(lab_ids) for idx in all_selected if idx >= len(lab_ids)],
                    dtype=int,
                )[:k]
                log_strategy_action(
                    f"Coreset appearance: selected {len(unlabeled_pos)} from {len(sample_ids)} candidates."
                )
                return sample_ids[unlabeled_pos].astype(int)
        except Exception as exc:
            log_strategy_action(
                f"Coreset appearance: labeled embedding failed ({exc}); falling back to unseeded."
            )

    selected_pos = k_center_greedy(norm_embeddings, k)
    log_strategy_action(f"Coreset appearance: selected {len(selected_pos)} from {len(sample_ids)} candidates.")
    return sample_ids[selected_pos].astype(int)
