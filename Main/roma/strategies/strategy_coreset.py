"""Coreset strategy: KMeans + k-center greedy on RoMa appearance embeddings."""

import numpy as np
from sklearn.cluster import KMeans
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action


def coreset_select(sample_ids: np.ndarray, embeddings: np.ndarray, k: int, cycle: int) -> np.ndarray:
    """Select k samples by KMeans-accelerated k-center greedy on appearance embeddings.

    Clusters embeddings into 4k centroids with KMeans, then runs k-center greedy on
    the centroids to pick k diverse cluster representatives. Falls back to direct
    k-center greedy if not enough samples to cluster.

    Args:
        sample_ids: (N,) pool indices corresponding to embeddings rows.
        embeddings: (N, D) appearance feature vectors.
        k:          number of samples to select.
        cycle:      current AL cycle (used to seed KMeans for reproducibility).

    Returns:
        (k,) selected pool indices.
    """
    sample_ids = np.asarray(sample_ids, dtype=int)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] == 0:
        return np.empty(0, dtype=int)
    if embeddings.shape[0] <= k:
        return sample_ids.astype(int)

    cluster_count = min(embeddings.shape[0], max(k + 1, min(4 * k, embeddings.shape[0])))
    kmeans = KMeans(n_clusters=cluster_count, random_state=int(cycle) + 784, n_init=10)
    kmeans.fit(embeddings)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    chosen_centroid_idx = k_center_greedy(centroids, k)
    chosen = []
    used_sample_pos = set()
    for centroid_idx in chosen_centroid_idx.tolist():
        centroid = centroids[centroid_idx]
        distances = np.linalg.norm(embeddings - centroid[None, :], axis=1)
        for pos in np.argsort(distances):
            pos = int(pos)
            if pos not in used_sample_pos:
                used_sample_pos.add(pos)
                chosen.append(int(sample_ids[pos]))
                break

    if len(chosen) < k:
        chosen_pos = np.asarray(sorted(used_sample_pos), dtype=int) if used_sample_pos else np.empty(0, dtype=int)
        extra_pos = k_center_greedy(embeddings, k, initial_idx=chosen_pos)
        for pos in extra_pos.tolist():
            pos = int(pos)
            if pos not in used_sample_pos:
                used_sample_pos.add(pos)
                chosen.append(int(sample_ids[pos]))
            if len(chosen) >= k:
                break

    log_strategy_action(
        f"Coreset: embedded {len(sample_ids)} samples, "
        f"ran KMeans with {cluster_count} centroids, picked {len(chosen)} samples."
    )
    return np.asarray(chosen[:k], dtype=int)


def run(strategy, k: int, model) -> np.ndarray:
    """Run the coreset strategy.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model used to compute appearance embeddings.

    Returns:
        (k,) selected pool indices.
    """
    if model is None:
        raise ValueError("model is required for coreset strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)
    sample_ids, embeddings = strategy._compute_fine_feature_embeddings(model, avail)
    return coreset_select(sample_ids, embeddings, k, strategy.cycle)
