"""Combined metric diversity strategy.

Uses modality-level metric fusion rather than raw feature concatenation.
Distances in appearance and geometry spaces are independently normalized
and summed before k-center greedy selection.

Motivated by diversification-based active learning literature.
"""

from __future__ import annotations

import numpy as np

from roma.strategies.strategy_combined_diversity import (
    _build_g_descriptors,
    _compute_single_H_descriptor,
)
from roma.strategies.strategy_geometry_diversity import normalize_geometric_descriptors
from roma.strategies.strategy_utils import log_strategy_action


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row independently."""
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def pairwise_l2(X: np.ndarray) -> np.ndarray:
    """Compute a full pairwise Euclidean distance matrix."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    sq_dist = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    np.sqrt(sq_dist, out=sq_dist)
    D = sq_dist.astype(np.float32, copy=False)
    np.fill_diagonal(D, 0.0)
    return D


def normalize_distance_matrix(D: np.ndarray) -> np.ndarray:
    """Normalize a pairwise distance matrix to [0, 1] using off-diagonal max."""
    D = np.asarray(D, dtype=np.float32)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")
    if D.shape[0] <= 1:
        return np.zeros_like(D, dtype=np.float32)

    off_diag = ~np.eye(D.shape[0], dtype=bool)
    max_val = float(np.max(D[off_diag])) if np.any(off_diag) else 0.0
    if max_val < 1e-12:
        return np.zeros_like(D, dtype=np.float32)

    D_norm = np.clip(D / max_val, 0.0, 1.0).astype(np.float32, copy=False)
    np.fill_diagonal(D_norm, 0.0)
    return D_norm


def k_center_greedy_from_distance_matrix(
    D: np.ndarray,
    k: int,
    initial_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Run k-center greedy directly on a precomputed distance matrix."""
    D = np.asarray(D, dtype=np.float32)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")

    N = D.shape[0]
    if N == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), N)

    selected: list[int] = []
    if initial_idx is not None:
        initial_idx = np.asarray(initial_idx, dtype=int)
        seen = set()
        for idx in initial_idx.tolist():
            idx = int(idx)
            if 0 <= idx < N and idx not in seen:
                selected.append(idx)
                seen.add(idx)

    selected_mask = np.zeros(N, dtype=bool)
    if selected:
        selected_mask[np.asarray(selected, dtype=int)] = True
        min_dist = np.min(D[:, np.asarray(selected, dtype=int)], axis=1)
    else:
        start_idx = int(np.argmax(D.mean(axis=1)))
        selected = [start_idx]
        selected_mask[start_idx] = True
        min_dist = D[:, start_idx].copy()

    min_dist = np.asarray(min_dist, dtype=np.float32)
    min_dist[selected_mask] = 0.0

    while len(selected) < k:
        next_idx = int(np.argmax(min_dist))
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True
        np.minimum(min_dist, D[:, next_idx], out=min_dist)
        min_dist[selected_mask] = 0.0

    return np.asarray(selected, dtype=int)


def _align_descriptor_rows(
    valid_ids: np.ndarray,
    G_raw: np.ndarray,
    emb_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter geometric descriptors to the ids that produced embeddings."""
    if np.array_equal(emb_ids, valid_ids):
        return valid_ids, G_raw

    emb_set = set(np.asarray(emb_ids, dtype=int).tolist())
    mask = np.array([int(pid) in emb_set for pid in valid_ids.tolist()], dtype=bool)
    return valid_ids[mask], G_raw[mask]


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs using normalized metric fusion across appearance and geometry."""
    if model is None:
        raise ValueError("model is required for combined_metric_diversity strategy")

    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    image_size = getattr(strategy, "_image_size", 560)
    alpha = float(getattr(strategy, "combined_metric_alpha", 1.0))
    beta = float(getattr(strategy, "combined_metric_beta", 1.0))

    valid_unlabeled_ids, G_raw_unlabeled = _build_g_descriptors(
        avail, strategy, model, image_size=image_size
    )
    if valid_unlabeled_ids.size == 0:
        return np.empty(0, dtype=int)

    emb_ids, f_emb = strategy._compute_fine_feature_embeddings(model, valid_unlabeled_ids)
    if f_emb.shape[0] == 0:
        return np.empty(0, dtype=int)

    valid_unlabeled_ids, G_raw_unlabeled = _align_descriptor_rows(
        valid_unlabeled_ids, G_raw_unlabeled, emb_ids
    )
    N_u = valid_unlabeled_ids.size
    if N_u == 0:
        return np.empty(0, dtype=int)
    if N_u == 1:
        return valid_unlabeled_ids[:1].astype(int)

    f_emb = _l2_normalize_rows(f_emb)

    labeled_idx = strategy.train_current_idx
    f_lab = None
    G_raw_labeled = None
    N_l = 0

    if labeled_idx.size > 0:
        try:
            valid_lab_ids, G_raw_lab_candidate = _build_g_descriptors(
                labeled_idx, strategy, model, image_size=image_size
            )
            if valid_lab_ids.size > 0:
                lab_emb_ids, f_lab_candidate = strategy._compute_fine_feature_embeddings(
                    model, valid_lab_ids
                )
                if f_lab_candidate.shape[0] > 0:
                    valid_lab_ids, G_raw_lab_candidate = _align_descriptor_rows(
                        valid_lab_ids, G_raw_lab_candidate, lab_emb_ids
                    )
                    if valid_lab_ids.size > 0:
                        f_lab = _l2_normalize_rows(f_lab_candidate)
                        G_raw_labeled = G_raw_lab_candidate
                        N_l = valid_lab_ids.size
        except Exception as exc:
            log_strategy_action(
                "Metric diversity: labeled descriptor failed "
                f"({exc}); using unseeded k-center."
            )

    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_unlabeled], axis=0)
        G_norm_all = normalize_geometric_descriptors(G_raw_all).astype(np.float32)
        G_norm_labeled = G_norm_all[:N_l]
        G_norm_unlabeled = G_norm_all[N_l:]
        f_all = np.concatenate([f_lab, f_emb], axis=0).astype(np.float32)
    else:
        G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled).astype(np.float32)
        G_norm_labeled = None
        f_all = f_emb.astype(np.float32)

    if G_norm_labeled is not None and N_l > 0:
        g_all = np.concatenate([G_norm_labeled, G_norm_unlabeled], axis=0).astype(np.float32)
    else:
        g_all = G_norm_unlabeled.astype(np.float32)

    if f_all.shape[0] != g_all.shape[0]:
        raise ValueError(
            f"Appearance and geometry counts must match, got {f_all.shape[0]} and {g_all.shape[0]}"
        )

    D_f = normalize_distance_matrix(pairwise_l2(f_all))
    D_g = normalize_distance_matrix(pairwise_l2(g_all))
    D_total = alpha * D_f + beta * D_g
    D_total = np.asarray(D_total, dtype=np.float32)
    np.fill_diagonal(D_total, 0.0)

    if N_l > 0:
        initial_idx = np.arange(N_l, dtype=int)
        total_needed = min(N_l + k, N_l + N_u)
        all_selected = k_center_greedy_from_distance_matrix(
            D_total, total_needed, initial_idx=initial_idx
        )
        unlabeled_pos = np.array(
            [idx - N_l for idx in all_selected if idx >= N_l], dtype=int
        )[:k]
        if unlabeled_pos.size < k:
            covered = set(unlabeled_pos.tolist())
            extras = [i for i in range(N_u) if i not in covered]
            unlabeled_pos = np.concatenate(
                [unlabeled_pos, np.asarray(extras[: k - unlabeled_pos.size], dtype=int)]
            )
    else:
        log_strategy_action("Metric diversity: no labeled pairs; using unseeded k-center.")
        unlabeled_pos = k_center_greedy_from_distance_matrix(D_total, k)

    log_strategy_action(
        f"Metric diversity: N_u={N_u}, N_l={N_l}, f_dim={f_emb.shape[1]}, "
        f"g_dim=8, alpha={alpha}, beta={beta}, selected={unlabeled_pos.size}"
    )
    return valid_unlabeled_ids[unlabeled_pos].astype(int)


__all__ = [
    "_build_g_descriptors",
    "_compute_single_H_descriptor",
    "k_center_greedy_from_distance_matrix",
    "normalize_distance_matrix",
    "pairwise_l2",
    "run",
]
