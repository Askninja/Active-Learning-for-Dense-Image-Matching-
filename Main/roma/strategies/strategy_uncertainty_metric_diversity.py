"""Hybrid uncertainty + metric diversity strategy.

Uses hs_cert uncertainty and modality-level metric fusion.
Appearance and geometry distances are independently normalized,
then combined with uncertainty into a max-min greedy selector.
"""

from __future__ import annotations

import numpy as np

from roma.strategies.strategy_combined_diversity import (
    _build_g_descriptors,
    _compute_single_H_descriptor,
)
from roma.strategies.strategy_combined_metric_diversity import (
    _l2_normalize_rows,
    k_center_greedy_from_distance_matrix,
    pairwise_l2,
)
from roma.strategies.strategy_geometry_diversity import normalize_geometric_descriptors
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores


def normalize_distance_matrix(D: np.ndarray) -> np.ndarray:
    """Normalize a distance matrix to [0, 1] using the off-diagonal 95th percentile."""
    D = np.asarray(D, dtype=np.float32)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got shape {D.shape}")
    if D.shape[0] <= 1:
        return np.zeros_like(D, dtype=np.float32)

    off_diag = D[~np.eye(D.shape[0], dtype=bool)]
    if off_diag.size == 0:
        return np.zeros_like(D, dtype=np.float32)

    scale = float(np.percentile(off_diag, 95))
    if scale < 1e-12:
        return np.zeros_like(D, dtype=np.float32)

    D_norm = np.clip(D / scale, 0.0, 1.0).astype(np.float32, copy=False)
    np.fill_diagonal(D_norm, 0.0)
    return D_norm


def _normalize_uncertainty(u: np.ndarray) -> np.ndarray:
    """Map uncertainty scores to [0, 1]."""
    u = np.asarray(u, dtype=np.float32)
    if u.size == 0:
        return u
    u = u - float(np.min(u))
    umax = float(np.max(u))
    if umax > 1e-12:
        u = u / umax
    return np.clip(u, 0.0, 1.0)


def _align_modalities(
    ids_ref: np.ndarray,
    emb_ids: np.ndarray,
    f_emb: np.ndarray,
    G_raw: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align ids, appearance, geometry, and uncertainty to the embedding ids."""
    ids_ref = np.asarray(ids_ref, dtype=int)
    emb_ids = np.asarray(emb_ids, dtype=int)
    f_emb = np.asarray(f_emb, dtype=np.float32)
    G_raw = np.asarray(G_raw, dtype=np.float64)
    u = np.asarray(u, dtype=np.float32)

    if not (ids_ref.shape[0] == G_raw.shape[0] == u.shape[0]):
        raise ValueError(
            f"Alignment inputs must have matching rows, got ids={ids_ref.shape[0]}, "
            f"G={G_raw.shape[0]}, u={u.shape[0]}"
        )

    pos_by_id = {int(pid): i for i, pid in enumerate(ids_ref.tolist())}
    keep_ids = []
    keep_f = []
    keep_g = []
    keep_u = []

    for row_idx, pid in enumerate(emb_ids.tolist()):
        pos = pos_by_id.get(int(pid))
        if pos is None:
            continue
        keep_ids.append(int(pid))
        keep_f.append(f_emb[row_idx])
        keep_g.append(G_raw[pos])
        keep_u.append(float(u[pos]))

    if not keep_ids:
        return (
            np.empty(0, dtype=int),
            np.empty((0, 0), dtype=np.float32),
            np.zeros((0, 8), dtype=np.float64),
            np.empty(0, dtype=np.float32),
        )

    return (
        np.asarray(keep_ids, dtype=int),
        np.asarray(keep_f, dtype=np.float32),
        np.asarray(keep_g, dtype=np.float64),
        np.asarray(keep_u, dtype=np.float32),
    )


def _compute_hs_cert_uncertainty(
    strategy,
    model,
    pair_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hs_cert_3 uncertainty for a set of pairs.

    Returns (valid_ids, uncertainties) where uncertainty = 1 - hs_cert_3 certainty.
    All pair_ids are returned as valid (degenerate pairs receive score 0).
    """
    pair_ids = np.asarray(pair_ids, dtype=int)
    hs_cert = _hs_cert_scores(strategy, model, pair_ids)
    uncertainties = _normalize_uncertainty(1.0 - hs_cert)
    return pair_ids, uncertainties


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs using hs_cert uncertainty and metric-fused diversity."""
    if model is None:
        raise ValueError("model is required for uncertainty_metric_diversity strategy")

    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    image_size = getattr(strategy, "_image_size", 560)
    alpha = float(getattr(strategy, "combined_metric_alpha", 1.0))
    beta = float(getattr(strategy, "combined_metric_beta", 1.0))
    gamma = float(getattr(strategy, "uncertainty_metric_gamma", 1.0))

    valid_unc_ids, u_unlabeled = _compute_hs_cert_uncertainty(strategy, model, avail)
    if valid_unc_ids.size == 0:
        return np.empty(0, dtype=int)

    valid_geo_ids, G_raw_unlabeled = _build_g_descriptors(
        valid_unc_ids, strategy, model, image_size=image_size
    )
    if valid_geo_ids.size == 0:
        return np.empty(0, dtype=int)

    geo_pos = {int(pid): i for i, pid in enumerate(valid_geo_ids.tolist())}
    unc_keep_mask = np.array([int(pid) in geo_pos for pid in valid_unc_ids.tolist()], dtype=bool)
    valid_joint_ids = valid_unc_ids[unc_keep_mask]
    u_joint = u_unlabeled[unc_keep_mask]
    G_raw_joint = np.asarray(
        [G_raw_unlabeled[geo_pos[int(pid)]] for pid in valid_joint_ids.tolist()],
        dtype=np.float64,
    )
    if valid_joint_ids.size == 0:
        return np.empty(0, dtype=int)

    emb_ids, f_emb = strategy._compute_fine_feature_embeddings(model, valid_joint_ids)
    if f_emb.shape[0] == 0:
        return np.empty(0, dtype=int)

    valid_unlabeled_ids, f_emb, G_raw_unlabeled, u_unlabeled = _align_modalities(
        valid_joint_ids, emb_ids, f_emb, G_raw_joint, u_joint
    )
    N_u = valid_unlabeled_ids.size
    if N_u == 0:
        return np.empty(0, dtype=int)
    if N_u == 1:
        return valid_unlabeled_ids[:1].astype(int)

    f_emb = _l2_normalize_rows(f_emb)
    u_unlabeled = _normalize_uncertainty(u_unlabeled)

    labeled_idx = strategy.train_current_idx
    f_lab = None
    G_raw_labeled = None
    u_labeled = np.empty(0, dtype=np.float32)
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
                    zeros_lab = np.zeros(valid_lab_ids.shape[0], dtype=np.float32)
                    valid_lab_ids, f_lab_candidate, G_raw_lab_candidate, u_lab_candidate = (
                        _align_modalities(
                            valid_lab_ids,
                            lab_emb_ids,
                            f_lab_candidate,
                            G_raw_lab_candidate,
                            zeros_lab,
                        )
                    )
                    if valid_lab_ids.size > 0:
                        f_lab = _l2_normalize_rows(f_lab_candidate)
                        G_raw_labeled = G_raw_lab_candidate
                        u_labeled = u_lab_candidate
                        N_l = valid_lab_ids.size
        except Exception as exc:
            log_strategy_action(
                "Uncertainty metric diversity: labeled descriptors failed "
                f"({exc}); using unseeded max-min."
            )

    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_unlabeled], axis=0)
        G_norm_all = normalize_geometric_descriptors(G_raw_all).astype(np.float32)
        G_norm_labeled = G_norm_all[:N_l]
        G_norm_unlabeled = G_norm_all[N_l:]
        f_all = np.concatenate([f_lab, f_emb], axis=0).astype(np.float32)
        g_all = np.concatenate([G_norm_labeled, G_norm_unlabeled], axis=0).astype(np.float32)
        u_all = np.concatenate([u_labeled, u_unlabeled], axis=0).astype(np.float32)
    else:
        G_norm_unlabeled = normalize_geometric_descriptors(G_raw_unlabeled).astype(np.float32)
        f_all = f_emb.astype(np.float32)
        g_all = G_norm_unlabeled.astype(np.float32)
        u_all = u_unlabeled.astype(np.float32)

    if not (f_all.shape[0] == g_all.shape[0] == u_all.shape[0]):
        raise ValueError(
            "Appearance, geometry, and uncertainty rows must match, got "
            f"{f_all.shape[0]}, {g_all.shape[0]}, {u_all.shape[0]}"
        )

    D_f = normalize_distance_matrix(pairwise_l2(f_all))
    D_g = normalize_distance_matrix(pairwise_l2(g_all))
    U_pair = 0.5 * (u_all[:, None] + u_all[None, :])
    D_total = gamma * U_pair + alpha * D_f + beta * D_g
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
            unlabeled_pos = np.concatenate([
                unlabeled_pos,
                np.asarray(extras[: k - unlabeled_pos.size], dtype=int),
            ])
    else:
        log_strategy_action(
            "Uncertainty metric diversity: no labeled pairs; using unseeded max-min."
        )
        unlabeled_pos = k_center_greedy_from_distance_matrix(D_total, k)

    log_strategy_action(
        f"Uncertainty metric diversity: N_u={N_u}, N_l={N_l}, f_dim={f_emb.shape[1]}, "
        f"g_dim=8, alpha={alpha}, beta={beta}, gamma={gamma}, "
        f"u_mean={float(u_unlabeled.mean()):.4f}, u_max={float(u_unlabeled.max()):.4f}, "
        f"selected={unlabeled_pos.size}"
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
