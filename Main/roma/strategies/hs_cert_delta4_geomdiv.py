"""HS-Cert delta-4 geometry diversity strategy.

Stage 1 ranks the unlabeled pool by hs_cert_3 uncertainty (inline
homography-spread via model.match / model.sample). Stage 2 keeps the top
4k uncertain pairs and selects k of them using geometry diversity via
k-center greedy on 8D corner-displacement descriptors.
"""

from __future__ import annotations

import cv2
import numpy as np

from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores
from roma.strategies.strategy_geometry_diversity import normalize_geometric_descriptors


DELTA = 4


def _compute_single_homography(
    matches: np.ndarray,
    confidences: np.ndarray,
    image_size: int = 560,
    top_k: int = 5000,
) -> np.ndarray:
    """Fit one benchmark-style homography from the highest-confidence matches."""
    if matches is None or confidences is None or len(matches) < 4:
        return np.eye(3, dtype=np.float64)

    matches = np.asarray(matches, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if len(matches) > top_k:
        top_idx = np.argpartition(confidences, -top_k)[-top_k:]
        matches = matches[top_idx]

    if len(matches) < 4:
        return np.eye(3, dtype=np.float64)

    try:
        H_mat, _ = cv2.findHomography(
            matches[:, :2],
            matches[:, 2:],
            method=cv2.RANSAC,
            confidence=0.99999,
            ransacReprojThreshold=3.0 * float(image_size) / 480.0,
        )
    except Exception:
        H_mat = None

    if H_mat is None:
        return np.eye(3, dtype=np.float64)
    return np.asarray(H_mat, dtype=np.float64)


def homography_to_geom_descriptor(H: np.ndarray, image_size: int = 560) -> np.ndarray:
    """Convert a single homography into an 8D corner-displacement descriptor."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return np.zeros(8, dtype=np.float64)
    s = float(image_size)
    m = max(s - 1.0, 1.0)
    corners = np.array([
        [0.0, 0.0],
        [m, 0.0],
        [0.0, m],
        [m, m],
    ], dtype=np.float64)

    descriptor = np.zeros(8, dtype=np.float64)
    for j, corner in enumerate(corners):
        pt_h = np.array([corner[0], corner[1], 1.0], dtype=np.float64)
        proj_h = H @ pt_h
        denom = proj_h[2]
        if not np.isfinite(denom) or abs(denom) < 1e-10:
            return np.zeros(8, dtype=np.float64)
        proj = proj_h[:2] / denom
        delta = (proj - corner) / m
        delta = np.where(np.isfinite(delta), delta, 0.0)
        descriptor[2 * j] = delta[0]
        descriptor[2 * j + 1] = delta[1]
    return descriptor



def _compute_descriptors(strategy, model, pair_ids: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build geometry descriptors by fitting one homography per requested pair id."""
    pair_ids = np.asarray(pair_ids, dtype=int)
    descriptors = []
    valid_ids = []
    for pair_id in pair_ids.tolist():
        try:
            matches, confidences = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"hs_cert_delta4_geomdiv: skipping pair {pair_id} because match extraction failed: {exc}"
            )
            continue
        H = _compute_single_homography(matches, confidences, image_size=image_size)
        descriptors.append(homography_to_geom_descriptor(H, image_size=image_size))
        valid_ids.append(int(pair_id))
    if not valid_ids:
        return np.zeros((0, 8), dtype=np.float64), np.empty(0, dtype=int)
    return np.asarray(descriptors, dtype=np.float64), np.asarray(valid_ids, dtype=int)


def run(strategy, k: int, model) -> np.ndarray:
    """Select k samples by hs_cert_3 uncertainty followed by geometry diversity."""
    if model is None:
        raise ValueError("model is required for hs_cert_delta4_geomdiv strategy")

    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    # hs_cert_3: certainty scores, low = uncertain; also populates homography_sets
    hs_cert = _hs_cert_scores(strategy, model, avail)
    # sort ascending on certainty = descending on uncertainty
    order = np.argsort(hs_cert)
    cand_size = min(avail.size, DELTA * k)
    candidate_ids = avail[order[:cand_size]].astype(int)
    log_strategy_action(
        f"hs_cert_delta4_geomdiv: stage1 top {cand_size} uncertain from {len(avail)}"
    )

    if candidate_ids.size <= k:
        return candidate_ids

    image_size = int(getattr(strategy, "_image_size", 560))
    G_raw_candidates, candidate_valid_ids = _compute_descriptors(
        strategy, model, candidate_ids, image_size=image_size
    )
    if candidate_valid_ids.size == 0:
        return candidate_ids[:k]

    candidate_ids = candidate_valid_ids
    k = min(k, candidate_ids.size)
    if candidate_ids.size <= k:
        return candidate_ids

    G_raw_labeled = None
    N_l = 0
    labeled_idx = np.asarray(getattr(strategy, "train_current_idx", np.empty(0, dtype=int)), dtype=int)
    if labeled_idx.size > 0:
        try:
            G_raw_labeled, valid_labeled_ids = _compute_descriptors(
                strategy, model, labeled_idx, image_size=image_size
            )
            N_l = int(valid_labeled_ids.size)
            if N_l == 0:
                G_raw_labeled = None
        except Exception as exc:
            log_strategy_action(
                f"hs_cert_delta4_geomdiv: labeled geometry seeding failed ({exc}); using unseeded k-center."
            )
            G_raw_labeled = None
            N_l = 0

    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_candidates], axis=0)
        G_norm_all = normalize_geometric_descriptors(G_raw_all)
        G_norm_labeled = G_norm_all[:N_l]
        G_norm_candidates = G_norm_all[N_l:]
        G_combined = np.concatenate([G_norm_labeled, G_norm_candidates], axis=0).astype(np.float32)
        initial_idx = np.arange(N_l, dtype=int)
        total_needed = min(N_l + k, G_combined.shape[0])
        all_selected = k_center_greedy(G_combined, total_needed, initial_idx=initial_idx)
        selected_local = np.asarray([idx - N_l for idx in all_selected if idx >= N_l], dtype=int)[:k]
    else:
        G_norm_candidates = normalize_geometric_descriptors(G_raw_candidates)
        if not np.any(np.abs(G_norm_candidates) > 0):
            return candidate_ids[:k]
        selected_local = k_center_greedy(G_norm_candidates, k)

    if selected_local.size < k:
        chosen = set(selected_local.tolist())
        extras = [i for i in range(candidate_ids.size) if i not in chosen]
        selected_local = np.concatenate(
            [selected_local, np.asarray(extras[:k - selected_local.size], dtype=int)]
        )

    selected = candidate_ids[selected_local[:k]].astype(int)
    log_strategy_action(
        f"hs_cert_delta4_geomdiv: stage2 selected {selected.size} via geometry k-center"
    )
    return selected
