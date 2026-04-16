"""Combined diversity strategy: VGG19 appearance + geometric transformation.

Selects image pairs that are simultaneously diverse in appearance space
and geometric transformation space, using k-center greedy on a combined
embedding matrix.

Two complementary signals:
    f(x): VGG19 fine feature embedding (128-dim, L2-normalized)
    g(x): corner displacement descriptor (8-dim) from a single high-confidence
          RANSAC homography fit — same as the benchmark evaluation script.

Both descriptors are independently normalized, then concatenated into one
combined embedding. K-center greedy on that embedding selects pairs that
cover both appearance space and geometric transformation space.

No uncertainty signal — pure diversity sampling.
"""

import numpy as np
import cv2
from roma.strategies.strategy_geometry_diversity import normalize_geometric_descriptors
from roma.strategies.strategy_utils import k_center_greedy, log_strategy_action
from roma.strategies.uncertainty_estimation import compute_uncertainty_and_homographies


# ---------------------------------------------------------------------------
# Single-homography geometric descriptor (benchmark style)
# ---------------------------------------------------------------------------

def _compute_single_H_descriptor(
    matches: np.ndarray,
    confs: np.ndarray,
    image_size: int = 560,
    top_k: int = 2000,
) -> np.ndarray:
    """Compute an 8-dim corner displacement descriptor from one RANSAC homography.

    Runs a single high-confidence RANSAC fit on all top-confidence matches,
    mirroring the benchmark evaluation script, rather than averaging K matrices.

    Args:
        matches:    (N, 4) pixel-space correspondences [xA, yA, xB, yB].
        confs:      (N,) confidence scores in [0, 1].
        image_size: side length of the square image in pixels.
        top_k:      number of top-confidence matches to use.

    Returns:
        (8,) descriptor.  Returns np.zeros(8) when RANSAC fails.
    """
    if matches is None or len(matches) < 8:
        return np.zeros(8, dtype=np.float64)

    matches = np.asarray(matches, dtype=np.float64)
    confs = np.asarray(confs, dtype=np.float64)

    if len(matches) > top_k:
        idx = np.argpartition(confs, -top_k)[-top_k:]
        matches = matches[idx]

    kpts_A = matches[:, :2]
    kpts_B = matches[:, 2:]

    H, _ = cv2.findHomography(
        kpts_A, kpts_B,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99999,
    )

    if H is None:
        return np.zeros(8, dtype=np.float64)

    s = float(image_size)
    corners = np.array([
        [0.,    0.,   1.],
        [s - 1, 0.,   1.],
        [0.,    s-1,  1.],
        [s - 1, s-1,  1.],
    ], dtype=np.float64)

    deltas = []
    for c in corners:
        p = H @ c
        denom = p[2] if abs(p[2]) >= 1e-10 else 1e-10
        p_cart = p[:2] / denom
        delta = np.clip((p_cart - c[:2]) / s, -3.0, 3.0)
        deltas.append(delta)

    g_raw = np.concatenate(deltas)
    return np.where(np.isfinite(g_raw), g_raw, 0.0)


# ---------------------------------------------------------------------------
# Per-batch descriptor builder
# ---------------------------------------------------------------------------

def _build_g_descriptors(
    pair_ids: np.ndarray,
    strategy,
    model,
    image_size: int = 560,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-H geometric descriptors for a set of pairs.

    Args:
        pair_ids:   (M,) pool indices to process.
        strategy:   ActiveLearningStrategy instance.
        model:      RoMa model.
        image_size: image side length in pixels.

    Returns:
        valid_ids:  (V,) pair indices that returned matches.
        G_raw:      (V, 8) raw geometric descriptors.
    """
    valid_ids = []
    G_raw = []

    for pair_id in pair_ids.tolist():
        try:
            matches, confs = strategy._get_matches_and_confidences(model, int(pair_id))
        except Exception as exc:
            log_strategy_action(
                f"Combined diversity: skipping pair {pair_id} — matches failed: {exc}"
            )
            continue

        if matches is None or len(matches) == 0:
            continue

        g = _compute_single_H_descriptor(
            matches, confs, image_size=image_size
        )
        valid_ids.append(int(pair_id))
        G_raw.append(g)

    if not valid_ids:
        return np.empty(0, dtype=int), np.zeros((0, 8), dtype=np.float64)

    return np.asarray(valid_ids, dtype=int), np.stack(G_raw, axis=0)


# ---------------------------------------------------------------------------
# Strategy entry point
# ---------------------------------------------------------------------------

def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs maximally diverse in appearance and geometric space.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        Number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices.
    """
    if model is None:
        raise ValueError("model is required for combined_diversity strategy")

    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    image_size = getattr(strategy, "_image_size", 560)

    # ------------------------------------------------------------------
    # Step 1 — geometric descriptors for unlabeled pool (single RANSAC H)
    # ------------------------------------------------------------------
    valid_unlabeled_ids, G_raw_unlabeled = _build_g_descriptors(
        avail, strategy, model, image_size=image_size
    )
    if valid_unlabeled_ids.size == 0:
        return np.empty(0, dtype=int)

    N_u = valid_unlabeled_ids.size

    # ------------------------------------------------------------------
    # Step 2 — appearance embeddings for the same valid pairs
    # ------------------------------------------------------------------
    emb_ids, f_emb = strategy._compute_fine_feature_embeddings(
        model, valid_unlabeled_ids
    )
    if f_emb.shape[0] == 0:
        return np.empty(0, dtype=int)

    # Align G_raw to emb_ids (may differ if embedding fails for some pairs)
    if not np.array_equal(emb_ids, valid_unlabeled_ids):
        emb_set = set(emb_ids.tolist())
        mask = np.array([pid in emb_set for pid in valid_unlabeled_ids.tolist()], dtype=bool)
        valid_unlabeled_ids = valid_unlabeled_ids[mask]
        G_raw_unlabeled = G_raw_unlabeled[mask]
        N_u = valid_unlabeled_ids.size

    if N_u == 1:
        return valid_unlabeled_ids[:1].astype(int)

    f_emb = f_emb.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3 — labeled pairs (optional seeding)
    # ------------------------------------------------------------------
    labeled_idx = strategy.train_current_idx
    G_raw_labeled = None
    f_lab = None
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
                    # Align
                    if not np.array_equal(lab_emb_ids, valid_lab_ids):
                        emb_set = set(lab_emb_ids.tolist())
                        mask = np.array(
                            [pid in emb_set for pid in valid_lab_ids.tolist()], dtype=bool
                        )
                        valid_lab_ids = valid_lab_ids[mask]
                        G_raw_lab_candidate = G_raw_lab_candidate[mask]

                    G_raw_labeled = G_raw_lab_candidate
                    f_lab = f_lab_candidate.astype(np.float32)
                    N_l = G_raw_labeled.shape[0]
        except Exception as exc:
            log_strategy_action(
                f"Combined diversity: labeled descriptor failed ({exc}); using unseeded k-center."
            )

    # ------------------------------------------------------------------
    # Step 4 — joint normalization of geometric descriptors
    # ------------------------------------------------------------------
    if G_raw_labeled is not None and N_l > 0:
        G_raw_all = np.concatenate([G_raw_labeled, G_raw_unlabeled], axis=0)
        G_norm_all = normalize_geometric_descriptors(G_raw_all)
        G_norm_lab = G_norm_all[:N_l].astype(np.float32)
        G_norm_unlab = G_norm_all[N_l:].astype(np.float32)
    else:
        G_norm_unlab = normalize_geometric_descriptors(G_raw_unlabeled).astype(np.float32)
        G_norm_lab = None

    # ------------------------------------------------------------------
    # Step 5 — concatenate appearance + geometric into one embedding
    # ------------------------------------------------------------------
    combined_unlab = np.concatenate([f_emb, G_norm_unlab], axis=1)   # (N_u, D+8)

    # ------------------------------------------------------------------
    # Step 6 — k-center greedy (seeded from labeled if available)
    # ------------------------------------------------------------------
    if G_norm_lab is not None and f_lab is not None and N_l > 0:
        combined_lab = np.concatenate([f_lab, G_norm_lab], axis=1)   # (N_l, D+8)
        combined_all = np.concatenate([combined_lab, combined_unlab], axis=0)
        initial_idx = np.arange(N_l, dtype=int)
        total_needed = min(N_l + k, N_l + N_u)
        all_selected = k_center_greedy(
            combined_all.astype(np.float32), total_needed, initial_idx=initial_idx
        )
        unlabeled_pos = np.array(
            [idx - N_l for idx in all_selected if idx >= N_l], dtype=int
        )[:k]
        if unlabeled_pos.size < k:
            covered = set(unlabeled_pos.tolist())
            extras = [i for i in range(N_u) if i not in covered]
            unlabeled_pos = np.concatenate([
                unlabeled_pos,
                np.asarray(extras[:k - unlabeled_pos.size], dtype=int),
            ])
    else:
        log_strategy_action(
            "Combined diversity: no labeled pairs; using unseeded k-center greedy."
        )
        unlabeled_pos = k_center_greedy(combined_unlab.astype(np.float32), k)

    log_strategy_action(
        f"Combined diversity: N_u={N_u}, N_l={N_l}, "
        f"f_dim={f_emb.shape[1]}, g_dim=8, selected {unlabeled_pos.size} pairs."
    )
    return valid_unlabeled_ids[unlabeled_pos].astype(int)
