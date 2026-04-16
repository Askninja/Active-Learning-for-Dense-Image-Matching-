"""UWE (Uncertainty-Weighted Embedding) with HS-Cert: h(x) = u(x) · f(x).

Faithful implementation of He et al. TMLR 2024:
  - f(x): raw (NOT L2-normalised) backbone pooled features
  - u(x): hs_cert uncertainty score in [0,1]  (u = 1 − c, c = 1/(1+s))
  - h(x) = u(x) · f(x) — uncertainty magnitude encoded directly
  - k-center greedy on h(x) in Euclidean space
"""

import numpy as np
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.uncertainty_estimation import compute_uncertainty_and_homographies


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by uncertainty-weighted embedding coreset (HS-Cert variant).

    Implements UWE from He et al. TMLR 2024:
        h(x) = u(x) · f(x)
    where u(x) ∈ [0,1] is the HS-Cert uncertainty (1 − certainty) and f(x)
    is the raw (unnormalized) backbone feature.  k-center greedy is then run
    on h(x) directly so that uncertain pairs sit farther from the origin and
    are preferentially selected.

    When labeled pairs exist the greedy search is seeded from their weighted
    embeddings (h_lab) so that selected pairs are diverse relative to what is
    already labeled — at the same scale.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices.
    """
    if model is None:
        raise ValueError("model is required for hs_cert_weighted_coreset strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    # ------------------------------------------------------------------
    # Step 1: Raw features f(x) — NOT L2-normalized
    # ------------------------------------------------------------------
    sample_ids, f_raw = strategy._compute_fine_feature_embeddings(
        model, avail, normalize=False
    )
    if f_raw.shape[0] == 0:
        return np.empty(0, dtype=int)
    f_raw = f_raw.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: Uncertainty scores u(x) ∈ [0,1]
    # HS-Cert: u = 1 − certainty,  certainty = 1/(1 + score)
    # compute_uncertainty_and_homographies returns uncertainties directly
    # ------------------------------------------------------------------
    uncertainties, _certainties, _homographies, valid_ids = compute_uncertainty_and_homographies(
        strategy, model, avail
    )
    if valid_ids.size == 0:
        return np.empty(0, dtype=int)

    score_map = {int(idx): float(u) for idx, u in zip(valid_ids.tolist(), uncertainties.tolist())}

    # Restrict f_raw to pairs that have valid hs_cert scores
    mask = np.array([int(idx) in score_map for idx in sample_ids.tolist()], dtype=bool)
    sample_ids = sample_ids[mask]
    f_raw = f_raw[mask]
    if sample_ids.size == 0:
        return np.empty(0, dtype=int)

    u = np.array(
        [score_map[int(idx)] for idx in sample_ids.tolist()],
        dtype=np.float32,
    )
    # u is already in [0,1] — use directly, no remapping

    # ------------------------------------------------------------------
    # Step 3: Weighted embedding h(x) = u(x) · f(x)
    # ------------------------------------------------------------------
    h = u[:, None] * f_raw  # (N, D)

    # ------------------------------------------------------------------
    # Step 4: Seeding from labeled pairs at consistent scale
    # ------------------------------------------------------------------
    labeled_idx = strategy.train_current_idx
    if labeled_idx.size > 0:
        try:
            lab_ids, f_lab_raw = strategy._compute_fine_feature_embeddings(
                model, labeled_idx, normalize=False
            )
            if f_lab_raw.shape[0] > 0:
                f_lab_raw = f_lab_raw.astype(np.float32)

                lab_uncertainties, _, _, lab_valid_ids = compute_uncertainty_and_homographies(
                    strategy, model, labeled_idx
                )
                if lab_valid_ids.size > 0:
                    lab_score_map = {
                        int(idx): float(u_val)
                        for idx, u_val in zip(lab_valid_ids.tolist(), lab_uncertainties.tolist())
                    }
                    lab_mask = np.array(
                        [int(idx) in lab_score_map for idx in lab_ids.tolist()], dtype=bool
                    )
                    lab_ids = lab_ids[lab_mask]
                    f_lab_raw = f_lab_raw[lab_mask]
                    u_lab = np.array(
                        [lab_score_map[int(idx)] for idx in lab_ids.tolist()],
                        dtype=np.float32,
                    )
                else:
                    # No valid hs_cert for labeled — fall back to zero weights
                    u_lab = np.zeros(len(lab_ids), dtype=np.float32)

                # Weighted labeled embeddings — same scale as unlabeled h
                h_lab = u_lab[:, None] * f_lab_raw  # (N_l, D)

                h_combined = np.concatenate([h_lab, h], axis=0)
                initial_idx = np.arange(len(lab_ids), dtype=int)
                total_needed = min(len(lab_ids) + k, h_combined.shape[0])

                all_selected = k_center_greedy(
                    h_combined.astype(np.float32),
                    total_needed,
                    initial_idx=initial_idx,
                )
                unlabeled_pos = np.array(
                    [idx - len(lab_ids) for idx in all_selected if idx >= len(lab_ids)],
                    dtype=int,
                )[:k]

                norms = np.linalg.norm(h, axis=1)
                correlation = float(np.corrcoef(u, norms)[0, 1]) if u.std() > 1e-8 else float("nan")
                log_strategy_action(
                    f"UWE hs_cert: N_u={len(sample_ids)}, N_l={len(lab_ids)}, "
                    f"selected {len(unlabeled_pos)}. "
                    f"Mean u(x)={u.mean():.3f}, mean ||h||={norms.mean():.3f}"
                )
                log_strategy_action(
                    f"UWE verification: corr(u, ||h||)={correlation:.3f} "
                    f"(should be ~1.0 if f_raw norms are similar across pairs)"
                )
                log_strategy_action(
                    f"UWE verification: ||h|| min={norms.min():.3f} "
                    f"max={norms.max():.3f} std={norms.std():.3f} "
                    f"(should NOT all be equal)"
                )
                return sample_ids[unlabeled_pos].astype(int)

        except Exception as exc:
            log_strategy_action(
                f"UWE hs_cert: labeled seeding failed ({exc}), "
                "falling back to unseeded."
            )

    # ------------------------------------------------------------------
    # Unseeded fallback
    # ------------------------------------------------------------------
    selected_pos = k_center_greedy(h.astype(np.float32), k)

    norms = np.linalg.norm(h, axis=1)
    correlation = float(np.corrcoef(u, norms)[0, 1]) if u.std() > 1e-8 else float("nan")
    log_strategy_action(
        f"UWE hs_cert: N_u={len(sample_ids)}, unseeded, "
        f"selected {len(selected_pos)}. "
        f"Mean u(x)={u.mean():.3f}, mean ||h||={norms.mean():.3f}"
    )
    log_strategy_action(
        f"UWE verification: corr(u, ||h||)={correlation:.3f} "
        f"(should be ~1.0 if f_raw norms are similar across pairs)"
    )
    log_strategy_action(
        f"UWE verification: ||h|| min={norms.min():.3f} "
        f"max={norms.max():.3f} std={norms.std():.3f} "
        f"(should NOT all be equal)"
    )
    return sample_ids[selected_pos].astype(int)
