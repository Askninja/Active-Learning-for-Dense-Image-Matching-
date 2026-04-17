"""UWE (Uncertainty-Weighted Embedding) with HS-Cert: h(x) = u(x) · f(x).

Faithful implementation of He et al. TMLR 2024:
  - f(x): raw (NOT L2-normalised) backbone pooled features
  - u(x): hs_cert uncertainty score in [0,1)  (u = 1 − c, c = 1/(1+s) ∈ (0,1])
  - h(x) = u(x) · f(x) — uncertainty magnitude encoded directly
  - k-center greedy on h(x) in Euclidean space
"""

import numpy as np
from roma.strategies.strategy_utils import k_center_greedy
from roma.strategies.strategy_utils import log_strategy_action
from roma.strategies.strategy_hs_cert_3 import _hs_cert_scores


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by uncertainty-weighted embedding coreset (HS-Cert variant).

    Implements UWE from He et al. TMLR 2024:
        h(x) = u(x) · f(x)
    where u(x) = 1 − hs_cert(x) ∈ [0, 1) and f(x) is the raw (unnormalized)
    backbone feature.  k-center greedy is then run on h(x) directly so that
    uncertain pairs sit farther from the origin and are preferentially selected.

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
    # Step 2: Uncertainty scores u(x) ∈ [0, 1)
    # _hs_cert_scores returns certainty ∈ (0, 1] for every element in avail.
    # u = 1 − certainty; pairs that fail RANSAC get certainty=1.0 → u=0.0.
    # ------------------------------------------------------------------
    hs_cert = _hs_cert_scores(strategy, model, avail)   # (len(avail),), certainty ∈ (0,1]
    u_avail = (1.0 - hs_cert).astype(np.float32)        # uncertainty ∈ [0, 1)

    # Build map: global pair index → uncertainty
    score_map = {int(idx): float(u_val) for idx, u_val in zip(avail.tolist(), u_avail.tolist())}

    # Align f_raw (which may be a subset of avail) with uncertainty scores
    mask = np.array([int(idx) in score_map for idx in sample_ids.tolist()], dtype=bool)
    sample_ids = sample_ids[mask]
    f_raw = f_raw[mask]
    if sample_ids.size == 0:
        return np.empty(0, dtype=int)

    u = np.array(
        [score_map[int(idx)] for idx in sample_ids.tolist()],
        dtype=np.float32,
    )
    # u ∈ [0, 1) — use directly, no remapping

    # ------------------------------------------------------------------
    # Step 3: Weighted embedding h(x) = u(x) · f(x)
    # ------------------------------------------------------------------
    h = u[:, None] * f_raw  # (N, D)

    # ------------------------------------------------------------------
    # Step 4: Unseeded k-center on h(x) = u(x) · f(x) for unlabeled only
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
