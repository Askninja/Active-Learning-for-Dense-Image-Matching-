"""UWE (Uncertainty-Weighted Embedding) with entropy: h(x) = u(x) · f(x).

Faithful implementation of He et al. TMLR 2024:
  - f(x): raw (NOT L2-normalised) backbone pooled features
  - u(x): binary certainty-map entropy, min-max normalized to [0,1] across the pool
  - h(x) = u(x) · f(x) — uncertainty magnitude encoded directly in the vector norm
  - k-center greedy on h(x) in Euclidean space

Features and entropy scores are extracted in a single forward pass per pair.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from roma.strategies.strategy_utils import (
    k_center_greedy,
    log_strategy_action,
    binary_entropy_score,
    normalize_weights,
)


def _extract_features_and_entropy(strategy, model, idx: np.ndarray):
    """Single forward pass per pair: returns backbone embedding and binary entropy score.

    Args:
        strategy: ActiveLearningStrategy instance.
        model:    RoMa model.
        idx:      (N,) int array of pair indices to process.

    Returns:
        ids:       (N,) int array of pair ids in processing order.
        embeddings: (N, D) float32 backbone features (raw, not L2-normalized).
        entropies:  (N,) float32 binary entropy scores in [0, 1].
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.empty(0, dtype=int), np.empty((0, 1), dtype=np.float32), np.empty(0, dtype=np.float32)

    dataset = strategy._create_remain_dataset(idx)
    # batch_size=1 avoids the _pair_embedding [0]-indexing bug with larger batches
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    device = next(model.parameters()).device
    model.eval()

    ids, embeddings, entropies = [], [], []
    with torch.no_grad():
        for pair_id, batch in zip(idx.tolist(), dataloader):
            batch = {
                k_: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k_, v in batch.items()
            }
            corresps = model(batch)

            # --- backbone embedding (raw, not normalized) ---
            feature_pyramid = model.extract_backbone_features(batch, batched=True, upsample=False)
            sorted_scales = sorted(feature_pyramid.keys())
            gaps_a, gaps_b = [], []
            for scale in sorted_scales:
                features = feature_pyramid[scale]           # (2, C, H, W) for a single pair
                feat_a, feat_b = features.chunk(2, dim=0)  # each (1, C, H, W)
                gaps_a.append(feat_a.flatten(2).mean(dim=2))   # (1, C) — raw, not normalized
                gaps_b.append(feat_b.flatten(2).mean(dim=2))
            emb_a = torch.cat(gaps_a, dim=1)  # (1, num_scales * C)
            emb_b = torch.cat(gaps_b, dim=1)
            emb = torch.cat((emb_a, emb_b), dim=1)[0].float().cpu().numpy()  # (D,)

            # --- binary entropy from finest-scale certainty ---
            finest_scale = min(corresps.keys())
            certainty = corresps[finest_scale]["certainty"]
            entropy = binary_entropy_score(certainty)

            ids.append(int(pair_id))
            embeddings.append(emb)
            entropies.append(entropy)

    return (
        np.asarray(ids, dtype=int),
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(entropies, dtype=np.float32),
    )


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs by uncertainty-weighted embedding coreset (entropy variant).

    Implements UWE from He et al. TMLR 2024:
        h(x) = u(x) · f(x)
    where u(x) ∈ [0,1] is the min-max normalized binary certainty-map entropy
    and f(x) is the raw (unnormalized) backbone feature.  k-center greedy is
    run on h(x) so that uncertain pairs sit farther from the origin and are
    preferentially selected.

    When labeled pairs exist the greedy search is seeded from their weighted
    embeddings so that selected pairs are diverse relative to what is already labeled.

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

    # ------------------------------------------------------------------
    # Step 1+2: Single pass — backbone features f(x) and entropy scores
    # ------------------------------------------------------------------
    sample_ids, f_raw, u_raw = _extract_features_and_entropy(strategy, model, avail)
    if sample_ids.size == 0:
        return np.empty(0, dtype=int)

    # min-max normalize entropy across the unlabeled pool → [0, 1]
    u = normalize_weights(u_raw)

    # ------------------------------------------------------------------
    # Step 3: Weighted embedding h(x) = u(x) · f(x)
    # ------------------------------------------------------------------
    h = u[:, None] * f_raw  # (N, D)

    # ------------------------------------------------------------------
    # Step 4: k-center seeded from labeled pairs
    # ------------------------------------------------------------------
    labeled_idx = strategy.train_current_idx
    if labeled_idx.size > 0:
        try:
            lab_ids, f_lab_raw, u_lab_raw = _extract_features_and_entropy(strategy, model, labeled_idx)
            if lab_ids.size > 0:
                u_lab = normalize_weights(u_lab_raw)
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
                    [i - len(lab_ids) for i in all_selected if i >= len(lab_ids)],
                    dtype=int,
                )[:k]

                log_strategy_action(
                    f"UWE entropy: N_u={len(sample_ids)}, N_l={len(lab_ids)}, "
                    f"selected {len(unlabeled_pos)}, mean_u={u.mean():.3f}"
                )
                return sample_ids[unlabeled_pos].astype(int)

        except Exception as exc:
            log_strategy_action(
                f"UWE entropy: labeled seeding failed ({exc}), falling back to unseeded."
            )

    # ------------------------------------------------------------------
    # Unseeded fallback
    # ------------------------------------------------------------------
    selected_pos = k_center_greedy(h.astype(np.float32), k)
    log_strategy_action(
        f"UWE entropy: N_u={len(sample_ids)}, unseeded, "
        f"selected {len(selected_pos)}, mean_u={u.mean():.3f}"
    )
    return sample_ids[selected_pos].astype(int)
