"""Entropy strategy: select top-k pairs by mean binary entropy of the dense certainty map.

For each pair the RoMa model produces a per-pixel certainty logit at the finest
scale (scale=1).  Applying sigmoid gives p = P(match) ∈ (0,1) for every pixel.
The binary entropy at each pixel is:

    H(x) = -(p(x)·log₂ p(x) + (1-p(x))·log₂(1-p(x)))  ∈ [0, 1]

The pair score is the spatial mean of H(x) over all H×W pixels.  Pairs with the
highest mean entropy are the most uncertain and are selected first.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from roma.strategies.strategy_utils import log_strategy_action, binary_entropy_score


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs with the highest mean binary entropy of the certainty map.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices sorted by descending entropy.
    """
    if model is None:
        raise ValueError("model is required for entropy strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    dataset = strategy._create_remain_dataset(avail)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    device = next(model.parameters()).device
    model.eval()
    scores = []
    with torch.no_grad():
        for sample_idx, batch in zip(avail.astype(int), dataloader):
            batch = {
                k_: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k_, v in batch.items()
            }
            corresps = model(batch)
            finest_scale = min(corresps.keys())          # scale=1
            certainty = corresps[finest_scale]["certainty"]  # (1, 1, H, W) logits
            score = binary_entropy_score(certainty)
            scores.append((int(sample_idx), score))

    if not scores:
        return np.empty(0, dtype=int)

    order = np.argsort([s for _, s in scores])[::-1]    # descending: highest entropy first
    chosen = [scores[pos][0] for pos in order[:k]]
    log_strategy_action(
        f"Entropy: scored {len(scores)} pairs, "
        f"mean_entropy={float(np.mean([s for _, s in scores])):.4f}, "
        f"selected top-{k}."
    )
    return np.asarray(chosen, dtype=int)
