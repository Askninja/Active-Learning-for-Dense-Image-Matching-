"""Entropy strategy: select top-k pairs by prediction entropy of the coarse GM classifier."""

import numpy as np
import torch
from roma.strategies.strategy_utils import mean_entropy_score
from roma.strategies.strategy_utils import log_strategy_action


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs with the highest mean entropy of the coarse GM class distribution.

    High entropy indicates the model is uncertain about which coarse match class each
    pixel belongs to, signalling pairs where the current model struggles most.

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

    from torch.utils.data import DataLoader
    dataset = strategy._entropy_dataset(avail)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    device = next(model.parameters()).device
    model.eval()
    scores = []
    selected_scale = None
    with torch.no_grad():
        for sample_idx, batch in zip(avail.astype(int), dataloader):
            batch = {k_: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k_, v in batch.items()}
            corresps = model(batch)
            if selected_scale is None:
                selected_scale = strategy._select_gm_cls_scale(corresps)
                log_strategy_action(f"Entropy: using coarse gm_cls scale={selected_scale}.")
            gm_cls = corresps[selected_scale]["gm_cls"].detach().float().cpu().numpy()
            scores.append((sample_idx, float(mean_entropy_score(gm_cls, temperature=strategy.temperature))))

    if not scores:
        return np.empty(0, dtype=int)
    order = np.argsort([s for _, s in scores])[::-1]
    chosen = [scores[pos][0] for pos in order[:k]]
    log_strategy_action(f"Entropy: scored {len(scores)} samples with temperature={strategy.temperature}.")
    return np.asarray(chosen, dtype=int)
