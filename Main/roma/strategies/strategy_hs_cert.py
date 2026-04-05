"""HS-Cert strategy: select top-k pairs by homography-spread uncertainty."""

import numpy as np
import torch
from roma.strategies.strategy_utils import log_strategy_action


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs with the highest homography-spread (HS) uncertainty.

    HS uncertainty measures how much RANSAC homographies vary across random subsets
    of the predicted matches. High variance means the model produces inconsistent
    correspondences, indicating a hard or ambiguous scene.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices sorted by descending uncertainty.
    """
    if model is None:
        raise ValueError("model is required for hs_cert strategy")
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
                selected_scale = max(corresps.keys())
                log_strategy_action(f"HS-Cert: using finest scale={selected_scale}.")
            flow = corresps[selected_scale]["flow"]
            certainty = corresps[selected_scale]["certainty"]
            H, W = flow.shape[-2:]
            M = strategy._get_matches(flow, certainty, H, W, num_matches=5000)
            scores.append((sample_idx, float(strategy._compute_hs_uncertainty(M, H, W, K=10, P=50))))

    if not scores:
        return np.empty(0, dtype=int)
    order = np.argsort([s for _, s in scores])[::-1]
    chosen = [scores[pos][0] for pos in order[:k]]
    log_strategy_action(f"HS-Cert: scored {len(scores)} samples.")
    return np.asarray(chosen, dtype=int)
