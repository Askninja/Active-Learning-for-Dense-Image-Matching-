"""HS-Cert strategy: select top-k pairs by homography-spread uncertainty."""

import numpy as np
from roma.strategies.uncertainty_estimation import compute_uncertainty_and_homographies
from roma.strategies.strategy_utils import log_strategy_action


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs with the highest homography-spread (HS) uncertainty.

    HS uncertainty measures how much RANSAC homographies vary across random subsets
    of the predicted matches. High variance means the model produces inconsistent
    correspondences, indicating a hard or ambiguous scene.

    The K=50 RANSAC homographies computed here are cached in
    ``strategy.homography_sets`` so that a subsequent geometry_diversity step
    (in a combined strategy) can reuse them without a second forward pass.

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

    uncertainties, _certainties, _homographies, valid_ids = (
        compute_uncertainty_and_homographies(strategy, model, avail)
    )
    if valid_ids.size == 0:
        return np.empty(0, dtype=int)

    # Sort by descending uncertainty and return the top-k pool indices
    order = np.argsort(uncertainties)[::-1]
    chosen = valid_ids[order[:k]].astype(int)
    log_strategy_action(
        f"HS-Cert: scored {valid_ids.size} samples, "
        f"mean_uncertainty={float(uncertainties.mean()):.4f}, selected {chosen.size}."
    )
    return chosen
