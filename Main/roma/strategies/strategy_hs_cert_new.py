"""HS-Cert-New strategy: homography-spread uncertainty via a 5x5 probe grid."""

import numpy as np
from roma.strategies.uncertainty_estimation import compute_uncertainty_and_homographies_grid
from roma.strategies.strategy_utils import log_strategy_action


def run(strategy, k: int, model) -> np.ndarray:
    """Select k pairs with the highest homography-spread uncertainty.

    Identical to hs_cert but probes a 5x5 uniform grid (25 points) across
    image A instead of just the 4 corners.  This captures instability in the
    interior of the image that corner-only variance can miss.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    RoMa model.

    Returns:
        (k,) selected pool indices sorted by descending uncertainty.
    """
    if model is None:
        raise ValueError("model is required for hs_cert_new strategy")
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    uncertainties, _certainties, _homographies, valid_ids = (
        compute_uncertainty_and_homographies_grid(strategy, model, avail, grid_size=5)
    )
    if valid_ids.size == 0:
        return np.empty(0, dtype=int)

    order = np.argsort(uncertainties)[::-1]
    chosen = valid_ids[order[:k]].astype(int)
    log_strategy_action(
        f"HS-Cert-New (5x5 grid): scored {valid_ids.size} samples, "
        f"mean_uncertainty={float(uncertainties.mean()):.4f}, selected {chosen.size}."
    )
    return chosen
