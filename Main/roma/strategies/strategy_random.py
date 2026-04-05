"""Random strategy: uniform random sampling from the unlabeled pool."""

import numpy as np


def run(strategy, k: int, model=None) -> np.ndarray:
    """Select k pairs uniformly at random from the available pool.

    Args:
        strategy: ActiveLearningStrategy instance.
        k:        number of samples to select.
        model:    unused.

    Returns:
        (k,) selected indices.
    """
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)
    return strategy.rng.choice(avail, size=k, replace=False).astype(int)
