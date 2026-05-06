import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_list(value):
    """Wrap value in list if needed."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
