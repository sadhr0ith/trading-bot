import os
import random
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except Exception:  # noqa: BLE001
    tf = None


def set_global_seeds(seed: Optional[int] = None) -> int:
    """
    Set seeds for reproducibility across random, numpy and tensorflow (if available).
    Returns the seed used.
    """
    resolved_seed = seed if seed is not None else int(os.getenv("TRADING_BOT_SEED", 42))
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    if tf is not None:
        try:
            tf.random.set_seed(resolved_seed)
        except Exception:  # noqa: BLE001
            pass
    return resolved_seed
