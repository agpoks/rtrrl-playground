"""One seeding entry point, so a run is reproducible from a single integer."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy's legacy global RNG and torch (if installed).

    Returns the :class:`numpy.random.Generator` that the agent and the
    environment should actually draw from -- the globals are seeded only so
    that a stray ``np.random.randn`` in user code is reproducible too.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency (only the BPTT baseline needs it)
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)
