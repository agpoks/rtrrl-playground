"""Import an algorithm from ``algos/<name>/algo.py`` by name.

The layout convention in this family of playgrounds is that each subject sits
in its own folder with a flat ``model.py``/``algo.py`` inside, and the runnable
example next to it imports it directly. That is lovely for reading one
algorithm and unhelpful the moment a tutorial wants two of them, so this is the
one place that knows how to reach across.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALGOS = {"rtrrl": "RTRRL", "ac_lambda": "ACLambda", "a2c_bptt": "A2CBPTT"}


def load_algo(name: str):
    """Return the agent class defined in ``algos/<name>/algo.py``."""
    if name not in ALGOS:
        raise KeyError(f"unknown algorithm '{name}'. Known: {', '.join(ALGOS)}")
    path = ROOT / "algos" / name / "algo.py"
    spec = importlib.util.spec_from_file_location(f"_algo_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, ALGOS[name])
