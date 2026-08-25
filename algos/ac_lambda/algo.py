"""AC(lambda) -- the same online actor-critic, with no memory. The control.

This is not a different algorithm. It is :class:`~algos.rtrrl.algo.RTRRL` with
the recurrent cell replaced by a feedforward one and the meta-RL inputs
switched off: same TD(lambda), same Dutch trace, same true-online critic
update, same entropy bonus, same everything. What is gone is the state that
carries information between timesteps.

That makes it the control experiment the rest of the repo leans on. Every
environment here is partially observable *by construction* -- no velocity is
ever in the observation -- so a policy that is a function of the current
observation alone is provably unable to reach the optimum, whatever its
learning rate. If a recurrent run beats this, the gap is what the memory was
worth; if it does not, no amount of talking about liquid time constants
matters.

Run it on ``cartpole-vel --obs-mode full`` to see the other half of the
argument: with the velocities put back in, the task is Markov, and this
agent is not handicapped at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtrrl"))

from algo import RTRRL  # noqa: E402  (algos/rtrrl/algo.py)


class ACLambda(RTRRL):
    """RTRRL with ``cell="mlp"``. Kept as its own name because it is a
    different *claim*, even though it is barely any different code."""

    def __init__(self, obs_dim: int, action_space, meta_inputs: bool = False, **kw):
        kw.pop("cell", None)
        kw.pop("estimator", None)
        super().__init__(obs_dim, action_space, cell="mlp", estimator="rflo",
                         meta_inputs=meta_inputs, **kw)
