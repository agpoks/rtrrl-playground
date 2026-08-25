"""rtrrl-playground -- Real-Time Recurrent Reinforcement Learning, from scratch.

The package holds everything the algorithms in ``algos/`` and the lessons in
``tutorial/`` share: a two-class space/environment protocol, four small
partially-observable environments (two of them RC-car driving tasks), the
CT-RNN cell with its three gradient modes (none / RFLO / exact RTRL), the
policy and value heads with hand-derived gradients, and the eligibility
traces.

Nothing here calls an autograd engine. Every derivative used by RTRRL is
written out by hand, because the entire point of the algorithm is that the
gradient is computed *forwards*, in the same pass as the activations, and
never by replaying a stored graph backwards. The one place autograd does
appear is ``algos/a2c_bptt``, the baseline RTRRL is meant to be compared
against.
"""

from __future__ import annotations

__version__ = "0.1.0"

from rtrrl_playground.spaces import Box, Discrete
from rtrrl_playground.envs import make_env, ENV_IDS
from rtrrl_playground.utils.seed import set_seed

__all__ = ["Box", "Discrete", "make_env", "ENV_IDS", "set_seed", "__version__"]
