"""The four environments, and one factory that builds any of them by id.

Two are the classic partially-observable RL yardsticks (they are what the
RTRRL paper reports on, so they are how you check the implementation is
right), and two are the RC-car tasks this repo actually exists for.

=====================  ====================================================
id                     what it tests
=====================  ====================================================
``memory-chain``       memory alone: nothing else in the observation matters
``cartpole-vel``       control under a hidden velocity, the paper's POMDP
``lanekeep``           drive: stay on a track, as fast as you can hold it
``overtake``           drive plus a slower car that never yields
=====================  ====================================================

All four are partially observable *by construction* -- no velocity is ever
in the observation. That is not a difficulty knob, it is the whole point:
a feedforward policy is provably unable to solve any of them, so the
recurrent state has to carry something, and RTRRL's job is to learn what.
"""

from __future__ import annotations

from rtrrl_playground.envs.cartpole import CartPoleVel
from rtrrl_playground.envs.lanekeep import LaneKeep
from rtrrl_playground.envs.memory_chain import MemoryChain
from rtrrl_playground.envs.overtake import Overtake
from rtrrl_playground.envs.track import Track, TRACKS

ENV_IDS = ["memory-chain", "cartpole-vel", "lanekeep", "overtake"]

_BUILDERS = {
    "memory-chain": MemoryChain,
    "cartpole-vel": CartPoleVel,
    "lanekeep": LaneKeep,
    "overtake": Overtake,
}


def make_env(env_id: str, **kwargs):
    """Build an environment by id. ``kwargs`` go straight to its constructor."""
    if env_id not in _BUILDERS:
        raise KeyError(f"unknown env '{env_id}'. Known ids: {', '.join(ENV_IDS)}")
    return _BUILDERS[env_id](**kwargs)


__all__ = ["make_env", "ENV_IDS", "MemoryChain", "CartPoleVel", "LaneKeep",
           "Overtake", "Track", "TRACKS"]
