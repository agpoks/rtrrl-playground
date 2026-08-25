"""MemoryChain: the smallest task that a memoryless policy cannot solve.

At ``t = 0`` the agent is shown a bit. For the next ``n - 1`` steps it is
shown nothing but a step counter. At ``t = n - 1`` it must reproduce the
bit it saw at the start; +1 if it does, -1 if it does not, 0 everywhere in
between.

From ``bsuite`` (Osband et al., "Behaviour Suite for Reinforcement
Learning", ICLR 2020), and reported in the RTRRL paper as MemoryChain-N.
It is the first environment in the tutorial for three reasons:

* it isolates *memory* from control -- there is no dynamics to get wrong;
* the reward is delayed by exactly ``n`` steps, so it is also the cleanest
  possible demonstration of what an eligibility trace buys you; and
* the optimal return is exactly 1, so "did it learn" is not a judgement call.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.spaces import Discrete, Env


class MemoryChain(Env):
    id = "memory-chain"

    def __init__(self, length: int = 10, seed: int | None = None):
        self.length = int(length)
        self.obs_dim = 3  # (context bit, time fraction, query flag)
        self.action_space = Discrete(2)
        self.max_steps = self.length
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._bit = 0

    def _obs(self) -> np.ndarray:
        first = self._t == 0
        last = self._t == self.length - 1
        return np.array([
            float(self._bit) if first else 0.0,
            self._t / self.length,
            1.0 if last else 0.0,
        ])

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._bit = int(self._rng.integers(2))
        return self._obs()

    def step(self, action):
        last = self._t == self.length - 1
        reward = 0.0
        if last:
            reward = 1.0 if int(action) == self._bit else -1.0
        self._t += 1
        terminated = last
        obs = self._obs() if not terminated else np.zeros(self.obs_dim)
        return obs, reward, terminated, False, {"bit": self._bit}
