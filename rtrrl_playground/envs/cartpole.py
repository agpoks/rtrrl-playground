"""CartPole with the velocities removed from the observation.

The dynamics are the textbook ones (Barto, Sutton & Anderson 1983, and the
exact constants Gymnasium's ``CartPole-v1`` uses, so numbers here are
comparable to anything you have run there). The only change is the
observation: ``obs_mode="vel"`` hides ``x_dot`` and ``theta_dot``, which is
the ``CartPole-vel`` POMDP the RTRRL paper reports.

Why that particular mutilation is interesting: cart position and pole angle
are still fully observed, so the agent knows *where* everything is and only
lacks *where it is going*. The missing quantity is exactly the derivative of
something it can see, which is the easiest possible thing for a recurrent
state to reconstruct -- and still completely impossible for a feedforward
policy. It is the cheapest honest test that the recurrence is doing work.

``obs_mode="full"`` restores the MDP, and is worth running once: it is the
control experiment that shows the memoryless baseline is not simply broken.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.spaces import Discrete, Env

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5  # actually half the pole's length
POLEMASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates
THETA_LIMIT = 12 * 2 * np.pi / 360
X_LIMIT = 2.4


class CartPoleVel(Env):
    id = "cartpole-vel"

    def __init__(self, obs_mode: str = "vel", max_steps: int = 500, seed: int | None = None):
        if obs_mode not in ("vel", "pos", "full"):
            raise ValueError("obs_mode must be 'vel' (hide velocities), 'pos' (hide positions) or 'full'")
        self.obs_mode = obs_mode
        self.obs_dim = 4 if obs_mode == "full" else 2
        self.action_space = Discrete(2)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(4)
        self._t = 0

    def _obs(self) -> np.ndarray:
        x, x_dot, th, th_dot = self._state
        if self.obs_mode == "full":
            return np.array([x, x_dot, th, th_dot])
        if self.obs_mode == "vel":  # hide the velocities
            return np.array([x, th])
        return np.array([x_dot, th_dot])  # hide the positions

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.uniform(-0.05, 0.05, size=4)
        self._t = 0
        return self._obs()

    def step(self, action):
        x, x_dot, th, th_dot = self._state
        force = FORCE_MAG if int(action) == 1 else -FORCE_MAG
        cos_th, sin_th = np.cos(th), np.sin(th)
        temp = (force + POLEMASS_LENGTH * th_dot ** 2 * sin_th) / TOTAL_MASS
        th_acc = (GRAVITY * sin_th - cos_th * temp) / (
            LENGTH * (4.0 / 3.0 - MASS_POLE * cos_th ** 2 / TOTAL_MASS)
        )
        x_acc = temp - POLEMASS_LENGTH * th_acc * cos_th / TOTAL_MASS
        # Euler, as in the reference implementation -- not because it is the
        # better integrator but because every published CartPole number uses it.
        self._state = np.array([
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            th + TAU * th_dot,
            th_dot + TAU * th_acc,
        ])
        self._t += 1
        x, _, th, _ = self._state
        terminated = bool(abs(x) > X_LIMIT or abs(th) > THETA_LIMIT)
        truncated = bool(self._t >= self.max_steps and not terminated)
        return self._obs(), 1.0, terminated, truncated, {}
