"""PhysicsLiGRU -- a few units that already know the vehicle, and the rest learning.

The observation is nine lidar beams. The agent's hardest job is to reconstruct
the state they leave out -- speed above all -- and it has to do that by
integrating how the beams change. But it is not actually short of information,
and this cell exists to make that point concrete:

**the agent knows what it commanded.** RTRRL's input is
``[observation, previous action, previous reward]``, and for these tasks the
previous action *is* a steering command and a throttle command. The vehicle's
response to those commands is partly known in advance -- a servo has a lag, a
motor has a gain, a car has drag -- and none of that has to be learned from a
lidar return.

So: reserve ``n_phys`` of the hidden units, give them the **known** update law
driven by the agent's own command, and let the remaining units be an ordinary
LiGRU learning whatever is left over.

    v_hat'   = (1 - a_d) v_hat   + a_a * throttle       an integrator for speed
    d_hat'   = (1 - a_t) d_hat   + a_t * steer          the servo's first-order lag
    p_hat'   = (1 - a_p) p_hat   + a_p * (v_hat d_hat)  a yaw-rate proxy, v * delta

This is a **physics prior, not a physics constraint**. The three rate constants
``a_a, a_d, a_t, a_p`` are learnable (through a sigmoid, so they stay in
``(0, 1)`` and the units stay stable), and they are *initialised from the
nominal vehicle*: ``a_a = accel_max dt / speed_max``, ``a_t = dt / steer_tau``,
``a_d = drag dt``. The cell therefore starts out already knowing roughly how the
car responds, and RTRRL can retune those constants online when the car turns
out to be a different car -- which is exactly the sim-to-real setting of
``tutorial/11``.

## Three properties worth checking against the rest of the package

**It is still local.** Each physics unit's parameters belong to it alone. The
yaw-rate proxy *reads* ``v_hat`` and ``d_hat``, but that is state coupling, and
state coupling is what ``D`` is for -- ``imm`` stays block diagonal, so RFLO,
SnAp-1, UORO and exact RTRL all apply unchanged.

**Its leak is bounded below 1 by construction**, for the same reason
``liquid_gru``'s is: ``leak = 1 - sigmoid(raw) < 1`` always. No numerical cap
needed on the physics units.

**It degrades to plain LiGRU.** Unless the input carries the repo's 3x3
driving action space -- so: no meta-inputs, or a task like ``memory-chain``
whose two actions are not a steering command -- ``n_phys`` is zero and this is
exactly ``nets/ligru.py``. It is a specialisation for the driving tasks and
refuses to pretend otherwise.

## Does it help? Measured, and no

===========================  ===========  ================  ==============
task                         ``ligru``    ``physics_ligru`` difference
===========================  ===========  ================  ==============
lanekeep, in sim                    504               484   -20, 0.4 SE
lanekeep, zero-shot                 407               392   -14, 0.2 SE
lanekeep, after adapting            383               397   +14, 0.2 SE
overtake (return)                   314               366   +52, 0.4 SE
overtake (passes)                  2.00              1.91   -0.09, 0.1 SE
===========================  ===========  ================  ==============

Eight seeds each; nothing separated, every comparison under half a standard
error. The prior works -- the units reconstruct the hidden state at r = 1.00
(steering) and 0.85 (speed) untrained, and keep doing it on a vehicle they were
not initialised for -- and the policy does not care.

The prediction was that it would pay on ``overtake``, the task where committing
to a pass needs a closing rate. It was run to find out, and it did not. The
reading that fits: state estimation is not the bottleneck here. ``lanekeep`` is
largely reactive, and on ``overtake`` these units estimate the *ego's* motion
while the quantity that decides a pass is the *opponent's* speed -- which dead
reckoning of your own commands cannot give you. Encoding the right physics into
the wrong half of the problem is a fair description.

Untested next thing: give the reserved units the *relative* dynamics, by
integrating the car-flagged beams into a closing-rate estimate.

## What it is not

It is not a full observer. There is no correction step, nothing compares the
prediction against the beams, and nothing here is a Kalman filter. It is
open-loop dead reckoning of the commanded input, offered to the learned units
as three extra features they would otherwise have to construct. If the vehicle
differs from the prior, these units are *wrong* in a structured way, and the
learned units have to fix it -- which is the interesting case, not a failure.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.envs.vehicle import VehicleParams
from rtrrl_playground.nets.cell import OnlineCell


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


class PhysicsLiGRU(OnlineCell):
    """``n_phys`` dead-reckoning units with a learnable prior, plus LiGRU."""

    name = "physics_ligru"
    N_PHYS = 3  # v_hat, delta_hat, yaw-rate proxy

    @property
    def exact_rows(self) -> int:
        """The physics block is closed under the state Jacobian, so ``hybrid``
        is exact on it -- see :meth:`OnlineCell.exact_rows`.

        Closed because of how the three units are written: ``v_hat`` and
        ``delta_hat`` read only the *command*, which arrives in the input, and
        the yaw-rate proxy reads only those two. Nothing in the block reads a
        learned unit, so ``D[:n_phys, n_phys:] == 0`` and there is no influence
        path from outside to have approximated away.
        """
        return self.n_phys

    def _build(self, n_obs: int | None = None, n_act: int | None = None,
               dt: float = 0.05, vehicle: VehicleParams | None = None,
               forget_bias: float = 1.0) -> None:
        nx = self.n_xi
        self.SLICES = {"W_z": (0, nx), "W_c": (nx, 2 * nx)}
        self.theta = np.concatenate([self._init_block(), self._init_block()], axis=1)
        self.theta[:, nx - 1] = forget_bias

        # Can we see the command, and does it *mean* steer and throttle? Both
        # have to hold. An earlier version checked only that an action block
        # fitted in the input, and happily switched itself on for MemoryChain,
        # where a two-action space decoded to a steering command that does not
        # exist. Require the repo's 3x3 driving action space explicitly.
        self.n_phys = 0
        if n_obs is not None and n_act == 9 and self.n >= self.N_PHYS:
            if n_obs + n_act <= self.n_in:
                self.n_phys = self.N_PHYS
                self._act_slice = slice(int(n_obs), int(n_obs) + int(n_act))
                # The discrete action index encodes (steer, throttle) as
                # (a // 3 - 1, a % 3 - 1); against a one-hot that is a fixed
                # linear map, so the physics units need no learned input weights
                # at all -- they read the command exactly.
                k = np.arange(int(n_act))
                self._steer_w = (k // 3 - 1).astype(np.float64)
                self._thr_w = (k % 3 - 1).astype(np.float64)

        if self.n_phys:
            p = vehicle or VehicleParams()
            self.theta[:self.n_phys, :] = 0.0
            # Columns 0..3 of a physics row are its rate constants. The rest of
            # the row is unused -- its `imm` entries are zero, so the gradient
            # is zero and those columns never move. Wasteful by
            # (p - 4) * n_phys floats, and worth it to keep one (n, p) array.
            self.theta[0, 0] = _logit(p.accel_max * dt / p.speed_max)  # a_a
            self.theta[0, 1] = _logit(p.drag * dt)                     # a_d
            self.theta[1, 2] = _logit(min(dt / p.steer_tau, 0.95))     # a_t
            self.theta[2, 3] = _logit(0.5)                             # a_p

    def _forward(self, xi: np.ndarray, need_D: bool):
        nx, n_in, n = self.n_xi, self.n_in, self.n
        Wz, Wc = self.theta[:, :nx], self.theta[:, nx:2 * nx]
        z = _sigmoid(Wz @ xi)
        c = np.tanh(Wc @ xi)
        h = self.h
        h_new = z * h + (1.0 - z) * c

        d_z = (h - c) * z * (1.0 - z)
        d_c = (1.0 - z) * (1.0 - c * c)
        imm = np.empty_like(self.theta)
        imm[:, :nx] = d_z[:, None] * xi[None, :]
        imm[:, nx:2 * nx] = d_c[:, None] * xi[None, :]
        leak = z.copy()

        D = None
        if need_D:
            sl = slice(n_in, n_in + n)
            D = np.diag(leak) + d_z[:, None] * Wz[:, sl] + d_c[:, None] * Wc[:, sl]

        if self.n_phys:
            self._overwrite_physics(xi, h, h_new, imm, leak, D)
        return h_new, imm, leak, D

    def _overwrite_physics(self, xi, h, h_new, imm, leak, D) -> None:
        """Replace the first ``n_phys`` rows with dead reckoning of the command."""
        onehot = xi[self._act_slice]
        steer = float(onehot @ self._steer_w)
        thr = float(onehot @ self._thr_w)
        raw = self.theta[:self.n_phys, :4]
        a = _sigmoid(raw[np.arange(self.n_phys), [0, 2, 3]])  # a_a, a_t, a_p
        a_d = _sigmoid(raw[0, 1])
        a_a, a_t, a_p = a
        v, delta, p = h[0], h[1], h[2]

        h_new[0] = (1.0 - a_d) * v + a_a * thr
        h_new[1] = (1.0 - a_t) * delta + a_t * steer
        h_new[2] = (1.0 - a_p) * p + a_p * (v * delta)

        imm[:self.n_phys, :] = 0.0
        imm[0, 0] = thr * a_a * (1 - a_a)          # d/d a_a  (through the sigmoid)
        imm[0, 1] = -v * a_d * (1 - a_d)           # d/d a_d
        imm[1, 2] = (steer - delta) * a_t * (1 - a_t)
        imm[2, 3] = (v * delta - p) * a_p * (1 - a_p)

        leak[0] = 1.0 - a_d
        leak[1] = 1.0 - a_t
        leak[2] = 1.0 - a_p
        if D is not None:
            D[:self.n_phys, :] = 0.0
            D[0, 0] = 1.0 - a_d
            D[1, 1] = 1.0 - a_t
            D[2, 2] = 1.0 - a_p
            # The yaw-rate proxy reads the other two. State coupling, not a
            # parameter leak -- it belongs in D and nowhere else.
            D[2, 0] = a_p * delta
            D[2, 1] = a_p * v
