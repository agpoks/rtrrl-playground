"""LiquidGRU -- a GRU gate reinterpreted as a conductance, plus a baseline leak.

Not from a paper. Built for this repo after checking that the nearby things --
CT-GRU (Mozer et al. 2017), LTC-SE's CT-GRU-style gates, and the
continuous-time readings of the GRU (Jordan, Sokol & Park, *Gated Recurrent
Units Viewed Through the Lens of Continuous Time Dynamical Systems*, Frontiers
in Computational Neuroscience 2021) -- are all motivated by *supervised*
modelling, and none of them is motivated by the thing that makes it worth
having here. Companion in spirit to ``liquid-nn-playground``'s Liquid-LSTM,
which does the same for the four LSTM gates.

## The cell

    xi   = [x ; h ; 1]
    z    = sigmoid(W_z xi)          a gate, but read as a *conductance*
    c    = tanh(W_c xi)             the state it is being pulled towards
    g    = 1/tau + z                total conductance: a fixed leak, plus the gate
    h'   = (h + dt g c) / (1 + dt g)

Semi-implicit Euler, the same fused step LTC uses, so it is stable at any step
size. Compare the two things it is between:

* **LiGRU** is ``h' = z h + (1-z) c``. The gate *interpolates*. Its effective
  time constant is set entirely by ``z``.
* **LTC** is ``h' = (h + dt f A)/(1 + dt(1/tau + f))``. Input-dependent leak,
  but the state is pulled towards a per-neuron constant ``A``, not towards a
  learned function of the input.

This takes the leak structure from LTC and the *target* from a GRU: the state
is pulled towards ``tanh(W_c xi)``, a full function of input and state, at a
rate the gate sets. In LTC terms the reversal potential became input-dependent;
in GRU terms the update gate became a conductance with a floor.

## Why it earns its place in an *online* repo

That floor is the point, and it is a fact about the learning rule rather than
about modelling.

RFLO carries an influence array updated as ``P <- leak * P + immediate``. That
is a geometric series, and it converges only while ``leak < 1``. A LiGRU whose
update gate saturates has ``leak = z = 1``: a unit that has learned to hold its
state perfectly, which is a perfectly reasonable thing for a memory task to
want -- and an influence sum that never decays and eventually overflows. Tens
of thousands of steps into a run, quietly. ``nets/cell.py`` currently patches
this with ``leak_max = 0.99``, an arbitrary numerical cap.

Here the leak is

    leak = 1 / (1 + dt (1/tau + z))  <=  1 / (1 + dt/tau)  <  1

**bounded below 1 by construction**, for any gate value, because ``1/tau > 0``
always. The arbitrary cap becomes a learned per-neuron parameter with a
physical meaning: ``tau`` is the longest this neuron is allowed to remember,
and gradient descent picks it. ``tests/test_gradients.py`` asserts the bound
holds with the cap switched off.

## Derivatives

With ``den = 1 + dt g`` and ``dh'/dg = dt (c - h') / den``:

=====================  =========================================================
``dh'/dW_z,ij``        ``dt (c_i - h'_i)/den_i * z_i(1-z_i) * xi_j``
``dh'/dW_c,ij``        ``dt g_i/den_i * (1 - c_i^2) * xi_j``
``dh'/dtau_i``         ``-dt (c_i - h'_i) / (den_i tau_i^2)``
``leak_i``             ``1 / den_i``
=====================  =========================================================

Every parameter belongs to exactly one neuron -- the property a full GRU's
reset gate breaks (see ``nets/ligru.py``) -- so RFLO, SnAp-1, UORO and exact
RTRL all apply unchanged.

## And what it costs, measured

The floor that buys the guarantee is also a ceiling on memory: a neuron cannot
hold its state for longer than about ``tau`` steps, however hard the gate
tries. On MemoryChain-8 -- a task that is *only* memory -- that price is
visible and monotone (200k steps, 3 seeds, optimum +1.0):

===================  =============  ================
``tau_init``         leak floor     MemoryChain-8
===================  =============  ================
``(1, 8)``           <= 0.89        +0.291 +/- 0.404
``(4, 40)``          <= 0.976       +0.552 +/- 0.393
``(10, 50)``         <= 0.980       +0.775 +/- 0.090
LiGRU (no floor)     1.0            +0.883 +/- 0.037
===================  =============  ================

Read that as the price list for numerical safety, in the currency of memory.
It approaches LiGRU as the floor approaches 1, which is exactly what the
algebra says it should do, and the seed spread collapses along the way.

On ``lanekeep`` the same three settings give 351 / 313 / 372 -- all inside the
seed spread, so the long time constants cost nothing on a task that does not
need deep memory. Hence the default of ``(10, 50)``.

**Being straight about the outcome: this cell does not win.** LiGRU beats it on
both tasks (+0.883 vs +0.775 on MemoryChain, 468 vs 372 on lanekeep). What it
has is a property neither of them has -- an influence series that provably
converges without a numerical cap -- and a measured price for it. That is worth
a file; it is not worth a claim.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.nets.cell import OnlineCell


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


class LiquidGRU(OnlineCell):
    name = "liquid_gru"

    def _build(self, dt: float = 1.0, tau_init=(10.0, 50.0), tau_min: float = 1.0,
               tau_max: float = 50.0, gate_bias: float = 0.0) -> None:
        self.dt = float(dt)
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        nx = self.n_xi
        self.SLICES = {"W_z": (0, nx), "W_c": (nx, 2 * nx), "tau": (2 * nx, 2 * nx + 1)}
        if np.isscalar(tau_init):
            tau = np.full((self.n, 1), float(tau_init))
        else:
            tau = np.exp(self.rng.uniform(np.log(float(tau_init[0])),
                                          np.log(float(tau_init[1])), (self.n, 1)))
        self.theta = np.concatenate([self._init_block(), self._init_block(), tau], axis=1)
        # No positive forget-bias here, unlike LiGRU. A gated cell normally
        # needs one so it starts out remembering; this one already does, because
        # tau provides the memory when the gate is shut.
        self.theta[:, nx - 1] = gate_bias

    def post_update(self) -> None:
        lo, hi = self.SLICES["tau"]
        self.theta[:, lo:hi] = np.clip(self.theta[:, lo:hi], self.tau_min, self.tau_max)

    def _forward(self, xi: np.ndarray, need_D: bool):
        nx, n_in, n = self.n_xi, self.n_in, self.n
        Wz, Wc, tau = self.theta[:, :nx], self.theta[:, nx:2 * nx], self.theta[:, 2 * nx]
        z = _sigmoid(Wz @ xi)
        c = np.tanh(Wc @ xi)
        g = 1.0 / tau + z                 # total conductance: baseline leak + gate
        den = 1.0 + self.dt * g
        h = self.h
        h_new = (h + self.dt * g * c) / den

        dg = self.dt * (c - h_new) / den  # dh'/dg, shared by the gate and by tau
        d_z = dg * z * (1.0 - z)
        d_c = self.dt * g / den * (1.0 - c * c)

        imm = np.empty_like(self.theta)
        imm[:, :nx] = d_z[:, None] * xi[None, :]
        imm[:, nx:2 * nx] = d_c[:, None] * xi[None, :]
        imm[:, 2 * nx] = -dg / tau ** 2
        leak = 1.0 / den                  # < 1 for any gate value, since 1/tau > 0

        D = None
        if need_D:
            sl = slice(n_in, n_in + n)
            D = np.diag(leak) + d_z[:, None] * Wz[:, sl] + d_c[:, None] * Wc[:, sl]
        return h_new, imm, leak, D
