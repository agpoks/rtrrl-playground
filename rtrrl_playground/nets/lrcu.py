"""LRCU -- Liquid-Resistance Liquid-Capacitance unit: both sides made liquid.

Farsang, Neubauer, Grosu, *"Liquid Resistance Liquid Capacitance Networks"*,
NeuroAI @ NeurIPS 2024 (arXiv:2403.08791). This is the cell that Lemmel,
Resch, Farsang, Hasani, Rus & Grosu (arXiv:2602.02236) found works best with
RTRRL when they fine-tuned a real 1:10 car online -- which is the single most
directly relevant result in the literature to what this playground is for, so
it gets its own cell here rather than a footnote.

The idea in one line: an LTC makes the *resistance* (the leak conductance)
depend on the input; an LRC additionally makes the **capacitance** -- strictly,
its reciprocal, the elastance ``eps`` -- depend on the input, and that second
liquid term multiplies the whole right-hand side::

    dh_i/dt = eps(w_i) * ( -sigmoid(f_i) h_i + tanh(u_i) e_i )

Solving that with one explicit Euler unfolding gives the LRCU update the paper
actually ships::

    h_i,t = (1 - eps(w_i) sigmoid(f_i)) h_i,t-1  +  eps(w_i) tanh(u_i) e_i

An LTC can only choose *how fast* to move; an LRCU can also choose *whether to
move at all this step*, uniformly across its own drive and its own decay. That
is what damps the oscillations LTCs are prone to under cheap solvers, and it
is why it survives being trained with a noisy online gradient.

**Simplified here in two respects, deliberately.** The paper's ``f``, ``u`` and
``w`` are sums over *per-synapse* nonlinearities, ``sum_j g_ji sigma(a_ji y_j +
b_ji)`` -- the biophysical-synapse construction from Lemmel & Grosu
(arXiv:2303.04944), which multiplies the parameter count per connection by
three. This implementation uses ordinary linear pre-activations,
``f = G xi``, and keeps the structural claim -- liquid resistance *and* liquid
capacitance -- while leaving the synapse model out. Every parameter still
belongs to exactly one neuron, so the influence bookkeeping in
``nets/cell.py`` applies unchanged; adding the per-synapse form would too, at
three times the width.

The second simplification is the leak conductance ``g_L`` that the paper adds
to both ``f`` and ``u``. It is not decoration and dropping it entirely does not
work -- see the note on ``leak_bias`` in ``_build``.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.nets.cell import OnlineCell


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


class LRCU(OnlineCell):
    name = "lrcu"

    def _build(self, e_init: float = 1.0, e_max: float = 2.0,
               leak_bias: float = 1.0) -> None:
        self.e_max = float(e_max)
        nx = self.n_xi
        self.SLICES = {"G": (0, nx), "K": (nx, 2 * nx), "O": (2 * nx, 3 * nx),
                       "e": (3 * nx, 3 * nx + 1)}
        e = self.rng.normal(0.0, e_init, size=(self.n, 1))
        self.theta = np.concatenate(
            [self._init_block(), self._init_block(), self._init_block(), e], axis=1)
        # A small negative bias on the elastance gate starts the unit
        # conservative: eps ~ 0.4 rather than 0.5, so an untrained network leaks
        # its state slowly instead of overwriting it every step.
        self.theta[:, 3 * nx - 1] = -0.5
        # A *positive* bias on the forget gate, which is this implementation's
        # stand-in for the paper's leak conductance g_L. It matters more than it
        # looks. The LRCU state settles at tanh(u) e / sigma(f), so a forget
        # gate that can close completely is an equilibrium that can run away --
        # measured, without this the state reached 14 on lanekeep while every
        # other cell here stayed inside [-1, 1], and the agent was hopeless.
        # Dropping g_L was the simplification; this is the part of it that had
        # to be put back.
        self.theta[:, nx - 1] = float(leak_bias)

    def post_update(self) -> None:
        """Bound the reversal potential, for the reason given in ``ltc.py``:
        the state is driven towards ``e``, so an unbounded ``e`` is an
        unbounded state and, eventually, a NaN."""
        lo, hi = self.SLICES["e"]
        self.theta[:, lo:hi] = np.clip(self.theta[:, lo:hi], -self.e_max, self.e_max)

    def _forward(self, xi: np.ndarray, need_D: bool):
        nx, n_in, n = self.n_xi, self.n_in, self.n
        G = self.theta[:, :nx]
        K = self.theta[:, nx:2 * nx]
        O = self.theta[:, 2 * nx:3 * nx]
        e = self.theta[:, 3 * nx]

        sf = _sigmoid(G @ xi)          # forget conductance
        tu = np.tanh(K @ xi)           # drive
        eps = _sigmoid(O @ xi)         # liquid elastance
        h = self.h
        h_new = (1.0 - eps * sf) * h + eps * tu * e

        d_f = -eps * sf * (1.0 - sf) * h
        d_u = eps * (1.0 - tu * tu) * e
        d_w = eps * (1.0 - eps) * (-sf * h + tu * e)

        imm = np.empty_like(self.theta)
        imm[:, :nx] = d_f[:, None] * xi[None, :]
        imm[:, nx:2 * nx] = d_u[:, None] * xi[None, :]
        imm[:, 2 * nx:3 * nx] = d_w[:, None] * xi[None, :]
        imm[:, 3 * nx] = eps * tu
        leak = 1.0 - eps * sf

        D = None
        if need_D:
            sl = slice(n_in, n_in + n)
            D = (np.diag(leak) + d_f[:, None] * G[:, sl]
                 + d_u[:, None] * K[:, sl] + d_w[:, None] * O[:, sl])
        return h_new, imm, leak, D
