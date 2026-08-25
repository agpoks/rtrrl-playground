"""LiGRU -- a gated discrete RNN, as the non-continuous-time control.

Ravanelli, Brakel, Omologo, Bengio, *"Light Gated Recurrent Units for Speech
Recognition"*, IEEE TETCI 2018 (arXiv:1803.10225): a GRU with the reset gate
removed::

    z     = sigmoid(W_z xi)
    c     = tanh(W_c xi)
    h_t+1 = z h_t + (1 - z) c

This is here as the baseline that is *gated but not continuous-time*: it can
choose how much to forget, exactly as an LTC can, but the choice is a learned
gate rather than a time constant with physical units, and there is no ODE
behind it. If a liquid cell beats a CT-RNN only because it has an
input-dependent leak, LiGRU will match it. If the continuous-time formulation
is doing something else, LiGRU will not.

**Why not a full GRU.** The reset gate computes ``c = tanh(W_c [x ; r * h])``,
so ``W_r`` belonging to neuron ``k`` reaches ``h_i`` for every ``i`` through
``W_c[i, k]``. That breaks the property every cell in this package relies on:
*each parameter belongs to exactly one neuron*. Without it, the immediate
Jacobian is no longer block-diagonal, RFLO's "keep only ``dh_i/dtheta_ij``"
throws away a first-order effect rather than a second-order one, and the
``(n, p)`` influence array in ``nets/cell.py`` cannot represent what is
happening. That is a real and underdiscussed constraint on which architectures
admit local online learning -- worth knowing before designing a cell you
intend to train this way -- so the boundary is drawn here explicitly instead
of being papered over with an approximation nobody would notice.

``tanh`` for the candidate rather than the paper's ReLU + batch norm: the
influence is carried for thousands of steps with no batch to normalise over,
and an unbounded state makes that a divergence waiting to happen.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.nets.cell import OnlineCell


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


class LiGRU(OnlineCell):
    name = "ligru"

    def _build(self, forget_bias: float = 1.0) -> None:
        nx = self.n_xi
        self.SLICES = {"W_z": (0, nx), "W_c": (nx, 2 * nx)}
        self.theta = np.concatenate([self._init_block(), self._init_block()], axis=1)
        # Positive bias on the update gate -> z ~ 0.73 at init, so the unit
        # starts out remembering. Standard practice for gated cells, and it
        # matters more here: an influence trace that is wiped every step never
        # accumulates enough of anything to learn from.
        self.theta[:, nx - 1] = forget_bias

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
        leak = z

        D = None
        if need_D:
            sl = slice(n_in, n_in + n)
            D = np.diag(leak) + d_z[:, None] * Wz[:, sl] + d_c[:, None] * Wc[:, sl]
        return h_new, imm, leak, D
