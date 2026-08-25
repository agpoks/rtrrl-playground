"""MLPCell -- the same interface with the recurrence deleted.

``h_t+1 = tanh(W [x_t ; 1])``. No dependence on ``h_t`` at all, which makes
this the control condition for every claim in the repo about what memory
buys: it plugs into the identical agent, uses the identical TD(lambda)
machinery, and differs from a CT-RNN in exactly one respect.

It also makes a quiet point about the gradient estimators. With no
recurrence, the influence is *just* the immediate Jacobian -- there is
nothing to carry, nothing to approximate, and RTRL, RFLO, SnAp and UORO all
collapse to the same thing, which is ordinary backpropagation through one
layer. Every difference between those four estimators is a statement about
time, and with no time there is no difference.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.nets.cell import OnlineCell


class MLPCell(OnlineCell):
    name = "mlp"

    def _build(self) -> None:
        self.SLICES = {"W": (0, self.n_xi)}
        w = self._init_block()
        w[:, self.n_in:self.n_in + self.n] = 0.0  # the recurrent block, held at zero
        self.theta = w

    def _forward(self, xi: np.ndarray, need_D: bool):
        W = self.theta
        phi = np.tanh(W @ xi)
        imm = (1.0 - phi * phi)[:, None] * xi[None, :]
        leak = np.zeros(self.n)
        D = np.zeros((self.n, self.n)) if need_D else None
        return phi, imm, leak, D

    def apply(self, dtheta: np.ndarray, lr: float) -> None:
        super().apply(dtheta, lr)
        if self.estimator != "none":  # keep the recurrent block dead
            self.theta[:, self.n_in:self.n_in + self.n] = 0.0
