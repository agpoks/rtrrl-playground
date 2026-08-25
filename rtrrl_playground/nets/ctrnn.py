"""CT-RNN -- a leaky integrator with a learned, input-independent time constant.

The cell the RTRRL paper uses::

    xi     = [x ; h ; 1]
    h_t+1  = h_t + (tanh(W xi) - h_t) / tau

``tau`` is per-neuron and learned, but it does *not* depend on the input --
that is precisely what makes this the non-liquid member of the set. A neuron's
memory horizon is a property of the neuron, fixed once training ends, and the
same whether the world is calm or violent. Everything in ``ltc.py`` and
``lrcu.py`` is a way of removing that restriction.

Derivatives, with ``phi = tanh(W xi)``:

===================  ==========================================
``dh'/dW_ij``        ``phi'_i xi_j / tau_i``
``dh'/dtau_i``       ``(h_i - phi_i) / tau_i^2``
``leak_i``           ``1 - 1/tau_i``
``D``                ``diag(leak) + (phi'/tau) W_rec``
===================  ==========================================

``leak`` is where RFLO's approximation lives: it is ``D`` with the recurrent
block deleted, so the influence decays at the neuron's own leak rate and never
learns about credit that travelled through a synapse.

Reference: Funahashi & Nakamura, Neural Networks 1993, for the continuous-time
RNN; the discretisation and the learned ``tau`` follow Lemmel & Grosu (2025).
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.nets.cell import OnlineCell


class CTRNN(OnlineCell):
    name = "ctrnn"

    def _build(self, tau_init=(1.0, 8.0), tau_min: float = 1.0,
               tau_max: float = 50.0, learn_tau: bool = True) -> None:
        """``tau_init`` is a pair (log-uniform spread) or a scalar (all equal).

        The spread is the default, and it matters more than it looks. ``tau``
        sets two things at once and pulls them in opposite directions: a small
        ``tau`` makes the *state* responsive (the neuron tracks its input
        within a step) and the *influence* forgetful (RFLO's trace decays at
        ``1 - 1/tau``, so at ``tau = 1`` it keeps nothing at all); a large
        ``tau`` does the reverse. One shared value forces a single compromise
        on the whole layer. Spreading them log-uniformly over 1 to 8 gives the
        actor fast units to react with and the learning rule slow units to
        carry credit in -- the same reason multi-timescale RNNs exist.
        """
        self.tau_min, self.tau_max, self.learn_tau = tau_min, tau_max, learn_tau
        self.SLICES = {"W": (0, self.n_xi), "tau": (self.n_xi, self.n_xi + 1)}
        if np.isscalar(tau_init):
            tau = np.full((self.n, 1), float(tau_init))
        else:
            lo, hi = float(tau_init[0]), float(tau_init[1])
            tau = np.exp(self.rng.uniform(np.log(lo), np.log(hi), (self.n, 1)))
        self.theta = np.concatenate([self._init_block(), tau], axis=1)

    def post_update(self) -> None:
        a, b = self.SLICES["tau"]
        self.theta[:, a:b] = np.clip(self.theta[:, a:b], self.tau_min, self.tau_max)

    def _forward(self, xi: np.ndarray, need_D: bool):
        W = self.theta[:, :self.n_xi]
        tau = self.theta[:, self.n_xi]
        phi = np.tanh(W @ xi)
        dphi = 1.0 - phi * phi
        inv_tau = 1.0 / tau
        h_new = self.h + (phi - self.h) * inv_tau

        imm = np.empty_like(self.theta)
        imm[:, :self.n_xi] = (dphi * inv_tau)[:, None] * xi[None, :]
        imm[:, self.n_xi] = ((self.h - phi) * inv_tau ** 2) if self.learn_tau else 0.0
        leak = 1.0 - inv_tau

        D = None
        if need_D:
            D = np.diag(leak) + (dphi * inv_tau)[:, None] * W[:, self.n_in:self.n_in + self.n]
        return h_new, imm, leak, D
