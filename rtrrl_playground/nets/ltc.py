"""LTC -- Liquid Time-Constant network: the time constant becomes an input.

Hasani, Lechner, Amini, Rus, Grosu, *"Liquid Time-constant Networks"*,
AAAI 2021 (arXiv:2006.04439).

The LTC ODE couples the leak rate to the network's own activity::

    dh/dt = -[ 1/tau + f(x, h) ] h  +  f(x, h) A

with ``f = sigmoid(W xi)`` a synaptic gate and ``A`` a per-neuron reversal
potential. The effective time constant is ``tau_eff = tau / (1 + tau f)``:
when the gate opens the neuron becomes fast, when it closes the neuron holds.
*That* is what "liquid" means -- the memory horizon is set at run time by what
is happening, not at training time by a parameter.

Solved with the paper's fused (semi-implicit) Euler step, which is stable for
any step size because the state appears on both sides::

    h_t+1 = ( h_t + dt f A ) / ( 1 + dt (1/tau + f) )

Writing ``den = 1 + dt(1/tau + f)`` and ``s' = f(1-f)``, the derivatives are:

=====================  ==========================================
``dh'/d(W xi)``        ``dt s' (A - h') / den``
``dh'/dW_ij``          the above times ``xi_j``
``dh'/dA_i``           ``dt f_i / den_i``
``dh'/dtau_i``         ``h'_i dt / (tau_i^2 den_i)``
``leak_i``             ``1 / den_i``
=====================  ==========================================

Note what ``leak`` now is: an *input-dependent* decay. RFLO's influence trace
on an LTC therefore forgets fast exactly when the neuron is being driven hard
and holds on when it is not, with no extra machinery -- which is a reasonable
guess at why liquid cells and local online rules get on well together.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.nets.cell import OnlineCell


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


class LTC(OnlineCell):
    name = "ltc"

    def _build(self, dt: float = 1.0, tau_init=(1.0, 8.0), tau_min: float = 0.5,
               tau_max: float = 50.0, a_init: float = 1.0, a_max: float = 2.0,
               unfold: int = 1) -> None:
        self.dt, self.tau_min, self.tau_max, self.a_max = float(dt), tau_min, tau_max, a_max
        # Algorithm 1 of the paper takes ``L`` fused steps per input step, at a
        # cost of O(L x T); this takes one by default. The fused solver is
        # unconditionally stable so L=1 does not diverge --- it is a coarser
        # integration of the same ODE, and for a controller holding a 50 ms
        # loop the extra steps are the most expensive thing in the cell. The
        # parameter exists so that trade can be measured rather than assumed.
        self.unfold = max(int(unfold), 1)
        # An LTC needs a bigger input gain than a tanh cell to get the same
        # amount of movement out of its state, and the reason is structural
        # rather than a tuning accident. In a CT-RNN the pre-activation *is* the
        # state, through a tanh that spans [-1, 1]. In an LTC it goes through a
        # sigmoid gate and then only *scales* the reversal potential:
        # h -> f A / (1/tau + f), so the same spread in W xi produces a
        # fraction of the state variation. Measured on lanekeep, the unit-to-
        # unit temporal spread goes 0.04 -> 0.09 across this factor.
        self.input_gain *= 3.0
        nx = self.n_xi
        self.SLICES = {"W": (0, nx), "A": (nx, nx + 1), "tau": (nx + 1, nx + 2)}
        # Gaussian, not +/-1. The LTC state settles at f*A/(1/tau + f), so A
        # is the *only* thing that sets a unit's amplitude -- with A = +/-1 the
        # whole layer is one signal and its negation, the hidden state is close
        # to rank one, and a linear readout on top of it has almost nothing to
        # work with. Measured: unit-to-unit spread went from 0.06 to 0.3, which
        # is the difference between a usable basis and a duplicated one.
        A = self.rng.normal(0.0, a_init, size=(self.n, 1))
        # Same log-uniform spread as the CT-RNN -- see the note there. An LTC
        # can move its effective time constant at run time through the gate,
        # but only relative to this baseline, so the baseline still has to
        # cover more than one order of magnitude.
        if np.isscalar(tau_init):
            tau = np.full((self.n, 1), float(tau_init))
        else:
            tau = np.exp(self.rng.uniform(np.log(float(tau_init[0])),
                                          np.log(float(tau_init[1])), (self.n, 1)))
        self.theta = np.concatenate([self._init_block(), A, tau], axis=1)

    def post_update(self) -> None:
        """Keep tau positive and the reversal potential bounded.

        The bound on ``A`` is not cosmetic. The state settles towards ``A``, so
        an unbounded reversal potential is an unbounded state, which feeds back
        through the recurrence and takes the whole cell to infinity -- over a
        long online run with no batch to average over, that is not a remote
        possibility, it is what happens. Bounding it is also the biophysically
        honest choice: a reversal potential is a voltage, not a free scale.
        """
        lo, hi = self.SLICES["tau"]
        self.theta[:, lo:hi] = np.clip(self.theta[:, lo:hi], self.tau_min, self.tau_max)
        lo, hi = self.SLICES["A"]
        self.theta[:, lo:hi] = np.clip(self.theta[:, lo:hi], -self.a_max, self.a_max)

    def _forward(self, xi: np.ndarray, need_D: bool):
        nx = self.n_xi
        W, A, tau = self.theta[:, :nx], self.theta[:, nx], self.theta[:, nx + 1]
        z = W @ xi
        f = _sigmoid(z)
        df = f * (1.0 - f)
        dt = self.dt / self.unfold
        inv_tau = 1.0 / tau
        den = 1.0 + dt * (inv_tau + f)
        # Algorithm 1's L fused steps, in closed form rather than a loop. The
        # gate f is computed once from xi and so is constant across the inner
        # steps, which makes the recursion h <- (h + dt f A)/den a geometric
        # series with an exact sum:
        #
        #     h_L = h_0 g + h_inf (1 - g),  g = den^-L,  h_inf = f A / (1/tau + f)
        #
        # Cheaper than iterating, and --- the reason it is written this way ---
        # differentiable in closed form. A loop gives the forward pass for
        # free but leaves ``imm`` describing only the last step, which measured
        # 50--75 % wrong against finite differences at L = 2 and L = 4.
        g = den ** (-self.unfold)
        h_inf = f * A / (inv_tau + f)
        h_new = self.h * g + h_inf * (1.0 - g)

        # Exact derivatives of the closed form above, for any L.
        dg_dz = -self.unfold * g / den * dt * df          # dg/dz
        dhinf_dz = A * df * inv_tau / (inv_tau + f) ** 2  # dh_inf/dz
        dz = (self.h - h_inf) * dg_dz + (1.0 - g) * dhinf_dz
        dg_dtau = self.unfold * g / den * dt * inv_tau ** 2
        imm = np.empty_like(self.theta)
        imm[:, :nx] = dz[:, None] * xi[None, :]
        imm[:, nx] = (1.0 - g) * f / (inv_tau + f)
        imm[:, nx + 1] = ((self.h - h_inf) * dg_dtau
                          + (1.0 - g) * f * A * inv_tau ** 2 / (inv_tau + f) ** 2)
        leak = g

        D = None
        if need_D:
            D = np.diag(leak) + dz[:, None] * W[:, self.n_in:self.n_in + self.n]
        return h_new, imm, leak, D
