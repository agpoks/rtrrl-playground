"""The base class every recurrent cell here shares, and the five ways to get a
gradient out of one without ever running backwards.

A cell is any update ``h_t+1 = F(h_t, x_t; theta)``. Training it online means
carrying the **influence** ``J_t = dh_t/dtheta`` forwards, in the same pass as
the activations, and turning it into a weight update the moment a learning
signal ``g = dL/dh`` arrives. Every cell in this package therefore supplies
exactly three things from one forward pass:

``h_new``
    the new state.
``imm`` ``(n, p)``
    the *immediate* influence: how ``h_t+1[i]`` depends on neuron ``i``'s own
    parameters, holding ``h_t`` fixed.
``leak`` ``(n,)`` and ``D`` ``(n, n)``
    how ``h_t+1`` depends on ``h_t`` -- ``leak`` is the direct, per-neuron part
    (the term that survives if you delete every recurrent connection), ``D``
    is the whole Jacobian.

**Every parameter belongs to exactly one neuron.** ``theta`` is an ``(n, p)``
array: row ``i`` is everything neuron ``i`` owns, whatever the cell calls those
numbers. That is not a packing convenience, it is the structural property that
makes local online learning possible at all -- see the note in
``nets/ligru.py`` about the one gate that breaks it.

Given those, the five estimators differ only in what they keep:

=========  ==============================================  ===============  ========
estimator  what it carries                                 memory           bias
=========  ==============================================  ===============  ========
``rtrl``   the exact ``J[i, k, j] = dh_i/dtheta_kj``        ``n^2 p``        none
``uoro``   a rank-1 random sketch ``J ~ s (x) theta~``      ``n + n p``      none*
``snap1``  ``J`` restricted to ``i == k``, true diagonal    ``n p``          yes
``rflo``   the same sparsity, decayed by ``leak`` only      ``n p``          yes
``hybrid`` exact over a *known* block, ``rflo`` elsewhere   ``n_E^2 p + n p``  none on E
``none``   nothing; the recurrent weights never move        ``0``            n/a
=========  ==============================================  ===============  ========

``hybrid`` is this package's own, and it is the one that is *not* a bargain.
See :meth:`OnlineCell.exact_rows`.

\\* UORO is unbiased *in expectation* and noisy in any single realisation,
which is the trade the other approximations refuse: they are wrong in a fixed
direction but quiet. Which of those two failure modes you prefer is the whole
argument, and ``tutorial/02_gradients_online.py`` measures it rather than
asserting it.

References -- Williams & Zipser 1989 (RTRL); Tallec & Ollivier, ICLR 2018
(UORO); Menick et al., ICLR 2021 (SnAp); Murray, eLife 2019 (RFLO). See
``papers/README.md``.
"""

from __future__ import annotations

import numpy as np

ESTIMATORS = ("rtrl", "uoro", "snap1", "rflo", "hybrid", "none")


class OnlineCell:
    """A recurrent cell that carries its own gradient forwards.

    Subclasses set ``self.theta`` (an ``(n, p)`` array) in ``_build`` and
    implement ``_forward``. Everything else -- influence bookkeeping, the five
    estimators, the parameter update -- lives here and is shared.
    """

    #: name -> (start, stop) columns of ``theta``. Filled in by each subclass.
    SLICES: dict[str, tuple[int, int]] = {}
    name = "cell"

    @property
    def exact_rows(self) -> int:
        """How many leading units form a block whose influence is known exactly.

        The premise of the ``hybrid`` estimator, and the reason it is not just
        another approximation:

        RFLO drops ``dh_i/dtheta_kj`` for ``i != k`` because carrying the whole
        thing costs ``n^2 p``. But if the first ``n_E`` units are a *model you
        already know* -- an integrator for speed, a servo lag, a solver iterate
        -- then their parameters are few, and the exact gradient with respect to
        *them* costs only ``n n_E p``: the slab ``J[i, k]`` over every unit
        ``i`` and every known owner ``k < n_E``.

        That slab is **closed under its own recursion** --
        ``J[i,k] <- sum_l D[i,l] J[l,k] + delta_ik imm_i`` needs ``J[l,k]``,
        which the slab holds -- so it is not an approximation. ``hybrid`` is
        therefore *exact for the parameters you know* and RFLO for the rest, at
        ``n / n_E`` of the cost of full RTRL.

        **The direction matters, and getting it wrong is silent.** The first
        version of this carried only ``n_E x n_E``, reasoning that the physics
        block reads nothing outside itself. True, but irrelevant: the *learned*
        units read the physics state, so they contribute to the physics
        parameters' gradient, and dropping them left an error 4x smaller than
        RFLO's rather than zero. The block has to be closed on the axis you are
        summing over, not the one you are propagating along.

        A cell returning 0 (the default) makes ``hybrid`` identical to RFLO.
        """
        return 0

    def __init__(self, n_in: int, n_hidden: int, estimator: str = "rflo",
                 rng: np.random.Generator | None = None, input_gain: float = 3.0,
                 rec_gain: float = 0.2, bias_scale: float = 0.5,
                 leak_max: float = 0.99, **kwargs):
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        self.rng = rng or np.random.default_rng(0)
        self.input_gain, self.rec_gain, self.bias_scale = input_gain, rec_gain, bias_scale
        self.leak_max = float(leak_max)
        self.n_in, self.n = int(n_in), int(n_hidden)
        self.n_xi = self.n_in + self.n + 1  # [x ; h ; 1]
        self.estimator = estimator
        self._build(**kwargs)
        self.p = self.theta.shape[1]
        self.h = np.zeros(self.n)
        self.reset_state()

    # -- subclass API ------------------------------------------------------
    def _build(self, **kwargs) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _forward(self, xi: np.ndarray, need_D: bool):  # pragma: no cover - interface
        """Return ``(h_new, imm, leak, D)``. ``D`` may be ``None`` if not needed."""
        raise NotImplementedError

    def post_update(self) -> None:
        """Hook for constraints (positive time constants, and so on)."""

    def _init_block(self) -> np.ndarray:
        """One ``(n, n_xi)`` weight block: input, recurrent and bias columns.

        Fan-in scaling (``1/sqrt(n_in)``, ``1/sqrt(n)``) keeps the
        pre-activation O(1) as either side grows. Two departures from the
        textbook init, both of which mattered a lot more than expected:

        ``input_gain = 3``
            Observations here are normalised to ``[-1, 1]`` *limits*, and a
            typical one sits well inside that -- a pole a tenth of the way to
            falling, a beam at half range. At unit gain the pre-activation is
            then small enough that every ``tanh`` is in its linear region, the
            recurrent state is an affine filter of the input, and a linear
            critic on top of it can only ever fit a linear value function.
            The policy survives that (a linear policy solves CartPole); the
            *critic* does not, and a critic stuck at a constant leaves a
            persistent positive TD error that drives the actor into whichever
            action it happened to prefer first. Nonlinearity is not a nicety
            here, it is what stops the entropy collapsing.

        ``bias_scale = 0.5``
            Random biases put different neurons at different points on the
            ``tanh``, which is the cheapest way to get a diverse basis out of
            an untrained recurrent layer.

        ``rec_gain = 0.2``
            A quarter of the usual ``1/sqrt(n)``. Until the cell has learned
            something, a strongly-driven recurrent block is mostly recycled
            noise that the actor and critic have to see through, and at these
            budgets they never do -- measured, on ``lanekeep``, dropping the
            recurrent gain from 1.0 to 0.2 is worth more than any other single
            change in this file. Raise it back if you are training for long
            enough that the recurrence earns its keep.
        """
        w = np.zeros((self.n, self.n_xi))
        w[:, :self.n_in] = self.rng.normal(
            0, self.input_gain / np.sqrt(self.n_in), (self.n, self.n_in))
        w[:, self.n_in:self.n_in + self.n] = self.rng.normal(
            0, self.rec_gain / np.sqrt(self.n), (self.n, self.n))
        if self.bias_scale:
            w[:, -1] = self.rng.normal(0, self.bias_scale, self.n)
        return w

    def slice(self, name: str) -> np.ndarray:
        a, b = self.SLICES[name]
        return self.theta[:, a:b]

    # -- influence bookkeeping --------------------------------------------
    @property
    def n_params(self) -> int:
        return self.theta.size

    @property
    def needs_D(self) -> bool:
        return self.estimator in ("rtrl", "uoro", "snap1", "hybrid")

    def reset_state(self, keep_influence: bool = False) -> np.ndarray:
        """Zero the state and (unless told otherwise) the carried influence.

        At an episode boundary both go. The new episode's state does not depend
        on the weights through any path the environment kept, and carrying the
        influence across a reset quietly leaks credit between episodes.
        """
        self.h = np.zeros(self.n)
        if keep_influence:
            return self.h
        e = self.estimator
        self.P = np.zeros((self.n, self.p)) if e in ("rflo", "snap1", "hybrid") else None
        if e == "hybrid":
            ne = self.exact_rows
            self.n_exact = ne
            # (n, n_E, p): the influence of *every* unit's state on the known
            # block's parameters. Not (n_E, n_E, p) -- see the note in
            # `exact_rows` about which direction the block has to be closed in.
            self.Je = np.zeros((self.n, ne, self.p)) if ne else None
        if e == "rtrl":
            self.J = np.zeros((self.n, self.n * self.p))
        elif e == "uoro":
            self.s = np.zeros(self.n)
            self.theta_tilde = np.zeros(self.n * self.p)
        return self.h

    def step(self, x: np.ndarray) -> np.ndarray:
        """Advance one tick and update the carried influence."""
        xi = np.concatenate([x, self.h, [1.0]])
        h_new, imm, leak, D = self._forward(xi, self.needs_D)
        e = self.estimator
        if e in ("rflo", "snap1"):
            # Cap the decay strictly below 1. A unit that has learned to hold
            # its state perfectly (an LRCU with its elastance gate closed, a
            # LiGRU with its update gate saturated, a CT-RNN with a huge tau)
            # has leak = 1, and an influence trace that never decays is a sum
            # that never converges -- it overflows, quietly, tens of thousands
            # of steps into a run. The cap is the same idea as lambda in an
            # eligibility trace: an explicit horizon, here about 100 steps.
            np.clip(leak, 0.0, self.leak_max, out=leak)
        if e == "rflo":
            self.P *= leak[:, None]
            self.P += imm
        elif e == "snap1":
            # SnAp-1: same sparsity as RFLO, but propagated through the true
            # diagonal of D -- which includes each neuron's self-recurrence,
            # exactly the term RFLO drops. One extra vector, no extra order.
            self.P *= np.clip(np.diag(D), -self.leak_max, self.leak_max)[:, None]
            self.P += imm
        elif e == "hybrid":
            # RFLO everywhere...
            self.P *= leak[:, None]
            self.P += imm
            ne = self.n_exact
            if ne:
                # ...and the exact influence of the *known block's parameters*.
                # J[i,k] for every unit i and every known parameter-owner
                # k < ne. The recursion needs J[l,k] for all l, which is exactly
                # what this slab holds, so it closes and is exact.
                self.Je = np.einsum("il,lkj->ikj", D, self.Je)
                self.Je[np.arange(ne), np.arange(ne)] += imm[:ne]
        elif e == "rtrl":
            self.J = D @ self.J
            self.J.reshape(self.n, self.n, self.p)[np.arange(self.n), np.arange(self.n)] += imm
        elif e == "uoro":
            self._uoro_update(imm, D)
        self.h = h_new
        return self.h

    def _uoro_update(self, imm: np.ndarray, D: np.ndarray) -> None:
        """Rank-1 unbiased sketch (Tallec & Ollivier 2018).

        ``J ~ s theta~^T``. Pushing that through the recurrence keeps it rank
        one; the immediate term is projected onto a fresh random direction,
        also rank one; the two are then squashed back to rank one with the
        normalisation that minimises the variance of the result.

        The projection is cheap here for the same reason everything else is:
        the immediate Jacobian is block diagonal, so ``nu^T dF/dtheta`` is a
        row-wise scaling of ``imm``, not a matrix product.
        """
        nu = self.rng.choice([-1.0, 1.0], size=self.n)
        nu_imm = (nu[:, None] * imm).ravel()
        Ds = D @ self.s
        eps = 1e-8
        rho0 = np.sqrt((np.linalg.norm(self.theta_tilde) + eps) / (np.linalg.norm(Ds) + eps))
        rho1 = np.sqrt((np.linalg.norm(nu_imm) + eps) / (np.sqrt(self.n) + eps))
        self.s = rho0 * Ds + rho1 * nu
        self.theta_tilde = self.theta_tilde / rho0 + nu_imm / rho1

    # -- gradients ---------------------------------------------------------
    def grad(self, g: np.ndarray) -> np.ndarray:
        """Turn a learning signal ``g = dL/dh`` into ``dL/dtheta``, shape ``(n, p)``."""
        e = self.estimator
        if e in ("rflo", "snap1"):
            return g[:, None] * self.P
        if e == "hybrid":
            dtheta = g[:, None] * self.P
            ne = self.n_exact
            if ne:
                # Column k collects *every* unit that k influences -- including
                # the learned ones, which read the known block's state. Summing
                # only over the block itself was the first version of this and
                # it was measurably wrong; see tests/test_gradients.py.
                dtheta[:ne] = np.einsum("i,ikj->kj", g, self.Je)
            return dtheta
        if e == "rtrl":
            return (g @ self.J).reshape(self.n, self.p)
        if e == "uoro":
            return float(g @ self.s) * self.theta_tilde.reshape(self.n, self.p)
        return np.zeros_like(self.theta)

    def apply(self, dtheta: np.ndarray, lr: float) -> None:
        """Gradient *ascent*: every objective in this repo is maximised."""
        if self.estimator == "none":
            return
        self.theta += lr * dtheta
        self.post_update()

    # -- introspection -----------------------------------------------------
    def influence_bytes(self) -> int:
        """Memory the carried gradient costs. Quoted in the tutorial's table."""
        e = self.estimator
        if e in ("rflo", "snap1"):
            return self.P.nbytes
        if e == "hybrid":
            return self.P.nbytes + (self.Je.nbytes if self.Je is not None else 0)
        if e == "rtrl":
            return self.J.nbytes
        if e == "uoro":
            return self.s.nbytes + self.theta_tilde.nbytes
        return 0

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_in={self.n_in}, n={self.n}, p={self.p}, "
                f"estimator={self.estimator!r}, params={self.n_params})")
