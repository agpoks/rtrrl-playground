"""Policy and value heads: every derivative written out, none of them autograd.

Three small linear maps sit on top of the recurrent state: a categorical or
Gaussian policy, and a scalar value. Their gradients are two lines of algebra
each, and writing them out rather than calling ``.backward()`` is the point --
in RTRRL these gradients are not the start of a backward pass through the
network, they are a *signal handed to the recurrent cell*, which already
knows how its own state depends on its own weights.

**Feedback alignment.** The recurrent cell needs ``dL/dh``, which is
``theta^T`` times the head's own gradient. Using the real ``theta`` means the
learning signal travelling back into the cell depends on the exact forward
weights -- the "weight transport problem", the standard objection to
backpropagation as a model of anything biological. Lillicrap et al. (Nature
Communications 2016) showed you can replace ``theta^T`` with a *fixed random
matrix* ``B`` and the forward weights will align themselves to it during
learning. RTRRL takes that option, so:

    feedback="random"     g_h = B^T (dL/dz)      B fixed at init, never trained
    feedback="symmetric"  g_h = theta^T (dL/dz)  the true gradient

``symmetric`` is the ablation. If a run improves a lot when you switch to it,
alignment is failing and that is worth knowing; in practice on these tasks it
barely moves, which is the whole surprise of the feedback-alignment result.
"""

from __future__ import annotations

import numpy as np

LOG_2PI = float(np.log(2 * np.pi))


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class CategoricalHead:
    """Softmax policy over ``n_act`` actions, with an entropy bonus."""

    def __init__(self, n_hidden: int, n_act: int, feedback: str = "random",
                 entropy_coef: float = 1e-5, sticky_prob: float = 0.0,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.n, self.n_act = int(n_hidden), int(n_act)
        self.theta = np.zeros((self.n_act, self.n))  # start uniform: no early bias
        self.bias = np.zeros(self.n_act)
        self.B = rng.normal(0, 1.0 / np.sqrt(self.n), (self.n_act, self.n))
        self.feedback = feedback
        self.entropy_coef = float(entropy_coef)
        # Sticky actions (Machado et al., "Revisiting the ALE", JAIR 2018):
        # with probability `sticky_prob`, repeat the previous action instead
        # of the fresh draw below, so consecutive actions are correlated
        # rather than i.i.d. -- a discrete stand-in for what Ornstein-
        # Uhlenbeck noise buys a continuous controller. `pi` is always the
        # *current* policy at `h`, so this changes only which sample of it
        # gets executed and credited: `grads` scores whatever action was
        # actually taken under the current `pi`, and the score-function
        # gradient does not care how that action was chosen. `sticky_prob`
        # defaults to 0.0, which draws exactly the one `rng` call per tick
        # this always did -- every existing result is reproduced exactly.
        self.sticky_prob = float(sticky_prob)
        self.prev_a: int | None = None

    @property
    def n_params(self) -> int:
        return self.theta.size + self.bias.size

    def act(self, h: np.ndarray, rng: np.random.Generator):
        pi = softmax(self.theta @ h + self.bias)
        a = int(rng.choice(self.n_act, p=pi))
        if (self.sticky_prob > 0.0 and self.prev_a is not None
                and rng.random() < self.sticky_prob):
            a = self.prev_a
        self.prev_a = a
        return a, pi

    def reset(self) -> None:
        """Clear the sticky-action memory at an episode boundary."""
        self.prev_a = None

    def grads(self, h: np.ndarray, a: int, pi: np.ndarray):
        """Gradients of ``log pi[a] + eta * H(pi)``.

        With ``z`` the logits::

            d log pi[a] / dz = onehot(a) - pi
            d H / dz_j       = -pi_j (log pi_j + H)

        The second one is worth deriving once rather than trusting: the naive
        ``-pi_j(log pi_j + 1)`` is what you get if you forget that every
        component of ``pi`` moves when one logit does.
        """
        dz = -pi.copy()
        dz[a] += 1.0
        if self.entropy_coef:
            logpi = np.log(np.clip(pi, 1e-12, None))
            H = -float(pi @ logpi)
            dz += self.entropy_coef * (-pi * (logpi + H))
        dtheta = np.outer(dz, h)
        back = self.B if self.feedback == "random" else self.theta
        return dtheta, dz, back.T @ dz

    def apply(self, dtheta, dbias, lr: float) -> None:
        self.theta += lr * dtheta
        self.bias += lr * dbias

    def entropy(self, pi: np.ndarray) -> float:
        return float(-(pi * np.log(np.clip(pi, 1e-12, None))).sum())

    def greedy(self, h: np.ndarray) -> int:
        """The argmax action. A method rather than the caller doing
        ``argmax(pi)``, because for a factored policy ``pi`` is several
        distributions laid end to end and its argmax is not an action."""
        return int(np.argmax(self.theta @ h + self.bias))


class FactoredCategoricalHead:
    """One softmax per action axis, sampled independently from the same ``h``.

    A grid action space costs a flat head the *product* of its axes. At the
    0.1 resolution this project wants -- 21 steering levels and 21 throttle
    levels -- that is a 441-way softmax fed one sample per 50 ms tick, and the
    score-function estimator has to tell 441 actions apart from single draws.
    Factoring it into 21 + 21 = 42 logits is not an approximation of that: if
    the two axes are conditionally independent given ``h`` then

        log pi(s, t | h) = log pi_s(s | h) + log pi_t(t | h)

    exactly, so the gradient is the concatenation of the per-axis gradients and
    the entropy is their sum. Independence given ``h`` is the real assumption,
    and it is the right one for two separate actuators: the recurrent state
    already carries whatever couples them.

    Laid out as one stacked ``theta`` of ``(sum(nvec), n_hidden)`` so every
    eligibility trace, clip and update in the agent works on it unchanged --
    the axes are slices of one parameter block, not separate heads.

    Actions are exchanged with the environment as the same flat index
    ``spaces.action_grid`` uses, so filters, replay of old traces and the
    one-hot encoding all keep working: axis ``i`` contributes
    ``idx[i] * prod(nvec[i+1:])``, which for ``(n_steer, n_throttle)`` is
    exactly ``a = i_steer * n_throttle + i_throttle``.
    """

    def __init__(self, n_hidden: int, nvec, feedback: str = "random",
                 entropy_coef: float = 1e-5, sticky_prob: float = 0.0,
                 smooth_t: float = 0.0, smooth_s: float = 0.0,
                 smooth_sigma: float = 0.1,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.nvec = tuple(int(x) for x in nvec)
        if any(x < 1 for x in self.nvec):
            raise ValueError(f"every axis needs >= 1 level, got {self.nvec}")
        self.n = int(n_hidden)
        # n_act is the stacked logit count, which is what the agent's traces
        # are shaped by; `n_joint` is the size of the action set it indexes
        self.n_act = int(sum(self.nvec))
        self.n_joint = int(np.prod(self.nvec))
        self.theta = np.zeros((self.n_act, self.n))
        self.bias = np.zeros(self.n_act)
        self.B = rng.normal(0, 1.0 / np.sqrt(self.n), (self.n_act, self.n))
        self.feedback = feedback
        self.entropy_coef = float(entropy_coef)
        self.sticky_prob = float(sticky_prob)
        self.prev_a: int | None = None
        # CAPS (Mysore et al. 2021) adapted to an ordinal categorical head:
        # the quantity an actuator sees is the *commanded level*, so the
        # penalty is on the expected level moving, not on the distribution
        # changing in some abstract sense. Two 2-level distributions that
        # swap all their mass have a large KL and the same mean; two that
        # shift one level have a small KL and a mean that moved. It is the
        # second that the tyre feels.
        self.smooth_t = float(smooth_t)
        self.smooth_s = float(smooth_s)
        self.smooth_sigma = float(smooth_sigma)
        self.prev_mean: np.ndarray | None = None
        # where each axis starts in the stacked block
        self.slices, o = [], 0
        for k in self.nvec:
            self.slices.append(slice(o, o + k))
            o += k
        # multipliers that turn per-axis indices into the flat grid index
        self.strides = np.ones(len(self.nvec), dtype=int)
        for i in range(len(self.nvec) - 2, -1, -1):
            self.strides[i] = self.strides[i + 1] * self.nvec[i + 1]
        # the command each level stands for, in the -1..1 the env decodes
        self.levels = [np.linspace(-1.0, 1.0, k) if k > 1 else np.zeros(1)
                       for k in self.nvec]

    @property
    def n_params(self) -> int:
        return self.theta.size + self.bias.size

    def flat(self, idx) -> int:
        return int(np.dot(np.asarray(idx, dtype=int), self.strides))

    def unflat(self, a: int):
        a, out = int(a), []
        for st in self.strides:
            out.append(a // st)
            a = a % st
        return out

    def act(self, h: np.ndarray, rng: np.random.Generator):
        z = self.theta @ h + self.bias
        pi = np.empty_like(z)
        idx = []
        for sl in self.slices:
            p = softmax(z[sl])
            pi[sl] = p
            # inverse-CDF for the same reason the flat head uses it
            j = int(np.searchsorted(np.cumsum(p), rng.random()))
            idx.append(min(j, sl.stop - sl.start - 1))
        a = self.flat(idx)
        if (self.sticky_prob > 0.0 and self.prev_a is not None
                and rng.random() < self.sticky_prob):
            a = self.prev_a
        self.prev_a = a
        return a, pi

    def reset(self) -> None:
        self.prev_a = None
        self.prev_mean = None   # the last episode's command is not this one's

    def means(self, pi: np.ndarray) -> np.ndarray:
        """Expected command per axis -- what the actuator is being asked for."""
        return np.array([float(pi[sl] @ u)
                         for sl, u in zip(self.slices, self.levels)])

    def smooth_grads(self, h: np.ndarray, pi: np.ndarray,
                     rng: np.random.Generator | None = None):
        """Gradient of the CAPS penalty, to be *subtracted* by the caller.

        Temporal term: the expected command must not move much between two
        ticks 50 ms apart. Spatial term: it must not move much for a small
        change of the recurrent state either -- which is the one that matters
        here, because a policy whose output swings on a small change of ``h``
        is a policy that will ring at whatever frequency ``h`` ripples at.
        CAPS perturbs the observation; perturbing ``h`` is the same condition
        one layer in, and costs a head evaluation rather than a cell step.

        Returns ``(dtheta, dz, g_back)`` with the same shapes ``grads`` uses,
        so the agent can fold them into the update it already builds. The
        side effect is remembering this tick's mean for the next one.
        """
        dz = np.zeros_like(pi)
        m = self.means(pi)
        # d(mean_i)/d(z in axis i) = p * (u - mean_i)
        dm = [pi[sl] * (u - m[i])
              for i, (sl, u) in enumerate(zip(self.slices, self.levels))]

        if self.smooth_t and self.prev_mean is not None:
            diff = m - self.prev_mean
            for i, sl in enumerate(self.slices):
                dz[sl] += 2.0 * self.smooth_t * diff[i] * dm[i]
        self.prev_mean = m

        dtheta = np.outer(dz, h)
        if self.smooth_s and rng is not None:
            hb = h + self.smooth_sigma * rng.standard_normal(h.shape)
            zb = self.theta @ hb + self.bias
            pib = np.empty_like(zb)
            for sl in self.slices:
                pib[sl] = softmax(zb[sl])
            mb = self.means(pib)
            diff = m - mb
            dzb = np.zeros_like(pi)
            for i, (sl, u) in enumerate(zip(self.slices, self.levels)):
                dz[sl] += 2.0 * self.smooth_s * diff[i] * dm[i]
                dzb[sl] -= 2.0 * self.smooth_s * diff[i] * pib[sl] * (u - mb[i])
            # the perturbed branch moves theta and the bias, but hb is not a
            # state the cell was ever in, so it must not be fed back into it
            dtheta = np.outer(dz, h) + np.outer(dzb, hb)
            dz_cell, dz = dz, dz + dzb
        else:
            dz_cell = dz

        back = self.B if self.feedback == "random" else self.theta
        return dtheta, dz, back.T @ dz_cell

    def grads(self, h: np.ndarray, a: int, pi: np.ndarray):
        """Per-axis ``onehot - pi`` stacked, which is the joint gradient.

        Because the joint log-probability is the sum of the per-axis ones and
        each axis's logits appear in only its own term, the cross blocks are
        zero and no coupling term is missing here.
        """
        dz = -pi.copy()
        idx = self.unflat(a)
        for j, sl in zip(idx, self.slices):
            dz[sl.start + j] += 1.0
        if self.entropy_coef:
            for sl in self.slices:
                p = pi[sl]
                logp = np.log(np.clip(p, 1e-12, None))
                H = -float(p @ logp)
                dz[sl] += self.entropy_coef * (-p * (logp + H))
        back = self.B if self.feedback == "random" else self.theta
        return np.outer(dz, h), dz, back.T @ dz

    def apply(self, dtheta, dbias, lr: float) -> None:
        self.theta += lr * dtheta
        self.bias += lr * dbias

    def entropy(self, pi: np.ndarray) -> float:
        """Joint entropy, which for independent axes is the sum of theirs."""
        tot = 0.0
        for sl in self.slices:
            p = pi[sl]
            tot += float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        return tot

    def greedy(self, h: np.ndarray) -> int:
        z = self.theta @ h + self.bias
        return self.flat([int(np.argmax(z[sl])) for sl in self.slices])

    def mean(self, h: np.ndarray) -> int:
        """The level nearest the distribution's *expected* level, per axis.

        The argmax of an ordinal distribution is not its centre, and on a
        signed axis that difference has a sign. Driving a straight, the
        policy is nearly indifferent between turning a little left and a
        little right: the expectation sits at zero -- hold the wheel -- while
        the mode hops between the neighbours either side of it, and the wheel
        changes direction every few ticks for a distribution that never
        actually moved. Measured on the straights of this track, the greedy
        policy crosses centre 4.3 to 8.7 times a second doing exactly that.

        Only meaningful because the levels are ordinal and evenly spaced,
        which is what `self.levels` is. For an unordered action set the mean
        of two actions means nothing and this would be wrong.
        """
        z = self.theta @ h + self.bias
        idx = []
        for sl, lv in zip(self.slices, self.levels):
            p = softmax(z[sl])
            idx.append(int(np.argmin(np.abs(lv - float(p @ lv)))))
        return self.flat(idx)


class OrdinalGaussianHead:
    """A location and a width over *ordered* levels, instead of free logits.

    The softmax head treats the 21 steering levels as unordered categories:
    level 10 and level 11 share no parameters, so "a little left" is not near
    "straight" in the model and nothing makes the distribution local.
    Measured on this car, it does not stay local -- the middle third of the
    rate axis, seven of the twenty-one levels, receives 2-6 % of the commands
    in every version tried, on the straights and in the corners alike. The
    policy is bang-bang in the rate, and that is why no cost on the output
    changed it: the prices were paid and the shape stayed. It is also why
    neither read-out of that distribution is any good -- its mode picks a
    side and saws, its expectation averages "hard left or hard right" into
    "barely turn" and cannot corner.

    So this head parameterises the distribution rather than the logits:

        mu    = tanh(w_mu . h + b)                    location, -1..1
        sigma = lo + (hi - lo) * sigmoid(w_s . h + b) width, same units
        p(j) proportional to exp(-(level_j - mu)^2 / 2 sigma^2)

    The width is squashed into its range rather than exponentiated and
    clipped: a clip has no gradient at the bound, so a policy that reached
    one could never come back, and the score function would stop matching
    the distribution exactly where it is pinned. Caught by the finite
    difference test, which is why it is written down here.

    Unimodal by construction, local by construction, and the quantity the
    gradient moves is where the wheel should go rather than the score of one
    level among twenty-one. The mode and the mean coincide, so there is no
    choice to get wrong at deployment. Two parameters an axis instead of 21.

    No entropy bonus. In this agent that term is added to the logit gradient
    and then multiplied by the TD error, so its sign follows `delta` -- it is
    not a bonus on entropy, and measured, cutting it tenfold made the policy
    *more* entropic. Here the spread is an explicit parameter with its own
    gradient, which is the honest way to have one.
    """

    def __init__(self, n_hidden: int, nvec, feedback: str = "random",
                 sticky_prob: float = 0.0, sigma_init: float = 0.35,
                 sigma_min: float = 0.04, sigma_max: float = 1.0,
                 rng: np.random.Generator | None = None, **_ignored):
        rng = rng or np.random.default_rng(0)
        self.nvec = tuple(int(x) for x in nvec)
        if any(x < 2 for x in self.nvec):
            raise ValueError(f"every axis needs >= 2 levels, got {self.nvec}")
        self.n = int(n_hidden)
        #: two parameters an axis: the location and the log width
        self.n_act = 2 * len(self.nvec)
        self.n_joint = int(np.prod(self.nvec))
        self.theta = np.zeros((self.n_act, self.n))
        self.bias = np.zeros(self.n_act)
        # start at the middle of the axis with a width of a few levels, so
        # the first actions explore locally rather than at the stops
        f = (float(sigma_init) - sigma_min) / (sigma_max - sigma_min)
        f = min(max(f, 1e-4), 1 - 1e-4)
        self.bias[1::2] = np.log(f / (1.0 - f))
        self.B = rng.normal(0, 1.0 / np.sqrt(self.n), (self.n_act, self.n))
        self.feedback = feedback
        self.sticky_prob = float(sticky_prob)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.prev_a: int | None = None
        self.levels = [np.linspace(-1.0, 1.0, k) for k in self.nvec]
        self.strides = np.ones(len(self.nvec), dtype=int)
        for i in range(len(self.nvec) - 2, -1, -1):
            self.strides[i] = self.strides[i + 1] * self.nvec[i + 1]
        # where each axis's probability vector sits in the stacked cache, so
        # the same figures and metrics read it the same way as the softmax
        self.slices, o = [], 0
        for k in self.nvec:
            self.slices.append(slice(o, o + k))
            o += k

    @property
    def n_params(self) -> int:
        return self.theta.size + self.bias.size

    def flat(self, idx) -> int:
        return int(np.dot(np.asarray(idx, dtype=int), self.strides))

    def unflat(self, a: int):
        a, out = int(a), []
        for st in self.strides:
            out.append(a // st)
            a = a % st
        return out

    def _params(self, h: np.ndarray):
        z = self.theta @ h + self.bias
        mu = np.tanh(z[0::2])
        g = 1.0 / (1.0 + np.exp(-z[1::2]))
        sig = self.sigma_min + (self.sigma_max - self.sigma_min) * g
        return mu, sig, g

    def _pmf(self, mu, sig):
        pi = np.empty(sum(self.nvec))
        for i, (sl, lv) in enumerate(zip(self.slices, self.levels)):
            pi[sl] = softmax(-0.5 * ((lv - mu[i]) / sig[i]) ** 2)
        return pi

    def act(self, h: np.ndarray, rng: np.random.Generator):
        mu, sig, g = self._params(h)
        pi = self._pmf(mu, sig)
        idx = []
        for sl in self.slices:
            j = int(np.searchsorted(np.cumsum(pi[sl]), rng.random()))
            idx.append(min(j, sl.stop - sl.start - 1))
        a = self.flat(idx)
        if (self.sticky_prob > 0.0 and self.prev_a is not None
                and rng.random() < self.sticky_prob):
            a = self.prev_a
        self.prev_a = a
        return a, (pi, mu, sig, g)

    def reset(self) -> None:
        self.prev_a = None

    @staticmethod
    def _pi(cache):
        return cache[0] if isinstance(cache, tuple) else cache

    def means(self, cache) -> np.ndarray:
        pi = self._pi(cache)
        return np.array([float(pi[sl] @ lv)
                         for sl, lv in zip(self.slices, self.levels)])

    def grads(self, h: np.ndarray, a: int, cache):
        """``d log pi(a) / d(location, log width)``, stacked per axis.

        A softmax over logits ``-(level - mu)^2 / 2 sigma^2``, so the usual
        score function in terms of those two parameters, with the tanh
        derivative on the location.
        """
        pi, mu, sig, g = cache
        span = self.sigma_max - self.sigma_min
        dz = np.zeros(self.n_act)
        for i, (sl, lv) in enumerate(zip(self.slices, self.levels)):
            p, j = pi[sl], self.unflat(a)[i]
            d = (lv - mu[i]) / sig[i] ** 2
            dz[2 * i] = (d[j] - float(p @ d)) * (1.0 - mu[i] ** 2)
            q = ((lv - mu[i]) ** 2) / sig[i] ** 2
            # d logp / d sigma, then sigma's own derivative through the squash
            dz[2 * i + 1] = ((q[j] - float(p @ q)) / sig[i]
                             * span * g[i] * (1.0 - g[i]))
        back = self.B if self.feedback == "random" else self.theta
        return np.outer(dz, h), dz, back.T @ dz

    def apply(self, dtheta, dbias, lr: float) -> None:
        self.theta += lr * dtheta
        self.bias += lr * dbias

    def entropy(self, cache) -> float:
        pi = self._pi(cache)
        tot = 0.0
        for sl in self.slices:
            p = pi[sl]
            tot += float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        return tot

    def greedy(self, h: np.ndarray) -> int:
        mu, _sig, _g = self._params(h)
        return self.flat([int(np.argmin(np.abs(lv - mu[i])))
                          for i, lv in enumerate(self.levels)])

    #: the distribution is symmetric about `mu`, so its expectation is its
    #: mode up to the discretisation -- the two read-outs agree here, which
    #: is the point of having a location parameter at all
    mean = greedy


class GaussianHead:
    """Diagonal Gaussian policy: state-dependent mean, state-independent log-sigma.

    Keeping sigma out of the state is the usual choice for continuous control
    and it earns its place here: sigma then has no influence path through the
    recurrent cell at all, so it is a plain parameter with its own gradient,
    and the cell's learning signal stays a single vector.
    """

    def __init__(self, n_hidden: int, n_act: int, feedback: str = "random",
                 entropy_coef: float = 1e-5, log_sigma_init: float = -0.5,
                 low: float = -1.0, high: float = 1.0, sticky_prob: float = 0.0,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.n, self.n_act = int(n_hidden), int(n_act)
        self.theta = np.zeros((self.n_act, self.n))
        self.bias = np.zeros(self.n_act)
        self.log_sigma = np.full(self.n_act, float(log_sigma_init))
        self.B = rng.normal(0, 1.0 / np.sqrt(self.n), (self.n_act, self.n))
        self.feedback = feedback
        self.entropy_coef = float(entropy_coef)
        self.low, self.high = float(low), float(high)
        # See `CategoricalHead`'s docstring on `sticky_prob`. Here the "action"
        # the trace was scored on is `raw`, the pre-clip sample, not just an
        # index into `pi` -- so a sticky tick has to substitute both `a` and
        # `raw` together, or `grads` would score the discarded fresh draw
        # instead of the action actually executed.
        self.sticky_prob = float(sticky_prob)
        self.prev_a = None
        self.prev_raw = None

    @property
    def n_params(self) -> int:
        return self.theta.size + self.bias.size + self.log_sigma.size

    def act(self, h: np.ndarray, rng: np.random.Generator):
        mu = self.theta @ h + self.bias
        sigma = np.exp(self.log_sigma)
        raw = mu + sigma * rng.normal(size=self.n_act)
        a = np.clip(raw, self.low, self.high)
        if (self.sticky_prob > 0.0 and self.prev_a is not None
                and rng.random() < self.sticky_prob):
            a, raw = self.prev_a, self.prev_raw
        self.prev_a, self.prev_raw = a, raw
        return a, (mu, sigma, raw)

    def reset(self) -> None:
        """Clear the sticky-action memory at an episode boundary."""
        self.prev_a = None
        self.prev_raw = None

    def grads(self, h: np.ndarray, a, cache):
        """Gradients of ``log pi[a] + eta * H``, ``H = sum(log sigma) + const``.

        The action used is the *pre-clip* sample, not the clipped one that the
        environment saw. Differentiating the log-density at the clipped value
        would attribute the clipping to the policy and push the mean outwards
        forever.
        """
        mu, sigma, raw = cache
        dmu = (raw - mu) / sigma ** 2
        dlog_sigma = (raw - mu) ** 2 / sigma ** 2 - 1.0 + self.entropy_coef
        dtheta = np.outer(dmu, h)
        back = self.B if self.feedback == "random" else self.theta
        return dtheta, dmu, back.T @ dmu, dlog_sigma

    def apply(self, dtheta, dbias, dlog_sigma, lr: float) -> None:
        self.theta += lr * dtheta
        self.bias += lr * dbias
        self.log_sigma = np.clip(self.log_sigma + lr * dlog_sigma, -4.0, 1.0)

    def entropy(self, cache) -> float:
        _, sigma, _ = cache
        return float((np.log(sigma) + 0.5 * (LOG_2PI + 1)).sum())


class ValueHead:
    """Scalar linear critic ``v = theta . h + b``.

    Its gradient with respect to ``h`` is just ``theta`` -- or, under feedback
    alignment, a fixed random vector ``B``, which is the ``g_C <- B_C 1`` line
    of the paper's Algorithm 1 and looks stranger than it is: the critic's
    "error signal" is a constant direction, and the TD error that multiplies
    it later is what carries the actual information.

    The bias lives in ``theta`` as a weight on a constant feature, so the
    critic is one flat parameter vector. That is not tidiness for its own
    sake: the Dutch trace has a term ``(e . x) x`` that couples every feature
    to every other, so bias and weights have to be traced *together* or the
    trace is not the one the true-online result is about.
    """

    def __init__(self, n_hidden: int, feedback: str = "random",
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.n = int(n_hidden)
        self.theta = np.zeros(self.n + 1)  # last entry is the bias
        self.B = rng.normal(0, 1.0 / np.sqrt(self.n), self.n)
        self.feedback = feedback

    @property
    def n_params(self) -> int:
        return self.theta.size

    @staticmethod
    def features(h: np.ndarray) -> np.ndarray:
        return np.concatenate([h, [1.0]])

    def value(self, h: np.ndarray) -> float:
        return float(self.theta[:-1] @ h + self.theta[-1])

    def back(self) -> np.ndarray:
        return self.B if self.feedback == "random" else self.theta[:-1]
