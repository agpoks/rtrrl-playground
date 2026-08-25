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
                 entropy_coef: float = 1e-5, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.n, self.n_act = int(n_hidden), int(n_act)
        self.theta = np.zeros((self.n_act, self.n))  # start uniform: no early bias
        self.bias = np.zeros(self.n_act)
        self.B = rng.normal(0, 1.0 / np.sqrt(self.n), (self.n_act, self.n))
        self.feedback = feedback
        self.entropy_coef = float(entropy_coef)

    @property
    def n_params(self) -> int:
        return self.theta.size + self.bias.size

    def act(self, h: np.ndarray, rng: np.random.Generator):
        pi = softmax(self.theta @ h + self.bias)
        a = int(rng.choice(self.n_act, p=pi))
        return a, pi

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


class GaussianHead:
    """Diagonal Gaussian policy: state-dependent mean, state-independent log-sigma.

    Keeping sigma out of the state is the usual choice for continuous control
    and it earns its place here: sigma then has no influence path through the
    recurrent cell at all, so it is a plain parameter with its own gradient,
    and the cell's learning signal stays a single vector.
    """

    def __init__(self, n_hidden: int, n_act: int, feedback: str = "random",
                 entropy_coef: float = 1e-5, log_sigma_init: float = -0.5,
                 low: float = -1.0, high: float = 1.0,
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

    @property
    def n_params(self) -> int:
        return self.theta.size + self.bias.size + self.log_sigma.size

    def act(self, h: np.ndarray, rng: np.random.Generator):
        mu = self.theta @ h + self.bias
        sigma = np.exp(self.log_sigma)
        a = mu + sigma * rng.normal(size=self.n_act)
        return np.clip(a, self.low, self.high), (mu, sigma, a)

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
