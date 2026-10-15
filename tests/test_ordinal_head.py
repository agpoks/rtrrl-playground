"""A location and a width over ordered levels, instead of 21 free logits.

The softmax head treats the steering levels as unordered categories, and
measured on the car it never learns the order: the middle third of the rate
axis -- seven of twenty-one levels -- takes 2-6 % of the commands in every
version tried, in the corners as well as on the straights. A bang-bang
distribution has no good deterministic read-out: its mode picks a side and
saws, its expectation averages "hard left or hard right" into "barely turn".

`OrdinalGaussianHead` fixes the shape rather than pricing its consequences.
These tests pin the three properties that make it worth having: the
distribution is unimodal and local, the mode and the mean agree, and the
score function is the true gradient of log pi(a).
"""

import numpy as np
import pytest

from rtrrl_playground.nets.heads import OrdinalGaussianHead

NVEC = (21, 21)


def _head(seed=0, **kw):
    return OrdinalGaussianHead(8, NVEC, rng=np.random.default_rng(seed), **kw)


def test_the_distribution_is_unimodal_and_centred_on_the_location():
    h = _head()
    hid = np.zeros(h.n)
    h.bias[0] = np.arctanh(0.5)            # location at half lock
    pi, mu, sig, _g = h.act(hid, np.random.default_rng(0))[1]
    p = pi[h.slices[0]]
    assert mu[0] == pytest.approx(0.5)
    # one mode, at the nearest level to mu, and monotone either side of it
    top = int(np.argmax(p))
    assert h.levels[0][top] == pytest.approx(0.5, abs=0.05)
    assert np.all(np.diff(p[:top + 1]) > 0)
    assert np.all(np.diff(p[top:]) < 0)


def test_most_of_the_mass_is_local():
    """The property the softmax lost: a moderate rate is representable."""
    h = _head()
    pi = h.act(np.zeros(h.n), np.random.default_rng(0))[1][0]
    p = pi[h.slices[0]]
    assert p[7:14].sum() > 0.6             # the middle third, measured 2-6 %
    assert p[0] + p[-1] < 0.02             # and the lock stops are not it


def test_the_mode_and_the_mean_agree():
    """So there is no read-out to get wrong at deployment."""
    h = _head()
    hid = np.zeros(h.n)
    for loc in (-0.8, -0.2, 0.0, 0.3, 0.9):
        h.bias[0] = np.arctanh(loc)
        assert h.unflat(h.greedy(hid))[0] == h.unflat(h.mean(hid))[0]


def test_the_score_function_is_the_true_gradient():
    rng = np.random.default_rng(3)
    h = _head(3)
    h.theta = rng.normal(0, 0.4, h.theta.shape)
    h.bias = rng.normal(0, 0.3, h.bias.shape)
    hid = rng.normal(0, 1, h.n)

    def logp(bias, a):
        g = _head(3)
        g.theta, g.bias = h.theta, bias
        mu, sig, _g = g._params(hid)
        pi = g._pmf(mu, sig)
        return sum(np.log(pi[sl][g.unflat(a)[i]])
                   for i, sl in enumerate(g.slices))

    a, cache = h.act(hid, rng)
    _dt, dz, _g = h.grads(hid, a, cache)
    eps, num = 1e-6, np.zeros_like(dz)
    for k in range(len(dz)):
        b1, b2 = h.bias.copy(), h.bias.copy()
        b1[k] += eps
        b2[k] -= eps
        num[k] = (logp(b1, a) - logp(b2, a)) / (2 * eps)
    assert np.allclose(dz, num, atol=1e-5)


def test_the_width_is_bounded_both_ways():
    """An unbounded width is a uniform policy; a zero one has no gradient.

    Squashed into the range rather than clipped at it, so the gradient
    survives at the ends -- a clipped policy that reached a bound could
    never come back."""
    h = _head(sigma_min=0.05, sigma_max=0.5)
    hid = np.zeros(h.n)
    for b, want in ((-20.0, 0.05), (20.0, 0.5)):
        h.bias[1] = b
        assert h._params(hid)[1][0] == pytest.approx(want)


def test_it_is_two_parameters_an_axis():
    h = _head()
    assert h.n_act == 2 * len(NVEC)
    assert h.n_params == h.n_act * (h.n + 1)
    # against 21 logits an axis plus their biases
    from rtrrl_playground.nets.heads import FactoredCategoricalHead
    soft = FactoredCategoricalHead(h.n, NVEC, rng=np.random.default_rng(0))
    assert h.n_params * 10 < soft.n_params


def test_caps_smoothing_is_refused_rather_than_ignored():
    from algos.rtrrl.algo import RTRRL
    from rtrrl_playground.spaces import Discrete
    with pytest.raises(ValueError, match="no meaning for head='ordinal'"):
        RTRRL(obs_dim=6, action_space=Discrete(441, nvec=NVEC), n_hidden=8,
              cell="ctrnn", seed=0, head="ordinal", smooth_t=2.0)


def test_the_agent_learns_with_it_end_to_end():
    from algos.rtrrl.algo import RTRRL
    from rtrrl_playground.spaces import Discrete
    rng = np.random.default_rng(0)
    ag = RTRRL(obs_dim=6, action_space=Discrete(441, nvec=NVEC), n_hidden=16,
               cell="ctrnn", seed=0, head="ordinal")
    before = ag.actor.theta.copy()
    ag.start(rng.normal(0, 1, 6))
    for t in range(100):
        ag.step(rng.normal(0, 1, 6), 1.0, False, False)
    assert np.any(ag.actor.theta != before)
    assert np.isfinite(ag.actor.theta).all()
