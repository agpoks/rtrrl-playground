"""The tests that matter: the hand-derived gradients are actually the gradients.

Everything else in this repo is a design choice you can argue with. These are
not -- an influence array or a policy gradient is either the derivative of the
thing it claims to be or it is a bug that trains slowly and blames the learning
rate. Each test below checks one against finite differences.

    pip install -e ".[dev]" && pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground.nets import CELLS, make_cell
from rtrrl_playground.nets.heads import CategoricalHead, GaussianHead, softmax

EPS = 1e-6
CELL_NAMES = sorted(CELLS)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_rtrl_influence_matches_finite_differences(cell_name):
    """dh_T/dtheta, carried forwards, equals the numerical derivative."""
    n_in, n, T = 3, 5, 7
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(T, n_in))

    def final_state(theta, estimator="none"):
        c = make_cell(cell_name, n_in, n, estimator=estimator,
                      rng=np.random.default_rng(0))
        c.theta = theta.copy()
        c.reset_state()
        for x in xs:
            c.step(x)
        return c

    ref = make_cell(cell_name, n_in, n, estimator="rtrl", rng=np.random.default_rng(0))
    theta0 = ref.theta.copy()
    run = final_state(theta0, "rtrl")
    J = run.J.reshape(n, n, run.p)

    worst = 0.0
    for k in range(n):
        for j in range(run.p):
            tp, tm = theta0.copy(), theta0.copy()
            tp[k, j] += EPS
            tm[k, j] -= EPS
            fd = (final_state(tp).h - final_state(tm).h) / (2 * EPS)
            worst = max(worst, float(np.abs(fd - J[:, k, j]).max()))
    assert worst < 1e-6, f"{cell_name}: worst error {worst:.2e}"


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_rflo_is_the_diagonal_of_rtrl_up_to_the_recurrent_term(cell_name):
    """RFLO should be a *approximation*, not a different quantity.

    After one step there is no history to disagree about, so RFLO's influence
    must equal the exact diagonal blocks exactly. If that fails, the two are
    computing different things and any later comparison is meaningless.
    """
    n_in, n = 4, 6
    x = np.random.default_rng(2).normal(size=n_in)
    exact = make_cell(cell_name, n_in, n, estimator="rtrl", rng=np.random.default_rng(0))
    approx = make_cell(cell_name, n_in, n, estimator="rflo", rng=np.random.default_rng(0))
    exact.step(x)
    approx.step(x)
    diag = exact.J.reshape(n, n, exact.p)[np.arange(n), np.arange(n)]
    assert np.allclose(diag, approx.P, atol=1e-12)


def test_uoro_is_unbiased():
    """UORO's expectation is the exact gradient; one sample is not.

    Averaging many samples has to converge on RTRL. This is the only property
    UORO claims, and it is the one worth testing.
    """
    n_in, n, T = 3, 6, 12
    rng = np.random.default_rng(3)
    xs = rng.normal(size=(T, n_in))
    g = rng.normal(size=n)

    def grad(estimator, seed):
        c = make_cell("ctrnn", n_in, n, estimator=estimator, rng=np.random.default_rng(0))
        if estimator == "uoro":  # a fresh stream of random directions per sample
            c.rng = np.random.default_rng(seed)
        c.reset_state()
        for x in xs:
            c.step(x)
        return c.grad(g)

    exact = grad("rtrl", 0)
    mean = np.mean([grad("uoro", 100 + s) for s in range(400)], axis=0)
    cos = float((mean * exact).sum() / (np.linalg.norm(mean) * np.linalg.norm(exact)))
    assert cos > 0.8, f"averaged UORO cosine to exact was only {cos:.3f}"


def test_categorical_head_gradient():
    """d[log pi(a) + eta H(pi)]/dz, including the entropy term."""
    rng = np.random.default_rng(4)
    head = CategoricalHead(6, 4, entropy_coef=0.3, rng=rng)
    head.theta = rng.normal(size=(4, 6))
    head.bias = rng.normal(size=4)
    h = rng.normal(size=6)
    a = 2
    z = head.theta @ h + head.bias
    _, dz, _ = head.grads(h, a, softmax(z))

    def objective(zz):
        p = softmax(zz)
        return float(np.log(p[a]) - 0.3 * (p * np.log(p)).sum())

    fd = np.array([(objective(z + EPS * np.eye(4)[i]) - objective(z - EPS * np.eye(4)[i]))
                   / (2 * EPS) for i in range(4)])
    assert np.abs(dz - fd).max() < 1e-6


def test_gaussian_head_gradient():
    """d log pi(a)/dmu and d log pi(a)/dlog_sigma."""
    rng = np.random.default_rng(5)
    head = GaussianHead(6, 2, entropy_coef=0.0, rng=rng)
    head.theta = rng.normal(size=(2, 6))
    head.bias = rng.normal(size=2)
    h = rng.normal(size=6)
    mu = head.theta @ h + head.bias
    sigma = np.exp(head.log_sigma)
    raw = mu + sigma * np.array([0.3, -0.7])
    _, dmu, _, dls = head.grads(h, raw, (mu, sigma, raw))

    def logp(mu_, log_sigma_):
        s = np.exp(log_sigma_)
        return float((-(raw - mu_) ** 2 / (2 * s ** 2) - log_sigma_
                      - 0.5 * np.log(2 * np.pi)).sum())

    for vec, analytic, wrt in ((mu, dmu, "mu"), (head.log_sigma, dls, "log_sigma")):
        fd = []
        for i in range(2):
            e = EPS * np.eye(2)[i]
            up = logp(mu + e, head.log_sigma) if wrt == "mu" else logp(mu, head.log_sigma + e)
            dn = logp(mu - e, head.log_sigma) if wrt == "mu" else logp(mu, head.log_sigma - e)
            fd.append((up - dn) / (2 * EPS))
        assert np.abs(analytic - np.array(fd)).max() < 1e-6, wrt


def test_liquid_gru_leak_is_bounded_below_one_by_construction():
    """The reason this cell exists, asserted rather than described.

    RFLO's influence is a geometric series ``P <- leak * P + immediate``, which
    converges only while ``leak < 1``. A LiGRU whose update gate saturates has
    ``leak = z = 1`` -- a unit that has learned to remember perfectly, and an
    influence sum that never decays. ``OnlineCell`` patches that with an
    arbitrary ``leak_max``; LiquidGRU does not need it, because its leak is
    ``1/(1 + dt(1/tau + z)) <= 1/(1 + dt/tau) < 1`` for *any* gate value.

    So this test runs with the cap switched off (``leak_max=1.0``) and drives
    the gates as far open as it can.
    """
    n_in, n = 6, 24
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(300, n_in)) * 3.0

    def max_leak(cell_name, gate_push):
        cell = make_cell(cell_name, n_in, n, estimator="rflo", leak_max=1.0,
                         rng=np.random.default_rng(0))
        cell.theta[:, :cell.n_xi] += gate_push  # saturate the first gate block
        cell.reset_state()
        worst = 0.0
        for x in xs:
            xi = np.concatenate([x, cell.h, [1.0]])
            _h, _imm, leak, _D = cell._forward(xi, False)
            worst = max(worst, float(leak.max()))
            cell.step(x)
        return worst, float(np.abs(cell.P).max())

    ligru_leak, _ = max_leak("ligru", 8.0)
    lgru_leak, lgru_p = max_leak("liquid_gru", 8.0)

    assert ligru_leak > 0.999, (
        "a saturated LiGRU gate should reach leak = 1 -- if it does not, this "
        "test is no longer demonstrating the problem it was written for")
    tau_max = make_cell("liquid_gru", n_in, n).tau_max
    bound = 1.0 / (1.0 + 1.0 / tau_max)
    assert lgru_leak <= bound + 1e-9, f"leak {lgru_leak} exceeded the bound {bound}"
    assert np.isfinite(lgru_p), "the influence array went non-finite despite the bound"
