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


def test_physics_ligru_only_activates_where_the_action_means_steer_and_throttle():
    """It is a specialisation for the driving tasks and must refuse elsewhere.

    An earlier version checked only that an action block fitted inside the
    input and switched itself on for MemoryChain, decoding a two-action space
    into a steering command that does not exist.
    """
    from rtrrl_playground import make_env
    from rtrrl_playground.utils.load import load_algo

    RTRRL = load_algo("rtrrl")
    expected = {"lanekeep": 3, "overtake": 3, "memory-chain": 0, "cartpole-vel": 0}
    for env_id, want in expected.items():
        env = make_env(env_id)
        agent = RTRRL(env.obs_dim, env.action_space, cell="physics_ligru", seed=0)
        assert agent.cell.n_phys == want, f"{env_id}: n_phys {agent.cell.n_phys}, wanted {want}"


def test_physics_units_dead_reckon_the_hidden_state():
    """Untrained, the reserved units should already track what the beams hide.

    That is the entire claim of the cell: the agent knows what it commanded, and
    the response to a command is partly known in advance, so the hidden speed
    and steering angle do not have to be learned from lidar.
    """
    from rtrrl_playground import make_env
    from rtrrl_playground.utils.load import load_algo

    env = make_env("lanekeep", grip_range=(1.0, 1.0))
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, cell="physics_ligru", seed=0)
    assert agent.cell.n_phys == 3

    obs = env.reset(seed=0)
    a = agent.start(obs)
    v_true, v_hat, d_true, d_hat = [], [], [], []
    for _ in range(400):
        obs, r, te, tr, _i = env.step(a)
        a = agent.step(obs, r, te, tr)
        v_true.append(env.v)
        d_true.append(env.delta)
        v_hat.append(agent.cell.h[0])
        d_hat.append(agent.cell.h[1])
        if a is None:
            a = agent.start(env.reset())

    r_v = float(np.corrcoef(v_true, v_hat)[0, 1])
    r_d = float(np.corrcoef(d_true, d_hat)[0, 1])
    assert r_v > 0.6, f"speed unit tracked the hidden speed at only r={r_v:.2f}"
    assert r_d > 0.9, f"steering unit tracked the true steering angle at only r={r_d:.2f}"


def test_physics_leak_is_bounded_below_one():
    """Same structural property as liquid_gru: leak = 1 - sigmoid(raw) < 1."""
    cell = make_cell("physics_ligru", 19, 8, estimator="rflo", n_obs=9, n_act=9,
                     leak_max=1.0, rng=np.random.default_rng(0))
    cell.theta[:cell.n_phys, :4] = -20.0  # drive the rate constants to zero
    cell.reset_state()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        x = np.zeros(19)
        x[9 + rng.integers(9)] = 1.0
        xi = np.concatenate([x, cell.h, [1.0]])
        _h, _imm, leak, _D = cell._forward(xi, False)
        worst = max(worst, float(leak[:cell.n_phys].max()))
        cell.step(x)
    assert worst < 1.0, f"physics leak reached {worst}"


def test_hybrid_is_exact_for_the_known_block():
    """The claim ``hybrid`` makes, and the one that makes it not-an-approximation.

    Exact for the parameters of the known block; RFLO is not. If this ever
    stops holding to machine precision the estimator has quietly become another
    approximation and the docs overclaim.
    """
    def build(est):
        return make_cell("physics_ligru", 19, 8, estimator=est, n_obs=9, n_act=9,
                         rng=np.random.default_rng(0))

    rng = np.random.default_rng(1)

    def inp():
        x = np.zeros(19)
        x[:9] = rng.random(9)
        x[9 + rng.integers(9)] = 1.0
        return x

    xs = [inp() for _ in range(8)]
    cells = {e: build(e) for e in ("rtrl", "hybrid", "rflo")}
    for c in cells.values():
        c.reset_state()
    for x in xs:
        for c in cells.values():
            c.step(x)

    g = rng.normal(size=8)
    grads = {e: c.grad(g) for e, c in cells.items()}
    ne = cells["rtrl"].n_phys
    assert ne == 3

    scale = max(np.abs(grads["rtrl"][:ne]).max(), 1e-12)
    err_hybrid = np.abs(grads["hybrid"][:ne] - grads["rtrl"][:ne]).max() / scale
    err_rflo = np.abs(grads["rflo"][:ne] - grads["rtrl"][:ne]).max() / scale
    assert err_hybrid < 1e-10, f"hybrid was not exact on the known block: {err_hybrid:.2e}"
    assert err_rflo > 1e-3, (
        "RFLO agreed with exact RTRL on the known block, so this test is no "
        "longer demonstrating anything")


def test_hybrid_degrades_to_rflo_without_a_known_block():
    """A cell that declares no known block must get exactly RFLO."""
    rng = np.random.default_rng(2)
    xs = rng.normal(size=(6, 4))
    a = make_cell("ctrnn", 4, 6, estimator="hybrid", rng=np.random.default_rng(0))
    b = make_cell("ctrnn", 4, 6, estimator="rflo", rng=np.random.default_rng(0))
    assert a.exact_rows == 0
    for c in (a, b):
        c.reset_state()
    for x in xs:
        a.step(x)
        b.step(x)
    g = rng.normal(size=6)
    assert np.allclose(a.grad(g), b.grad(g))


def test_hybrid_costs_less_than_exact_rtrl():
    n, ne = 8, 3
    sizes = {}
    for est in ("rtrl", "hybrid", "rflo"):
        c = make_cell("physics_ligru", 19, n, estimator=est, n_obs=9, n_act=9,
                      rng=np.random.default_rng(0))
        sizes[est] = c.influence_bytes()
    assert sizes["rflo"] < sizes["hybrid"] < sizes["rtrl"], sizes


def test_spike_gating_defers_updates_without_losing_them():
    """Event-triggered learning: the update is deferred, not dropped.

    With a threshold the parameter write happens only when integrated |delta|
    crosses it, but the eligibility traces and the influence recursion still
    advance every tick -- so what is deferred is the write, not the
    information. A gated run must therefore touch the weights strictly less
    often than an ungated one, and still learn.
    """
    import numpy as np
    from rtrrl_playground import make_env
    from rtrrl_playground.train import train
    from rtrrl_playground.utils.load import load_algo

    counts = {}
    for thr in (0.0, 1.0):
        env = make_env("lanekeep", seed=0)
        ag = load_algo("rtrrl")(env.obs_dim, env.action_space, cell="ligru",
                                seed=0, spike_threshold=thr)
        train(env, ag, 3000, progress=False, seed=0)
        counts[thr] = ag.n_updates
        assert np.isfinite(ag.critic.theta).all()
    assert counts[1.0] < counts[0.0], "gating did not reduce the update count"
    assert counts[1.0] > 0, "gating suppressed every update"
