"""Tests for the predictive safety filter.

The first two are the ones that matter. A safety filter has exactly one job,
and it is falsifiable: put the worst possible policy behind it and see whether
the constraint is ever violated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env
from rtrrl_playground.envs.scripted import WallFollower
from rtrrl_playground.safety import PredictiveSafetyFilter, make_safe
from rtrrl_playground.train import rollout
from rtrrl_playground.utils.load import load_algo


def _drive(env, filt, policy, n_ep=8, seed0=100):
    off, iv_steps, rets = 0, 0, []
    for ep in range(n_ep):
        obs = env.reset(seed=seed0 + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        R, info = 0.0, {}
        for _ in range(env.max_steps):
            a = policy(obs)
            if filt is not None:
                a, iv = filt(np.array([env.x, env.y, env.psi, env.v, env.delta]), a)
                iv_steps += int(iv)
            obs, r, te, tr, info = env.step(a)
            R += r
            if te or tr:
                break
        off += bool(info.get("off_track"))
        rets.append(R)
    return off, float(np.mean(rets)), iv_steps


def test_filter_keeps_a_random_policy_on_the_track():
    """The whole point, on the worst policy available."""
    env = make_env("lanekeep")
    rng = np.random.default_rng(0)
    policy = lambda obs: int(rng.integers(9))  # noqa: E731

    off_unfiltered, _, _ = _drive(env, None, policy)
    assert off_unfiltered > 0, "a random policy that never crashes means the task is broken"

    filt = PredictiveSafetyFilter(env.track, dt=env.dt)
    off_filtered, _, _ = _drive(env, filt, policy)
    assert off_filtered == 0, f"the filter let {off_filtered} episodes off the track"


def test_filter_is_invisible_to_a_competent_policy():
    """A filter that constantly overrides a good driver is a controller, not a filter."""
    env = make_env("lanekeep")
    filt = PredictiveSafetyFilter(env.track, dt=env.dt)
    _off, ret_filtered, iv = _drive(env, filt, WallFollower())
    _off2, ret_plain, _ = _drive(env, None, WallFollower())
    assert filt.intervention_rate < 0.05, f"intervened on {filt.intervention_rate:.0%} of steps"
    assert ret_filtered > 0.9 * ret_plain, f"{ret_filtered:.0f} vs {ret_plain:.0f} unfiltered"


def test_scalar_and_vectorised_certificates_agree():
    """The scalar fast path must decide exactly what the batched one decides."""
    env = make_env("lanekeep")
    filt = PredictiveSafetyFilter(env.track, dt=env.dt)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    checked = 0
    for _ in range(60):
        state = np.array([env.x, env.y, env.psi, env.v, env.delta])
        s0 = np.repeat(state[None, :], 9, axis=0)
        batched = filt._certify(filt._first.step(s0, filt._grid[:, 0], filt._grid[:, 1]), None)
        for a in range(9):
            assert bool(batched[a]) == filt._certify_scalar(state, a, None), \
                f"disagreement on action {a} at state {state}"
            checked += 1
        _o, _r, te, tr, _i = env.step(int(rng.integers(9)))
        if te or tr:
            env.reset()
    assert checked > 400


def test_optimistic_grip_assumption_is_not_safe():
    """Assuming more grip than the tyres have breaks the guarantee.

    Not a bug -- the certificate is only as good as the model, and this is the
    test that says so out loud rather than leaving it as a caveat in prose.
    """
    env = make_env("lanekeep", grip_range=(0.6, 0.6))
    rng = np.random.default_rng(1)
    policy = lambda obs: int(rng.integers(9))  # noqa: E731
    honest = PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=0.6)
    off_honest, _, _ = _drive(env, honest, policy, n_ep=10)
    assert off_honest == 0, "a filter that knows the grip should be sound"

    rng = np.random.default_rng(1)
    optimistic = PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=1.4)
    off_optimistic, _, _ = _drive(env, optimistic, policy, n_ep=10)
    assert off_optimistic >= off_honest


def test_safe_agent_wraps_an_agent_and_trains():
    from rtrrl_playground.train import train

    for credit in ("executed", "proposed"):
        env = make_env("lanekeep")
        agent = load_algo("rtrrl")(env.obs_dim, env.action_space, seed=0)
        safe = make_safe(agent, env, credit=credit)
        out = train(env, safe, 400, progress=False, seed=0)
        assert out["steps"] == 400
        assert "filtered" in safe.stats
        ev = rollout(env, safe.eval_policy(), n_episodes=2, seed=1)
        assert len(ev["returns"]) == 2


def test_filter_respects_obstacles():
    """On overtake the certificate has to include the other cars."""
    env = make_env("overtake")
    env.reset(seed=0)
    filt = PredictiveSafetyFilter(env.track, dt=env.dt)
    state = np.array([env.x, env.y, env.psi, env.v, env.delta])
    # A wall of obstacles right in front must make full-throttle-straight unsafe.
    ahead = np.array([[env.x + d * np.cos(env.psi), env.y + d * np.sin(env.psi)]
                      for d in (0.6, 0.9, 1.2)])
    assert not filt._certify_scalar(state, 5, ahead), "drove through a car"


# --- control barrier functions ---------------------------------------------

def test_cbf_keeps_a_random_policy_on_the_track():
    """Same falsifiable job as the predictive filter, different criterion."""
    from rtrrl_playground.cbf import DiscreteCBFFilter

    env = make_env("lanekeep")
    rng = np.random.default_rng(0)
    policy = lambda obs: int(rng.integers(9))  # noqa: E731

    filt = DiscreteCBFFilter(env.track, dt=env.dt, h_kind="braking")
    off, _ret, _iv = _drive(env, filt, policy, n_ep=8)
    assert off == 0, f"the CBF let {off} episodes off the track"


def test_naive_positional_barrier_is_myopic():
    """h = w - |d| cannot see a braking distance, and this asserts it.

    Not a bug in the method -- a statement about the barrier, and the reason
    ``h_kind="braking"`` exists. If this test ever passes with zero crashes the
    demonstration is gone and the docs claim needs re-checking.
    """
    from rtrrl_playground.cbf import DiscreteCBFFilter

    env = make_env("lanekeep")
    rng = np.random.default_rng(0)
    policy = lambda obs: int(rng.integers(9))  # noqa: E731

    naive = DiscreteCBFFilter(env.track, dt=env.dt, h_kind="lateral")
    off_naive, _r, _i = _drive(env, naive, policy, n_ep=10)

    rng = np.random.default_rng(0)
    good = DiscreteCBFFilter(env.track, dt=env.dt, h_kind="braking")
    off_good, _r, _i = _drive(env, good, policy, n_ep=10)

    assert off_naive > off_good, (
        f"the naive barrier crashed {off_naive} times and the closing-rate one "
        f"{off_good}: the myopia this pair exists to demonstrate is not showing")


def test_cbf_is_forward_invariant_on_its_own_condition():
    """Whenever the filter accepts an action, the CBF inequality must hold."""
    from rtrrl_playground.cbf import DiscreteCBFFilter

    env = make_env("lanekeep")
    filt = DiscreteCBFFilter(env.track, dt=env.dt, h_kind="braking")
    rng = np.random.default_rng(3)
    obs = env.reset(seed=0)
    checked = 0
    for _ in range(600):
        state = np.array([env.x, env.y, env.psi, env.v, env.delta])
        h_now = float(filt.h(state)[0])
        a, _iv = filt(state, int(rng.integers(9)))
        nxt = filt.model.step(state[None, :], filt._grid[a, 0:1], filt._grid[a, 1:2])
        h_next = float(filt.h(nxt)[0])
        if h_now > 0 and filt.n_no_safe_action == 0:
            assert h_next >= (1 - filt.alpha) * h_now - 1e-9, (
                f"accepted an action violating the CBF condition: "
                f"h {h_now:.4f} -> {h_next:.4f}")
            checked += 1
        obs, r, te, tr, _i = env.step(a)
        if te or tr:
            env.reset()
    assert checked > 100


def test_cbf_wraps_an_agent_and_trains():
    from rtrrl_playground.cbf import make_safe_cbf
    from rtrrl_playground.train import train

    env = make_env("lanekeep")
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, seed=0)
    safe = make_safe_cbf(agent, env)
    out = train(env, safe, 400, progress=False, seed=0)
    assert out["steps"] == 400
    ev = rollout(env, safe.eval_policy(), n_episodes=2, seed=1)
    assert len(ev["returns"]) == 2


def test_cbf_respects_obstacles():
    from rtrrl_playground.cbf import DiscreteCBFFilter

    env = make_env("overtake")
    env.reset(seed=0)
    filt = DiscreteCBFFilter(env.track, dt=env.dt, h_kind="braking")
    state = np.array([env.x, env.y, env.psi, env.v, env.delta])
    ahead = np.array([[env.x + d * np.cos(env.psi), env.y + d * np.sin(env.psi)]
                      for d in (0.3, 0.5)])
    assert filt.h(state, ahead)[0] < filt.h(state, None)[0], \
        "a car right in front did not lower the barrier"
