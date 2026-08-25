"""Contract tests: the environments keep their promises and every agent runs.

Cheap, and they catch the class of bug that is otherwise found three hundred
thousand steps into a training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import ENV_IDS, make_env
from rtrrl_playground.envs.scripted import SCRIPTED
from rtrrl_playground.nets import CELLS, ESTIMATORS
from rtrrl_playground.train import rollout, train
from rtrrl_playground.utils.load import load_algo


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_contract(env_id):
    env = make_env(env_id)
    obs = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    rng = np.random.default_rng(0)
    saw_terminal = False
    for _ in range(2000):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample(rng))
        assert obs.shape == (env.obs_dim,)
        assert np.isfinite(obs).all()
        assert np.isfinite(r)
        assert not (terminated and truncated), "an episode cannot both end and be cut short"
        if terminated or truncated:
            saw_terminal = True
            obs = env.reset()
    assert saw_terminal, "no episode ended in 2000 steps -- max_steps is not being applied"


def test_reset_is_deterministic_given_a_seed():
    for env_id in ENV_IDS:
        env = make_env(env_id)
        assert np.allclose(env.reset(seed=7), env.reset(seed=7))


def test_lap_progress_is_continuous_across_the_start_line():
    """The reward must not spike by a lap length when s wraps to zero."""
    env = make_env("lanekeep")
    env.reset(seed=0)
    rewards = []
    for _ in range(2000):
        _o, r, te, tr, _i = env.step(4)
        rewards.append(r)
        if te or tr:
            env.reset()
    assert max(rewards) < 2.0, "a reward spike this large means a lap wrap leaked in"


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("estimator", list(ESTIMATORS))
def test_every_cell_and_estimator_runs(cell, estimator):
    env = make_env("cartpole-vel")
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, cell=cell,
                               estimator=estimator, seed=0)
    out = train(env, agent, 400, progress=False, seed=0)
    assert np.isfinite(agent.cell.h).all()
    assert out["steps"] == 400


@pytest.mark.parametrize("algo", ["rtrrl", "ac_lambda"])
def test_agents_train_and_evaluate(algo):
    env = make_env("lanekeep")
    agent = load_algo(algo)(env.obs_dim, env.action_space, seed=0)
    train(env, agent, 400, progress=False, seed=0)
    ev = rollout(env, agent.eval_policy(), n_episodes=2, seed=1)
    assert len(ev["returns"]) == 2


def test_scripted_policies_beat_random():
    rng = np.random.default_rng(0)
    for env_id, cls in SCRIPTED.items():
        env = make_env(env_id)
        scripted = rollout(env, cls(), n_episodes=5, seed=3)["returns"].mean()
        random = rollout(env, lambda o: int(rng.integers(9)), n_episodes=5, seed=3)["returns"].mean()
        assert scripted > 2 * random, f"{env_id}: scripted {scripted:.0f} vs random {random:.0f}"


def test_imitation_moves_the_policy_towards_the_expert():
    env = make_env("overtake")
    expert = SCRIPTED["overtake"]()
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, seed=0)
    obs = env.reset(seed=0)
    agent.start(obs)

    def agreement(n):
        matched = 0
        for _ in range(n):
            nonlocal obs
            a_expert = expert(obs)
            matched += int(agent.a == a_expert)
            obs, r, te, tr, _ = env.step(a_expert)
            if agent.imitate(obs, a_expert, r, te, tr, lr=3e-2) is None:
                obs = env.reset()
                expert.reset()
                agent.start(obs)
        return matched / n

    before = agreement(500)
    for _ in range(6):
        agreement(500)
    after = agreement(500)
    assert after > before, f"cloning did not improve agreement: {before:.2f} -> {after:.2f}"
