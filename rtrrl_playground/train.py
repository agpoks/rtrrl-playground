"""The training loop -- one function, shared by every algorithm here.

There is deliberately only one, and it is deliberately this small. Online
reinforcement learning is a loop with a step in it; the moment a repo grows a
Trainer class with callbacks, the thing a reader came to understand is
distributed across four files. The agent interface it expects is two methods::

    action = agent.start(obs)                                   # new episode
    action = agent.step(obs, reward, terminated, truncated)     # or None at the end

``rollout`` is the same loop with learning switched off, for evaluation.

The one subtlety is the episode boundary. ``terminated`` (the pole fell, the
car left the track) and ``truncated`` (we hit the step limit) both end an
episode, but only the first means the return really stopped -- so they are
passed through separately and it is the *agent's* business what to bootstrap
from. Collapsing them into one ``done`` flag is the most common silent bug in
implementations of this loop, and it teaches the agent that the world ends
whenever the harness gets bored.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


def train(env, agent, total_steps: int, log_every: int = 5000, seed: int = 0,
          window: int = 20, progress: bool = True, on_episode=None):
    """Run ``total_steps`` environment steps of online learning.

    Returns a dict with the per-episode returns, lengths, the wall-clock time
    and whatever the agent last reported in ``agent.stats``.
    """
    obs = env.reset(seed=seed)
    action = agent.start(obs)
    returns: list[float] = []
    lengths: list[int] = []
    infos: list[dict] = []
    recent: deque[float] = deque(maxlen=window)
    ep_return, ep_len = 0.0, 0
    t0 = time.perf_counter()
    curve: list[tuple[int, float]] = []

    for t in range(1, total_steps + 1):
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_len += 1
        action = agent.step(obs, reward, terminated, truncated)
        if action is None:  # the agent reports the episode is over
            returns.append(ep_return)
            lengths.append(ep_len)
            infos.append(dict(info))
            recent.append(ep_return)
            if on_episode is not None:
                on_episode(len(returns), ep_return, info)
            ep_return, ep_len = 0.0, 0
            obs = env.reset()
            action = agent.start(obs)
        if progress and t % log_every == 0:
            mean = float(np.mean(recent)) if recent else float("nan")
            curve.append((t, mean))
            stats = getattr(agent, "stats", {})
            extra = " ".join(f"{k}={v:+.3f}" for k, v in stats.items())
            print(f"  step {t:>8,}  episodes {len(returns):>5}  "
                  f"return({window}-ep mean) {mean:8.2f}   {extra}", flush=True)
        elif t % log_every == 0:
            curve.append((t, float(np.mean(recent)) if recent else float("nan")))

    return {
        "returns": np.array(returns),
        "lengths": np.array(lengths),
        "infos": infos,
        "curve": curve,
        "train_time_s": time.perf_counter() - t0,
        "steps": total_steps,
    }


def rollout(env, policy, n_episodes: int = 10, seed: int = 0, keep_history: bool = False):
    """Evaluate ``policy`` for ``n_episodes`` without learning.

    ``policy`` is anything callable on an observation, with two optional hooks
    that this loop calls if they exist:

    ``policy.reset()``
        at every episode boundary. Forgetting it is how a recurrent policy ends
        up evaluated with a hidden state left over from the previous episode.
    ``policy.observe(reward)``
        after every step. A meta-RL agent takes the previous reward as an
        *input*; evaluating it with that input pinned to zero is evaluating a
        different network.

    ``agent.eval_policy()`` returns an object with both. The scripted policies
    in ``envs/scripted.py`` have ``reset`` and do not need ``observe``.
    """
    returns, lengths, infos, history = [], [], [], None
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        R, n, info = 0.0, 0, {}
        for _ in range(env.max_steps):
            obs, r, terminated, truncated, info = env.step(policy(obs))
            if hasattr(policy, "observe"):
                policy.observe(r)
            R += r
            n += 1
            if terminated or truncated:
                break
        returns.append(R)
        lengths.append(n)
        infos.append(dict(info))
        if keep_history and ep == 0:
            history = list(getattr(env, "history", []))
    return {"returns": np.array(returns), "lengths": np.array(lengths),
            "infos": infos, "history": history}


def result_line(model: str, metric_name: str, metric: float, params: int,
                train_time_s: float, **extra) -> str:
    """The one line every example prints, and the benchmark runner parses.

    Same convention as the other playgrounds in this family, so the same kind
    of comparison table falls out.
    """
    tail = "".join(f" {k}={v}" for k, v in extra.items())
    return (f"RESULT: model={model} metric_name={metric_name} metric={metric:.4f} "
            f"params={params} train_time_s={train_time_s:.2f}{tail}")
