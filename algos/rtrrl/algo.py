"""RTRRL -- Real-Time Recurrent Reinforcement Learning.

Julian Lemmel, Radu Grosu, *"Real-Time Recurrent Reinforcement Learning"*,
AAAI 2025 (arXiv:2311.04830). See ``papers/README.md``.

Three pieces, each of which is separately old, and the paper's contribution
is that they compose into something that learns from a single stream of
experience with no replay, no batch, and no backward pass:

1. **A meta-RL recurrent architecture.** One CT-RNN whose input at each step
   is ``[observation, previous action, previous reward]`` and whose state
   feeds a linear actor and a linear critic. Feeding the last action and
   reward back in is what makes it "meta": the network can, in principle,
   read its own recent history of consequences out of its own input.
2. **TD(lambda) actor-critic with eligibility traces.** The outer learning
   rule. One scalar TD error per step, multiplied into a trace per parameter
   group, so a reward can credit actions taken many steps earlier without
   anything having been stored.
3. **RFLO.** The recurrent cell's own gradient, carried *forwards* -- see
   ``rtrrl_playground/nets/ctrnn.py``.

What that buys: memory and compute per step are constant in episode length,
the update is available *during* the episode rather than after it, and
nothing in the loop needs a copy of the past. What it costs: the gradient
is approximate twice over (RFLO drops most of the true Jacobian; feedback
alignment replaces the transpose with a random matrix), and the whole thing
runs at batch size one.

Timing is the part that is easy to get wrong, so, explicitly -- at step ``t``
the agent holds ``h_t`` and the influence ``J_t`` of ``h_t`` on the weights.
It draws ``a_t`` from ``h_t``, the environment answers with ``r_t`` and
``o_{t+1}``, and only then does the cell advance to ``h_{t+1}``. The TD error
compares ``v(h_{t+1})`` to ``v(h_t)``, but every *trace* is updated with the
quantities from time ``t``: ``h_t``, ``J_t``, and the policy gradient at the
action actually taken. Pair ``J_{t+1}`` with ``h_t`` and the credit is one
step out of alignment -- which does not crash, does not warn, and quietly
costs you most of the learning.

One deviation from Algorithm 1 as printed: the paper's loop draws its first
action from ``h = 0``, before the first observation has been fed in, so
``o_0`` is never seen. Here the cell consumes ``o_0`` first and acts from
``h_0``. Same algorithm, one fewer wasted step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rtrrl_playground.nets import make_cell  # noqa: E402
from rtrrl_playground.nets.heads import CategoricalHead, GaussianHead, ValueHead  # noqa: E402
from rtrrl_playground.spaces import Box, Discrete  # noqa: E402
from rtrrl_playground.traces import accumulating_trace, dutch_trace  # noqa: E402


def _clip(g: np.ndarray, max_norm: float) -> np.ndarray:
    """Global-norm clip that also refuses to pass a non-finite gradient on.

    Without the ``nan_to_num``, one inf turns the norm into inf, the scale into
    zero, and the whole array into NaN -- which then lives in an eligibility
    trace forever and takes every parameter with it. Silently zeroing is the
    wrong default in general; here the alternative is a run that produces
    NaN-shaped garbage a hundred thousand steps later.
    """
    if max_norm <= 0:
        return g
    if not np.isfinite(g).all():
        g = np.nan_to_num(g, nan=0.0, posinf=max_norm, neginf=-max_norm)
    n = float(np.linalg.norm(g))
    return g * (max_norm / n) if n > max_norm else g


class RTRRL:
    """The agent. One object, one environment, one update per timestep.

    The paper's Table 5 values are 32 units, gamma 0.99, all three lambdas
    0.9, alpha_A 1e-2, alpha_C 1.0, alpha_R 1e-3, entropy 1e-5. Four of those
    are not the defaults here, because on these environments they do not work
    -- measured, with the sweeps in ``benchmarks/``. See "Deviations from the
    paper" in ``algos/rtrrl/README.md``; every one of them is still reachable
    from a flag.
    """

    def __init__(self, obs_dim: int, action_space, n_hidden: int = 32,
                 gamma: float = 0.99, lam_actor: float = 0.9, lam_critic: float = 0.9,
                 lam_rnn: float = 0.9, lr_actor: float = 1e-3, lr_critic: float = 0.03,
                 lr_rnn: float = 1e-5, entropy_coef: float = 0.03,
                 cell: str = "ctrnn", estimator: str = "rflo", feedback: str = "random",
                 critic_update: str = "true-online", critic_lr_mode: str = "normalized",
                 meta_inputs: bool = True, clip: float = 1.0,
                 reward_scale: float = 1.0, seed: int = 0, **cell_kwargs):
        if critic_update not in ("true-online", "paper", "accumulating"):
            raise ValueError("critic_update must be 'true-online', 'paper' or 'accumulating'")
        if critic_lr_mode not in ("normalized", "fixed"):
            raise ValueError("critic_lr_mode must be 'normalized' or 'fixed'")
        self.rng = np.random.default_rng(seed)
        self.action_space = action_space
        self.meta_inputs = bool(meta_inputs)
        self.obs_dim = int(obs_dim)
        n_in = obs_dim + (action_space.flat_dim + 1 if meta_inputs else 0)

        self.cell = make_cell(cell, n_in, n_hidden, estimator=estimator,
                              rng=self.rng, **cell_kwargs)
        head_kw = dict(feedback=feedback, entropy_coef=entropy_coef, rng=self.rng)
        if isinstance(action_space, Discrete):
            self.actor = CategoricalHead(n_hidden, action_space.n, **head_kw)
        elif isinstance(action_space, Box):
            self.actor = GaussianHead(n_hidden, action_space.dim,
                                      low=action_space.low, high=action_space.high, **head_kw)
        else:
            raise TypeError(f"unsupported action space {action_space!r}")
        self.critic = ValueHead(n_hidden, feedback=feedback, rng=self.rng)

        self.gamma, self.clip, self.reward_scale = gamma, clip, reward_scale
        self.lam_actor, self.lam_critic, self.lam_rnn = lam_actor, lam_critic, lam_rnn
        self.lr_actor, self.lr_critic, self.lr_rnn = lr_actor, lr_critic, lr_rnn
        self.critic_update = critic_update
        self.critic_lr_mode = critic_lr_mode
        self.n_params = self.cell.n_params + self.actor.n_params + self.critic.n_params
        self._zero_traces()
        self.stats: dict[str, float] = {}

    # -- bookkeeping ------------------------------------------------------
    def _zero_traces(self) -> None:
        n, n_act = self.cell.n, getattr(self.actor, "n_act")
        self.e_critic = np.zeros(n + 1)
        self.e_actor = np.zeros((n_act, n))
        self.e_actor_b = np.zeros(n_act)
        self.e_rnn = np.zeros_like(self.cell.theta)

    def _input(self, obs: np.ndarray, prev_a, prev_r: float) -> np.ndarray:
        if not self.meta_inputs:
            return np.asarray(obs, dtype=np.float64)
        return np.concatenate([np.asarray(obs, dtype=np.float64),
                               self.action_space.encode(prev_a),
                               [prev_r * self.reward_scale]])

    # -- interaction ------------------------------------------------------
    def start(self, obs: np.ndarray):
        """Begin an episode: clear state and traces, consume ``o_0``, act."""
        self.cell.reset_state()
        self._zero_traces()
        self.h = self.cell.step(self._input(obs, None, 0.0))
        # V_old starts at V(s_0) rather than at 0: the true-online correction
        # term is a *difference* between successive weight vectors evaluated at
        # the same state, and seeding it with 0 injects one spurious update
        # worth of the initial value at every episode boundary.
        self.v_old = self.critic.value(self.h)
        self.a, self.cache = self.actor.act(self.h, self.rng)
        return self.a

    def step(self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool):
        """Consume one transition, apply one update, return the next action.

        ``terminated`` and ``truncated`` are not interchangeable: a terminal
        state has value zero by definition and must not be bootstrapped from,
        while a time-limit cut-off is an artefact of the harness and the next
        state's value is still the best estimate available.
        """
        r = float(reward) * self.reward_scale
        h_t = self.h
        phi_t = self.critic.features(h_t)
        v_now = float(self.critic.theta @ phi_t)  # V(s_t) under the *current* weights
        # The Dutch trace is only stable while alpha * ||phi||^2 <= 1, and with
        # n tanh units ||phi||^2 drifts from ~1 at init (h starts at zero, only
        # the bias feature is on) up towards ~n as the state saturates. So the
        # paper's alpha_C = 1.0 is fine for the first few hundred steps and then
        # blows the critic up. Dividing by ||phi||^2 -- the normalised-LMS step
        # -- keeps that product at exactly alpha_C forever, and is what makes
        # the paper's number usable rather than something to quietly retune.
        alpha_c = self.lr_critic
        if self.critic_lr_mode == "normalized":
            alpha_c = self.lr_critic / max(1.0, float(phi_t @ phi_t))

        # --- 1. what the heads want from the state, at time t ---------------
        if isinstance(self.actor, GaussianHead):
            dtheta_a, dz, g_actor, dlog_sigma = self.actor.grads(h_t, self.a, self.cache)
        else:
            dtheta_a, dz, g_actor = self.actor.grads(h_t, self.a, self.cache)
            dlog_sigma = None
        # The cell is told what the actor wants (through B_A) plus what the
        # critic wants (through B_C). It is never told *why*: the sign and size
        # of the actual correction arrive later, as the scalar delta.
        g = g_actor + self.critic.back()

        # --- 2. turn that into a weight gradient, BEFORE advancing the cell --
        # The influence the cell is carrying right now is dh_t/dtheta, and g is
        # dL/dh_t. One `cell.step` from now the influence will belong to
        # h_{t+1} and this product will be off by one timestep -- which does
        # not crash, does not warn, and costs most of the learning. Order
        # matters here more than anywhere else in the file.
        dW = self.cell.grad(g)

        # --- 3. now advance the recurrent state (and its influence) ----------
        self.h = self.cell.step(self._input(obs, self.a, reward))
        v_next = 0.0 if terminated else self.critic.value(self.h)
        delta = r + self.gamma * v_next - v_now

        # --- 3b. traces -- every one of them at time t ----------------------
        if self.critic_update == "accumulating":
            self.e_critic = accumulating_trace(self.e_critic, phi_t, self.gamma, self.lam_critic)
        else:
            self.e_critic = dutch_trace(self.e_critic, phi_t, self.gamma,
                                        self.lam_critic, alpha_c)
        self.e_actor = accumulating_trace(self.e_actor, dtheta_a, self.gamma, self.lam_actor)
        self.e_actor_b = accumulating_trace(self.e_actor_b, dz, self.gamma, self.lam_actor)
        # Clipped *before* it enters the trace, not only at the update. The
        # trace is a geometric sum with ratio gamma*lambda, so one unbounded
        # gradient does not merely make one bad update -- it stays in the trace
        # for ~1/(1-gamma*lambda) steps, which is enough to overflow.
        self.e_rnn = accumulating_trace(self.e_rnn, _clip(dW, self.clip),
                                        self.gamma, self.lam_rnn)

        # --- 4. updates ----------------------------------------------------
        if self.critic_update == "accumulating":
            self.critic.theta += alpha_c * delta * self.e_critic
        elif self.critic_update == "paper":
            # Algorithm 1 exactly as printed.
            self.critic.theta += (delta * self.e_critic
                                  + alpha_c * (self.v_old - v_now) * phi_t)
        else:
            # Full true-online TD(lambda), van Seijen et al. (2016) Alg. 1. The
            # difference from "paper" is the (V - V_old) e term, which is what
            # makes the online weights match the offline lambda-return solution.
            self.critic.theta += ((delta + v_now - self.v_old) * self.e_critic
                                  - alpha_c * (v_now - self.v_old) * phi_t)
        self.v_old = v_next

        step_a = self.lr_actor * delta
        if dlog_sigma is None:
            self.actor.apply(_clip(step_a * self.e_actor, self.clip),
                             _clip(step_a * self.e_actor_b, self.clip), 1.0)
        else:
            self.actor.apply(_clip(step_a * self.e_actor, self.clip),
                             _clip(step_a * self.e_actor_b, self.clip),
                             _clip(step_a * dlog_sigma, self.clip), 1.0)
        self.cell.apply(_clip(delta * self.e_rnn, self.clip), self.lr_rnn)

        # --- 5. carry forward ----------------------------------------------
        if not np.isfinite(self.h).all():
            raise FloatingPointError(
                "the recurrent state went non-finite. Something diverged -- in "
                "order of likelihood: lr_rnn too high, a cell parameter that "
                "drives the state (LTC's A, LRCU's e) not being bounded by its "
                "post_update, or clip=0. Lower --lr-rnn first.")
        self.a, self.cache = self.actor.act(self.h, self.rng)
        self.stats = {"delta": float(delta), "value": float(v_now),
                      "entropy": self.actor.entropy(self.cache)}
        if terminated or truncated:
            return None
        return self.a

    # -- supervised pretraining -------------------------------------------
    def imitate(self, obs, expert_action, reward: float, terminated: bool,
                truncated: bool, lr: float = 1e-2, lr_rnn: float | None = None):
        """One online behavioural-cloning step: copy ``expert_action``.

        The same forward-mode machinery, with the learning signal swapped. In
        RL the signal is ``delta * d[log pi(a_sampled)]``; here it is
        ``d[log pi(a_expert)]`` with no ``delta`` at all, because a
        demonstration is a label and not a reward. Everything else -- the
        influence carried in the cell, the feedback-aligned routing of the
        head's gradient back into it -- is untouched.

        This is the offline half of Lemmel et al. (2026): behavioural cloning
        first, RTRRL fine-tuning afterwards, on the same weights. Doing the
        cloning *with RFLO too* rather than with BPTT is not required by that
        paper, but it keeps the promise this repo is about -- one pass, one
        update per step, constant memory -- across both phases.

        Returns the expert's action, so a caller can execute it.
        """
        a = int(expert_action)
        dtheta_a, dz, g_actor = self.actor.grads(self.h, a, self.cache)
        dW = self.cell.grad(g_actor)
        self.actor.apply(_clip(lr * dtheta_a, self.clip), _clip(lr * dz, self.clip), 1.0)
        self.cell.apply(_clip(dW, self.clip), self.lr_rnn if lr_rnn is None else lr_rnn)
        self.h = self.cell.step(self._input(obs, a, reward))
        self.a, self.cache = self.actor.act(self.h, self.rng)
        self.stats = {"entropy": self.actor.entropy(self.cache)}
        if terminated or truncated:
            return None
        return a

    def begin_finetune(self, reset_critic: bool = True) -> None:
        """Hand a cloned policy over to RL: clear the traces, zero the critic.

        The critic has never seen a reward, so whatever is in it is noise with
        the authority of a value function. Zeroing it costs the few thousand
        steps it takes to refit and avoids the cloned policy being wrecked by a
        confident wrong baseline in the first hundred.
        """
        self._zero_traces()
        if reset_critic:
            self.critic.theta[:] = 0.0
        self.v_old = self.critic.value(self.h)

    # -- evaluation -------------------------------------------------------
    def greedy(self, obs: np.ndarray, prev_a=None, prev_r: float = 0.0):
        """One greedy action, advancing the recurrent state. Stateless caller.

        Prefer :meth:`eval_policy` for anything more than a single step --
        this signature makes the caller responsible for threading the previous
        action and reward back in, and a caller that forgets (they all forget)
        evaluates a meta-RL agent with its meta-inputs pinned to zero.
        """
        self.h = self.cell.step(self._input(obs, prev_a, prev_r))
        if isinstance(self.actor, GaussianHead):
            mu = self.actor.theta @ self.h + self.actor.bias
            return np.clip(mu, self.actor.low, self.actor.high)
        pi = self.actor.act(self.h, self.rng)[1]
        return int(np.argmax(pi))

    def eval_policy(self):
        """A greedy policy for :func:`~rtrrl_playground.train.rollout`.

        Evaluating a recurrent, meta-RL agent correctly needs two things that
        a plain ``policy(obs)`` callable cannot do, and getting either wrong
        changes the number without changing anything visible:

        * the recurrent state must be **reset at every episode boundary**,
          or episode two starts with episode one's state still in it; and
        * the **previous action and reward have to be fed back in**, because
          they are inputs the network was trained with. Pinning them to zero
          at evaluation time is evaluating a different network.

        So this returns a small stateful object with ``reset()`` and
        ``observe(reward)`` hooks, which ``rollout`` calls at the right moments.
        """
        return _GreedyPolicy(self)


class _GreedyPolicy:
    """Stateful greedy wrapper -- see :meth:`RTRRL.eval_policy`."""

    def __init__(self, agent: "RTRRL"):
        self.agent = agent
        self.reset()

    def reset(self) -> None:
        self.agent.cell.reset_state()
        self.prev_a, self.prev_r = None, 0.0

    def observe(self, reward: float) -> None:
        self.prev_r = float(reward)

    def __call__(self, obs):
        a = self.agent.greedy(obs, self.prev_a, self.prev_r)
        self.prev_a = a
        return a
