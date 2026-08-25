"""Lesson 4 -- the whole of RTRRL, in one file, with nothing imported from the repo.

    python tutorial/04_rtrrl_from_scratch.py
    python tutorial/04_rtrrl_from_scratch.py --env lanekeep --steps 300000

Every other file in this repo is a generalisation of what is below: five
gradient estimators instead of one, four cells instead of one, flags for the
ablations. This one is the algorithm and nothing else -- about a hundred lines
of NumPy, no classes you have to chase, no autograd. If you read one file
here, read this one.
"""

# %% [markdown]
# # Lesson 4 — RTRRL from scratch
#
# Lesson 2 gave us `dh/dtheta` carried forwards (RFLO). Lesson 3 gave us a way
# to reach backwards in time with one scalar (eligibility traces). RTRRL is
# what happens when you put them together and let the scalar be a TD error.
#
# The network:
#
# ```
# xi_t   = [ o_t ; a_{t-1} ; r_{t-1} ; h_t ; 1 ]      <- meta-RL input
# h_t+1  = h_t + (tanh(W xi_t) - h_t) / tau
# pi_t   = softmax(A h_t + b)                          <- actor
# v_t    = c . h_t + d                                 <- critic
# ```
#
# The learning, all of it, per step:
#
# ```
# delta  = r + gamma v(h_t+1) - v(h_t)                 <- one scalar
# P     <- (1 - 1/tau) P + (1/tau) tanh'(W xi) xi^T    <- RFLO influence
# g      = B_A^T dlog(pi)/dz + B_C                     <- feedback alignment
# e_W   <- gamma lam e_W + g * P                       <- traces
# W     <- W + alpha_R delta e_W
# ```
#
# That is the entire algorithm. No replay buffer, no batch, no backward pass,
# no stored history, and the update happens *while* the episode is running.

# %%
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtrrl_playground import make_env  # the environment only -- nothing else

# %% [markdown]
# ## The agent
#
# Read `step` from top to bottom. The one thing to keep straight is *which
# timestep each quantity belongs to*: the action was drawn from `h_t`, so the
# policy gradient and the influence `P` both belong to `t`, while the TD error
# needs `v(h_{t+1})`, which only exists after the cell has advanced. Every
# trace below is fed the time-`t` quantities; only `delta` looks forward.

# %%
class RTRRLFromScratch:
    def __init__(self, obs_dim, n_act, n=32, gamma=0.99, lam=0.9,
                 lr_actor=1e-3, lr_critic=0.1, lr_rnn=1e-3, entropy=0.03,
                 tau=2.0, seed=0):
        rng = self.rng = np.random.default_rng(seed)
        self.n, self.n_act, self.obs_dim = n, n_act, obs_dim
        n_in = obs_dim + n_act + 1                 # observation, last action, last reward
        self.n_xi = n_in + n + 1                   # ... plus the state and a bias
        self.n_in = n_in

        # Recurrent weights. The input gain of 3 is not decoration: observations
        # here are normalised to their *limits*, so a typical one is small, and
        # at unit gain every tanh sits in its linear region -- which leaves the
        # critic unable to fit anything but a linear value function.
        self.W = np.zeros((n, self.n_xi))
        self.W[:, :n_in] = rng.normal(0, 3.0 / np.sqrt(n_in), (n, n_in))
        self.W[:, n_in:n_in + n] = rng.normal(0, 1.0 / np.sqrt(n), (n, n))
        self.W[:, -1] = rng.normal(0, 0.5, n)
        self.tau = np.full(n, tau)

        self.A = np.zeros((n_act, n))              # actor, starts uniform
        self.b = np.zeros(n_act)
        self.c = np.zeros(n + 1)                   # critic, bias in the last slot

        # Fixed random feedback -- never trained. This is the feedback-alignment
        # trick: the cell is told what the heads want through a random matrix
        # rather than through the heads' own transposed weights.
        self.B_A = rng.normal(0, 1 / np.sqrt(n), (n_act, n))
        self.B_C = rng.normal(0, 1 / np.sqrt(n), n)

        self.gamma, self.lam, self.entropy = gamma, lam, entropy
        self.lr_actor, self.lr_critic, self.lr_rnn = lr_actor, lr_critic, lr_rnn
        self.n_params = self.W.size + self.tau.size + self.A.size + self.b.size + self.c.size

    # -- forward -----------------------------------------------------------
    def _cell(self, x):
        """One CT-RNN tick, and the RFLO influence update that goes with it."""
        xi = np.concatenate([x, self.h, [1.0]])
        phi = np.tanh(self.W @ xi)
        dphi = 1.0 - phi * phi
        inv_tau = 1.0 / self.tau
        # P is dh/dW, carried forwards. One outer product, same shape as W.
        self.P *= (1.0 - inv_tau)[:, None]
        self.P += (dphi * inv_tau)[:, None] * xi[None, :]
        self.h = self.h + (phi - self.h) * inv_tau
        return self.h

    def _input(self, obs, prev_a, prev_r):
        onehot = np.zeros(self.n_act)
        if prev_a is not None:
            onehot[prev_a] = 1.0
        return np.concatenate([obs, onehot, [prev_r]])

    def _policy(self):
        z = self.A @ self.h + self.b
        e = np.exp(z - z.max())
        return e / e.sum()

    def _value(self, h):
        return float(self.c[:-1] @ h + self.c[-1])

    # -- interaction -------------------------------------------------------
    def start(self, obs):
        self.h = np.zeros(self.n)
        self.P = np.zeros_like(self.W)
        self.e_W = np.zeros_like(self.W)
        self.e_A = np.zeros_like(self.A)
        self.e_b = np.zeros_like(self.b)
        self.e_c = np.zeros(self.n + 1)
        self._cell(self._input(obs, None, 0.0))
        self.v_old = self._value(self.h)
        self.pi = self._policy()
        self.a = int(self.rng.choice(self.n_act, p=self.pi))
        return self.a

    def step(self, obs, reward, terminated, truncated):
        h_t, pi_t, a_t = self.h, self.pi, self.a
        phi_t = np.concatenate([h_t, [1.0]])
        P_t = self.P.copy()                  # the influence of h_t, before advancing
        v_now = float(self.c @ phi_t)

        self._cell(self._input(obs, a_t, reward))
        v_next = 0.0 if terminated else self._value(self.h)
        delta = reward + self.gamma * v_next - v_now

        # --- what the heads want, in terms of the logits ---
        dz = -pi_t.copy()
        dz[a_t] += 1.0                                        # d log pi[a] / dz
        logpi = np.log(np.clip(pi_t, 1e-12, None))
        H = -float(pi_t @ logpi)
        dz += self.entropy * (-pi_t * (logpi + H))            # d H / dz
        g = self.B_A.T @ dz + self.B_C                        # ... routed back randomly

        # --- traces, all at time t ---
        gl = self.gamma * self.lam
        alpha_c = self.lr_critic / max(1.0, float(phi_t @ phi_t))
        self.e_c = gl * self.e_c + alpha_c * phi_t - alpha_c * gl * float(self.e_c @ phi_t) * phi_t
        self.e_A = gl * self.e_A + np.outer(dz, h_t)
        self.e_b = gl * self.e_b + dz
        self.e_W = gl * self.e_W + g[:, None] * P_t           # <- RFLO meets the trace

        # --- updates ---
        self.c += (delta + v_now - self.v_old) * self.e_c - alpha_c * (v_now - self.v_old) * phi_t
        self.v_old = v_next
        self.A += self.lr_actor * delta * self.e_A
        self.b += self.lr_actor * delta * self.e_b
        self.W += self.lr_rnn * delta * self.e_W

        self.pi = self._policy()
        self.a = int(self.rng.choice(self.n_act, p=self.pi))
        self.entropy_now = H
        self.delta = delta
        return None if (terminated or truncated) else self.a


# %% [markdown]
# ## Run it
#
# Same loop as every online agent: act, observe, learn, repeat. Note that there
# is nothing between "observe" and "learn" — no buffer to fill, no batch to
# wait for, no end of episode to reach.

# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cartpole-vel")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    env = make_env(args.env)
    agent = RTRRLFromScratch(env.obs_dim, env.action_space.n, seed=args.seed)
    print(f"  {args.env}: obs_dim={env.obs_dim}, {env.action_space.n} actions, "
          f"{agent.n_params} parameters, influence array {agent.W.nbytes / 1024:.1f} KiB")

    obs = env.reset(seed=args.seed)
    action = agent.start(obs)
    returns, R = [], 0.0
    for t in range(1, args.steps + 1):
        obs, r, terminated, truncated, _ = env.step(action)
        R += r
        action = agent.step(obs, r, terminated, truncated)
        if action is None:
            returns.append(R)
            R = 0.0
            action = agent.start(env.reset())
        if t % (args.steps // 10) == 0:
            recent = np.mean(returns[-20:]) if returns else float("nan")
            print(f"  step {t:>8,}  episodes {len(returns):>5}  "
                  f"return (last 20) {recent:8.2f}  entropy {agent.entropy_now:.3f}")
    return returns


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Things to try
#
# * Set `lr_actor = 1e-2` and watch the entropy collapse inside a few thousand
#   steps. The critic starts at zero, so the early TD error is persistently
#   positive, which reinforces *whatever the policy already prefers* regardless
#   of whether it was any good. Two-timescale — critic fast, actor slow — is
#   not a nicety here, it is the difference between learning and not.
# * Set `entropy = 0` for the same failure from the other side.
# * Delete the `self.P *= (1 - inv_tau)` line, so the influence is only ever
#   the current step's immediate term. That is RFLO with a memory of one, and
#   it is roughly what "just do backprop on the current step" would give you.
# * Replace `g = self.B_A.T @ dz + self.B_C` with `g = self.A.T @ dz + self.c[:-1]`
#   to turn feedback alignment off and use the true gradient. On these tasks
#   it barely moves — which is the whole surprise of the alignment result.
