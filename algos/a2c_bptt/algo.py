"""Recurrent A2C with truncated BPTT -- the baseline RTRRL exists to argue with.

Same cell (a CT-RNN, same equations as ``rtrrl_playground/nets/ctrnn.py``),
same actor and critic heads, same partially-observable environments. The only
thing that changes is *when and how the gradient is obtained*:

============  ==========================================  =========================
              RTRRL                                       A2C + truncated BPTT
============  ==========================================  =========================
gradient      carried forwards, one influence array       replayed backwards
              (``O(n p)`` for RFLO)                       through ``T`` stored steps
update timing every step, during the episode              every ``T`` steps, after
memory        constant in ``T``                           linear in ``T``
credit reach  unbounded (traces decay, nothing is cut)    exactly ``T`` steps
framework     hand-derived, NumPy                         ``torch.autograd``
============  ==========================================  =========================

The last row is the interesting one for a reader. This file calls
``loss.backward()``; nothing else in the repo does. That single line is what
RTRRL is buying its way out of, and having the alternative sitting here in
sixty lines makes the cost concrete rather than rhetorical -- as does the
truncation window ``T``, which is the knob BPTT has and traces do not: raise
it and credit reaches further back at a proportional cost in memory, and the
graph for a 900-step driving episode does not fit in the budget an RC car has
between control ticks.

This is the only file in the repo that needs PyTorch: ``pip install -e
".[torch]"``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTRNNTorch(nn.Module):
    """The CT-RNN of ``nets/ctrnn.py``, written for autograd instead of RFLO."""

    def __init__(self, n_in: int, n_hidden: int, tau_init: float = 2.0,
                 input_gain: float = 3.0):
        super().__init__()
        self.n_in, self.n = n_in, n_hidden
        self.W_in = nn.Parameter(torch.randn(n_hidden, n_in) * input_gain / np.sqrt(n_in))
        self.W_rec = nn.Parameter(torch.randn(n_hidden, n_hidden) / np.sqrt(n_hidden))
        self.bias = nn.Parameter(torch.randn(n_hidden) * 0.5)
        self.log_tau = nn.Parameter(torch.full((n_hidden,), float(np.log(tau_init))))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        phi = torch.tanh(F.linear(x, self.W_in) + F.linear(h, self.W_rec) + self.bias)
        tau = self.log_tau.exp().clamp(1.0, 50.0)
        return h + (phi - h) / tau


class A2CBPTT:
    """Recurrent advantage actor-critic, updated every ``truncation`` steps.

    Exposes the same ``start`` / ``step`` interface as the online agents so it
    drops into the same training loop -- the buffering is hidden inside
    ``step``, which is exactly the asymmetry being illustrated: the caller
    cannot tell that this one is not learning online, but the machine can.
    """

    def __init__(self, obs_dim: int, action_space, n_hidden: int = 32,
                 gamma: float = 0.99, lam: float = 0.9, lr: float = 3e-4,
                 truncation: int = 32, value_coef: float = 0.5,
                 entropy_coef: float = 1e-2, clip: float = 1.0,
                 meta_inputs: bool = True, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.action_space = action_space
        self.n_act = action_space.n
        self.meta_inputs = meta_inputs
        self.device = torch.device(device)
        n_in = obs_dim + (self.n_act + 1 if meta_inputs else 0)

        self.cell = CTRNNTorch(n_in, n_hidden).to(self.device)
        self.actor = nn.Linear(n_hidden, self.n_act).to(self.device)
        self.critic = nn.Linear(n_hidden, 1).to(self.device)
        nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.weight); nn.init.zeros_(self.critic.bias)
        self.params = list(self.cell.parameters()) + list(self.actor.parameters()) \
            + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)

        self.gamma, self.lam, self.T = gamma, lam, truncation
        self.value_coef, self.entropy_coef, self.clip = value_coef, entropy_coef, clip
        self.n_params = sum(p.numel() for p in self.params)
        self.stats: dict[str, float] = {}
        self.peak_graph_steps = 0

    # -- helpers ----------------------------------------------------------
    def _input(self, obs, prev_a, prev_r) -> torch.Tensor:
        parts = [np.asarray(obs, dtype=np.float32)]
        if self.meta_inputs:
            parts += [self.action_space.encode(prev_a).astype(np.float32),
                      np.array([prev_r], dtype=np.float32)]
        return torch.from_numpy(np.concatenate(parts)).to(self.device)

    def _act(self):
        logits = self.actor(self.h)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        self.buf["logp"].append(dist.log_prob(a))
        self.buf["ent"].append(dist.entropy())
        self.buf["v"].append(self.critic(self.h).squeeze(-1))
        self.a = int(a.item())
        return self.a

    def _new_buffer(self):
        self.buf = {k: [] for k in ("logp", "ent", "v", "r", "done")}

    # -- interface --------------------------------------------------------
    def start(self, obs):
        self.h = torch.zeros(self.cell.n, device=self.device)
        self._new_buffer()
        self.h = self.cell(self._input(obs, None, 0.0), self.h)
        return self._act()

    def step(self, obs, reward, terminated, truncated):
        self.buf["r"].append(float(reward))
        self.buf["done"].append(bool(terminated))
        self.h = self.cell(self._input(obs, self.a, reward), self.h)
        if terminated or truncated or len(self.buf["r"]) >= self.T:
            self._learn(bootstrap=not terminated)
            if terminated or truncated:
                return None
            # Detach: the next window starts from this state's *value*, not
            # from its history. This is the truncation, and it is the reason
            # BPTT cannot credit an action further back than T steps.
            self.h = self.h.detach()
            self._new_buffer()
        return self._act()

    def _learn(self, bootstrap: bool):
        n = len(self.buf["r"])
        if n == 0:
            return
        self.peak_graph_steps = max(self.peak_graph_steps, n)
        v = torch.stack(self.buf["v"])
        with torch.no_grad():
            last = self.critic(self.h).squeeze(-1) if bootstrap else torch.zeros(1)[0]
        adv = torch.zeros(n, device=self.device)
        gae, next_v = 0.0, last
        for t in reversed(range(n)):
            nonterm = 0.0 if self.buf["done"][t] else 1.0
            delta = self.buf["r"][t] + self.gamma * next_v * nonterm - v[t].detach()
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
            next_v = v[t].detach()
        returns = adv + v.detach()

        logp = torch.stack(self.buf["logp"])
        ent = torch.stack(self.buf["ent"])
        loss = (-(logp * adv).mean()
                + self.value_coef * F.mse_loss(v, returns)
                - self.entropy_coef * ent.mean())
        self.opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.opt.step()
        self.h = self.h.detach()
        with torch.no_grad():
            self.stats = {"delta": float(adv.mean()), "value": float(v.mean().detach()),
                          "entropy": float(ent.mean().detach())}

    def greedy(self, obs, prev_a=None, prev_r=0.0):
        with torch.no_grad():
            self.h = self.cell(self._input(obs, prev_a, prev_r), self.h)
            return int(self.actor(self.h).argmax().item())

    def eval_policy(self):
        """Same contract as the online agents -- see ``RTRRL.eval_policy``."""
        agent = self

        class _Greedy:
            def reset(self):
                agent.h = torch.zeros(agent.cell.n, device=agent.device)
                self.prev_a, self.prev_r = None, 0.0

            def observe(self, reward):
                self.prev_r = float(reward)

            def __call__(self, obs):
                a = agent.greedy(obs, self.prev_a, self.prev_r)
                self.prev_a = a
                return a

        pol = _Greedy()
        pol.reset()
        return pol
