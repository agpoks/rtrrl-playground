"""Lesson 12 -- LiGRU and LRCU from scratch: the equations, the derivatives, the checks.

    python tutorial/12_cells_from_scratch.py

Every recurrent cell in this repo, written out in twenty lines each, with its
paper, its update equation, its hand-derived Jacobians, and a finite-difference
check that those Jacobians are what they claim to be. Nothing is imported from
``rtrrl_playground`` except for the final cross-check.
"""

# %% [markdown]
# # Lesson 12 — The cells from scratch
#
# Lesson 2 showed how to carry a gradient forwards. It treated the cell as a
# black box: *something* supplies `imm`, `leak` and `D`, and the estimator does
# the rest. This lesson opens the box.
#
# For every cell there are exactly four things to know, and they are all that
# `nets/cell.py` ever asks for:
#
# | | |
# |---|---|
# | **the update** | $h_{t+1} = F(h_t, x_t; \theta)$ |
# | **`imm`** $(n, p)$ | $\partial h_{t+1}[i]\,/\,\partial\theta_{ij}$ — how neuron $i$'s state depends on neuron $i$'s **own** parameters, holding $h_t$ fixed |
# | **`leak`** $(n,)$ | $\partial h_{t+1}[i]\,/\,\partial h_t[i]$ — the *direct* part, with every recurrent connection deleted |
# | **`D`** $(n, n)$ | the whole state-to-state Jacobian |
#
# RFLO uses `imm` and `leak`. SnAp-1 swaps `leak` for $\mathrm{diag}(D)$. Exact
# RTRL uses `imm` and all of `D`. So deriving a cell means writing down four
# things, and checking a cell means checking that they are the derivatives of
# the update.

# %%
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sigmoid(z):
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


# %% [markdown]
# ## The one structural rule
#
# **Every parameter must belong to exactly one neuron.** `theta` is an
# $(n, p)$ array; row $i$ is everything neuron $i$ owns. That is what makes
# `imm` an $(n, p)$ array rather than an $(n, n, p)$ tensor, and it is what
# makes RFLO's approximation a *second*-order omission rather than a
# first-order one.
#
# It is also a real constraint on architecture, and the clearest way to see it
# is the cell that fails it.
#
# ### Why a full GRU is not in this repo
#
# The GRU's candidate is $c = \tanh(W_c [x\,;\, r \odot h])$ where
# $r = \sigma(W_r \xi)$. So $W_r$ belonging to neuron $k$ reaches $h_i$ for
# **every** $i$, through $W_c[i, k]$. The immediate Jacobian is no longer block
# diagonal, and $(n, p)$ cannot represent it.
#
# LiGRU's contribution is precisely to delete that gate.

# %% [markdown]
# ## 1. LiGRU
#
# Ravanelli, Brakel, Omologo & Bengio, *"Light Gated Recurrent Units for Speech
# Recognition"*, IEEE TETCI 2018 — [arXiv:1803.10225](https://arxiv.org/abs/1803.10225).
#
# A GRU with the reset gate removed. Their argument was empirical (for speech,
# the reset gate is redundant with the update gate, and removing it trains
# faster); the consequence that matters here is structural — what is left is
# local.
#
# $$z = \sigma(W_z \xi), \qquad c = \tanh(W_c \xi), \qquad
#   h_{t+1} = z \odot h_t + (1 - z) \odot c$$
#
# with $\xi = [x_t\,;\,h_t\,;\,1]$. The gate **interpolates** between keeping
# the state and replacing it.
#
# Derivatives, writing $d_z = (h - c)\,z(1-z)$ and $d_c = (1-z)(1-c^2)$:
#
# $$\frac{\partial h'_i}{\partial W_{z,ij}} = d_{z,i}\,\xi_j, \qquad
#   \frac{\partial h'_i}{\partial W_{c,ij}} = d_{c,i}\,\xi_j, \qquad
#   \mathrm{leak}_i = z_i$$
#
# $$D = \mathrm{diag}(z) + d_z\,W_z^{\text{rec}} + d_c\,W_c^{\text{rec}}$$
#
# Note $\mathrm{leak} = z$: a saturated gate gives $\mathrm{leak} = 1$. Hold
# that thought.

# %%
def ligru(h, x, theta, n_in):
    """Returns (h_next, imm, leak, D). ``theta`` is (n, 2*n_xi): [W_z | W_c]."""
    n = len(h)
    xi = np.concatenate([x, h, [1.0]])
    nx = len(xi)
    Wz, Wc = theta[:, :nx], theta[:, nx:2 * nx]
    z = sigmoid(Wz @ xi)
    c = np.tanh(Wc @ xi)
    h_new = z * h + (1.0 - z) * c

    d_z = (h - c) * z * (1.0 - z)
    d_c = (1.0 - z) * (1.0 - c * c)
    imm = np.concatenate([d_z[:, None] * xi[None, :], d_c[:, None] * xi[None, :]], axis=1)
    leak = z
    rec = slice(n_in, n_in + n)
    D = np.diag(leak) + d_z[:, None] * Wz[:, rec] + d_c[:, None] * Wc[:, rec]
    return h_new, imm, leak, D


# %% [markdown]
# ## 2. LRCU
#
# Farsang, Neubauer & Grosu, *"Liquid Resistance Liquid Capacitance Networks"*,
# NeuroAI @ NeurIPS 2024 — [arXiv:2403.08791](https://arxiv.org/abs/2403.08791).
#
# The paper's neural ODE is
#
# $$\dot h_i = \epsilon(w_i)\,\big(-\sigma(f_i)\,h_i + \tanh(u_i)\,e_i\big)$$
#
# and solving it with one explicit Euler unfolding gives the **LRCU**:
#
# $$h_{t+1} = \big(1 - \epsilon\,\sigma(f)\big)\odot h_t \;+\; \epsilon\odot\tanh(u)\odot e$$
#
# Three gates, not two, and they do different jobs:
#
# * $\sigma(f)$ — the **forget conductance** (liquid *resistance*: an
#   input-dependent leak, which is what LTC already had);
# * $\tanh(u)\odot e$ — the drive, towards a per-neuron reversal potential;
# * $\epsilon = \sigma(w)$ — the **elastance** (liquid *capacitance*), which
#   multiplies *both* of the others. This is what LRC adds over LTC: an LTC can
#   only choose how fast to move, an LRCU can also choose whether to move at
#   all this step, uniformly across its own drive and its own decay.
#
# **Two simplifications here, both deliberate.** The paper's $f$, $u$, $w$ are
# sums over *per-synapse* nonlinearities, $\sum_j g_{ji}\,\sigma(a_{ji}y_j +
# b_{ji})$ — the biophysical-synapse construction from
# [Lemmel & Grosu 2023](https://arxiv.org/abs/2303.04944), which triples the
# parameters per connection. This uses ordinary linear pre-activations. And a
# positive bias stands in for the paper's leak conductance $g_L$, without which
# the state settles at $\tanh(u)e/\sigma(f)$ and runs away — measured at 14 on
# `lanekeep` while every other cell stayed inside $[-1, 1]$.
#
# Derivatives, with $s_f = \sigma(f)$, $t_u = \tanh(u)$, $\epsilon = \sigma(w)$:
#
# $$\frac{\partial h'}{\partial f} = -\epsilon\,s_f(1-s_f)\,h, \qquad
#   \frac{\partial h'}{\partial u} = \epsilon\,(1-t_u^2)\,e, \qquad
#   \frac{\partial h'}{\partial w} = \epsilon(1-\epsilon)\,(-s_f h + t_u e)$$
#
# $$\frac{\partial h'_i}{\partial e_i} = \epsilon_i t_{u,i}, \qquad
#   \mathrm{leak}_i = 1 - \epsilon_i s_{f,i}$$

# %%
def lrcu(h, x, theta, n_in):
    """``theta`` is (n, 3*n_xi + 1): [G | K | O | e]."""
    n = len(h)
    xi = np.concatenate([x, h, [1.0]])
    nx = len(xi)
    G, K, O = theta[:, :nx], theta[:, nx:2 * nx], theta[:, 2 * nx:3 * nx]
    e = theta[:, 3 * nx]
    sf = sigmoid(G @ xi)          # forget conductance -- liquid resistance
    tu = np.tanh(K @ xi)          # drive
    eps = sigmoid(O @ xi)         # elastance -- liquid capacitance
    h_new = (1.0 - eps * sf) * h + eps * tu * e

    d_f = -eps * sf * (1.0 - sf) * h
    d_u = eps * (1.0 - tu * tu) * e
    d_w = eps * (1.0 - eps) * (-sf * h + tu * e)
    imm = np.concatenate([d_f[:, None] * xi[None, :],
                          d_u[:, None] * xi[None, :],
                          d_w[:, None] * xi[None, :],
                          (eps * tu)[:, None]], axis=1)
    leak = 1.0 - eps * sf
    rec = slice(n_in, n_in + n)
    D = (np.diag(leak) + d_f[:, None] * G[:, rec]
         + d_u[:, None] * K[:, rec] + d_w[:, None] * O[:, rec])
    return h_new, imm, leak, D


# %% [markdown]
# ## 3. LiquidGRU — this repo's own
#
# Now the thought from LiGRU. Its $\mathrm{leak} = z$, so a saturated gate
# gives $\mathrm{leak} = 1$ — and RFLO's influence
# $P \leftarrow \mathrm{leak}\cdot P + \mathrm{imm}$ is a geometric series that
# then never converges.
#
# Take LTC's leak structure and a GRU's target: pull the state towards
# $\tanh(W_c\xi)$ at a rate that is a gate **plus a floor**.
#
# $$g = 1/\tau + z, \qquad h_{t+1} = \frac{h_t + \Delta t\,g\,c}{1 + \Delta t\,g}$$
#
# $$\mathrm{leak} = \frac{1}{1 + \Delta t\,g} \;\le\; \frac{1}{1 + \Delta t/\tau} \;<\; 1$$
#
# bounded below 1 for **any** gate value. The arbitrary `leak_max` cap in
# `nets/cell.py` becomes a learned per-neuron $\tau$.

# %%
def liquid_gru(h, x, theta, n_in, dt=1.0):
    """``theta`` is (n, 2*n_xi + 1): [W_z | W_c | tau]."""
    n = len(h)
    xi = np.concatenate([x, h, [1.0]])
    nx = len(xi)
    Wz, Wc, tau = theta[:, :nx], theta[:, nx:2 * nx], theta[:, 2 * nx]
    z = sigmoid(Wz @ xi)
    c = np.tanh(Wc @ xi)
    g = 1.0 / tau + z
    den = 1.0 + dt * g
    h_new = (h + dt * g * c) / den

    dg = dt * (c - h_new) / den
    d_z = dg * z * (1.0 - z)
    d_c = dt * g / den * (1.0 - c * c)
    imm = np.concatenate([d_z[:, None] * xi[None, :],
                          d_c[:, None] * xi[None, :],
                          (-dg / tau ** 2)[:, None]], axis=1)
    leak = 1.0 / den
    rec = slice(n_in, n_in + n)
    D = np.diag(leak) + d_z[:, None] * Wz[:, rec] + d_c[:, None] * Wc[:, rec]
    return h_new, imm, leak, D


# %% [markdown]
# ## Check every derivative
#
# Three separate claims, and each is checkable:
#
# * **`imm`** is $\partial h'/\partial\theta$ at fixed $h$;
# * **`D`** is $\partial h'/\partial h$;
# * **`leak`** is $\mathrm{diag}(D)$ *with the recurrent block deleted* — which
#   is not the same as $\mathrm{diag}(D)$, and the difference is exactly what
#   separates RFLO from SnAp-1.

# %%
CELLS = {
    "ligru": (ligru, lambda nx: 2 * nx),
    "lrcu": (lrcu, lambda nx: 3 * nx + 1),
    "liquid_gru": (liquid_gru, lambda nx: 2 * nx + 1),
}


def check(name, n_in=3, n=5, eps=1e-6, seed=0):
    fn, psize = CELLS[name]
    rng = np.random.default_rng(seed)
    nx = n_in + n + 1
    theta = rng.normal(0, 0.4, (n, psize(nx)))
    if name == "lrcu":
        theta[:, 3 * nx - 1] += 1.0          # the leak-conductance bias
    if name == "liquid_gru":
        theta[:, 2 * nx] = rng.uniform(2, 8, n)   # tau must be positive
    h = rng.normal(0, 0.3, n)
    x = rng.normal(0, 0.5, n_in)

    h_new, imm, leak, D = fn(h, x, theta, n_in)

    d_imm = 0.0
    for k in range(n):
        for j in range(theta.shape[1]):
            tp, tm = theta.copy(), theta.copy()
            tp[k, j] += eps
            tm[k, j] -= eps
            fd = (fn(h, x, tp, n_in)[0] - fn(h, x, tm, n_in)[0]) / (2 * eps)
            # only neuron k's own state may move: that is the locality rule
            assert np.abs(np.delete(fd, k)).max() < 1e-7, f"{name}: parameter leaked to another neuron"
            d_imm = max(d_imm, abs(fd[k] - imm[k, j]))

    d_D = 0.0
    for k in range(n):
        hp, hm = h.copy(), h.copy()
        hp[k] += eps
        hm[k] -= eps
        fd = (fn(hp, x, theta, n_in)[0] - fn(hm, x, theta, n_in)[0]) / (2 * eps)
        d_D = max(d_D, float(np.abs(fd - D[:, k]).max()))

    # leak = the diagonal with the recurrent block removed
    theta_norec = theta.copy()
    for blk in range(theta.shape[1] // nx):
        theta_norec[:, blk * nx + n_in: blk * nx + n_in + n] = 0.0
    d_leak = 0.0
    for k in range(n):
        hp, hm = h.copy(), h.copy()
        hp[k] += eps
        hm[k] -= eps
        fd = (fn(hp, x, theta_norec, n_in)[0] - fn(hm, x, theta_norec, n_in)[0]) / (2 * eps)
        d_leak = max(d_leak, abs(fd[k] - fn(h, x, theta_norec, n_in)[2][k]))
    return d_imm, d_D, d_leak


print(f"  {'cell':<12}{'imm':>11}{'D':>11}{'leak':>11}")
for name in CELLS:
    a, b, c = check(name)
    print(f"  {name:<12}{a:>11.2e}{b:>11.2e}{c:>11.2e}")

# %% [markdown]
# ## The same cells in the package agree
#
# The point of writing them out here is that you can read them. The point of
# this check is that the package is not doing something else.

# %%
from rtrrl_playground.nets import make_cell  # noqa: E402

print()
for name in CELLS:
    fn, _ = CELLS[name]
    pkg = make_cell(name, 3, 5, estimator="rtrl", rng=np.random.default_rng(0))
    if name == "liquid_gru":
        pkg.theta[:, 2 * pkg.n_xi] = 4.0
    h = np.zeros(5)
    x = np.random.default_rng(7).normal(size=3)
    pkg.h = h.copy()
    got = pkg.step(x)
    mine = fn(h, x, pkg.theta, 3)[0]
    print(f"  {name:<12} package vs from-scratch: max |diff| = {np.abs(got - mine).max():.2e}")

# %% [markdown]
# ## What each mechanism actually does
#
# Equations are one thing. Here is the behaviour they buy, on the same input:
# a step that switches on at $t=20$ and off at $t=60$.

# %%
def response(name, n=1, T=100, seed=1, saturate=0.0):
    fn, psize = CELLS[name]
    rng = np.random.default_rng(seed)
    n_in = 1
    nx = n_in + n + 1
    theta = np.zeros((n, psize(nx)))
    if name == "ligru":
        theta[:, 0] = 2.0          # input drives the gate open
        theta[:, nx] = 3.0         # and the candidate
        theta[:, nx - 1] = 2.0 + saturate   # forget-bias: hold by default
    elif name == "lrcu":
        theta[:, 0] = -2.0         # input closes the forget conductance
        theta[:, nx] = 3.0
        theta[:, 3 * nx - 1] = 0.5
        theta[:, nx - 1] = 1.0
        theta[:, 3 * nx] = 1.0     # reversal potential
    else:
        theta[:, 0] = 2.0
        theta[:, nx] = 3.0
        theta[:, nx - 1] = saturate    # same gate bias, for the saturation demo
        theta[:, 2 * nx] = 20.0        # a long but finite tau
    h = np.zeros(n)
    xs = np.zeros(T)
    xs[20:60] = 1.0
    out, leaks = [], []
    for t in range(T):
        h, _imm, leak, _D = fn(h, np.array([xs[t]]), theta, n_in)
        out.append(h[0])
        leaks.append(leak[0])
    return xs, np.array(out), np.array(leaks)


print()
rows = [(n, 0.0) for n in CELLS] + [("ligru", 8.0), ("liquid_gru", 8.0)]
for name, sat in rows:
    xs, h, leak = response(name, saturate=sat)
    tag = f"{name} (gate saturated)" if sat else name
    print(f"  {tag:<26} pulse ends t=60 {h[60]:+.3f} -> t=99 {h[99]:+.3f}"
          f"   leak max {leak.max():.4f}")
print("\n  The last two rows are the point. Drive the gate hard and LiGRU's leak")
print("  reaches 1.0000: it now holds its state forever -- here the zero it started")
print("  with, because a gate that never forgets also never lets anything in. That")
print("  is a reachable configuration, not a pathological one, and it is fatal for")
print("  RFLO: the influence sum P <- leak*P + imm becomes a geometric series with")
print("  ratio 1 and never converges. LiquidGRU under the same drive is unmoved --")
print("  its leak cannot exceed 1/(1 + dt/tau) whatever the gate does.")

# %% [markdown]
# ## References
#
# * **LiGRU** — Ravanelli, Brakel, Omologo & Bengio, *Light Gated Recurrent
#   Units for Speech Recognition*, IEEE TETCI 2(2):92–102, 2018.
#   [arXiv:1803.10225](https://arxiv.org/abs/1803.10225)
# * **LRC / LRCU** — Farsang, Neubauer & Grosu, *Liquid Resistance Liquid
#   Capacitance Networks*, NeuroAI @ NeurIPS 2024.
#   [arXiv:2403.08791](https://arxiv.org/abs/2403.08791)
# * **LTC** — Hasani, Lechner, Amini, Rus & Grosu, *Liquid Time-constant
#   Networks*, AAAI 2021. [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
# * **Biophysical synapses** (the per-synapse nonlinearity LRCU simplifies here)
#   — Lemmel & Grosu, AAAI 2023.
#   [arXiv:2303.04944](https://arxiv.org/abs/2303.04944)
# * **GRU** — Cho et al., *Learning Phrase Representations using RNN
#   Encoder–Decoder*, EMNLP 2014. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
# * **RFLO** — Murray, *Local online learning in recurrent networks with random
#   feedback*, eLife 2019. [doi:10.7554/eLife.43299](https://doi.org/10.7554/eLife.43299)
#
# Full list with why each one is here: [`papers/README.md`](../papers/README.md).
