"""Lesson 3 -- eligibility traces: how one number credits an action ten steps back.

    python tutorial/03_traces.py

The other half of RTRRL. Lesson 2 dealt with getting a gradient across time
inside the network; this one deals with getting *reward* across time, which is
a separate problem with a separate mechanism.
"""

# %% [markdown]
# # Lesson 3 — Eligibility traces
#
# An online agent sees one transition at a time and then throws it away. So
# when a reward finally arrives, the actions that caused it are gone: not in a
# buffer, not in a graph, gone.
#
# An eligibility trace is the answer. Alongside each parameter, keep a decaying
# record of how recently and how strongly that parameter was involved:
#
# ```
# e  <-  gamma * lambda * e  +  (gradient at this step)
# theta  <-  theta + alpha * delta * e
# ```
#
# One scalar TD error `delta` now updates *everything the agent has recently
# done*, weighted by how recently it did it. Memory cost: one number per
# parameter, constant in time. This is the mechanism that lets an algorithm
# with no history still assign delayed credit.

# %%
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground.traces import accumulating_trace, dutch_trace

# %% [markdown]
# ## What a trace looks like
#
# One feature, switched on at step 3 and again at steps 10-12. Watch the trace
# rise while the feature is active and decay after — that decaying tail is the
# window in which a reward can still find the action.

# %%
gamma, lam = 0.99, 0.9
active = np.zeros(30)
active[3] = 1.0
active[10:13] = 1.0
e = 0.0
trace = []
for t in range(30):
    e = gamma * lam * e + active[t]
    trace.append(e)
print("  step:  " + "".join(f"{t:5d}" for t in range(0, 20)))
print("  on:    " + "".join(f"{int(a):5d}" for a in active[:20]))
print("  trace: " + "".join(f"{v:5.2f}" for v in trace[:20]))
print(f"\n  gamma*lambda = {gamma * lam:.3f}, so the trace halves every "
      f"{np.log(0.5) / np.log(gamma * lam):.1f} steps and is effectively "
      f"{1 / (1 - gamma * lam):.0f} steps long.")

# %% [markdown]
# ## Accumulating vs Dutch
#
# The textbook trace above just adds. Visit the same state repeatedly and it
# piles up without bound, so the effective step size on that feature is larger
# than the one you configured — sometimes much larger.
#
# The **Dutch** trace of true online TD(λ) subtracts exactly the part of the
# accumulation the new observation makes redundant:
#
# ```
# e <- gamma*lambda*e + alpha*x - alpha*gamma*lambda*(e . x)*x
# ```
#
# Below, the same feature is on for twenty consecutive steps. The accumulating
# trace grows towards `1/(1-gamma*lambda)`; the Dutch trace settles.

# %%
x = np.array([1.0])
alpha = 0.3
ea = np.zeros(1)
ed = np.zeros(1)
rows = []
for t in range(20):
    ea = accumulating_trace(ea, x, gamma, lam)
    ed = dutch_trace(ed, x, gamma, lam, alpha)
    rows.append((t, float(ea[0]) * alpha, float(ed[0])))
print(f"  {'step':>5} {'alpha * accumulating':>22} {'dutch':>10}")
for t, a, d in rows[::4]:
    print(f"  {t:>5} {a:>22.3f} {d:>10.3f}")

# %% [markdown]
# ## Does it matter? Delayed credit, measured
#
# A chain of `n` states. One action at the start decides a reward delivered
# `n` steps later; everything in between is silent. Learn the value of the
# first state with TD(λ), and see how many episodes each λ needs.
#
# λ = 0 is TD(0): the reward moves back exactly one state per episode, so it
# takes `n` episodes for the information to reach the start. λ = 0.9 carries
# it most of the way in one.

# %%
def chain_value(lam_, n=10, episodes=60, alpha=0.1, gamma=1.0, seed=0):
    """Learn V(s_0) on a deterministic chain paying +1 at the end."""
    theta = np.zeros(n + 1)
    curve = []
    for ep in range(episodes):
        e = np.zeros(n + 1)
        for t in range(n):
            x = np.zeros(n + 1)
            x[t] = 1.0
            xn = np.zeros(n + 1)
            xn[t + 1] = 1.0
            r = 1.0 if t == n - 1 else 0.0
            terminal = t == n - 1
            v, vn = float(theta @ x), (0.0 if terminal else float(theta @ xn))
            delta = r + gamma * vn - v
            e = accumulating_trace(e, x, gamma, lam_)
            theta += alpha * delta * e
        curve.append(float(theta[0]))
    return curve


print(f"\n  V(s_0) after k episodes on a 10-step chain (true value = 1.0)")
print(f"  {'episodes':>9}" + "".join(f"{f'lambda={l}':>13}" for l in (0.0, 0.5, 0.9, 0.99)))
curves = {l: chain_value(l) for l in (0.0, 0.5, 0.9, 0.99)}
for k in (1, 2, 5, 10, 20, 60):
    print(f"  {k:>9}" + "".join(f"{curves[l][k - 1]:>13.4f}" for l in (0.0, 0.5, 0.9, 0.99)))

# %% [markdown]
# ## The plot

# %%
def plot(curves, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for l, c in curves.items():
        ax.plot(c, label=f"lambda = {l}")
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8, label="true value")
    ax.set_xlabel("episodes")
    ax.set_ylabel("V(s_0)")
    ax.set_title("How fast delayed credit reaches the start of a 10-step chain")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        print(f"  wrote {path}")


outdir = ROOT / "runs"
outdir.mkdir(exist_ok=True)
plot(curves, outdir / "lesson03_traces.png")

# %% [markdown]
# ## What to take from this
#
# * A trace is **`O(number of parameters)` memory and buys unbounded reach** —
#   it decays, but nothing is truncated. Compare with BPTT's window in
#   lesson 2, which is a hard cliff paid for in stored activations.
# * **λ is a bias-variance dial**, not a horizon: `λ=0` bootstraps aggressively
#   (biased, low variance), `λ=1` is Monte-Carlo (unbiased, high variance).
# * The **Dutch** trace is what makes the online updates equal to the offline
#   λ-return solution, which ordinary accumulating traces do not guarantee.
#   That is why the RTRRL paper uses it for the critic — and why
#   `--critic-update accumulating` is there, so you can see the difference.
#
# Lesson 4 puts lessons 2 and 3 together: RFLO supplies `dh/dtheta`, the trace
# supplies the reach, and one TD error drives both.
