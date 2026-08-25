"""Lesson 2 -- five ways to get a recurrent gradient, measured against the truth.

    python tutorial/02_gradients_online.py

Backpropagation through time is not the only way to differentiate a recurrent
network, and on a robot it is close to the worst: it needs the whole episode in
memory and cannot produce an update until the episode is over. This lesson
builds the alternatives on a supervised task where the exact gradient is
computable, and measures what each approximation actually costs you.
"""

# %% [markdown]
# # Lesson 2 — Five ways to get a recurrent gradient
#
# A recurrent network's gradient has to cross time. The textbook route is to
# store every activation and replay them backwards at the end of the sequence
# (BPTT). The alternative is to carry the derivative **forwards**: keep an
# *influence* array `J = dh/dtheta` alongside the state, and update it in the
# same pass. Then the gradient is available at every step, memory is constant
# in sequence length, and there is nothing to replay.
#
# The catch is that the exact forward version, RTRL, needs `n^2 * p` numbers.
# Everything interesting in this area is an approximation of that, and each one
# is a different bargain. Here we build five and grade them against the exact
# answer.

# %%
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground.nets import make_cell

# %% [markdown]
# ## The task: report a bit you saw a while ago
#
# A single input channel carries a value at `t = 0` and noise afterwards. At
# the end of a `T`-step sequence the network's first unit should equal that
# value. Loss is squared error at the last step only, so **all** of the
# gradient has to travel `T` steps backwards through the recurrence. That is
# the quantity these estimators disagree about, so it is worth making it the
# only quantity in the experiment.

# %%
def make_sequence(rng, T, n_in, delay):
    """A sequence whose target depends on the input `delay` steps before the end."""
    xs = rng.normal(0, 0.3, size=(T, n_in))
    signal = float(rng.choice([-1.0, 1.0]))
    xs[T - 1 - delay, 0] = signal
    return xs, signal


def loss_grad_of_state(h, target):
    """dL/dh for L = 0.5 (h_0 - target)^2, which is what the cell gets told."""
    g = np.zeros_like(h)
    g[0] = -(h[0] - target)  # ascent convention: every cell here maximises
    return g


def gradient(cell_name, estimator, xs, target, theta, seed=0, truncate=None):
    """Run one sequence and return the estimator's dL/dtheta."""
    cell = make_cell(cell_name, xs.shape[1], theta.shape[0],
                     estimator="rtrl" if truncate else estimator,
                     rng=np.random.default_rng(seed))
    cell.theta = theta.copy()
    cell.reset_state()
    T = len(xs)
    for t, x in enumerate(xs):
        # Truncated BPTT-k, computed the cheap way: an exact influence that is
        # *zeroed* k steps before the end -- keeping the hidden state, dropping
        # only its history -- sees exactly the dependencies a k-step BPTT window
        # sees, and nothing earlier. That is what truncation means.
        if truncate is not None and t == T - truncate:
            cell.J[:] = 0.0
        cell.step(x)
    return cell.grad(loss_grad_of_state(cell.h, target)), cell


# %% [markdown]
# ## Is the exact one actually exact?
#
# Before comparing anything to RTRL, check RTRL against finite differences.
# An influence array is exactly the kind of code that is subtly wrong in a way
# that still trains, slowly, and blames the learning rate.

# %%
def check_rtrl(cell_name="ctrnn", n_in=3, n=5, T=7, seed=1):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(T, n_in))
    ref = make_cell(cell_name, n_in, n, estimator="rtrl", rng=np.random.default_rng(0))
    theta = ref.theta.copy()

    def final_state(th):
        c = make_cell(cell_name, n_in, n, estimator="none", rng=np.random.default_rng(0))
        c.theta = th.copy()
        c.reset_state()
        for x in xs:
            c.step(x)
        return c.h.copy()

    ref.reset_state()
    for x in xs:
        ref.step(x)
    J = ref.J.reshape(n, n, ref.p)
    eps, worst = 1e-6, 0.0
    for k in range(n):
        for j in range(ref.p):
            tp, tm = theta.copy(), theta.copy()
            tp[k, j] += eps
            tm[k, j] -= eps
            fd = (final_state(tp) - final_state(tm)) / (2 * eps)
            worst = max(worst, float(np.abs(fd - J[:, k, j]).max()))
    return worst


for cell_name in ("ctrnn", "ltc", "lrcu", "ligru"):
    print(f"  {cell_name:6s} RTRL vs finite differences, worst abs error: "
          f"{check_rtrl(cell_name):.2e}")

# %% [markdown]
# ## How aligned is each estimator with the truth?
#
# Cosine similarity between each estimate and the exact RTRL gradient, as a
# function of how far back the dependency reaches. Cosine rather than error
# norm because for gradient descent only the *direction* has to be right —
# the magnitude is what the learning rate is for.
#
# What the numbers below actually say, which is not quite what you might
# guess: RFLO and SnAp-1 land around 0.7 and stay there as the dependency
# lengthens. They are biased, but the bias does not grow without limit,
# because the part of the gradient they *do* keep -- the immediate influence of
# a neuron's parameters on its own state -- is also the largest part. UORO
# scores far worse per sample than either, which is exactly what being
# unbiased-but-rank-one looks like, and the next cell shows why that number is
# misleading. Truncated BPTT is near-perfect inside its window.

# %%
def alignment(cell_name="ctrnn", n_in=4, n=16, T=40, delays=(1, 2, 4, 8, 16, 32),
              n_seq=40, seed=0):
    rng = np.random.default_rng(seed)
    base = make_cell(cell_name, n_in, n, estimator="rtrl", rng=np.random.default_rng(0))
    theta = base.theta.copy()
    methods = ["rflo", "snap1", "uoro", "bptt-4", "bptt-16"]
    out = {m: [] for m in methods}
    for delay in delays:
        acc = {m: [] for m in methods}
        for s in range(n_seq):
            xs, target = make_sequence(rng, T, n_in, delay)
            exact, _ = gradient(cell_name, "rtrl", xs, target, theta, seed=0)
            for m in methods:
                if m.startswith("bptt-"):
                    g, _ = gradient(cell_name, "rtrl", xs, target, theta, seed=0,
                                    truncate=int(m.split("-")[1]))
                else:
                    g, _ = gradient(cell_name, m, xs, target, theta, seed=1000 + s)
                num = float((g * exact).sum())
                den = float(np.linalg.norm(g) * np.linalg.norm(exact)) + 1e-12
                acc[m].append(num / den)
        for m in methods:
            out[m].append(float(np.mean(acc[m])))
    return delays, out


delays, align = alignment()
print("\n  cosine similarity to the exact RTRL gradient")
print("  delay:      " + "".join(f"{d:>8}" for d in delays))
for m, vals in align.items():
    print(f"  {m:<11}" + "".join(f"{v:8.3f}" for v in vals))

# %% [markdown]
# ## What each one costs
#
# The numbers that decide whether an algorithm fits on a car: bytes of carried
# influence, and microseconds per step.

# %%
import time

def cost_table(cell_name="ctrnn", n_in=20, n=32, steps=300):
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(steps, n_in))
    rows = []
    for est in ("rtrl", "uoro", "snap1", "rflo", "none"):
        cell = make_cell(cell_name, n_in, n, estimator=est, rng=np.random.default_rng(0))
        cell.reset_state()
        t0 = time.perf_counter()
        for x in xs:
            cell.step(x)
        dt = (time.perf_counter() - t0) / steps * 1e6
        rows.append((est, cell.influence_bytes(), dt))
    return rows


print(f"\n  cell = ctrnn, n = 32, n_in = 20, p = {make_cell('ctrnn', 20, 32).p}")
print(f"  {'estimator':<10} {'influence':>12} {'us/step':>10}")
for est, nbytes, dt in cost_table():
    print(f"  {est:<10} {nbytes / 1024:>9.1f} KiB {dt:>10.1f}")

# %% [markdown]
# ## The plot
#
# Same numbers, as a picture. The x axis is how many steps the credit has to
# travel; the y axis is how much of the true gradient direction survives.

# %%
def plot(delays, align, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"rflo": "-o", "snap1": "-s", "uoro": "-^", "bptt-4": "--v", "bptt-16": "--d"}
    for m, vals in align.items():
        ax.plot(delays, vals, styles.get(m, "-o"), label=m)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("how many steps back the credit has to travel")
    ax.set_ylabel("cosine similarity to the exact gradient")
    ax.set_title("What each online gradient estimator keeps (CT-RNN, n=16)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        print(f"  wrote {path}")
    return fig


# %% [markdown]
# ## UORO scored 0.1. Is it broken?
#
# No -- it is unbiased, which is a statement about the *average* of many
# samples and says nothing about any one of them. Average `k` independent UORO
# estimates of the same gradient and watch the alignment climb. Nothing else
# here does that: RFLO and SnAp-1 average to their own biased answer no matter
# how many samples you take, and get no better.
#
# Which of those you want depends on what you are doing. Over a long run with
# many small updates, noise averages out and bias does not, and UORO's
# guarantee is real. At batch size one on a car, with an update every 50 ms
# and no second chance, a quiet wrong direction may well beat a loud right one.

# %%
def uoro_averaging(cell_name="ctrnn", n_in=4, n=16, T=40, delay=8, ks=(1, 4, 16, 64, 256), seed=0):
    rng = np.random.default_rng(seed)
    base = make_cell(cell_name, n_in, n, estimator="rtrl", rng=np.random.default_rng(0))
    theta = base.theta.copy()
    xs, target = make_sequence(rng, T, n_in, delay)
    exact, _ = gradient(cell_name, "rtrl", xs, target, theta, seed=0)
    biased, _ = gradient(cell_name, "rflo", xs, target, theta, seed=0)

    def cos(g):
        return float((g * exact).sum() / (np.linalg.norm(g) * np.linalg.norm(exact) + 1e-12))

    rows, running, done = [], np.zeros_like(exact), 0
    for k in ks:
        while done < k:
            g, _ = gradient(cell_name, "uoro", xs, target, theta, seed=9000 + done)
            running += g
            done += 1
        rows.append((k, cos(running / done)))
    return rows, cos(biased)


rows, rflo_cos = uoro_averaging()
print("\n  averaging k independent UORO samples of the SAME gradient:")
for k, c in rows:
    print(f"    k = {k:>4}   cosine to exact = {c:6.3f}")
print(f"    RFLO, for comparison, does not improve with k: {rflo_cos:6.3f}")

# %%
outdir = ROOT / "runs"
outdir.mkdir(exist_ok=True)
plot(delays, align, outdir / "lesson02_gradient_alignment.png")

# %% [markdown]
# ## What to take from this
#
# * **RFLO and SnAp-1 are cheap and biased**, and the bias is bounded rather
#   than catastrophic: they keep the influence of a neuron's parameters on its
#   own state, which is the bulk of the gradient, and throw away credit that
#   travelled through a synapse. More samples never fix it.
# * **SnAp-1 barely differs from RFLO here.** It keeps one term more — each
#   neuron's *self*-recurrence, `W_rec[i,i]` — and at this initialisation those
#   diagonal weights are `O(1/sqrt(n))`, so there is very little there to keep.
#   Re-run with a cell whose self-recurrence is large and the two separate;
#   that is the honest scope of the difference, and it is smaller than the
#   number of papers about it suggests.
# * **UORO is unbiased and extremely noisy.** One sample is a rank-one sketch
#   and scores badly; the average of a few hundred is excellent. Which is the
#   better deal depends entirely on how many updates you get to average over.
# * **Truncated BPTT is exact inside its window and blind outside it** — a
#   cliff, not a decay — and it pays for that window in stored activations and
#   in not being able to update until the window closes.
#
# RTRRL takes the RFLO bargain. Lesson 4 puts it together with a reward.
