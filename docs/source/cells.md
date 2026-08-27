# The recurrent cells

Four cells plus a control, chosen so each adds exactly one mechanism to the one
before. All of them plug into any of the five {doc}`estimators`, because the
cell decides what the state does and the estimator decides what you are allowed
to know about how it got there.

| `--cell` | update | adds | reference |
|---|---|---|---|
| `ctrnn` | $h_{t+1} = h_t + (\tanh(W\xi) - h_t)/\tau$ | a learned but input-*independent* time constant | Funahashi & Nakamura 1993 |
| `ltc` | $h_{t+1} = (h_t + \Delta t\, f A)/(1 + \Delta t(1/\tau + f))$ | $\tau$ becomes a function of the input -- *liquid* | [Hasani et al. 2021](https://arxiv.org/abs/2006.04439) |
| `lrcu` | $h_{t+1} = (1 - \epsilon\sigma(f))h_t + \epsilon\tanh(u)\,e$ | the *capacitance* becomes one too | [Farsang et al. 2024](https://arxiv.org/abs/2403.08791) |
| `ligru` | $h_{t+1} = z h_t + (1-z)\tanh(W_c\xi)$ | gating without continuous time, as the control | [Ravanelli et al. 2018](https://arxiv.org/abs/1803.10225) |
| `liquid_gru` | $h_{t+1} = (h_t + \Delta t\, g\, c)/(1 + \Delta t\, g)$, $g = 1/\tau + z$ | the gate becomes a *conductance* with a floor | this repo |
| `mlp` | $h_{t+1} = \tanh(W x_t)$ | nothing; no recurrence at all | -- |

**LRCU is here because of the hardware paper.** Lemmel, Resch, Farsang,
Hasani, Rus & Grosu ([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)) put
RTRRL on a real 1:10 RoboRacer with an event camera and found the LRC cell the
one that gets on best with it -- which makes it the most directly relevant
result in the literature to what this playground is for.

## `liquid_gru`: the repo's own hybrid, and why it exists

Not from a paper. The nearby things — CT-GRU (Mozer et al. 2017), LTC-SE's
CT-GRU-style gates, and the
[continuous-time readings of the GRU](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.678158/full)
— are all motivated by *supervised* modelling. This one is motivated by the
learning rule, which is a different argument.

RFLO carries an influence updated as $P \leftarrow \text{leak}\cdot P +
\text{immediate}$. That is a geometric series and it converges only while
$\text{leak} < 1$. A LiGRU whose update gate saturates has $\text{leak} = z =
1$ — a unit that has learned to remember perfectly, which is a reasonable thing
for a memory task to want, and **an influence sum that never decays**. It
overflows quietly, tens of thousands of steps into a run. `OnlineCell` patches
that with `leak_max = 0.99`, an arbitrary numerical cap.

Take the leak structure from LTC and the *target* from a GRU — pull the state
towards $\tanh(W_c \xi)$ rather than towards a constant, at a rate
$g = 1/\tau + z$ — and the leak becomes

$$\text{leak} = \frac{1}{1 + \Delta t(1/\tau + z)} \le \frac{1}{1 + \Delta t/\tau} < 1$$

**bounded below 1 by construction, for any gate value.** The arbitrary cap
becomes a learned per-neuron parameter with a physical meaning. A test asserts
this with the cap switched off: driven to saturation, LiGRU's leak reaches
1.0000 and `liquid_gru`'s stops at 0.844.

**The price, measured.** That floor is also a ceiling on memory. On
MemoryChain-8 (200k steps, 3 seeds, optimum +1.0):

| `tau_init` | leak floor | MemoryChain-8 |
|---|---|---|
| (1, 8) | ≤ 0.89 | +0.291 ± 0.404 |
| (4, 40) | ≤ 0.976 | +0.552 ± 0.393 |
| **(10, 50)** — the default | ≤ 0.980 | **+0.775 ± 0.090** |
| LiGRU (no floor) | 1.0 | +0.883 ± 0.037 |

Monotone, and it approaches LiGRU as the floor approaches 1 — exactly what the
algebra says. On `lanekeep` the same three settings give 351 / 313 / 372, all
inside the seed spread, so long time constants are free there; hence the
default.

**It does not win**, and the shape of the loss needed eight seeds rather than
three to see:

| task | `ligru` | `liquid_gru` | difference |
|---|---|---|---|
| MemoryChain-8 (optimum +1.0) | **+0.835** ± 0.092 | +0.565 ± 0.322 | +0.27, **2.1 SE — separated** |
| lanekeep | 436 ± 87 | 337 ± 158 | +99, 1.5 SE — **not separated** |

*(8 seeds each. SE is the standard error of the difference of means.)*

So: LiGRU is genuinely better **on the memory task**, and on the driving task
the two are **not distinguishable at eight seeds** — an earlier three-seed run
here claimed otherwise and was wrong.

The per-seed numbers say something the means hide. `liquid_gru` on
MemoryChain-8 is **bimodal**: `[0.84, 0.83, 0.65, 0.62, -0.00, 0.65, 0.06, 0.86]`.
Five of eight seeds land between 0.62 and 0.86; two collapse to zero. It is not
uniformly worse than LiGRU, it is *less reliable* — which is a different defect
and points at initialisation rather than at the update equation.

What it has that neither LiGRU nor LRCU has is an influence series that provably
converges with no numerical cap, and a measured price for it. Worth a file, not
worth a claim.

## Two things worth knowing before you write your own cell

**Every parameter must belong to exactly one neuron.** `theta` is an $(n, p)$
array: row $i$ is everything neuron $i$ owns. That is not packing convenience,
it is the structural property that makes local online learning possible --
without it the immediate Jacobian is not block-diagonal, RFLO's "keep only
$\partial h_i/\partial\theta_{ij}$" throws away a first-order effect rather
than a second-order one, and the influence array cannot represent what is
happening.

That is why `ligru` is here and a full **GRU is not**. The GRU's reset gate
computes $c = \tanh(W_c[x; r \odot h])$, so $W_r$ belonging to neuron $k$
reaches $h_i$ for every $i$ through $W_c[i,k]$. The boundary is drawn
explicitly in `nets/ligru.py` rather than papered over, because it is a real
and underdiscussed constraint on which architectures admit this kind of
training.

**Bound whatever drives the state.** An LTC settles towards its reversal
potential $A$; an LRCU towards $e$. Leave those unbounded and the state is
unbounded, and over a long online run with no batch to average over that is not
a remote possibility -- it is what happens, at about step 200,000, as a NaN.
Both are clipped in `post_update`, which is also the biophysically honest
choice: a reversal potential is a voltage, not a free scale.

There is a third, subtler one, and it caught this repo too. A unit that has
learned to hold its state perfectly has leak $= 1$, and an influence trace that
never decays is a sum that never converges. `OnlineCell` caps the decay at
`leak_max = 0.99`, an explicit influence horizon of about 100 steps -- the same
idea as $\lambda$ in an eligibility trace.

## Deliberately not here

**CfC** and **Liquid-S4** already live in
[`liquid-nn-playground`](https://github.com/agpoks/liquid-nn-playground) with
proper supervised benchmarks. What is new in this repo is not the cells, it is
training them one timestep at a time from a reward.
