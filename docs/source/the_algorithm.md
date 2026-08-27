# The algorithm

RTRRL is three old pieces composed. Each is separately unremarkable; the claim
is that together they give an agent that learns from a single stream of
experience at constant memory and constant cost per step, with the update
available *during* the episode.

Reference: Julian Lemmel, Radu Grosu, *"Real-Time Recurrent Reinforcement
Learning"*, AAAI 2025, [arXiv:2311.04830](https://arxiv.org/abs/2311.04830).

```{contents}
:local:
:depth: 2
```

## The whole thing in one figure

```{image} _static/diagrams/rtrrl_architecture.png
:alt: the RTRRL architecture, showing what is carried forward and what is not
:width: 100%
```

Every arrow points forward in time. The only things that cross a timestep
boundary are the **influence matrix** $\partial h_t/\partial\theta$ and the
eligibility traces — both fixed-size, both updated in place. There is no
backward pass, no replay buffer and no batch, which is the entire difference
between this and every other recurrent actor-critic.

The dashed return path is the meta-RL loop: the previous action and reward are
fed back as *inputs*, not only used as targets. On `lanekeep` that is worth 24%
of the return ([Variants](variants.md#meta-rl-inputs)) — the previous reward is
the only observation the agent ever gets of the grip it is never told.

The figure is also in `docs/tikz/fig_rtrrl.tex` as TikZ, in the greyscale IEEE
style, for dropping into a paper.

## 1. A meta-RL recurrent architecture

One recurrent cell whose input at each step is the observation, the **previous
action**, and the **previous reward**:

$$\xi_t = [\,o_t\;;\;a_{t-1}\;;\;r_{t-1}\;;\;h_t\;;\;1\,]$$

and whose state drives a linear actor and a linear critic:

$$\pi_t = \mathrm{softmax}(A h_t + b), \qquad v_t = c^\top h_t + d.$$

Feeding the last action and reward back in is what makes it *meta*: the network
can in principle read its own recent history of consequences out of its own
input, which is what lets it adapt within an episode to something it cannot
observe -- the grip level in `lanekeep`, the speed of the car ahead in
`overtake`.

## 2. The recurrent gradient, carried forwards

The cell keeps an **influence** array $J_t = \partial h_t / \partial\theta$
and updates it in the same pass as the activations. For a CT-RNN with
$h_{t+1} = h_t + (\phi(W\xi_t) - h_t)/\tau$, the exact recursion is

$$J_{t+1} = D_t J_t + \frac{1}{\tau}\phi'(W\xi_t)\,\xi_t^\top,
\qquad D_t = \frac{\partial h_{t+1}}{\partial h_t},$$

which costs $O(n^2 p)$ memory. **RFLO** (Murray, eLife 2019) -- what RTRRL
uses -- keeps only the terms where the weight belongs to the neuron whose state
is being differentiated, and replaces $D_t$ by its leak:

$$P_{t+1} = \Big(1 - \tfrac{1}{\tau}\Big) P_t + \tfrac{1}{\tau}\,\phi'(W\xi_t)\,\xi_t^\top$$

One array the same shape as $W$, one outer product per step, and entirely
local: the update to $W_{ij}$ needs neuron $i$'s own activity and input $j$.
Four other options are implemented; see {doc}`estimators`.

## 3. TD(λ) actor-critic with eligibility traces

One scalar per step,

$$\delta_t = r_t + \gamma\, v(h_{t+1}) - v(h_t),$$

multiplied into a trace per parameter group. The recurrent cell's trace is
where the two halves meet:

$$g_t = B_A^\top \nabla_z\big[\log \pi_t[a_t] + \eta H(\pi_t)\big] + B_C,
\qquad e_W \leftarrow \gamma\lambda\, e_W + g_t \odot P_t,
\qquad W \leftarrow W + \alpha_R\, \delta_t\, e_W.$$

$B_A$ and $B_C$ are **fixed random matrices** -- feedback alignment (Lillicrap
et al. 2016), so the learning signal reaching the cell never depends on the
heads' own weights. `--feedback symmetric` uses the true transpose instead; on
these tasks it barely moves, which is the whole surprise of that result.

## Timing, which is the part that is easy to get wrong

At step $t$ the agent holds $h_t$ and the influence $J_t$ **of** $h_t$. It
draws $a_t$ from $h_t$; the environment answers with $r_t$ and $o_{t+1}$; only
then does the cell advance. The TD error needs $v(h_{t+1})$, but **every trace
is fed the time-$t$ quantities** -- $h_t$, $J_t$, and the policy gradient at
the action actually taken.

Pairing $J_{t+1}$ with $h_t$ does not crash, does not warn, and costs most of
the learning. It was a real bug in this repo, found by the difference between
the from-scratch lesson (which got the order right) and the library agent
(which did not). The fix is why `algo.py` computes `dW = cell.grad(g)` on the
line *before* `cell.step(...)`, with a comment saying so.

## What it looks like

```{image} _static/plots/lanekeep.png
:alt: a scripted driver and a learned one, coloured by speed
:width: 100%
```

Both lift in the corners -- yellow on the straights, green through the turns.
The learned agent found that trade-off without ever observing its own speed.

```{image} _static/plots/learning_curve.png
:alt: return against environment steps
:width: 85%
:align: center
```

## What it is not

A reproduction. This is a compact reimplementation for taking apart, on small
environments that run on a laptop. Several of the paper's hyperparameters do
not work here and the reasons are written down in
`algos/rtrrl/README.md`; the environments are not the paper's; the numbers in
{doc}`benchmarks` are this repo's own.
