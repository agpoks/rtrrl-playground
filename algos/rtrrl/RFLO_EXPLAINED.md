# RFLO + eligibility traces + the event trigger, explained simply

`README.md` tells you what the pieces are and how to run them.
[`../../papers/README.md`](../../papers/README.md) links the papers. This is
the missing middle layer: what each equation in `algo.py` actually computes and
why it's shaped that way, for someone who wants to follow the code line by line
without re-deriving TD(λ) or RFLO from scratch first.

Everything here uses the same symbols as
[`event-driven-rtrl/paper/main.tex`](../../../event-driven-rtrl/paper/main.tex)
(Section "Method"), so the two documents cross-reference cleanly. `algo.py`
additionally implements the event trigger (`spike_threshold`), which the paper
formalises as Algorithm 1 — that's the part not in `event-driven-rtrl`'s own
port (`edrtrl/agent.py` deliberately leaves it out, to keep the two mechanisms
measurable one at a time).

## The one thing to hold in your head

At every timestep the agent does **four things**, in a fixed order, and does
them **once each**, on **one transition**, with **no history stored**:

1. push the transition through the recurrent cell (forward pass),
2. update a scalar surprise measure (the TD error),
3. update a handful of running sums (eligibility traces),
4. nudge the weights in the direction those sums point.

There is no backward pass anywhere. Credit for something that happened five
steps ago is not computed later by unrolling the past — it is already sitting
in a trace, decaying a little every step, ready to be used the instant a
reward arrives. That's the whole trick, and it's why this runs forever at
constant memory instead of needing a growing buffer.

## 1. The forward pass, and why it needs a shortcut

A recurrent cell's state update is $h_{t+1} = f(h_t, \xi_t; \theta)$. To learn
$\theta$ online (no replay, no unrolling), you eventually need
$\partial h_t / \partial\theta$ — how the state right now depends on every
weight the network has ever used. The honest, exact version of that quantity
(RTRL) is an $n \times n \times p$ object and costs $O(n^3)$ per step. That's
fine for toy networks and hopeless for anything that has to run every 20 ms.

**RFLO's shortcut** (Murray 2019): keep only the *diagonal* piece — for each
unit $i$, keep $\partial h_{t,i}/\partial\theta_i$, the influence of unit
$i$'s *own* weights on its *own* state, and throw away how unit $i$'s weights
affect unit $j$'s state. That collapses the object to $n \times p$ — one row
per neuron, as wide as the number of parameters *one* neuron owns. This is
biased (it's not the true gradient) but it's cheap, and empirically it's
enough to learn with.

```
P_i  <-  leak_i * P_i  +  immediate_i
```

In plain words: **row $i$ of the influence matrix is an exponential moving
average of how unit $i$'s recent inputs have been pushing on unit $i$'s own
weights.** `leak_i` is whatever this timestep's version of "how much of the
old state survives" is for that unit (a CT-RNN's time constant, an LTC's
input-dependent one, a LIF's membrane decay — see below). `immediate_i` is
this tick's fresh contribution, e.g. $\partial(\text{drive}_i)/\partial\theta_i$.
This is exactly the pattern of an eligibility trace, and it's the same
mechanism the traces below use again for the heads — RFLO is really "give
every neuron its own eligibility trace over its own weights."

**Order matters, and it's the easiest bug to write.** At tick $t$ you use
$P_t$ (paired with $h_t$) to build a gradient *before* advancing the cell.
Advance first and you're one step out of alignment: $P_{t+1}$ paired with
$h_t$. This doesn't crash or warn. It just quietly costs most of the
learning — see the timing note in `README.md`.

## 2. The TD error: one number that carries all the credit

```
delta_t = r_t + gamma * V(h_{t+1}) - V(h_t)
```

This is "how surprised was the critic." If the world went better than the
critic expected, `delta_t > 0`; worse, `delta_t < 0`; exactly as expected,
`delta_t ~ 0`. Every weight update in the whole system — critic, actor,
cell — is `delta_t` times *something else* (a trace). `delta_t` is the only
place the *reward* enters; everything downstream of it is bookkeeping about
*which* weights deserve credit, not about whether there's credit to give.

One subtlety worth keeping straight: `terminated` means the episode is
genuinely over and $V(h_{t+1})$ is zero by definition (there's no future to
bootstrap from). `truncated` means the harness cut the episode off (a time
limit) — the world didn't end, so $V(h_{t+1})$ is still the best guess
available and must **not** be zeroed. Confusing the two makes the critic
learn that every time-limit cutoff was a disaster.

## 3. Eligibility traces: "how much does this weight deserve, right now"

A trace is a running sum that lets one scalar (`delta_t`) reach backward to
weights that were used several steps ago, without storing those steps.
Two flavours are used here, for different reasons.

**Accumulating trace** — the plain one:

```
e <- gamma * lambda * e + x
```

Read this as "keep `gamma*lambda` of what you already had (a discount factor
times a recency factor) and add this tick's fresh evidence `x`." It's a
leaky sum: recent contributions count close to fully, older ones fade
geometrically. This is what the **actor** uses (`x` = the policy's own score
function, $\nabla_{\theta_A}\log\pi(a_t\mid h_t)$) and what the **cell** uses
(`x` = the gradient RFLO just handed it, $g_{t,i} P_{t,i}$).

**Dutch trace** — the corrected one, for the critic only:

```
e <- gamma*lambda*e + alpha*x - alpha*gamma*lambda*(e . x)*x
```

The first two terms are the same accumulating idea, scaled by the learning
rate `alpha` up front (that's why `alpha` lives inside the trace here and not
at the update site, unlike the actor and cell). The third term is the part
that makes this "true-online" TD(λ) rather than ordinary TD(λ): it subtracts
out exactly the part of the new observation `x` that the trace already
implied, so that stepping through the recursion online gives the *same
answer* as if you'd computed the whole $\lambda$-return offline, after the
fact, with perfect hindsight. Without that correction term, online TD(λ) and
offline TD(λ) slowly diverge from each other as training goes on; with it,
they're provably identical at every step (van Seijen et al., 2016).

Why does only the critic get this treatment? Because the critic's update is
also paired with a second correction, the $(V_t - V^{\text{old}})$ term below
— together they're what "true-online" means, and the actor and cell traces
don't have an analogous exact target to match, so the simpler accumulating
trace is what's used for them.

## 4. The three updates

**Critic** (true-online TD(λ), van Seijen et al. 2016, Algorithm 1):

```
w <- w + (delta_t + V_t - V_old) * e_C - alpha_C * (V_t - V_old) * phi_t
V_old <- V_{t+1}          # for next tick
```

`V_old` is the value this *same state* had under the *previous* set of
weights — it's what lets the update correct for the fact that the weights
themselves have been moving while the trace accumulated. This is the term
that's easy to leave out (plain TD(λ) with a Dutch trace omits it) and the
one that makes the online result match the offline $\lambda$-return exactly.

**Actor** (policy gradient, credit assigned via the trace):

```
theta_A <- theta_A + alpha_A * delta_t * e_A
```

Read this as: "move the policy's parameters in the direction that would have
made recently-taken actions more likely, scaled by how much better or worse
things turned out than expected, and weighted by how recently each action was
taken (that's what `e_A`'s decay encodes)."

**Cell** (RFLO's weight update):

```
theta <- theta + alpha_R * delta_t * e_theta,   e_theta = accumulating_trace(e_theta, g_i * P_i)
```

Same shape as the actor's update, but the "fresh evidence" going into the
trace is $g_{t,i} P_{t,i}$ — the learning signal from the heads
(`g_{t,i}`) multiplied by the influence RFLO already computed
(`P_{t,i}`), one factor per neuron.

**Where `g` comes from — feedback alignment.** The exact backprop signal into
the cell would need the actor's and critic's *output* weight matrices
transposed and multiplied through. Feedback alignment (Lillicrap et al. 2016)
replaces those transposed weights with **fixed, random** matrices $B_A, B_C$
that never change:

```
g_t = B_A^T . grad_z(log pi(a_t | h_t)) + B_C
```

This is not an approximation of backprop's gradient — it's a *different*,
biased direction that nonetheless reliably points the cell's weights the right
way in practice (the mechanism, empirically, is that the forward weights
rotate to align themselves with the fixed random feedback matrices over
training, not the other way around). It matters here specifically because it
keeps the whole computation forward-only: there is no backward graph to walk,
so there's nothing to unroll and nothing to store.

## 5. All three normalised by the same thing, and why

`alpha_C` and `alpha_A` are both divided by $\|\phi_t\|^2$ (with
$\phi_t = [h_t; 1]$) rather than used as fixed constants. This is the
normalised-LMS trick: as training progresses the recurrent state fills out
and $\|\phi_t\|^2$ drifts upward (from ~1 at initialisation to ~$n$ once every
unit is saturated), and a fixed learning rate that's stable early becomes
unstable later. Dividing by $\|\phi_t\|^2$ pins the *effective* step size at
exactly `alpha_C` (or `alpha_A`) for the whole run, rather than only for the
first few hundred steps.

## 6. The event trigger: defer the write, not the computation

Steps 1–3 above (forward pass, TD error, traces) run **every tick, always** —
nothing about them is optional. What the trigger gates is only step 4, the
moment the pending updates actually get written into the weights:

```
m <- m + |delta_t|
if m >= threshold:
    w         <- w + sum(pending critic updates)
    theta_A   <- theta_A + sum(pending actor updates)
    theta     <- theta + sum(pending cell updates)
    m <- 0
else:
    keep accumulating; don't discard anything
```

Think of `m` exactly like a leaky-integrate-and-fire neuron's membrane
potential, because that's precisely the model it's borrowed from: it
integrates a magnitude (`|delta_t|` instead of a synaptic current) and
"spikes" — releasing the accumulated update — when it crosses a threshold,
then resets to zero. Two things distinguish this from the lazy alternative
("just apply the update every $N$ steps"):

- **Nothing is thrown away.** Between two flushes, every tick's would-be
  update is still summed into the pending totals above (`e_C`, `e_A`,
  `e_theta` keep decaying and accumulating exactly as if nothing were being
  withheld). The trigger changes *when* the sum is applied, never *what* the
  sum is.
- **The waiting time is set by the data, not a clock.** A stretch where the
  world is behaving as the critic predicted produces small `|delta_t|` every
  tick, so `m` crawls toward the threshold slowly and the weights are left
  alone. A surprise produces one or two large `|delta_t|` values and `m`
  crosses the threshold almost immediately. `threshold = 0` fires on every
  tick and recovers plain RTRRL exactly — same code path, nothing skipped.

This is event-triggered control (Heemels et al. 2012) applied to the
*learning update* rather than to actuation, and it's a genuinely different
knob from the state-gating cell in `edrtrl/cell.py`: that one skips
*computing* a row of $P$ for a silent unit; this one always computes
everything and only defers *writing* it.

## The whole thing, as one loop

```
state:  h, P, e_C, e_A, e_theta, (pending sums), m      # all zero at episode start

every tick:
    a_t = sample from pi(. | h_t);  apply it;  observe r_t, o_{t+1}
    phi_t = [h_t; 1];  V_t = w . phi_t
    g_t = B_A^T . grad(log pi(a_t|h_t)) + B_C            # feedback alignment
    d_theta_t = g_t * P_t                                # RFLO's row-wise product
    h_{t+1}, P_{t+1} = cell_step(xi_t, theta)             # advance state AND influence together
    V_{t+1} = 0 if terminal else w . [h_{t+1}; 1]
    delta_t = r_t + gamma * V_{t+1} - V_t

    e_C     = dutch_trace(e_C, phi_t, ...)
    e_A     = accumulating_trace(e_A, grad(log pi), ...)
    e_theta = accumulating_trace(e_theta, d_theta_t, ...)

    pending_w      += (critic update built from e_C and delta_t)
    pending_thetaA += alpha_A * delta_t * e_A
    pending_theta  += alpha_R * delta_t * e_theta

    m += |delta_t|
    if m >= threshold:
        w, theta_A, theta += pending_w, pending_thetaA, pending_theta
        pending_* = 0;  m = 0
```

Everything above the `m += |delta_t|` line runs unconditionally, every tick,
whether or not the flush fires below it. That's the property the paper
depends on when it says the trigger "changes no state and no estimate — it
only defers a write."

The LaTeX version of this same loop, numbered and typeset as
Algorithm 1, is in `event-driven-rtrl/paper/main.tex`
(`\label{alg:etrtrl}`, Section "The full step, in one place").
