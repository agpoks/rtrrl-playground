# Every RTRRL variant, listed and measured

RTRRL as implemented here is **six independent choices**, not one algorithm.
The docs used to tabulate two of them; the other four were defaults that had
never been shown to be the right ones. This page is the full inventory and the
sweep that checks each.

```{contents}
:local:
:depth: 2
```

## The six axes

| axis | values | what it changes |
|---|---|---|
| `cell` | `ctrnn` `ltc` `lrcu` `ligru` `liquid_gru` `physics_ligru` `mlp` | the recurrent state's update law |
| `estimator` | `rflo` `snap1` `uoro` `rtrl` `hybrid` `none` | how the influence $\partial h_t/\partial\theta$ is carried |
| `feedback` | `random` `symmetric` | whether the actor's error travels back through a fixed random matrix or the true weights |
| `critic_update` | `true-online` `paper` `accumulating` | which TD($\lambda$) trace |
| `critic_lr_mode` | `normalized` `fixed` | whether $\alpha_C$ is divided by $\|\phi\|^2$ |
| `meta_inputs` | `True` `False` | whether the previous action and reward are fed back as input |

That is $7 \times 6 \times 2 \times 3 \times 2 \times 2 = 1008$ distinct agents.
Sweeping the product is not useful; each axis below is swept **with the others
at their defaults**, so a row reads "this choice, against the default, on this
task".

## How to run it

```bash
python benchmarks/all_variants.py --env lanekeep    --steps 150000 --seeds 4
python benchmarks/all_variants.py --env memory-chain --steps 200000 --seeds 4
```

Results land in `benchmarks/results/variants_<env>.json`, which is what the
tables below are generated from.

## LaneKeep — 150k steps, 4 seeds

Defaults are `ligru` / `rflo` / `random` / `true-online` / `normalized` /
`meta_inputs=True`, and appear as the first row of every block.

### Cells

| cell | return | sd |
|---|---|---|
| **`ligru`** | **530.1** | 65.6 |
| `mlp` | 513.4 | 24.0 |
| `physics_ligru` | 510.5 | 52.9 |
| `liquid_gru` | 400.0 | 14.9 |
| `ctrnn` | 389.0 | 35.4 |
| `lrcu` | 301.2 | 202.4 |
| `ltc` | 295.1 | 173.4 |

**`mlp` — no recurrence at all — comes second.** That is the single most
important row on this page and it is not a flattering one: `lanekeep` is
largely a reactive task, the nine beams contain most of what a controller
needs, and memory buys about 3% over having none. Any claim that a recurrent
cell is *necessary* has to be made on `memory-chain`, not here.

`lrcu` and `ltc` have standard deviations of 200 — half the mean. Those are not
worse cells so much as less stable ones at these hyperparameters.

### Gradient estimators

| estimator | return | sd |
|---|---|---|
| **`rflo`** | **530.1** | 65.6 |
| `hybrid` | 526.3 | 62.8 |
| `rtrl` (exact) | 523.8 | 31.7 |
| `none` (reservoir) | 484.8 | 62.3 |
| `snap1` | 416.7 | 243.5 |
| `uoro` | 397.6 | 221.0 |

Two results here, and both are awkward for the premise of a gradient zoo.

**Exact RTRL is not better than RFLO** — 523.8 against 530.1, comfortably
inside one standard deviation. RFLO's diagonal approximation throws away every
off-diagonal term of the influence matrix and loses nothing measurable.

**`none` — no gradient through the recurrence at all, a reservoir — scores
485**, 8% below the best. Most of what the recurrent parameters contribute on
this task is available from a random recurrent layer with only the readout
trained.

Together with the `mlp` row, the honest summary of `lanekeep` is that it does
not test what an online recurrent gradient estimator is for.

### Feedback path

| feedback | return | sd |
|---|---|---|
| **`random`** (feedback alignment) | **530.1** | 65.6 |
| `symmetric` (the true gradient) | 440.2 | 103.2 |

**The fixed random feedback matrix beats the true gradient by 20%.** This is
the feedback-alignment result, and it is a genuine surprise every time: the
network learns to align its forward weights with a matrix that was drawn once
and never touched. It is also why the default is `random` and not an
approximation apologised for.

### Critic

| `critic_update` | return | sd | | `critic_lr_mode` | return | sd |
|---|---|---|---|---|---|---|
| **`true-online`** | **530.1** | 65.6 | | **`normalized`** | **530.1** | 65.6 |
| `accumulating` | 509.3 | 59.9 | | `fixed` | 443.0 | 62.5 |
| `paper` | 507.1 | 7.5 | | | | |

The trace variant is worth ~4% and is inside the spread. The learning-rate mode
is worth **20%** and is not: dividing $\alpha_C$ by $\|\phi\|^2$ is the
difference between a critic that tracks and one that oscillates, and it is the
single most valuable non-obvious default in the implementation.

### Meta-RL inputs

| `meta_inputs` | return | sd |
|---|---|---|
| **`True`** | **530.1** | 65.6 |
| `False` | 426.6 | 50.9 |

Feeding the previous action and reward back as input is worth 24%. This is the
"meta" in meta-RL and it is doing real work even on a task with no explicit
task distribution — the previous reward is the only observation the agent gets
of the grip it is never told.

## What no sweep found

**Every axis is won by its default.** That is expected — the defaults were
tuned — but it is worth stating plainly rather than presenting the table as a
discovery. The value of the sweep is the *sizes*: it says the cell choice is
worth 3% against no memory at all, the estimator choice is worth less than its
spread, and the two boring knobs (`critic_lr_mode`, `meta_inputs`) are worth
20–24% each. Effort spent choosing a cell is effort not spent on the things
that actually move the number.

## Memory-chain — 200k steps, 4 seeds

`lanekeep` cannot distinguish a recurrent agent from a memoryless one, so the
same sweep is run on `memory-chain`, where optimal is $+1.0$ and guessing is
$0.0$.

### Cells — the only axis that separates

| cell | return | sd |
|---|---|---|
| **`ligru`** | **+1.00** | 0.00 |
| **`physics_ligru`** | **+1.00** | 0.00 |
| `lrcu` | +0.80 | 0.40 |
| `liquid_gru` | +0.30 | 0.40 |
| `mlp` | +0.30 | 0.40 |
| `ctrnn` | +0.20 | 0.50 |
| `ltc` | −0.10 | 0.10 |

This is the table to cite for any claim about *memory*. `ligru` solves the task
outright on every seed; `ctrnn` — RTRRL's own cell — reaches +0.20, and `ltc`
does not learn it at all at these hyperparameters.

**`mlp` scoring +0.30 needs an explanation, and it is not that a memoryless
network has memory.** With `meta_inputs=True` the previous action and reward are
fed back as inputs, which gives *any* architecture a one-step memory. The MLP is
memoryless in its weights and not in its inputs. A genuinely memoryless
comparison is `mlp` with `meta_inputs=False`, and the honest reading of this row
is that one step of history is worth about a third of the task.

### Everything else saturates

| axis | spread across all values |
|---|---|
| `estimator` | +1.00 for all but `uoro` (+0.80) |
| `feedback` | +1.00 for both |
| `critic_update` | +1.00 for all three |
| `critic_lr_mode` | +1.00 for both |
| `meta_inputs` | +1.00 / +0.80 |

At 200k steps with a `ligru` cell the task is solved by every configuration
tried, so **this benchmark separates cells and nothing else**. That is a
statement about the benchmark, not about the axes: a sweep whose rows are all
1.00 has no resolving power left and needs a harder task or a shorter budget
before it can be used to compare anything.

Taken with the LaneKeep tables, the two benchmarks are complementary and
individually misleading: `lanekeep` cannot see memory, and `memory-chain`
cannot see anything else.

## Not swept here

`n_hidden`, the learning rates, `gamma`, the three $\lambda$s, `clip`,
`reward_scale` and `entropy_coef` are continuous hyperparameters rather than
variants, and sweeping them belongs in
[`benchmarks/sweep.py`](benchmarks.md). The tuning that produced the current
defaults — including that the paper's $\alpha_R = 10^{-3}$ is about 100x too
high on these tasks — is in [Measured results](benchmark_results.md).
