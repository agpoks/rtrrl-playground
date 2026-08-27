# The safety filter

Learning by trial and error means, definitionally, making the errors. On a
simulated car that is free; on a real one it is a broken car, and it is the
single biggest reason online RL does not get deployed.

`rtrrl_playground/safety.py` implements a **predictive safety filter**
(Wabersich & Zeilinger, [Automatica 2021](https://arxiv.org/abs/1812.05506)),
from scratch and with no solver dependency.

```bash
python tutorial/10_safety_filter.py
```

## What it is

The filter sits between the agent and the actuator and answers one question per
step:

> If I apply this action, does a **safe backup plan** still exist afterwards?

Yes → apply it unchanged. No → apply the nearest action for which one does.
Formally, at every step:

$$\min_{u_0}\;\|u_0 - u_\text{learner}\| \quad\text{s.t.}\quad
  x_{k+1}=f(x_k,u_k),\; x_k \in \mathcal{X},\; u_k \in \mathcal{U},\;
  x_N \in \mathcal{X}_\text{safe}$$

$\mathcal{X}_\text{safe}$ is **"stopped, and on the track"** — a car at rest
inside the boundary can stay there forever, so it is control-invariant, and
reaching it certifies that the episode need not end badly. The backup that gets
there is full braking with the steering pointed back at the centreline.

Its virtue is what it does *not* do. No reward shaping, no restricted action
space, no constrained policy class. The learner is untouched in the interior
and constrained only at the boundary.

## Measured

20 episodes on `lanekeep`:

| policy | off-track | return | filtered |
|---|---|---|---|
| random | **100%** | 14 | — |
| random + filter | **0%** | 15 | 16.4% |
| wall-follower | 0% | 569 | — |
| wall-follower + filter | 0% | **569** | **0.0%** |

Both rows matter, and they say opposite-sounding things:

**The random policy goes from leaving the track every episode to never.** The
guarantee, on the worst policy available.

**The competent policy is filtered 0% of the time**, at an unchanged return. A
safety filter doing its job is *invisible* to a driver who was not going to
crash. If yours intervenes constantly on a good policy, it is not a filter, it
is a controller — and the agent is now learning against it rather than against
the task.

## With and without, while learning

```{image} _static/plots/results_safety.png
:alt: crashes and return with and without a safety filter
:width: 100%
```


RTRRL on `lanekeep`, 200k steps, 4 seeds. "Crashes during training" is the
fraction of training episodes that ended in a wall — on a real vehicle, the
number that matters most.

| filter | eval return | crashes **during training** | crashes at eval | steps overridden |
|---|---|---|---|---|
| none | 428 ± 97 | 61.3% | 30.0% | — |
| `assumed_grip=1.0` (default), `credit=executed` | 354 ± 136 | 21.3% | 6.2% | 15.8% |
| `assumed_grip=1.0`, `credit=proposed` | 356 ± 216 | 20.6% | 13.8% | 30.7% |
| **`assumed_grip=0.6` (worst case)** | **449 ± 155** | **0.0%** | **0.0%** | 17.5% |
| `assumed_grip=1.4` (optimistic) | 329 ± 53 | 60.8% | 51.2% | 7.8% |

Three things in that table, and the third is the one to take away.

**Safety was free — better than free.** The worst-case filter crashed *zero*
times in 200k steps of learning and still finished with the **highest** return
of any row. That is not a trade-off being won narrowly; it is the observation
that an episode which ends in a wall is an episode that stopped paying, so
never crashing is worth something to the return as well as to the car.

**The default filter is not safe, and says so.** `assumed_grip=1.0` still
crashed 21% of training episodes, because the environment draws grip from
0.6–1.4 and a filter assuming 1.0 is optimistic half the time. A safety filter
inherits its guarantee from its model and nothing else.

**An optimistic filter is worse than no filter.** At `assumed_grip=1.4` the
return is *below* the unfiltered agent and the crash rate is no better. It
certifies corners the car cannot take, and overrides good actions on the way.
A filter you cannot justify the model of is not a conservative choice.

**And `credit` matters more than expected.** Telling the agent about the action
that was actually executed halves the evaluation crash rate against telling it
about the one it proposed (6.2% vs 13.8%) and halves the intervention rate
(15.8% vs 30.7%). Neither is theoretically correct, but the agent evidently
learns more from what happened than from what it wanted.

## Three limitations, not caveats

**It is privileged.** The filter runs on the vehicle state, not on the agent's
nine beams. That is not cheating — on a real car it sits on the state
estimator, which is where it belongs — but the guarantee is only ever as good
as that estimate, and a filter tested against ground truth in simulation has
not been tested in the part that usually fails.

**It does not know the grip either.** `LaneKeep` redraws tyre grip every
episode and never observes it; the filter predicts with `assumed_grip`. Set it
above the truth and it certifies corners the car cannot take — crashes happen
*through* the filter. `tests/test_safety.py` asserts this rather than leaving
it as prose. Set it to the worst case and you buy safety with timidity. There
is no setting that is both fast and guaranteed.

**It makes the update off-policy.** The action reaching the environment is not
always the one the policy chose, and TD(λ) has no term for "the action was
replaced". `credit="executed"` tells the agent about what happened (the filter
becomes part of the environment); `credit="proposed"` tells it about what it
chose (on-policy gradient, misreported dynamics). Neither is correct, the
literature on learning through a shield does not agree either, and both are
implemented so the difference can be measured instead of argued.

## Two implementation notes

**Enumerate, don't solve.** With nine discrete actions the minimisation above
is not a solver problem — it is enumerate-and-check, ordered by distance from
the proposed action, so the argmin is exact. This is a real predictive safety
filter, not an approximation of one.

**The fast path is scalar Python, on purpose.** The certificate is a 25-step
*sequential* recursion on five numbers. Through NumPy that is ~750 array calls
of one element each, and the per-call overhead — not the arithmetic — costs
about 4 ms per control tick. The identical arithmetic in Python floats costs
about 0.3 ms. Since the proposed action is safe essentially always on a
competent policy, that is the cost of the filter in practice. The vectorised
nine-candidate version is kept for the fallback, and a test asserts the two
certificates agree exactly.

## Not implemented, deliberately

**Control barrier functions** (Ames et al., [ECC 2019](https://arxiv.org/abs/1903.11199))
are the pointwise alternative: certify safety through a scalar function of the
current state, with no prediction. The trade is offline design effort — finding
a valid CBF — against online compute. On a task where the constraint is a
bitmap rather than an analytic set, exhibiting a backup trajectory is much
easier than finding the function.
