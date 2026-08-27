# Safe learning

Learning by trial and error means, definitionally, making the errors. On a
simulated car that is free; on a real one it is a broken car, and it is the
single biggest reason online RL does not get deployed.

Two answers are implemented here, opposites in method: one certifies safety by
**exhibiting a plan**, the other by **evaluating a function**. Both wrap an
agent identically, so they can be compared on the same task, the same nine
actions and the same model.

## Why a filter, and not one of the alternatives

There are four standard ways to stop an RL agent from doing something
catastrophic, and they differ in *where they pay*.

| approach | how it works | what it costs |
|---|---|---|
| reward shaping | a large negative reward for crashing | pays everywhere, guarantees nothing, and the agent must crash to learn it |
| constrained policy class | parameterise so unsafe actions are unreachable | pays everywhere; usually costs the optimum too |
| Lagrangian / CMDP | constrain the *expected* cost | a constraint in expectation is not a constraint; rare violations are allowed by construction |
| **safety filter** | leave the policy alone, veto individual actions | pays only at the boundary; guarantees hold per-step, not in expectation |

The last row is the one implemented here, twice. Its distinguishing property is
that it is **not a learning method at all** — it makes no assumption about the
agent, so the same filter wraps RTRRL, a wall-follower, or a uniform random
policy, and the guarantee does not depend on which. That is also the honest
statement of its weakness: everything it promises comes from its *model* of the
vehicle, and nothing from the learning.

`rtrrl_playground/safety.py` implements a **predictive safety filter**
(Wabersich & Zeilinger, [Automatica 2021](https://arxiv.org/abs/1812.05506)),
from scratch and with no solver dependency.

```bash
python tutorial/10_safety_filter.py
```

## Part 1 — The predictive safety filter

### What it is

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

### The certificate, in one picture

```{image} _static/plots/safety_certificate.png
:alt: nine candidate actions, each followed by its braking backup
:width: 100%
```

Nine lines, one per action. Each is **not** a plan the car intends to follow —
it is the emergency stop that would still be available *if that action were
taken*. Green if that escape route stays on the track and ends stopped, red if
it does not. The right panel is the same nine answers arranged as the action
grid the agent actually chooses from, which is the form the verdict reaches the
learner in.

The state is not hand-placed: it was found by running a random policy *behind*
the filter and stopping at the first step where the nine answers were not
unanimous. A competent policy never goes near the boundary, and an unfiltered
random one leaves the track before it gets interesting, so that regime — a
policy pushing at the limit and being held off it — is the only place the
picture has any content.

Read off what the filter has decided at 2.3 m/s pointed at the inside of a
corner: **steering into the wall is allowed only if you also brake.** Nobody
wrote that rule. It falls out of asking whether a stop still exists.

### Why the terminal set is not optional

The single design decision that separates this from an N-step lookahead is the
requirement that the backup **ends stopped**. It is worth spelling out what it
buys, because "check N steps ahead" sounds like the same thing and is not.

Let $\pi_b$ be the backup policy (full braking, steering to the centreline),
$\mathcal{X}$ the constraint set (on track, clear of obstacles), and

$$\mathcal{X}_\text{safe} \;=\; \{x \in \mathcal{X} : v \le v_\text{stop}\}$$

Call a state **certified** if the $\pi_b$-trajectory from it stays in
$\mathcal{X}$ for $N-1$ steps and lands in $\mathcal{X}_\text{safe}$.

*Claim.* If $x_{t+1}$ is certified, then at the next step the filter always has
at least one action it can admit — so it can never paint itself into a corner.

*Proof.* Take the candidate $u = \pi_b(x_{t+1})$, the first move of the very
backup that certified $x_{t+1}$. It leads to $x_{t+2}$, from which the same
backup trajectory continues, stays in $\mathcal{X}$ for its remaining $N-2$
steps, and reaches $\mathcal{X}_\text{safe}$. The certificate at $t+1$ asks for
$N-1$ steps, which is one more than remains — and that is exactly what
$\mathcal{X}_\text{safe}$ is for. A stopped car under full braking stays
stopped and stays where it is, so $\mathcal{X}_\text{safe}$ is invariant under
$\pi_b$ and the extra steps cost nothing. Therefore $u$ is certified, and the
filter is never empty. $\square$

That is **recursive feasibility**, and it is an induction, not a horizon. Drop
the terminal condition and it collapses immediately: an $N$-step lookahead will
happily approve full throttle at a wall $N+1$ steps away, then approve it again
one step later, and again, until the wall is $N$ steps away and every action is
already too late. The filter would have been feasible at every single step and
crashed anyway.

Two caveats, both real:

**The induction is about the model.** Everything above is a statement about
$f$, not about the car. If $f$ is optimistic the whole argument is sound and
irrelevant — see [the grip sweep](#it-does-not-know-the-grip-either), where an
optimistic filter is recursively feasible right up to the wall.

**The action set is discrete.** $\pi_b(x)$ is a continuous steering value and
the filter must round it onto the nine-action grid. Strictly, the claim holds
for the rounded backup only if the rounded backup is itself certifiable, which
is checked, not assumed — the fallback branch in `__call__` counts the states
where nothing at all is certifiable (`n_no_safe_action`) rather than pretending
they cannot occur.

### The algorithm, in full

```text
filter(x, u_learner):
    if certify(step(x, u_learner)):        # scalar fast path, ~0.3 ms
        return u_learner, not_intervened

    S = step(x, u)  for all nine u         # one vectorised control-rate step
    ok = certify(S)                        # nine backups at once
    if none(ok):
        n_no_safe_action += 1
        return brake_toward_centreline(x), intervened
    return nearest u with ok[u], intervened   # nearest in (steer, throttle)

certify(s):                                # s has already had the candidate applied
    alive = s in X
    repeat N-1 times:
        s = f(s, pi_b(s))                  # full braking, steer to centreline
        alive &= s in X
    return alive and v(s) <= v_stop        # <- the terminal set
```

Two details in there are load-bearing. The candidate's own step is taken at the
**control** rate even when the backup is predicted on a coarser grid, because
that step is really going to happen. And the search order is by distance in
`(steer, throttle)` from what the learner asked for, so an override is the
*smallest* change that restores a backup rather than an arbitrary safe action —
which is what makes the filter's effect on the policy as small as it can be.

### Every knob, and what it costs

| parameter | default | what it does | failure mode when wrong |
|---|---|---|---|
| `horizon` | 25 | steps the backup must survive | too short: the terminal set is unreachable at speed, so almost everything is refused (see below). Too long: cost, and pessimism accumulates |
| `assumed_grip` | 1.0 | the filter's belief about the tyres | **the one that matters.** Optimistic → the guarantee is void. See the sweep below |
| `assumed_vehicle` | env's own | mass, wheelbase, servo lag, drag | same shape of failure as grip, spread over more parameters |
| `margin` | 0.05 m | shrinks the legal corridor | absorbs model error and state-estimate error; the cheapest robustness there is |
| `stop_speed` | 0.25 m/s | what counts as "stopped" | too large and the terminal set is not really invariant; too small and it is unreachable |
| `predict_dt_scale` | 1.0 | coarsens the backup's prediction grid | 2× is twice as fast and **three times** as interventionist (37% vs 13%): a braking curve predicted in 0.1 s jumps is pessimistic about where the car stops |
| `credit` | `executed` | what the agent is told it did | changes what is learned, not what is safe. Measured below |
| `obstacle_radius` | 0.44 m | clearance kept from other cars | on `overtake`; too small and the filter certifies a pass that clips |

### The horizon, swept

```{image} _static/plots/safety_knobs.png
:alt: intervention rate and crash rate against horizon and against alpha
:width: 100%
```

Left panel, under full throttle and random steering — a policy deliberately
trying to crash. The reading is not the obvious one.

A **short horizon looks safest**, and is not. At $N=3$ the filter overrides 98%
of actions and the car never leaves the track, because at that horizon nothing
that carries speed can reach a stop, so almost nothing is certifiable and the
car is forced to crawl. It is not safe because it sees further; it is safe
because it has taken the car away from the learner entirely. An intervention
rate near 100% means the filter *is* the controller and the agent is no longer
learning the task.

The residual 7% of episodes off-track at $N=25$ and $N=35$ is not a horizon
problem at all — it is `assumed_grip=1.0` against a car whose grip is drawn from
$U(0.6, 1.4)$, which is the next section.

### Measured

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

### With and without, while learning

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

### Three limitations, not caveats

#### It is privileged

The filter runs on the vehicle state, not on the agent's
nine beams. That is not cheating — on a real car it sits on the state
estimator, which is where it belongs — but the guarantee is only ever as good
as that estimate, and a filter tested against ground truth in simulation has
not been tested in the part that usually fails.

#### It does not know the grip either

`LaneKeep` redraws tyre grip every episode from $U(0.6, 1.4)$ and never
observes it; the filter predicts with a fixed `assumed_grip`. This is the
limitation that decides whether any of the rest is worth anything, so it is
swept rather than asserted.

```{image} _static/plots/safety_grip.png
:alt: crash rate and intervention rate against the filter's assumed grip
:width: 90%
```

| `assumed_grip` | episodes off-track | steps overridden |
|---|---|---|
| **0.6** — the worst case | **0%** | 40.3% |
| 0.8 | 0% | 37.6% |
| 1.0 — the *mean* | 7% | 36.5% |
| 1.2 | 71% | 33.1% |
| 1.4 — the best case | 79% | 38.7% |
| 2.4 — badly wrong | **100%** | 17.4% |

Full throttle with random steering, 14 episodes each. That policy is used
deliberately: a uniformly random policy averages 0.9 m/s on this track, and at
that speed the grip limit binds on **0.1%** of steps — the car is never going
fast enough for grip to be what stops it, so the identical sweep run under a
random policy returns a flat line at zero and measures nothing. Under full
throttle the limit binds on 81% of steps.

Three things fall out of that table.

**Assuming the mean is not good enough.** At `assumed_grip=1.0` — the expected
value of the true parameter, and the obvious choice — 7% of episodes end in a
wall. The filter is optimistic on every episode where the draw came in below
1.0, which is half of them. A guarantee that holds in expectation is not a
guarantee; you have to assume the worst case, and at 0.6 the crash rate is
exactly zero.

**The failure is a cliff, not a slope.** Between 1.0 and 1.2 the crash rate goes
from 7% to 71%. There is no gentle degradation to notice in testing and no
margin to discover by being slightly careful.

**Wrong filters intervene *less*.** The override rate *falls* from 40% to 17% as
the filter becomes more optimistic. This is the property that makes an
optimistic filter genuinely dangerous rather than merely useless: it does not
announce itself by becoming annoying. It gets quieter, feels better tuned, and
is certifying corners the car cannot take. **Intervention rate is not a safety
metric** — it is a cost metric, and reading it as the former inverts the sign.

There is no setting that is both fast and guaranteed. `tests/test_safety.py`
asserts the optimistic failure rather than leaving it as prose.

#### It makes the update off-policy

The action reaching the environment is not
always the one the policy chose, and TD(λ) has no term for "the action was
replaced". `credit="executed"` tells the agent about what happened (the filter
becomes part of the environment); `credit="proposed"` tells it about what it
chose (on-policy gradient, misreported dynamics). Neither is correct, the
literature on learning through a shield does not agree either, and both are
implemented so the difference can be measured instead of argued.

### Two implementation notes

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

## Part 2 — Control barrier functions

`rtrrl_playground/cbf.py` implements the **pointwise** alternative, so the two
approaches can be compared on the same task, the same action set, the same
model and the same wrapper.

### What a CBF does instead

Both filters answer "may I apply this action?". They answer it in
fundamentally different ways.

The predictive filter **exhibits a trajectory**: roll the model forward
twenty-five steps under a braking backup, check the whole path is legal and
ends stopped. Safety is certified by *producing the plan that would save you*.

A CBF **evaluates a function**. Define $h(x) > 0$ on the safe set and require
one algebraic inequality of the action — in discrete time
(Agrawal & Sreenath, RSS 2017):

$$h(x_{t+1}) \;\ge\; (1-\alpha)\, h(x_t), \qquad 0 < \alpha \le 1$$

Satisfy that every step and $h$ can never cross zero, so the safe set is
forward invariant. **No horizon, no backup policy: one model step instead of
twenty-five.**

#### Where that inequality comes from

In continuous time a control barrier function asks the safe set's boundary to
be repelling: for an extended class-$\mathcal{K}$ function $\gamma$,

$$\sup_{u \in \mathcal{U}} \dot h(x, u) \;\ge\; -\gamma\big(h(x)\big)$$

Take the linear choice $\gamma(h) = \gamma h$. Then $\dot h \ge -\gamma h$, and
by Grönwall $h(t) \ge h(0)e^{-\gamma t}$ — so if $h(0) > 0$, $h$ stays positive
for all time. The constraint permits $h$ to shrink, but only geometrically, and
geometric decay never reaches zero.

The discrete-time version (Agrawal & Sreenath) replaces the derivative with a
difference. Writing $\Delta h = h(x_{t+1}) - h(x_t)$ and asking
$\Delta h \ge -\alpha\,h(x_t)$ rearranges to the condition above, and the same
induction gives

$$h(x_t) \;\ge\; (1-\alpha)^t\, h(x_0) \;>\; 0 \quad \text{whenever } h(x_0) > 0$$

which is where $0 < \alpha \le 1$ comes from: at $\alpha = 1$ the condition is
just $h(x_{t+1}) \ge 0$, the weakest thing that is still forward invariant, and
as $\alpha \to 0$ it approaches "$h$ may never decrease at all".

Two things about this are worth being clear-eyed about. The bound is
$(1-\alpha)^t h(x_0)$, which **tends to zero** — the guarantee is that $h$ is
never negative, not that it stays comfortably positive, so the car is permitted
to converge on the boundary forever. And the induction, exactly like the
predictive filter's, is a statement about the model $f$ used to evaluate
$h(x_{t+1})$, so it inherits the grip problem unchanged.

#### Why it is enumerate-and-check here too

With a continuous input the constraint is linear in the control and the filter
is the QP $\min \|u - u_L\|^2$ subject to the CBF condition. This repo's
action space is nine discrete actions, so that QP degenerates into
enumerate-and-check — exactly as the predictive filter does. The argmin is
still exact, and **the only thing that differs between the two filters is the
criterion**, which is what makes the comparison below clean.

### The barrier matters more than the method

```{image} _static/plots/barrier.png
:alt: the two barriers, as safe sets over the track
:width: 100%
```

Above: $h(x)$ over the track for a car at 3 m/s pointed 25° off the path, black
line at $h=0$. On the left the safe set is the whole corridor -- a pure
position constraint, blind to the fact that the car is *moving at a wall*. On
the right it has collapsed to two islands on the straights, which is what a car
in that state genuinely has.

The same difference, as a function of the two variables that matter:

```{image} _static/plots/safety_barrier_field.png
:alt: h over lateral offset and heading error, for both barriers
:width: 100%
```

$h_\text{lateral}$ has **flat contours**: it takes the same value whether the
car is running parallel to the wall or driving straight into it, and it is
identical at 1 m/s and at 2.5 m/s. That is the myopia, drawn. $h_\text{braking}$
tilts with heading error and the tilt steepens with speed — the certified set
narrows as the car commits to a direction, which is the behaviour a barrier for
a *moving* vehicle has to have.


The obvious barrier for staying on a track is $h = w - |d|$, with $d$ the
lateral offset. It is also **myopic**: it permits driving flat out straight at
a wall until the step before contact, because until then $h$ is still positive
and still falling slowly. A one-step condition cannot see a braking distance.

The fix is to put the vehicle's dynamics *into the barrier*:

$$h = w - |d| - T_{\text{look}}\,\big|v \sin e_\psi\big|$$

subtracting the lateral ground the car will cover in $T_{\text{look}}$ seconds
at its current lateral closing rate. The barrier now shrinks when you are
moving *towards* a wall, not merely when you are near one.

Both are implemented (`h_kind="lateral"` / `"braking"`) and both are measured,
because *"CBFs are unsafe here"* and *"that barrier was unsafe here"* are very
different claims and only the second is true.

### Measured, head to head

15 episodes on `lanekeep`. "filter µs" is the filter's own cost, with the
environment's ~120 µs subtracted. Both filters get the same
proposed-action-first shortcut, or the comparison would measure an
implementation asymmetry rather than the criteria.

```{image} _static/plots/results_cbf.png
:alt: predictive filter versus control barrier function
:width: 100%
```

#### Under a random policy — the stress test

| filter | off-track | overridden | filter µs |
|---|---|---|---|
| none | **100%** | — | — |
| CBF, $h = w-|d|$ (naive) | **47%** | 14.5% | 264 |
| CBF, $h$ with the closing-rate term | **0%** | 13.7% | 280 |
| predictive, 25-step rollout | **0%** | 10.7% | 565 |

The naive barrier **fails outright**. The corrected one is sound and costs half
what the rollout costs.

#### Under a competent policy — the deployment case

| filter | off-track | return | overridden | filter µs |
|---|---|---|---|---|
| CBF, closing-rate barrier | 0% | 569 | **8.7%** | 279 |
| predictive, 25-step rollout | 0% | 570 | **0.0%** | 172 |

And here it inverts. The predictive filter is *cheaper* — because when no
intervention is needed it is a single scalar rollout, and for a competent
driver it is never needed — and it is **far less conservative**: it overrides
nothing, while the CBF clips 8.7% of the actions of a driver that was never
going to crash.

### $\alpha$, swept

The right-hand panel of [the conservatism figure](#the-horizon-swept) sweeps
$\alpha$ under the same full-throttle stress test, and the result is a flat
line: the override rate sits near 60% at every value from 1.0 to 0.1, while the
crash rate wanders between 7% and 28% without an obvious trend.

That is worth reporting precisely because it is a non-result. $\alpha$ is the
CBF's only conservatism dial and on this task **it barely does anything**, for a
structural reason: the binding constraint is almost always $h(x_{t+1}) \ge 0$
rather than the decay rate, so tightening $(1-\alpha)h(x_t)$ changes which
actions are admitted only in the narrow band of states where $h$ is small. The
predictive filter's horizon, by contrast, moves the override rate from 36% to
98%. If you need a dial you can actually turn, that asymmetry is the practical
difference between the two methods on this problem.

### What the comparison actually says

**Neither method is safer than the other. The barrier design is what carries
the safety**, and a bad barrier fails visibly (47%) where the same method with
a good one does not fail at all.

**The pointwise method is more conservative, and structurally so.** A one-step
condition cannot tell that a *plan* exists — only that the next state is
acceptable — so it refuses actions a rollout certifies. That is the price of
not having a horizon.

**Their cost profiles are opposite.** The CBF is ~2× cheaper when it has to
intervene, and ~1.6× more expensive when it does not, because "nothing needed
here" is one scalar rollout for the predictive filter and still nine barrier
evaluations for the CBF. On a vehicle running a policy that is usually right,
that favours the rollout — which is the opposite of the usual summary that CBFs
are the cheap option.

**Both inherit the same three limitations**, unchanged, because they follow
from being a filter and not from the criterion: both are privileged, both are
only as good as their model of the car (including the grip neither knows), and
both make the learner's update off-policy.

## Choosing between them

| if you… | use | because |
|---|---|---|
| have a credible backup controller (braking, a stop, a safe pose) | predictive | the terminal set is where the guarantee comes from, and you already have it |
| cannot name a backup, but can write down what "safe" means as a function | CBF | it needs $h$, not $\pi_b$ |
| run a policy that is usually right | predictive | it costs one scalar rollout when nothing is needed, and it overrode a competent policy **0%** of the time against the CBF's 8.7% |
| are hard real-time with a tight worst case | CBF | one model step, bounded; the rollout's worst case is $N$ times its best |
| have continuous inputs and a QP solver | CBF | the constraint is linear in $u$; the predictive form needs an NLP |
| have a discrete action set | either | both degenerate to enumerate-and-check and the argmin is exact |
| do not trust your vehicle model | **neither, yet** | both guarantees are statements about $f$. Fix the model first; see the grip sweep |

The last row is not a rhetorical flourish. On this task the difference between
the two methods is worth a few percent of intervention rate, and the difference
between a correct and an optimistic grip estimate is worth **0% versus 71%** of
episodes ending in a wall. Effort spent choosing between the criteria is effort
not spent on the thing that actually decides the outcome.

## Failure modes

Collected in one place, because each of these was observed here rather than
anticipated.

**The model is optimistic.** Both filters certify and both are wrong. The
symptom is counterintuitive: the intervention rate goes *down*. See the grip
sweep.

**The filter becomes the controller.** An override rate near 100% (the $N=3$
column) is a filter that has taken the task away from the learner. The car does
not crash and the agent does not learn either — check the intervention rate
before believing a low crash rate.

**The barrier is myopic.** A position-only $h$ permits full speed at a wall
until the step before contact: 47% of episodes off-track for a filter that is
functioning exactly as specified. This is a failure of the barrier, not of the
method.

**The terminal set is dropped.** An $N$-step lookahead with no terminal
condition is feasible at every step and still crashes, because feasibility at
$t$ says nothing about feasibility at $t+1$ without the induction.

**The agent learns to lean on it.** Measured on `overtake`: an agent trained
behind a filter scored 344 against 194 for one trained without, and then
*under-performed* when the filter was removed. The filter is part of the
environment the policy was fitted to. If it will not be there at deployment,
it must not be there at training either — or the policy must be evaluated
without it, which is what the `credit` ablation exists to expose.

**The state estimate is wrong.** Not measured here, and the one most likely to
bite on hardware. Both filters read the true state from the simulator. Every
guarantee above is conditional on that, and a filter validated only against
ground truth has not been tested in the component that usually fails.

## Reproducing the figures

```bash
python scripts/make_safety_figures.py               # all four
python scripts/make_safety_figures.py --only grip   # just the grip sweep
```

Everything is seeded. The certificate figure searches for its own state rather
than being given one, so it will move if the filter's defaults change — which
is the point.

## Not implemented

The **continuous-input QP** form. With nine discrete actions there is nothing
for a QP solver to do, and adding one would mean adding a dependency to
demonstrate an equivalence rather than a difference. `--action-mode continuous`
exists in the environments, so it is the obvious extension.
