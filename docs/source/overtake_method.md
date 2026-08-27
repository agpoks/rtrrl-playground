# Overtaking: the method

`lanekeep` is where you check that an agent can drive. `overtake` is the one
this repo is actually about, because it is the task where the missing state
stops being a teaching device and becomes the difficulty.

This page is the method: what the problem is, what the agent is given, what
reward it gets, what a hand-written solution looks like, and where the learned
one is measured against it.

```{image} _static/plots/overtake.png
:alt: overtake — a pass is the ego line crossing a traffic trail
:width: 100%
```

## The problem

The same 1:10 car as `lanekeep`, on the same oval, with **two slower cars
holding the racing line**. They do not brake, do not steer, and do not react —
the same dumb traffic as `scuderia_gym_jax`'s `examples/overtake.py`, for the
same reason: if the traffic reacts, you can no longer tell whose behaviour you
are looking at.

Their speeds are drawn fresh each episode from 1.5–2.3 m/s, and the ego's
ceiling is 4.0 m/s, so a pass is always physically available. The track is
widened to 0.95 m half-width, because a pass needs the two cars more than
`2 × CAR_RADIUS = 0.44 m` apart laterally while both stay inside the boundary,
and 0.75 m does not comfortably allow that.

## What the agent is given

18 numbers. Nine lidar beams, **and nine flags saying whether each return is a
car or a wall.**

That second channel is not a convenience. Without it a car two metres ahead and
a wall two metres ahead are byte-identical, and no amount of recurrence recovers
a distinction the sensor never made. With it, the agent has a lidar plus the
world's crudest object classifier — a fair caricature of what the real car has
downstream of `obstacle_perception`.

## What is hidden, and why it is the whole task

**The speed of the car in front.** Never observed, redrawn every episode.

A closing rate is a *derivative of a range*, so it cannot be read off one frame
at any resolution. And committing to a pass depends on it entirely: at 0.3 m/s
of closure you have time to set up, at 2 m/s you do not. This is the point in
the repo where the argument for a recurrent policy stops being pedagogical.

The ego's own speed is hidden too, as in `lanekeep`, and so is the per-episode
grip.

## Reward

| event | reward |
|---|---|
| every step | arc-length progress, normalised so full speed ≈ 1.0 |
| passing a car | **+2** |
| contact | **−5**, episode ends |
| leaving the track | −1, episode ends |

A pass is detected as the ego's arc length crossing an opponent's *from
behind* — the gap going from positive to negative — with a guard on `|gap| <
length/4` so that the wrap-around half a lap away, where the sign also flips,
is not paid out as an overtake.

**Return is the wrong headline number here**, and the benchmark tables say so.
An agent that never passes anybody and never crashes scores respectably,
because progress alone pays. The numbers that say whether it learned to
*overtake* are the pass count and the crash rate, so both are always reported.

## The scripted reference

`rtrrl_playground/envs/scripted.py` has a fifteen-line `Overtaker`: follow the
wall; when a *car*-flagged beam blocks the middle third, pick the side that is
open and car-free, and **hold it**.

Committing to a side is the part that is easy to get wrong, and it is the same
trap as in `scuderia_gym_jax`'s example: recompute the side every frame and the
moment the ego draws level with the car it is passing, "go round the free side"
flips, and it steers back into it. So the side is latched when the manoeuvre
starts and released only once the middle beams are clear.

It scores **~313 return, 1.6 passes, and crashes 65% of the time** — and the
crashes are not a bug in it. It commits from a single frame and has no idea how
fast it is closing. That defect is the specification for what a learned policy
is supposed to fix.

## Results

See {doc}`benchmark_results` for the full table across every recurrent cell.
Two things worth knowing before reading it:

* **The seed spread is large** — larger than on `lanekeep`, because an episode
  either finds a clean pass or ends in contact, and the mean of those two
  describes neither. Every number is over multiple seeds with its spread.
* **The physics-prior cell does not help here**, which was a prediction that
  got tested and failed. Its reserved units dead-reckon the *ego's* motion,
  while the quantity that decides a pass is the *opponent's* speed. Encoding
  the right physics into the wrong half of the problem — see {doc}`cells`.

## Where this goes

`overtake` is a toy of a real thing. The real version is the multi-agent side
of `scuderia_gym_jax`, where `num_agents` cars share one `State`, one `step`
integrates all of them, and a pairwise SAT check runs on their actual
rectangles — with ST/STD vehicle models underneath instead of a kinematic
bicycle. See {doc}`to_scuderia_gym_jax`.
