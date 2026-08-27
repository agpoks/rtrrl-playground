# The environments

The *dynamics* these run on are documented separately in {doc}`physics`;
this page is about what each task asks of an agent.

Four, all partially observable **by construction**: no velocity is ever in an
observation. That is not a difficulty knob, it is the premise -- an RC car's
sensors give you positions and your decisions need rates.

| id | observation | action | what it tests |
|---|---|---|---|
| `memory-chain` | a bit, a clock, a query flag | 2 | memory alone; optimal return is exactly 1 |
| `cartpole-vel` | cart position, pole angle | 2 | control with the velocities hidden |
| `lanekeep` | 9 lidar beams | 9 (3 steer x 3 throttle) | drive a 1:10 car at the grip limit |
| `overtake` | 9 beams + 9 is-it-a-car flags | 9 | drive, past traffic whose speed is hidden |

Plus `ScuderiaLaneKeep`, which is the same interface wrapped around
`scuderia_gym_jax`'s real vehicle models -- see {doc}`to_scuderia_gym_jax`.

## MemoryChain

From bsuite (Osband et al., ICLR 2020). See a bit at $t=0$, nothing for $n-1$
steps, reproduce it at the end: $+1$ right, $-1$ wrong, $0$ in between. There
is no dynamics to get wrong and the optimum is exactly 1, which makes it the
only environment here where "did it learn" is not a judgement call. For a
memoryless policy it is not a hard task, it is an impossible one.

## CartPole with the velocities removed

Textbook dynamics and Gymnasium's exact constants, with `obs_mode="vel"`
hiding $\dot x$ and $\dot\theta$ -- the `CartPole-vel` POMDP the RTRRL paper
reports. The missing quantity is exactly the derivative of something visible,
which is the easiest possible thing for a recurrent state to reconstruct and
still impossible for a feedforward one. `obs_mode="full"` restores the MDP and
is the control worth running once.

The observations are **scaled to their limits**, which sounds cosmetic and is
not: raw CartPole hands you a cart position in $[-2.4, 2.4]$ next to a pole
angle in $[-0.21, 0.21]$, so a network with one input scale sees the variable
that decides the episode at a tenth the amplitude of the one that barely
matters.

## LaneKeep -- "learn to drive"

A kinematic bicycle at RC scale (0.33 m wheelbase, 0.4 rad of lock, 4 m/s top
speed) on a 27 m oval with 2.5 m corners, at 20 Hz. Reward is arc length
covered along the centreline, normalised so full speed is 1.0 per step;
leaving the track ends the episode with $-1$.

Two details make it a task rather than a demo:

**A grip limit.** The yaw rate is capped at what $0.6g \times \text{grip}$ of
lateral acceleration can produce, so cornering speed is bounded and the
throttle choice matters. Without it a kinematic bicycle takes any corner at any
speed and flat out is always right -- a scripted wall-follower laps it at full
throttle, which was the first version of this environment and a useless one.

**Grip is redrawn every episode and never observed.** At grip 0.6 the corner
limit is 3.0 m/s; at 1.4 it is 4.6 m/s, and the difference is worth about a
hundred points. The only way to find out which one you are on is to drive: ask
for more lateral acceleration than the tyres have and the car runs wide.

**Be honest about how much that buys.** It is not enough to make `lanekeep`
*require* memory, and that is in the benchmark table rather than quietly
omitted: nine beams are a lot of information, and reacting to the forward one
is a decent speed controller by itself. `lanekeep` is where you check that an
agent can drive at all.

## Overtake -- "learn to overtake"

`lanekeep` plus two slower cars holding the racing line. They do not brake, do
not steer and do not react -- the same dumb traffic as `scuderia_gym_jax`'s
`examples/overtake.py`, for the same reason: if the traffic reacts, you can no
longer tell whose behaviour you are looking at.

The observation gains one channel per beam: **is this return a wall or a car?**
Without it, a car two metres ahead and a wall two metres ahead are identical,
and no amount of recurrence recovers a distinction the sensor never made.

What stays hidden is the thing that decides everything: **the speed of the car
in front**, drawn fresh each episode from 1.5-2.3 m/s. A closing rate can only
be had by watching a range change over several frames. This is where memory
stops being a teaching device -- the scripted overtaker in
`envs/scripted.py` is sensible, fifteen lines, and crashes about half the time
for exactly this reason.

Reward: progress as before, $+2$ per car passed, $-5$ and episode over on
contact. **Return is the wrong headline number here** and the benchmark table
says so: an agent that never passes anybody and never crashes scores
respectably, because progress alone pays. The pass count and the crash rate are
what say whether it learned to overtake.

## Two implementation notes

**`terminated` and `truncated` are not interchangeable.** Terminated means the
MDP ended and bootstrapping must stop; truncated means the harness got bored
and the next state's value is still a legitimate estimate. Collapsing them into
one `done` flag teaches the agent that the world ends after `max_steps`, which
on a looping race track is exactly the wrong lesson.

**The track is a bitmap.** The boundary is implied by $|d| \le w$ rather than
stored as a polygon, so there is nothing to intersect; the track is rasterised
once into an on-track grid and a beam is a strided lookup. That is also what
`f1tenth_gym` and `scuderia_gym_jax` do -- ray-march against an occupancy image
-- so it is a step towards the real thing rather than a shortcut away from it.
