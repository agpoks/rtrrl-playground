# The physical models

Every equation in this repo that describes a moving thing, in one place, with
its parameters — and an honest statement of where each model stops being a
vehicle model.

## The ladder

There are three rungs, and it is worth knowing which one you are standing on:

| rung | where | lateral dynamics | use it for |
|---|---|---|---|
| **kinematic bicycle + yaw-rate cap** | `lanekeep`, `overtake` | none (a hard cap standing in for understeer) | learning experiments that finish in a minute |
| **CartPole** | `cartpole-vel` | n/a | a POMDP with a known-correct reference implementation |
| **ST / STD / STD4W with real tyres** | [`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax) via `envs/scuderia.py` | slip angles, Pacejka / brush / Dugoff, load transfer | any number you intend to believe |

This repo's own models are on the first rung on purpose. They are fast enough
that a tutorial lesson runs on a laptop, and they are **not vehicle models**.
{doc}`to_scuderia_gym_jax` is the rung up.

## The kinematic bicycle

State $[x, y, \psi, v, \delta]$, control $[\text{steer}, \text{throttle}]$ each
in $[-1, 1]$, integrated with explicit Euler at 20 Hz:

$$
\begin{aligned}
\dot\delta &= \big(\text{steer}\cdot\delta_{\max} + \delta_{\text{bias}} - \delta\big)\,/\,\tau_\delta \\
\dot v &= \text{throttle}\cdot a_{\max}\cdot k_{\text{throttle}} - c_d\, v \\
\dot\psi &= \operatorname{clip}\!\left(\frac{v}{L}\tan\delta,\; \pm\frac{\mu\, a_{\text{lat,max}}}{v}\right) \\
\dot x &= v\cos\psi, \qquad \dot y = v\sin\psi
\end{aligned}
$$

Three things in there are not textbook, and each earns its place:

**The steering is a first-order lag, not an instant input.** A servo takes
about 80 ms to get where it was told. That is hidden state — the commanded
angle and the actual one differ during a transient — and it is one of the
things the recurrent policy has to learn to carry.

**The yaw rate is capped** at what $\mu\,a_{\text{lat,max}}$ of lateral
acceleration can produce. This is understeer in its crudest possible form: ask
for more than the tyres have and you simply do not get it, and the car runs
wide. Without it a kinematic bicycle takes any corner at any speed, the
throttle choice is free, and a scripted wall-follower laps the track flat out —
which was the first version of `lanekeep` and a useless one. **It is not a
slip-angle model and must not be mistaken for one.**

**$\mu$ (grip) is redrawn every episode** from `grip_range` (default 0.6–1.4)
and is *never observed*. At 0.6 the corner limit is 3.0 m/s; at 1.4 it is
4.6 m/s. The only way to find out which one you are on is to drive and notice
that the car ran wider than the steering asked for. See {doc}`environments`.

## The parameters

All of them live in `rtrrl_playground.envs.vehicle.VehicleParams`, and the
environment takes one, so a second vehicle is one argument away:

```python
from rtrrl_playground import make_env, VehicleParams, REAL_VEHICLE
env = make_env("lanekeep", vehicle=VehicleParams(wheelbase=0.35, drag=0.22))
env = make_env("lanekeep", vehicle=REAL_VEHICLE)     # the shipped "other" car
```

| parameter | default | what it is |
|---|---|---|
| `wheelbase` | 0.33 m | 1:10 scale; gives a 0.78 m minimum geometric turn radius at full lock |
| `steer_max` | 0.40 rad | steering limit |
| `steer_tau` | 0.08 s | servo lag |
| `accel_max` | 4.0 m/s² | drive |
| `speed_max` | 4.0 m/s | top speed |
| `drag` | 0.15 1/s | rolling + aero, so coasting is not free |
| `a_lat_max` | 6.0 m/s² | ≈ 0.6 g — an RC tyre on smooth concrete |

and four **defects**, all off by default, which exist so a second vehicle can
be imperfect realistically rather than arbitrarily:

| defect | what it models |
|---|---|
| `steer_bias` | a servo trim that is not quite centred — the commanded zero is not the car's zero |
| `throttle_scale` | motor and battery: what fraction of the commanded acceleration actually arrives |
| `beam_noise` | additive lidar noise, in metres |
| `beam_dropout` | probability a beam returns max range instead of its hit |

`beam_dropout` returns `BEAM_RANGE` — the same value a genuinely empty corridor
returns — because that is what a real lidar does, and it is the reason dropout
is worth simulating: the agent cannot tell a miss from open space.

## The sensor

Nine beams over ±60°, ray-marched at 15 cm steps out to 5 m against a
**rasterised occupancy bitmap** of the track, not against its geometry. That is
the same thing `f1tenth_gym` and `scuderia_gym_jax` do, so it is a step towards
the real sensor rather than a shortcut away from it — and it means a real lidar
scan and a simulated one are literally the same nine numbers (see
{doc}`real_data`).

Nine beams and not five: at 15° spacing, adjacent rays are 0.65 m apart at
2.5 m — about one car's detection diameter. With five beams the gap was 1.5 m
and traffic genuinely disappeared *between* rays, which is a sensor bug
masquerading as a hard exploration problem.

The beams are quantised to the march step, so a smooth approach to a wall
arrives at the agent as a staircase.

## CartPole

Textbook (Barto, Sutton & Anderson 1983) with Gymnasium's exact constants, so
numbers here are comparable to anything you have run there: cart 1.0 kg, pole
0.1 kg and 0.5 m half-length, force ±10 N, Euler at 50 Hz. Euler and not
something better, because every published CartPole number uses it.

The one change is the **observation**: `obs_mode="vel"` hides $\dot x$ and
$\dot\theta$. Everything is scaled to its limit, which sounds cosmetic and is
not — raw CartPole hands you a cart position in $[-2.4, 2.4]$ next to a pole
angle in $[-0.21, 0.21]$, so a network with one input scale sees the variable
that decides the episode at a tenth the amplitude of the one that barely
matters.

## The model the *network* is given

`--cell physics_ligru` puts a piece of the model above **inside the recurrent
cell**. Three units integrate the agent's own command through the known
first-order response — motor gain and drag for speed, servo lag for steering,
and $v\delta$ as a yaw-rate proxy — with the rate constants initialised from
`VehicleParams` and then learnable. The rest of the layer is an ordinary LiGRU
learning what the prior gets wrong.

The point is that the input already contains what such a unit needs: RTRRL is
fed the previous action, and for these tasks the previous action is a steering
and throttle command. Untrained, the steering unit tracks the true (hidden)
steering angle at correlation 1.00 and the speed unit tracks the true speed at
0.85. See {doc}`cells`.

## The models the *controllers* believe

Two places in this repo hold a model that is deliberately allowed to be wrong
about the plant, because that mismatch is the subject:

**The safety filter** (`safety.py`) predicts with its own `assumed_grip` and
`assumed_vehicle`. Give it the truth and it is sound; give it an optimistic
grip and it certifies corners the car cannot take. Measured in {doc}`safety`.

**The MPCC** in [`mpcc-online-tuning`](https://github.com/agpoks/mpcc-online-tuning)
does not model the grip limit at all. Compensating for a limit the controller
does not know about is precisely what the online tuner is asked to do.

## What is not modelled here, and where it is

No slip angle. No tyre force curve. No load transfer, no combined-slip
ellipse, no differential, no aero map, no suspension, no motor/ESC dynamics
beyond a scalar, no steering-rate limit, no actuation delay beyond the servo
lag.

All of those are in `scuderia_gym_jax`: five vehicle models (KS, ST-linear, ST,
STD, STD4W) and four tyre models (full 32-parameter Pacejka, simplified
Pacejka, brush/tanh, Dugoff), fitted to real RC-car recordings and validated
against a numba reference. `rtrrl_playground/envs/scuderia.py` puts the agents
on them without changing a line of agent code — see
{doc}`to_scuderia_gym_jax`, and note the honest cost: a step there is ~12 ms
against `lanekeep`'s ~150 µs.
