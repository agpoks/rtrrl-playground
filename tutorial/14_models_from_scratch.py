"""Lesson 14 -- the vehicle and sensor models from scratch, with the algebra.

    python tutorial/14_models_from_scratch.py

Every equation the simulator runs on, derived, implemented in a dozen lines,
and checked against something that is true independently of the code.
"""

# %% [markdown]
# # Lesson 14 — The models from scratch
#
# `docs/source/physics.md` states the equations. This derives them, and then
# checks each one against a fact that does not come from the implementation —
# a circle's radius, a known terminal velocity, a lap length.
#
# Three rungs, and it matters which one you are standing on:
#
# | rung | lateral dynamics | where |
# |---|---|---|
# | kinematic bicycle + yaw-rate cap | none — a hard cap standing in for understeer | **this repo** |
# | CartPole | n/a | this repo |
# | ST / STD / STD4W, Pacejka / brush / Dugoff tyres | slip angles, load transfer | `scuderia_gym_jax` |
#
# What follows is rung one. It is **not a vehicle model**, and the checks below
# are the ones that a vehicle model would also have to pass, not a substitute
# for it.

# %%
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# %% [markdown]
# ## 1. The kinematic bicycle
#
# Two wheels, no slip: each wheel rolls along the direction it points. Put the
# reference point at the **rear axle**, wheelbase $L$, steering angle $\delta$.
#
# The rear wheel points along the body, so the reference point moves along
# $\psi$:
#
# $$\dot x = v\cos\psi, \qquad \dot y = v\sin\psi$$
#
# For the yaw rate: the no-slip front wheel points along $\psi + \delta$, and
# the vehicle rotates about the instantaneous centre where the two wheel axes
# meet. That point is a distance $R = L/\tan\delta$ from the rear axle,
# perpendicular to the body, so
#
# $$\dot\psi = \frac{v}{L}\tan\delta$$
#
# **That is the whole model, and its one assumption is that tyres do not slip.**
# Everything a real tyre does — slip angle, saturation, load transfer — is
# outside it by construction.

# %%
L = 0.33          # wheelbase [m], 1:10 scale
STEER_MAX = 0.40  # [rad]
STEER_TAU = 0.08  # servo lag [s]
ACCEL_MAX = 4.0   # [m/s^2]
SPEED_MAX = 4.0   # [m/s]
DRAG = 0.15       # [1/s]
A_LAT_MAX = 6.0   # [m/s^2] ~ 0.6 g


def bicycle(state, steer, throttle, dt, grip=1.0, cap=True, lag=True):
    """One tick. ``state = [x, y, psi, v, delta]``, inputs in [-1, 1]."""
    x, y, psi, v, delta = state
    target = steer * STEER_MAX
    delta = delta + (target - delta) * dt / STEER_TAU if lag else target
    v = min(max(v + (throttle * ACCEL_MAX - DRAG * v) * dt, 0.0), SPEED_MAX)
    psi_dot = v / L * np.tan(delta)
    if cap and v > 1e-3:
        lim = A_LAT_MAX * grip / v
        psi_dot = min(max(psi_dot, -lim), lim)
    return np.array([x + v * np.cos(psi) * dt, y + v * np.sin(psi) * dt,
                     psi + psi_dot * dt, v, delta])


# %% [markdown]
# ### Check 1 — constant steering must draw a circle of radius $L/\tan\delta$
#
# Nothing in the integrator knows that. If it comes out right, the geometry is
# right.

# %%
def measure_radius(steer_frac, v, r_hint, cap=True, dt=1e-3, circles=3):
    """Radius of the circle the car actually drives.

    Integrate for a whole number of circles, not a fixed number of steps.
    Estimating the centre as the mean of a *partial* arc biases it towards the
    chord, and the measured radius then comes out too small -- which is exactly
    the artefact this function exists to avoid, and which the first version of
    this check fell into.
    """
    arc = circles * 2 * np.pi * r_hint
    n = max(int(arc / (v * dt)), 200)
    s = np.array([0.0, 0.0, 0.0, v, steer_frac * STEER_MAX])
    pts = []
    for _ in range(n):
        s = bicycle(s, steer_frac, 0.0, dt, cap=cap, lag=False)
        s[3] = v                          # hold speed, isolate the geometry
        pts.append(s[:2].copy())
    pts = np.array(pts)
    return float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())


print("  constant steering, no cap, no lag, held at speed:\n")
print(f"  {'delta [rad]':>12}{'L/tan(delta)':>15}{'measured R':>13}{'error':>10}")
for frac in (0.25, 0.5, 1.0):
    delta = frac * STEER_MAX
    r_true = L / np.tan(delta)
    r_meas = measure_radius(frac, v=2.0, r_hint=r_true, cap=False)
    print(f"  {delta:>12.3f}{r_true:>15.3f}{r_meas:>13.3f}{abs(r_meas - r_true):>10.4f}")

# %% [markdown]
# ### Check 2 — drag must give the right terminal velocity
#
# $\dot v = a - c_d v$ settles at $v_\infty = a/c_d$. At full throttle that is
# $4.0/0.15 = 26.7$ m/s, far above the 4 m/s clamp — which is worth knowing:
# **the top speed of this car is the clamp, not the physics.** Drag only shapes
# the approach and makes coasting cost something.

# %%
s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
for _ in range(2000):
    s = bicycle(s, 0.0, 1.0, 0.01)
print(f"\n  full throttle, terminal speed: {s[3]:.3f} m/s "
      f"(clamp {SPEED_MAX}, drag-implied {ACCEL_MAX / DRAG:.1f})")
s[4] = 0.0
for _ in range(2000):
    s = bicycle(s, 0.0, 0.0, 0.01)
print(f"  coasting from there, after 20 s: {s[3]:.3f} m/s "
      f"(exp(-c_d t) -> {SPEED_MAX * np.exp(-DRAG * 20):.3f})")

# %% [markdown]
# ## 2. The grip limit — one line, and the reason the task exists
#
# A kinematic bicycle takes **any corner at any speed**: nothing in
# $\dot\psi = (v/L)\tan\delta$ says the tyres cannot deliver the implied lateral
# acceleration $a_\text{lat} = v\dot\psi$. So flat out is always the right
# answer, a scripted wall-follower laps the track at top speed, and the
# throttle half of the action space is not worth learning.
#
# The cap is the crudest possible repair:
#
# $$\dot\psi \;=\; \operatorname{clip}\!\left(\frac{v}{L}\tan\delta,\;
#   \pm\frac{\mu\, a_\text{lat,max}}{v}\right)$$
#
# Ask for more lateral acceleration than the tyres have and you do not get it —
# **understeer, with no slip angle anywhere.** It implies a maximum cornering
# speed for a given radius:
#
# $$v_\text{max}(R) = \sqrt{\mu\, a_\text{lat,max}\, R}$$

# %%
print("\n  grip-limited cornering speed, v = sqrt(mu * a_lat_max * R):\n")
print(f"  {'R [m]':>7}{'mu=0.6':>10}{'mu=1.0':>10}{'mu=1.4':>10}")
for R in (2.0, 2.5, 4.0, 8.0):
    row = "".join(f"{min(np.sqrt(mu * A_LAT_MAX * R), SPEED_MAX):>10.2f}" for mu in (0.6, 1.0, 1.4))
    print(f"  {R:>7.1f}{row}")
print(f"\n  The oval's corners are 2.5 m, so cornering speed swings from 3.0 to 4.0 m/s")
print("  across the grip range -- and grip is redrawn every episode and never observed.")

# %% [markdown]
# ### Check 3 — the cap must actually bind
#
# Drive a constant-radius circle at increasing speed and measure the radius the
# car *achieves*. Below the limit it should track $L/\tan\delta$; above it, the
# car should run wide by exactly the ratio the cap implies.

# %%
print("\n  full lock, increasing speed -- where the tyres give up:\n")
print(f"  {'v [m/s]':>9}{'geometric R':>14}{'achieved R':>13}{'runs wide by':>14}")
for v in (1.0, 2.0, 3.0, 4.0):
    r_geo = L / np.tan(STEER_MAX)
    # The capped radius is v^2 / (mu a_lat_max) once the cap binds; hint with
    # the larger of the two so the integration still covers whole circles.
    r_hint = max(r_geo, v ** 2 / A_LAT_MAX)
    r_meas = measure_radius(1.0, v=v, r_hint=r_hint)
    print(f"  {v:>9.1f}{r_geo:>14.3f}{r_meas:>13.3f}{r_meas / r_geo:>13.2f}x"
          f"   (v^2/a_lat = {v ** 2 / A_LAT_MAX:.2f})")

# %% [markdown]
# ## 3. The steering servo — the first piece of hidden state
#
# The command is not the angle. A servo is a first-order lag:
#
# $$\dot\delta = \frac{\delta_\text{cmd} - \delta}{\tau_\delta},
#   \qquad \tau_\delta = 80\ \text{ms}$$
#
# At 20 Hz, one control tick moves $\delta$ about
# $\Delta t/\tau_\delta = 62\%$ of the way to the command. So during any
# transient the commanded and the actual angle differ — and the actual one is
# **not in the observation**. It is one of the things the recurrent state has
# to carry, and `physics_ligru` reserves a unit that reconstructs it exactly
# (correlation 1.00, untrained) precisely because this equation is known.

# %%
s = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
print("\n  step the steering command to full lock at t = 0:\n")
print(f"  {'t [s]':>7}{'delta':>9}{'fraction':>10}{'1-exp(-t/tau)':>15}")
for i in range(1, 8):
    s = bicycle(s, 1.0, 0.0, 0.05)
    t = i * 0.05
    print(f"  {t:>7.2f}{s[4]:>9.3f}{s[4] / STEER_MAX:>10.2f}{1 - np.exp(-t / STEER_TAU):>15.2f}")

# %% [markdown]
# ## 4. The sensor — ray marching against a bitmap
#
# Nine beams over $\pm 60°$, marched in 15 cm steps to 5 m. The boundary is
# *implied* by $|d| \le w$ rather than stored as a polygon, so there is nothing
# to intersect analytically — the track is rasterised once into an on-track
# bitmap and a beam becomes a strided lookup.
#
# That is also what `f1tenth_gym` and `scuderia_gym_jax` do — they march against
# an occupancy image — so the bitmap is a step *towards* the real sensor, not a
# shortcut away from it, and a real lidar scan and a simulated one end up being
# literally the same nine numbers (lesson 9).
#
# ### Check 4 — the beams must agree with geometry on a straight

# %%
from rtrrl_playground import make_env  # noqa: E402
from rtrrl_playground.envs.lanekeep import BEAM_ANGLES, BEAM_RANGE  # noqa: E402

env = make_env("lanekeep")
env.reset(seed=0)
# put the car in the middle of the bottom straight, pointed along it
k = 5
env.x, env.y = env.track.center[k]
env.psi = float(env.track.heading[k])
obs = env._obs()
w = env.track.half_width
print("\n  on the centreline of a straight, the beam at angle a should read")
print("  w / |sin a| if it hits the wall before 5 m:\n")
print(f"  {'angle':>8}{'measured':>11}{'w/|sin a|':>12}")
for ang, r in zip(BEAM_ANGLES, obs[:9] * BEAM_RANGE):
    pred = w / abs(np.sin(ang)) if abs(np.sin(ang)) > 1e-6 else np.inf
    pred = min(pred, BEAM_RANGE)
    print(f"  {np.degrees(ang):>7.0f}°{r:>11.2f}{pred:>12.2f}")
print("\n  Agreement to the 15 cm march step, which is the resolution by construction.")

# %% [markdown]
# ## 5. What this model is not
#
# No slip angle. No tyre force curve. No load transfer, no combined-slip
# ellipse, no differential, no aero map, no suspension, no motor or ESC
# dynamics beyond a scalar, no steering-rate limit.
#
# All of those are in
# [`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax) — five
# vehicle models and four tyre models, fitted to real RC-car recordings and
# validated against a numba reference — and
# `rtrrl_playground/envs/scuderia.py` puts the agents on them without changing
# a line of agent code (lesson 8).
#
# The honest cost of going up a rung: a step there is ~12 ms against
# `lanekeep`'s ~150 µs.
#
# ## References
#
# * Rajamani, *Vehicle Dynamics and Control*, Springer 2012 — ch. 2 for the
#   kinematic bicycle and its assumptions
# * Barto, Sutton & Anderson, IEEE SMC 1983 — the CartPole equations
# * Liniger, Domahidi & Morari, *Optimization-based autonomous racing of 1:43
#   scale RC cars*, 2015 — the same scale, with a dynamic model
# * `docs/source/physics.md` — the same equations as a reference page
