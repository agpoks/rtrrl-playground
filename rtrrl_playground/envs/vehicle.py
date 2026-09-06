"""The vehicle's physical parameters, in one place, so two of them can differ.

Everything about the car that is a *number* rather than an equation lives here.
That is not tidiness: it is what makes the sim-to-real lesson possible at all.
A "real vehicle" in this repo is the same equations with a different
:class:`VehicleParams` -- a slightly longer wheelbase, a slower servo, a
steering trim that is not quite centred, a motor that gives 90% of what it is
asked. Those are the differences that actually separate a simulator from the
car on the bench, and none of them is visible in an observation.

The defaults are a 1:10 RC car, roughly a Traxxas Slash on smooth concrete, and
they are the numbers the module-level constants in
:mod:`rtrrl_playground.envs.lanekeep` still export for anything that wants them
as plain floats.

## What is *not* here

The tyre. There is no slip angle, no load transfer, no Pacejka curve -- the
lateral dynamics are a kinematic bicycle with a hard cap on yaw rate, which is
understeer in its crudest possible form. That is a deliberate ceiling on this
repo's ambitions: it is enough to make "how fast can I take this corner" a real
question, and it is nowhere near enough to be a vehicle model. The real ones
are in `scuderia_gym_jax` (ST, STD, STD4W with Pacejka, brush or Dugoff tyres,
fitted to actual recordings), and
:mod:`rtrrl_playground.envs.scuderia` is the adapter that puts the agents on
them. See ``docs/source/physics.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters of the 1:10 car.

    ``wheelbase`` .. ``a_lat_max`` are the model. The four below them are
    *defects* -- they default to "perfect" and exist so a second vehicle can be
    made imperfect in a way that is realistic rather than arbitrary.
    """

    wheelbase: float = 0.33      # m
    # 0.40 rad (22.9 deg), a 0.78 m minimum turn radius at this wheelbase.
    # Raised to 0.44 and reverted: it does not unlock the imported circuits
    # (a scripted controller stalls on ICRA26 T1 and T2 at every lock from
    # 0.40 to 0.80), and mpcc-online-tuning measured the same from the other
    # side -- 0.50 was tried there and reverted, 4x worse at the long
    # horizon, because extra steering authority is worth nothing when the
    # plan is wrong. Every published number here is at 0.40.
    steer_max: float = 0.40      # rad
    steer_tau: float = 0.08      # s, first-order servo lag
    accel_max: float = 4.0       # m/s^2
    speed_max: float = 4.0       # m/s
    drag: float = 0.15           # 1/s, rolling + aero, so coasting is not free
    a_lat_max: float = 6.0       # m/s^2, about 0.6 g -- an RC tyre on smooth concrete

    # --- the dynamic bicycle, off by default ----------------------------
    # ``dynamics="dynamic"`` swaps the kinematic model plus yaw-rate clip for a
    # lateral model with brush tyres, which makes sideslip and yaw rate real
    # states rather than a saturation. That is what the beta--r phase plane
    # needs: on the kinematic car "sliding" is not a state the car can be in,
    # it is a clip that binds on 70 % of steps, which is why the trigger
    # measures blind to slip (0.96x chance). Off by default because every
    # published number in this project is on the kinematic car.
    dynamics: str = "kinematic"  # or "dynamic"
    mass: float = 3.5            # kg
    inertia: float = 0.0952      # kg m^2, the neutral estimate m*a*b
    a_cg: float = 0.16           # m, centre of mass to front axle
    c_f: float = 290.0           # N/rad, front cornering stiffness
    c_r: float = 310.0           # N/rad, rear cornering stiffness
    n_substeps: int = 10         # the lateral mode is faster than the 20 Hz tick
    # The dynamic bicycle is singular as speed goes to zero: the slip angles
    # divide by v_x, so the lateral eigenvalue C/(m v_x) grows without bound
    # and any explicit integrator goes unstable. Every serious vehicle
    # simulator blends to the kinematic model at low speed for this reason
    # (Liniger 2015). Measured here: without the blend the car leaves the start
    # line already sideways, at a median sideslip of 65 degrees.
    #
    # A tanh, not the clamped linear ramp this first used: a ramp has a
    # discontinuous derivative at both ends, and mpcc-online-tuning replaced
    # exactly that ramp with exactly this tanh for exactly this reason.
    #
    # 0.70/0.15 are scuderia's own *plant* constants. That project also
    # measured that a *controller* wants a higher midpoint (1.40) because it
    # must additionally linearise the model -- but this is a plant, so the
    # plant number is the right one to start from.
    v_blend_mid: float = 0.70    # m/s, centre of the tanh
    v_blend_width: float = 0.15  # m/s
    # Bias, not a numerical epsilon. The slip angles are atan((vy +- l r)/vx),
    # whose derivative goes like 1/vx, so a guard of 1e-3 is no guard at all at
    # 0.8 m/s -- it prevents only a division by exactly zero. Biasing the
    # denominator as sqrt(vx^2 + bias^2) caps that derivative everywhere. It
    # makes the model deliberately wrong by a few percent at speed, which is
    # the trade: a slightly wrong model that integrates beats an exact one that
    # returns NaN. From mpcc-online-tuning, where it was the fix for a QP that
    # reported NaN at 0.79 m/s.
    v_bias: float = 0.35         # m/s
    # Which axle the drive force goes through. 1.0 is rear drive --- the
    # single-track *drift* model the phase-plane work is about, and the layout
    # whose (beta, r) portrait has saddle points worth staying inside. ``None``
    # splits by static load, which is four-wheel drive; that car is easier and
    # does not drift, because derating both axles preserves the balance and it
    # keeps understeering. Measured: at full lock and full throttle, sideslip
    # is 0.1 deg under four-wheel drive and -44 deg at a rear share of 0.7.
    #
    # The choice is not free either way: a rear-driven car cannot accelerate
    # harder than its rear tyre grips, which here is 2.9 m/s^2 rather than the
    # 4.0 of ``accel_max``. That limit is enforced in the integrator, and it
    # is a fact about the layout rather than a tuning knob.
    rear_drive_share: float | None = 1.0
    # What fraction of the driven axle's grip full throttle is allowed to
    # spend going forwards. This is not a comfort setting, it is what makes
    # the model drivable at all.
    #
    # ``accel_max`` is 4.0 while a rear-driven axle here can deliver 2.91, so
    # the traction clamp put full throttle at exactly kappa_r = 1.0 --- the
    # whole rear tyre used longitudinally, nothing left sideways. With three
    # discrete throttle levels that means "accelerate" *is* "spin", and it was:
    # a scripted wall-follower went from 20.5 laps on the kinematic car to
    # 0.16 on this one, stalling on 100 % of episodes at 39 degrees of
    # sideslip. Sweeping the fraction, the car becomes drivable at 0.35 for
    # every drive split (12.5--14.3 laps, no stalls, 2.6--6.4 degrees of
    # sideslip) and is undrivable at 1.0 for all of them.
    #
    # A real driver does not use the whole rear tyre to accelerate mid-corner
    # either, so this is the model being honest rather than being helped.
    throttle_grip_share: float = 0.35

    # --- defects, all zero/one by default -------------------------------
    steer_bias: float = 0.0      # rad, a servo trim that is not quite centred
    throttle_scale: float = 1.0  # motor/battery: what fraction of the commanded accel arrives
    beam_noise: float = 0.0      # m, std of additive lidar noise
    beam_dropout: float = 0.0    # probability a beam returns max range instead of a hit

    def perturbed(self, **kw) -> "VehicleParams":
        """A copy with some parameters changed. ``params.perturbed(drag=0.25)``."""
        return replace(self, **kw)

    def diff(self, other: "VehicleParams") -> dict:
        """What differs between two vehicles, as ``{name: (mine, theirs)}``."""
        return {f: (getattr(self, f), getattr(other, f))
                for f in self.__dataclass_fields__
                if getattr(self, f) != getattr(other, f)}


#: A plausible "the simulator was optimistic" vehicle: 6% longer wheelbase, a
#: servo half again as slow, 1.7 degrees of steering trim, a motor down on
#: power, more drag, less grip, and a noisy lidar. Every one of these is a thing
#: that is true of a real car and false of the model of it, and none of them is
#: observable -- which is the entire point.
REAL_VEHICLE = VehicleParams(
    wheelbase=0.35,
    steer_tau=0.12,
    accel_max=3.4,
    drag=0.22,
    a_lat_max=5.2,
    steer_bias=0.03,
    throttle_scale=0.9,
    beam_noise=0.04,
    beam_dropout=0.02,
)
