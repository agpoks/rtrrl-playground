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
    steer_max: float = 0.40      # rad, giving a 0.78 m minimum geometric turn radius
    steer_tau: float = 0.08      # s, first-order servo lag
    accel_max: float = 4.0       # m/s^2
    speed_max: float = 4.0       # m/s
    drag: float = 0.15           # 1/s, rolling + aero, so coasting is not free
    a_lat_max: float = 6.0       # m/s^2, about 0.6 g -- an RC tyre on smooth concrete

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
