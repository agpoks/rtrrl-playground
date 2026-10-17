"""An adapter that puts the agents in this repo onto ``scuderia_gym_jax``.

The two driving environments here are toys, and deliberately so: a kinematic
bicycle with a crude grip limit, a bitmap track, and a nine-beam sensor, all
sized so a lesson finishes on a laptop in a minute. The real vehicle models --
ST, STD, STD4W, with Pacejka or brush tyres fitted to actual RC-car recordings
-- live in `scuderia_gym_jax <https://github.com/agpoks/scuderia_gym_jax>`_,
and that is where a result should eventually be reported.

This is the bridge. It wraps a ``ScuderiaEnvJax`` in the same eight-line
:class:`~rtrrl_playground.spaces.Env` interface everything else here speaks, so
the agent code does not change at all::

    from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep
    env = ScuderiaLaneKeep(model="st", map_name="berlin")
    agent = RTRRL(env.obs_dim, env.action_space, cell="lrcu")

Three things are worth knowing before trusting a number that comes out of it:

**The reward is distance travelled, not progress along a racing line.** The
maps that ship with the simulator are occupancy images with no centreline, so
there is no arc length to differentiate. Distance-without-crashing is the
standard stand-in for f1tenth-style RL baselines and it is a genuinely
different objective -- it will happily reward a fast lap of a small loop inside
a wide corridor. If you want the real thing, bring a centreline and replace
:meth:`ScuderiaLaneKeep._reward`.

**One environment, one step at a time.** ``scuderia_gym_jax`` is built to be
``vmap``ped over thousands of cars and ``scan``ned over whole rollouts without
Python in the loop; driving it one step at a time from a Python agent gives up
almost all of that. RTRRL is a batch-size-one algorithm, so there is nothing to
vmap over -- but it does mean the simulator is being used against its grain,
and a step here costs far more than a step of ``lanekeep``.

**The velocity is still hidden.** ``get_obs`` in the simulator returns the full
state vector, which includes the speed; this adapter throws that away and
returns downsampled lidar only, because a POMDP is the point. Pass
``observe_speed=True`` to keep it and turn the task Markov.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.envs.vehicle import VehicleParams
from rtrrl_playground.spaces import (N_ACCEL, N_STEER, Box, Discrete,
                                     Env, action_grid)

STEER_MAX = 0.4  # rad, the steering-angle setpoint the simulator's PID takes
SPEED_MAX = 4.0  # m/s
#: How many of the simulator's beams reach the agent, by default. Nine is what
#: ``lanekeep`` gives and it is what every earlier result here was measured on
#: -- but it is a *small fraction* of the real sensor, and measured on a track
#: whose corners are tighter than the car's turning circle it costs a factor of
#: four in distance covered: 0.16 laps at 9 beams over 120 degrees against 0.61
#: at 61 over 270. Raise ``n_beams_out`` for anything that has to place the car
#: on a line rather than merely avoid a wall.
N_BEAMS_OUT = 9
BEAM_RANGE = 10.0


class ScuderiaLaneKeep(Env):
    """Drive a ``scuderia_gym_jax`` car on one of its shipped maps."""

    id = "scuderia-lanekeep"

    # Class-level defaults for the steering-jerk and corridor state. The unit
    # tests build bare instances with `object.__new__` to avoid starting a
    # simulator and loading a map, and every field added to `__init__` used to
    # break every one of those helpers at once. Defining the defaults here
    # means a bare instance behaves like an env with the knob switched off,
    # which is also what an old pickled env would do.
    steer_jerk_rate = None
    steer_jerk_corridor = False
    _steer_rate_cmd = 0.0
    _steer_jerk = 0.0
    corridor_steer_penalty = 0.0
    corridor_lo = 3.0
    corridor_hi = 6.0
    corridor_feature = False
    _corridor_m = 6.0
    _corridor_gate = 0.0
    de_features = ""
    de_cone_deg = 6.0
    de_margin = 0.0
    _de_prev = None

    def __init__(self, model: str = "st", map_name: str = "berlin",
                 map_ext: str = ".png", tire_model: int | None = None,
                 action_mode: str = "discrete",
                 observe_speed: bool = False, num_beams: int = 108,
                 n_beams_out: int = N_BEAMS_OUT, centreline=None,
                 half_width=None, width_scale: float = 1.0,
                 max_steps: int = 2000, control_repeat: int = 5,
                 start_pose=(0.0, 0.0, 0.0), grip: float | None = None,
                 track_half_width: float | None = None,
                 n_steer: int = N_STEER, n_accel: int = N_ACCEL,
                 lookahead=None, aim_feature: bool = False,
                 steer_mode: str = "absolute", steer_rate: float = 2.0,
                 steer_filter: float = 0.0,
                 drive_features: bool = False, proprio: bool = False,
                 grip_feature: bool = False, steer_feature: bool = False,
                 grip_penalty: float = 0.0, grip_target: float = 0.8,
                 reverse_penalty: float = 0.0,
                 steer_move_penalty: float = 0.0,
                 accel_rate: float | None = None,
                 steer_jerk_rate: float | None = None,
                 steer_jerk_corridor: bool = False,
                 corridor_steer_penalty: float = 0.0,
                 corridor_lo: float = 3.0,
                 corridor_hi: float = 6.0,
                 corridor_feature: bool = False,
                 de_features: str = "",
                 de_cone_deg: float = 6.0, de_margin: float = 0.0,
                 accel_jerk_penalty: float = 0.0,
                 accel_max: float = 1.0, brake_max: float | None = None,
                 accel_mode: str = "ramp", accel_cmd: float = 10.0,
                 n_throttle: int | None = None, throttle_mode: str | None = None,
                 rolling_start: float = 1.0, random_start: bool = False,
                 respawn_at_crash: bool = False, respawn_backup: float = 3.0,
                 respawn_crash_prob: float = 1.0,
                 seed: int = 0, **make_kwargs):
        try:
            import jax
            import jax.numpy as jnp
            import scuderia_gym_jax as sgj
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ScuderiaLaneKeep needs scuderia_gym_jax and its dependencies:\n"
                "    pip install jax chex\n"
                "    pip install -e /path/to/scuderia_gym_jax\n"
                f"(the import failed with: {exc})"
            ) from exc
        self._jax, self._jnp = jax, jnp

        # tire_model is left to the car config unless asked for: the packed
        # parameter arrays carry a tyre code and ScuderiaEnvJax refuses a spec
        # that disagrees with them, which is the right call and an easy way to
        # get an unhelpful error out of an adapter that guessed.
        if tire_model is not None:
            make_kwargs["tire_model"] = tire_model
        # Before `make`, not after: the simulator is constructed on the next
        # statement and a kwarg set later never reaches it. Setting it late
        # left `ModelSpec.ctrl_mode` at 0, so what looked like acceleration
        # control was speed control being handed +/-10 as a *setpoint* -- the
        # PID saturated and the car appeared to brake at a_max, which is a
        # flattering measurement of the wrong thing.
        # Deprecated spellings, accepted and translated. The car has no
        # throttle -- the axis is a longitudinal acceleration (m/s^2 in
        # ctrl_mode=1, amps in ctrl_mode=2) -- so the names went, but a run
        # already in flight has `n_throttle=`/`throttle_mode=` pickled into
        # its worker queue and must not start failing halfway through.
        _dep_n_throttle, _dep_throttle_mode = n_throttle, throttle_mode
        if _dep_n_throttle is not None:
            n_accel = _dep_n_throttle
        if _dep_throttle_mode is not None:
            accel_mode = _dep_throttle_mode
        if accel_mode not in ("ramp", "accl"):
            raise ValueError(
                f"accel_mode must be ramp or accl, got {accel_mode!r}")
        self.accel_mode = str(accel_mode)
        if self.accel_mode == "accl":
            # the int, not the name: `make` writes whatever it is handed onto
            # the ModelSpec without passing it through `ctrl_code_from_name`,
            # so the string "accl" never equals CTRL_ACCL and is silently
            # ignored. Measured: string and default give identical rollouts.
            make_kwargs.setdefault("ctrl_mode", 1)
        # map_ext matters: the maps that ship with the simulator are a mix of
        # .png and .pgm, and the loader takes the extension as an argument
        # rather than looking. ``levine`` is a .pgm; ``berlin``, ``skirk``,
        # ``vegas`` and ``stata_basement`` are .png.
        self.env = sgj.make(model=model, num_agents=1, produce_scans=True,
                            num_beams=num_beams, map_name=map_name,
                            map_ext=map_ext, collision_on=True, **make_kwargs)
        self.model, self.map_name = model, map_name
        self.num_beams = num_beams
        self.control_repeat = int(control_repeat)  # simulator ticks per agent action
        self.dt = float(self.env.timestep) * self.control_repeat
        self.observe_speed = bool(observe_speed)
        self.action_mode = action_mode
        self.n_beams_out = int(n_beams_out)
        # see `lanekeep`'s note: one normalised heading error per distance,
        # the quantity beams cannot carry. Needs a centreline, which is the
        # same thing it needs to pay a progress reward at all.
        self.lookahead = ([float(lookahead)] if np.isscalar(lookahead)
                          else None if lookahead is None
                          else [float(v) for v in lookahead])
        # the pose-free sibling of `lookahead` -- see lanekeep's note
        self.aim_feature = bool(aim_feature)
        #: Three scan-derived cues -- how far it is to the next obstruction
        #: ahead, where the corridor opens, and which way it bends beyond that.
        #: The speed cue is the point: a progress reward has nothing to slow
        #: the car for a corner, and the policy currently has no feature from
        #: which it could learn to anyway.
        self.drive_features = bool(drive_features)
        #: What the car's own sensors publish: speed (VESC), steering angle,
        #: yaw rate and body accelerations (IMU). Deliberately *not* sideslip.
        #: beta is available on the car as atan2(vy, vx) from `/ekf/state`, but
        #: that vy is itself inferred from a bicycle model plus the IMU -- so
        #: feeding beta here, where it is a true simulator state, trains a
        #: policy on a quantity the car can only estimate. Giving the recurrent
        #: cell the same raw signals the EKF integrates lets it form its own
        #: estimate, which is what the memory is for and what transfers.
        self.proprio = bool(proprio)
        #: The wheel angle the policy is currently commanding, as a fraction
        #: of lock. In `delta` mode the action is a *rate*: the adapter keeps
        #: `_steer_cmd` and integrates `action * steer_rate * dt` into it. That
        #: setpoint is a state of the loop, and without this feature it is a
        #: *hidden* one -- the policy drives an integrator it cannot see, and
        #: "hold the wheel where it is" is then not a thing it can express,
        #: because the zero-rate action means "keep whatever angle I am at"
        #: and it does not know what that is. Measured on the straights, where
        #: the lidar error signal is smallest and so the sign of the feedback
        #: flips fastest, the wheel crosses centre 2.3 times a second.
        #:
        #: It is free on the car: this is the number the servo was last told,
        #: not a measurement, so there is nothing to estimate and nothing to
        #: transfer. In `absolute` mode it is the previous action, which the
        #: cell already receives as a meta input, and so adds nothing.
        self.steer_feature = bool(steer_feature)
        #: How much of the tyres' friction budget is in use right now, as a
        #: fraction: sqrt(ax^2 + ay^2) / (mu g). The project has a fitted tyre
        #: model -- `rc10_default.yaml` records mu = 1.10 measured from
        #: Fy_peak = 18 N at Fz = 16.4 N -- and uses it in exactly one place,
        #: the phase-plane barrier, to decide *when to learn*. Nothing tells
        #: the policy how much grip it has left.
        #:
        #: This does, from one scalar, and it is the rare feature that is
        #: *easier* on the car than in simulation: both accelerations come
        #: straight off the IMU, so no model, no state estimator and no map is
        #: involved. A policy that can see it has the information to keep a
        #: margin; measured, the car departs at a grip usage of about 1.0,
        #: which is at the limit rather than far beyond it.
        #: Price the friction budget in the reward. Off by default (0.0) and
        #: that default is load-bearing: every number this project has
        #: published is on a pure progress reward.
        #:
        #: The gap it closes: progress pays about +1 per step and nothing else
        #: is priced, so going flat out is *optimal* right up to the crash.
        #: Measured, the policy drives at 3.4-3.9 m/s and departs with the
        #: tyres saturated (grip usage 0.90 at 41-64 % of exits), while pure
        #: pursuit laps the same track holding 69 % of the tyres in reserve.
        #: Every feature added so far -- free_ahead, the lookahead, the grip
        #: reading itself -- gives the policy the information to keep a margin.
        #: None of them gives it a reason to.
        #:
        #: Penalises only the part above `grip_target`, so ordinary cornering
        #: is free and the term is silent until the car is actually leaning on
        #: the tyres. Uses `a_lat = v r` against the *measured* peak friction
        #: rather than a generic constant.
        self.grip_penalty = float(grip_penalty)
        #: Price a *reversal of the steered wheel*, per reversal, in the same
        #: units as the progress reward.
        #:
        #: Measured on the 21x21 delta arm: the wheel changes direction 6.4
        #: times a second and the driven path carries 3.3x the curvature of
        #: the centreline, so more than half the tyre budget goes into the
        #: weave rather than into the corner. A flat cost per reversal prices
        #: that directly -- genuine corner-to-corner changes happen a few
        #: times a lap and cost almost nothing, a 6 Hz limit cycle costs
        #: 6 * reverse_penalty every second.
        #:
        #: Deliberately nothing here prices *negative acceleration*. On this
        #: car that is negative motor current, which is the brake, and the
        #: policy already under-uses it (1 % of steps).
        self.reverse_penalty = float(reverse_penalty)
        #: A price on the wheel *moving*, paid per tick as
        #: `steer_move_penalty * |action|`. In delta mode the action is the
        #: rate, so this is a price per degree of wheel travel: holding an
        #: angle is free, and reaching one costs in proportion to how far the
        #: wheel has to go. That is the shape the other two smoothing costs
        #: do not have. `reverse_penalty` only charges once the wheel has
        #: already swung all the way across centre, and CAPS (`smooth_t`)
        #: charges for the rate *changing*, so holding a constant full rate
        #: -- which drives the wheel into the stop and keeps it there -- is
        #: free under both. Measured on the straights, the wheel travels
        #: about 2 rad/s for no net angle at all, and neither cost sees it.
        self.steer_move_penalty = float(steer_move_penalty)
        #: |action| on the steering axis for the tick just decoded, which is
        #: what that price is charged on.
        self._steer_move = 0.0
        self._steer_flip = False
        #: The steering axis has had a rate limit (`steer_rate`) and a price
        #: on reversing (`reverse_penalty`) since the smoothing work began.
        #: The acceleration axis had neither, so the commanded acceleration
        #: could jump the full +10 to -10 m/s^2 in one 50 ms tick, which no
        #: ESC can follow and which the recorded traces show it doing. These
        #: are the two matching mechanisms.
        #:
        #: `accel_rate` is a jerk limit in m/s^3 on the *command* (amps per
        #: second over 0.106 in current mode). None leaves the axis as it
        #: was, so every earlier result still reproduces.
        self.accel_rate = None if accel_rate is None else float(accel_rate)
        #: charged per step on |change in command| as a fraction of full
        #: scale, so it means the same at any accel_cmd
        self.accel_jerk_penalty = float(accel_jerk_penalty)
        self._accel_cmd = 0.0
        self._accel_jerk = 0.0
        self.grip_target = float(grip_target)
        self.grip_feature = bool(grip_feature)
        self._prev_vxy = None
        self._prev_g = None
        # How the steering half of an action is read. "absolute" places the
        # servo (the historical behaviour, and every published number); "delta"
        # moves it at `steer_rate` rad/s, so the command is a rate and the
        # angle is the integral of the policy's choices.
        #
        # The case for "delta" is that the accel axis in this adapter has always
        # worked that way -- `_v_cmd` ramps rather than being placed -- so
        # absolute steering was the inconsistent half. It matters because with
        # three absolute levels there is no steering angle between 0 and 22.9
        # deg in the action set at all, so holding a gentle curve requires
        # dithering between the lock stops: measured, the sign reverses every
        # 3 steps and the car weaves +/-0.5 m. With three *rates* the policy can
        # walk to any angle and stay there.
        if steer_mode not in ("absolute", "delta"):
            raise ValueError("steer_mode must be 'absolute' or 'delta'")
        self.steer_mode, self.steer_rate = steer_mode, float(steer_rate)
        #: Ticks of first-order low-pass on the angle handed to the servo.
        #: Not a rate limit -- `steer_rate` is one of those, and at 2 and 3
        #: rad/s it stopped the car cornering without changing how often the
        #: wheel reversed. A lag attenuates by frequency instead: at tau = 3
        #: ticks a corner's half-hertz demand passes nearly intact while the
        #: 4 Hz dither measured on the straights is cut severalfold. It is
        #: also the one place this can be fixed on the real car without the
        #: policy's cooperation, because it lives in the driver, not the
        #: reward. Eleven interventions on the policy left the straight-line
        #: crossing rate at 7-9 a second; this one does not ask it to learn.
        self.steer_filter = float(steer_filter)
        self._steer_out = 0.0
        #: Jerk limit on the *rate* command, in rad/s^2 -- the steering
        #: analogue of `accel_rate`, and until now the one axis with nothing
        #: of the kind. `steer_rate` caps how fast the setpoint may move;
        #: nothing capped how fast that rate may *change*, so the policy
        #: could ask for +6 rad/s and then -6 rad/s on the next 50 ms tick,
        #: and the traces show it doing exactly that. The acceleration axis
        #: got its limit first and it is the only smoothing cost in this
        #: project that did not cost distance.
        #:
        #: It belongs on the rate because in delta mode the rate *is* the
        #: action; limiting the angle instead would just be `steer_rate`
        #: again, which was tried at 4, 3 and 2 rad/s and either changed
        #: nothing or stopped the car cornering.
        self._check_steer_jerk(steer_jerk_rate, self.steer_mode)
        self.steer_jerk_rate = (None if steer_jerk_rate is None
                                else float(steer_jerk_rate))
        #: Apply `steer_jerk_rate` in proportion to the corridor ahead:
        #: the full limit down a straight, none of it in a hairpin.
        #:
        #: Measured, this is the reason a flat limit fails. The steering
        #: jerk the policy asks for is the same in both places -- median
        #: 144 rad/s^2 on a straight against 132 in a corner, above 30 on
        #: 57 % of straight steps against 55 % of corner steps -- so a
        #: single limit that quiets the straights also binds on more than
        #: half of the corner commands, and at 30 rad/s^2 the car stopped
        #: driving (149 m -> 16 m, 0/5 seeds). It is the same shape of
        #: failure as the three prices: the quantity being limited does not
        #: distinguish where the behaviour is wanted from where it is not.
        #: The corridor does distinguish them, so it is what should decide
        #: how hard the limit bites.
        self.steer_jerk_corridor = bool(steer_jerk_corridor)
        #: the largest jerk the action space can even ask for: full rate one
        #: way to full rate the other in one tick. Used as "no limit" at the
        #: open end of the gate, so the two ends interpolate in one unit.
        self._jerk_unlimited = 2.0 * float(steer_rate) / max(float(self.dt), 1e-9)
        self._steer_rate_cmd = 0.0
        self._steer_jerk = 0.0
        #: A price on moving the wheel, weighted by how long the corridor
        #: ahead is: free in a hairpin, full price down a straight.
        #:
        #: Three unconditional prices on command magnitude have already
        #: failed (reversal, CAPS, wheel travel), and the reason each failed
        #: is that the car genuinely needs the whole lock in master_cup's
        #: hairpins, so any cost big enough to quiet the straights is big
        #: enough to stop it cornering. This one is conditional, and on a
        #: quantity the car can measure: on a long straight the right rate
        #: command is zero, so a policy that holds the wheel pays nothing.
        #:
        #: The gate is the minimum range in a +/-6 deg cone, in metres. That
        #: cone and not the +/-12 deg one `drive_features` already publishes:
        #: measured over 6,000 steps of master_cup, +/-6 deg separates
        #: straights (R > 10 m) from corners (R < 3 m) with AUC 0.842 and
        #: means of 5.30 m against 2.99, where +/-12 deg manages 0.745 and
        #: 3.20 against 2.49. The wider cone subtends more than the corridor
        #: is wide -- master_cup pinches to 0.37 m -- so its minimum reads
        #: the side wall rather than the end of the straight.
        self.corridor_steer_penalty = float(corridor_steer_penalty)
        self.corridor_lo, self.corridor_hi = float(corridor_lo), float(corridor_hi)
        #: The policy has to be able to see the gate or this is a price on
        #: something unobservable, which is what `steer_feature` was added to
        #: stop. Separate flag so the feature and the price can be told apart.
        self.corridor_feature = bool(corridor_feature)
        self._corridor_m = float(corridor_hi)
        self._corridor_gate = 0.0
        #: Which of the six disparity-extender features to append, as a
        #: subset of ``track.DE_LETTERS``. The extender itself is not new --
        #: `free_space_heading` has driven on it since the start and
        #: `aim_feature` publishes its heading -- but every number it
        #: computes *besides* the heading was being thrown away. ``"b"``,
        #: the length of the chosen straight, is the one the corridor work
        #: needed: a gate on the minimum of a raw cone reads the side wall
        #: wherever the corridor is narrower than the cone is wide, and
        #: needs two thresholds in metres that have to be calibrated to
        #: whichever driver produced the distribution.
        from rtrrl_playground.envs.track import DE_LETTERS
        bad = set(de_features or "") - set(DE_LETTERS)
        if bad:
            raise ValueError(
                f"de_features got unknown letter(s) {sorted(bad)}; the six "
                f"are {DE_LETTERS!r} -- see envs.track.de_features")
        self.de_features = str(de_features or "")
        self.de_cone_deg = float(de_cone_deg)
        self.de_margin = float(de_margin)
        self._de_prev = None
        self.obs_dim = (self.n_beams_out + int(observe_speed)
                        + (0 if self.lookahead is None else len(self.lookahead))
                        + int(self.aim_feature)
                        + 3 * int(self.drive_features)
                        + 5 * int(self.proprio)
                        + int(self.grip_feature)
                        + int(self.steer_feature)
                        + int(self.corridor_feature)
                        + len(self.de_features))
        # one definition of what a discrete action means, shared with the
        # filters -- see `spaces.action_grid`
        self.action_grid = action_grid(n_steer, n_accel)
        self.n_steer, self.n_accel = int(n_steer), int(n_accel)
        self.action_space = (
            Discrete(len(self.action_grid), nvec=(int(n_steer), int(n_accel)))
            if action_mode == "discrete" else Box(2))
        self.max_steps = int(max_steps)
        self.start_pose = np.asarray(start_pose, dtype=float)
        self._key = jax.random.key(seed)
        # its own stream, so turning random starts on does not shift the
        # simulator's noise draws and change everything else too
        self._rs_rng = np.random.default_rng(seed + 90210)
        # Which raw beams to keep. Evenly spaced across the full field of view,
        # so the nine numbers the agent sees mean the same thing they do in
        # envs/lanekeep.py: right to left, straight ahead in the middle.
        self._idx = np.linspace(0, num_beams - 1, self.n_beams_out).astype(int)
        # The simulator's real field of view, read from it rather than assumed.
        # It is 4.7 rad (269 deg) -- the same span the car's own /scan reports
        # -- not the +/-pi/2 that several callers here have guessed. Beam
        # angles built on the wrong span do not fail loudly; they just point
        # every derived direction at the wrong part of the world.
        _fov = float(getattr(getattr(self.env, "scan_sim", None), "fov", 4.7))
        self.scan_fov = _fov
        self.scan_angles = np.linspace(-_fov / 2, _fov / 2, num_beams)
        self.beam_angles = self.scan_angles[self._idx]
        # A centreline turns "distance travelled" into "progress along the
        # track", which is what lanekeep pays and therefore the only reward
        # under which a number here is comparable with one from there. Without
        # it the agent is rewarded for a fast lap of any loop inside a wide
        # corridor. Pass an (N, 2) array of metres.
        self.centreline = None if centreline is None else np.asarray(
            centreline, dtype=float)
        if self.centreline is not None:
            d = np.diff(np.vstack([self.centreline, self.centreline[:1]]), axis=0)
            self._cl_s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
            self.lap_length = float(self._cl_s[-1])
        self._prev_s = None
        # ``half_width`` ends an episode on lateral deviation from the
        # centreline rather than on hitting something in the map, the way
        # ``lanekeep`` already does. It defaults to None -- the map-collision
        # behaviour below is unchanged -- because on a *shipped* map collision
        # is a fine terminator.
        #
        # It is not fine on an imported one, which is the whole reason this
        # exists. A SLAM occupancy grid marks only the walls the lidar
        # actually saw: measured, 1.6 % of pixels on master_cup against ~30 %
        # for a shipped map. A car that leaves the circuit then drives into
        # open space, never collides, is never terminated and earns nothing --
        # so the run looks like training and teaches nothing. A logical
        # boundary is immune to holes, unclosed walls and the edge of the
        # image, and it is the same quantity a real car's tracking error is
        # measured against, which is why it also transfers to hardware.
        # ``half_width`` takes a scalar or one value per centreline point. Per
        # point is the honest form: a real corridor is not a constant width,
        # and collapsing it to one number has to pick between two bad options
        # -- the narrowest, which terminates a car that is comfortably on
        # track wherever the track is wide, or the median, which lets it leave
        # the track entirely at the pinch points. Measured on master_cup, the
        # per-point half-width runs 0.37 m to 0.78 m, and the 5th percentile
        # (0.54 m) ended a *perfectly tracking* pure-pursuit run in 86 steps.
        #
        # ``width_scale`` multiplies whatever was passed. It exists for
        # training rather than for geometry: starting at 1.3 and annealing to
        # 1.0 gives a policy room to recover early without ever changing the
        # track the numbers are reported against. It does not move the walls.
        if lookahead is not None and self.centreline is None:
            raise ValueError(
                "lookahead needs a centreline to look along: pass "
                "centreline=(N, 2) metres alongside it.")
        if half_width is not None and self.centreline is None:
            raise ValueError(
                "half_width terminates on distance from the centreline, so it "
                "needs one: pass centreline=(N, 2) metres alongside it.")
        if half_width is None:
            self.half_width = None
        else:
            hw = np.atleast_1d(np.asarray(half_width, dtype=float))
            if hw.size not in (1, len(self.centreline)):
                raise ValueError(
                    f"half_width must be a scalar or one value per centreline "
                    f"point ({len(self.centreline)}), got {hw.size}")
            self.half_width = hw * float(width_scale)
        self.width_scale = float(width_scale)
        # jit the bound method once. Without this every control tick re-enters
        # the tracer and a step costs tens of milliseconds -- the simulator is
        # designed to be jitted around a whole `lax.scan` rollout, and stepping
        # it from Python is exactly the usage that does not get that for free.
        self._step_env = self._jax.jit(self.env.step_env)
        self._state = None
        self._x = None
        self.history: list[dict] = []
        self._track = None
        self.track_half_width = (None if track_half_width is None
                                 else float(track_half_width))
        #: This environment terminates on the *per-point* lateral limit while a
        #: safety filter certifies against ``track``'s single scalar, so the two
        #: boundaries are set independently and the scalar is the tighter one.
        #: That leaves a band -- 0.49 m to a median 0.78 m on master_cup -- in
        #: which the episode is still alive and the filter can certify nothing,
        #: and a stopped car there cannot be recovered by braking it harder.
        #: ``lanekeep`` cannot reach such a state, which is why the filter's
        #: recovery is off by default and on here.
        self.filter_recovers_at_standstill = True
        # What a safety filter is told about the car. Most of ``VehicleParams``
        # already describes this vehicle -- 0.40 rad of lock, 4 m/s -- because
        # both simulators model the same 1:10 car. Two fields do not, and
        # neither is cosmetic:
        #
        # ``accel_max`` is 4.0 m/s^2 there and **1.0 m/s^2 here**, measured:
        # full acceleration from rest gives 1.13 m/s^2 and full brake from 4 m/s
        # gives 1.00, because ``_decode`` ramps a speed *setpoint* at
        # ``accel * 1.0 * dt`` and the simulator's PID tracks it. A filter
        # left at 4.0 believes it can shed speed four times faster than the car
        # can, certifies a backup plan that stops in a quarter of the distance,
        # and produces exactly the "crashes *through* the filter" that
        # ``bridges.safety.make_safe_agent`` warns about for optimistic grip.
        #
        # ``drag`` is 0.0 for the same reason: the measured deceleration was a
        # flat 1.00 m/s^2 from 4 m/s down, with no velocity-proportional term
        # (0.15/s would have added 0.6 m/s^2 at that speed and did not).
        #
        # ``wheelbase`` is the simulator's own 0.1705 + 0.1515.
        #: How fast the speed setpoint may ramp, up and down, m/s^2. These were
        #: one number -- `_decode` ramped at a symmetric 1.0 -- which capped
        #: deceleration at 9 % of the 10.8 m/s^2 the tyres can supply. The car
        #: coasted rather than braked.
        #:
        #: The asymmetry is the physical case, not a tuning knob. Accelerating
        #: is limited by the drivetrain (1.13 m/s^2 measured) and braking by the
        #: tyres (mu g = 10.8), so a single rate has to be wrong for one of
        #: them. It was wrong for braking, and it cost this track's two hairpins:
        #: at R = 0.82 m the apex allows 2.88 m/s, and shedding 3.9 -> 2.88 at
        #: 1.0 m/s^2 takes 3.47 m while the longest lookahead feature sees 2.0.
        #: The policy could not see the corner until it was inside its own
        #: stopping distance.
        #:
        #: `brake_max=None` keeps the symmetric behaviour, so every result
        #: measured before this existed still reproduces.
        self.accel_max = float(accel_max)
        self.brake_max = float(accel_max if brake_max is None else brake_max)
        #: What the second action axis *means*. The car has no throttle:
        #: it is current-controlled, so the quantity is an acceleration.
        #:
        #: ``"ramp"`` is the ROS2 Ackermann convention the simulator defaults
        #: to: the action nudges a speed setpoint and a PID chases it. It is
        #: the wrong interface for this car. The setpoint can only move at
        #: ``accel_max``, so the whole axis spans +/-1 m/s^2 however many
        #: levels it is given -- 9 % of the 10.8 the tyres supply and 9 A of
        #: the ESC's 100 -- and the policy cannot ask for a force at all, only
        #: for a speed it would like to have soon.
        #:
        #: ``"accl"`` commands longitudinal acceleration directly, bypassing
        #: the PID (``ctrl_mode=CTRL_ACCL``). That is the quantity the
        #: friction circle is written in, the quantity ``bridges/friction.py``
        #: filters actions against, and one constant away from the motor
        #: current the real ESC takes: a = K_tG * I / (r m) = 0.106 I, so the
        #: +/-``accel_cmd`` range below is +/-94 A of a 100 A controller.
        #:
        #: ``accel_cmd`` defaults to 10.0 m/s^2, just inside mu g = 10.8, so
        #: the axis spans what the tyres can actually deliver and the friction
        #: circle -- not the action set -- is what bounds it.
        self.accel_cmd = float(accel_cmd)
        #: m/s the car is placed at on reset in accl mode, matching the speed
        #: the ramp-mode setpoint produces. 0.0 disables it.
        self.rolling_start = float(rolling_start)
        #: Start each episode at a uniformly random point on the centreline
        #: rather than always at the line.
        #:
        #: Always starting at the line means the policy only ever sees the
        #: part of the circuit it can already reach, and on this one the
        #: first corner is R = 1.04 m at s = 7 m and needs near-full lock.
        #: A 21-level policy commands that by chance in 0 % of random
        #: episodes against 3 % for a 3-level one, so it never gets past,
        #: never sees a reward beyond 9 m, and has nothing to learn from --
        #: every fine-resolution arm run so far has died there. The coarse
        #: action set was not better at driving, it was better at stumbling
        #: through the first corner.
        #:
        #: Training only. Evaluation keeps the fixed start, or "furthest
        #: reached" stops meaning the same thing between arms.
        self.random_start = bool(random_start)
        #: Restart the next episode just before where the last one ended,
        #: rather than at the line or at random.
        #:
        #: What a race weekend does. The car goes off at one corner, and the
        #: next run is spent on that corner, not on the straight before it.
        #: Resetting to the line instead spends almost all of a long episode
        #: re-driving track the policy already handles, so the sector that
        #: actually ends runs gets a few samples per episode and the rest
        #: gets thousands -- and online RTRRL has no replay buffer to even
        #: that out, so what the policy sees *is* what it learns from.
        #:
        #: The weights are never touched: this changes only where the next
        #: episode begins, so the crash stays in the learning exactly as it
        #: would otherwise.
        self.respawn_at_crash = bool(respawn_at_crash)
        #: metres back up the centreline from the crash, so the car arrives
        #: at the corner with a run-up instead of being spawned mid-mistake
        self.respawn_backup = float(respawn_backup)
        #: how often to take the crash point rather than the ordinary start.
        #:
        #: 1.0 is the pure version -- always restart at the last failure --
        #: and it can trap: measured on a random policy, six episodes in a row
        #: respawned inside the same 157-162 m sector and the rest of the
        #: circuit was never seen again. Practising the corner that ends runs
        #: is the point; practising only that corner is not, because the
        #: policy then forgets the track it could already drive and there is
        #: no replay buffer here to remind it. Below 1.0 the remainder falls
        #: through to `random_start` or the line.
        self.respawn_crash_prob = float(respawn_crash_prob)
        self._last_end = None
        if self.accel_mode == "accl":
            # The kernels read the mode off the ModelSpec, and `make()` writes
            # whatever it is handed: passing the *string* "accl" leaves it
            # uncoverted and the comparison against CTRL_ACCL silently fails,
            # so the car keeps running in speed mode. Measured: string and
            # default give byte-identical rollouts. Pass the int.
            make_kwargs.setdefault("ctrl_mode", 1)
        # The filter deliberately keeps the *conservative* figure: one that
        # underestimates braking certifies plans the car can over-deliver on,
        # and the reverse is what `bridges.safety.make_safe_agent` warns about.
        self.vehicle = VehicleParams(wheelbase=0.322, accel_max=1.0, drag=0.0)
        self.grip = self._read_grip() if grip is None else float(grip)

    def _read_grip(self) -> float:
        """The simulator's own friction coefficient, not a guess at it.

        ``mu`` is packed into the tyre parameter arrays rather than exposed as
        a field; ``simple[13]`` is the one ``scuderia_gym_jax``'s own
        ``tests/test_tire_parity.py`` reads, and on the shipped RC-10 config it
        agrees with ``st[0]``, ``brush[2]`` and ``dugoff[2]`` at 1.1. Reading it
        matters: a barrier certified at 1.0 against tyres worth 1.1 is
        conservative by 10 % everywhere, and one certified the other way round
        is confidently wrong -- see ``bridges.safety.make_safe_agent``'s note on
        a filter given a grip the road does not have.
        """
        try:
            return float(np.asarray(self.env.params.tire.simple).ravel()[13])
        except Exception:
            return 1.0

    # -- the dynamic state, under the names the rest of this project uses --
    #
    # The single-track state is ``[x, y, delta, v, psi, yaw_rate, beta]``,
    # verified against the running simulator rather than read off a docstring.
    # Sideslip and yaw rate are therefore *already here*; they were simply
    # never given names, and that alone is what kept
    # ``bridges.stability.StabilityTrigger`` and
    # ``bridges.safety.make_safe_agent`` off this environment -- both look for
    # ``env.beta`` / ``env.yaw_rate`` / ``env.delta`` / ``env.vx`` /
    # ``env.grip``, find nothing, and either never fire or refuse to attach.
    #
    # Nothing is cached: ``_x`` is refreshed by ``step``/``reset``, and these
    # are read a handful of times per tick against the ~12 ms the simulator
    # itself costs.
    _ST = dict(x=0, y=1, delta=2, v=3, psi=4, yaw_rate=5, beta=6)

    def _st(self, name: str) -> float:
        if self._x is None:
            raise RuntimeError("reset() this environment before reading its state")
        return float(self._x[0, self._ST[name]])

    @property
    def x(self) -> float:
        return self._st("x")

    @property
    def y(self) -> float:
        return self._st("y")

    @property
    def psi(self) -> float:
        return self._st("psi")

    @property
    def delta(self) -> float:
        """Steering angle, rad -- the servo's position, not the setpoint."""
        return self._st("delta")

    @property
    def accel_command(self) -> float:
        """The longitudinal command actually issued this step, m/s^2.

        The action is a *request*: in ``accl`` mode it is scaled by
        ``accel_cmd`` and then, if ``accel_rate`` is set, clipped to one
        jerk-limited step from the last command. Reading the action instead of
        this is reading the request, and with a rate limit in force the two
        are different signals -- which is the same mistake that made a
        steering *rate* get plotted as a steering angle. The steering axis has
        ``delta`` for this; the acceleration axis had nothing.
        """
        return self._accel_cmd

    @property
    def speed_setpoint(self) -> float:
        """The speed the simulator's PID is tracking, m/s."""
        return self._v_cmd

    @property
    def v(self) -> float:
        """Velocity *magnitude*, m/s. ``vx`` is its longitudinal component."""
        return self._st("v")

    @property
    def vx(self) -> float:
        # beta is the angle between the velocity vector and the body axis, so
        # the longitudinal component is v cos(beta). At the sideslips this car
        # reaches the difference is under a percent -- but the phase-plane
        # barrier divides by this, and "close enough" is how a barrier ends up
        # certified against a speed the car is not doing.
        return self._st("v") * float(np.cos(self._st("beta")))

    @property
    def yaw_rate(self) -> float:
        return self._st("yaw_rate")

    @property
    def beta(self) -> float:
        """Sideslip angle, rad."""
        return self._st("beta")

    @property
    def track(self):
        """A :class:`~rtrrl_playground.envs.track.Track`, for a safety filter.

        Built once, on demand, from the centreline this adapter was given.
        ``Track`` carries a *scalar* half-width while this environment may hold
        one per point, so one number has to stand for the corridor, and the
        choice is not cosmetic. The filter's feasible set has a hard edge at
        ``half_width - margin``: certifiability does not degrade across it, it
        falls off a cliff. Measured on master_cup at 1.5 m/s, over 30 points
        round the lap, as certifiable actions out of nine:

        ==========  ======  ======  ======  ======
        filter hw   d=0.3   d=0.4   d=0.5   d=0.6
        ==========  ======  ======  ======  ======
        0.369 min      8.9     0.0     0.0     0.0
        0.537 p5       9.0     9.0     0.1     0.0
        0.781 median   9.0     9.0     9.0     9.0
        ==========  ======  ======  ======  ======

        So the narrowest value is the *worst* default, not the safest one. This
        environment terminates on the per-point width -- a median of 0.78 m --
        and a filter pinned at the 0.37 m pinch point declares a car
        unrecoverable while the task is still perfectly happy with it. It then
        brakes, the backup controller drives, and with ``credit="executed"`` the
        learner is trained on the backup rather than on its own proposal, which
        is the bias ``experiments/trigger_arms.py``'s revision (1) exists to
        record. Over-conservatism here does not buy safety; it buys a filter
        that is always on.

        The default is therefore the 5th percentile -- the same statistic
        ``bridges.mapimport.save_centreline`` writes into the file's header as
        *the* half-width, so the filter's boundary is the number the track is
        described by. Pass ``track_half_width`` for anything else.

        ``None`` without a centreline, so ``make_safe_agent`` raises its own
        clear error instead of certifying against a track that is not there.
        """
        if self.centreline is None:
            return None
        if self._track is None:
            from rtrrl_playground.envs.track import Track
            if self.track_half_width is not None:
                hw = float(self.track_half_width)
            elif self.half_width is None:
                hw = 1.1
            else:
                hw = float(np.percentile(self.half_width, 5))
            self._track = Track(self.centreline[:, 0], self.centreline[:, 1],
                                half_width=hw)
        return self._track

    # -- helpers ----------------------------------------------------------
    def _split(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    def _decode(self, action) -> np.ndarray:
        """Action -> ``[steering angle, speed setpoint]``, the simulator's input."""
        if self.action_mode == "discrete":
            steer, accel = self.action_grid[int(action)]
        else:
            a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            steer, accel = float(a[0]), float(a[1])
        if self.accel_mode == "accl":
            # a direct acceleration command; no integration, and deliberately
            # no clip to SPEED_MAX -- that clip is a property of a *setpoint*,
            # and applying it here would cap the command at 4 (m/s^2 here,
            # amps in current mode) and quietly make the car slower
            a = float(accel) * self.accel_cmd
            if self.accel_rate is not None:
                step = self.accel_rate * self.dt
                a = float(np.clip(a, self._accel_cmd - step,
                                  self._accel_cmd + step))
            # on full scale, so the penalty is independent of accel_cmd
            self._accel_jerk = (abs(a - self._accel_cmd)
                                / max(self.accel_cmd, 1e-9))
            self._accel_cmd = a
            # Braking stops the car; it does not reverse it. Ramp mode clipped
            # the setpoint to [0, SPEED_MAX], so reverse was unreachable there
            # and the two modes would not otherwise be comparable -- measured,
            # a uniform-random policy in accl mode settles at -0.83 m/s and
            # runs the lap backwards. Capped at exactly the deceleration that
            # reaches standstill within this step.
            v_now = float(self._x[0, 3]) if self._x is not None else 0.0
            # Ramp mode clipped its setpoint to [0, SPEED_MAX], which gave the
            # car a top speed and a reverse floor as a side effect of the
            # parameterisation. Commanding acceleration has neither unless it
            # is said, and leaving it unsaid made accl a 15 m/s car racing a
            # 4 m/s one: it arrived at the first corner -- R = 1.04 m at
            # s = 7 m, good for 3.25 m/s -- doing 6.57 against ramp's 3.90,
            # and 46-90 % of episodes ended `overspeed`. The envelope is the
            # same in both modes; only the way of expressing it differs.
            hi = (SPEED_MAX - v_now) / max(self.dt, 1e-6)
            lo = -v_now / max(self.dt, 1e-6)
            self._v_cmd = float(np.clip(a, lo, hi))
        else:
            rate = self.accel_max if accel >= 0.0 else self.brake_max
            self._v_cmd = float(np.clip(self._v_cmd + accel * rate * self.dt,
                                        0.0, SPEED_MAX))
        prev_steer = self._steer_cmd
        self._steer_move = abs(float(steer))
        # Read here, not after the simulator runs: `self._scans` is still the
        # scan the current observation was built from, so the gate charges
        # against the corridor the policy actually saw when it chose this.
        if (self.corridor_steer_penalty or self.corridor_feature
                or self.steer_jerk_corridor):
            self._corridor_m = self._corridor_ahead()
            self._corridor_gate = float(np.clip(
                (self._corridor_m - self.corridor_lo)
                / max(self.corridor_hi - self.corridor_lo, 1e-9), 0.0, 1.0))
        if self.steer_mode == "delta":
            u = float(steer) * self.steer_rate              # rad/s requested
            if self.steer_jerk_rate is not None:
                lim = self.steer_jerk_rate
                if self.steer_jerk_corridor:
                    # gate 1 (a straight) is the full limit, gate 0 (a
                    # hairpin) is no limit at all
                    free = getattr(self, "_jerk_unlimited",
                                   2.0 * self.steer_rate / max(self.dt, 1e-9))
                    lim = free + (lim - free) * self._corridor_gate
                step_ = lim * self.dt
                u = float(np.clip(u, self._steer_rate_cmd - step_,
                                  self._steer_rate_cmd + step_))
            # on full scale, so the number means the same at any steer_rate
            self._steer_jerk = (abs(u - self._steer_rate_cmd)
                                / max(self.steer_rate, 1e-9))
            self._steer_rate_cmd = u
            self._steer_cmd = float(np.clip(self._steer_cmd + u * self.dt,
                                            -STEER_MAX, STEER_MAX))
        else:
            self._steer_cmd = float(steer) * STEER_MAX
        # the wheel crossed centre and is now pointed the other way. Read off
        # the commanded angle rather than off the action, so it means the same
        # in both steering modes: in delta mode the action is a rate, whose
        # sign changes on every normal corner exit.
        self._steer_flip = bool(prev_steer * self._steer_cmd < 0.0)
        if self.steer_filter:
            a = 1.0 / max(self.steer_filter, 1.0)
            self._steer_out = (1.0 - a) * self._steer_out + a * self._steer_cmd
            return np.array([self._steer_out, self._v_cmd])
        return np.array([self._steer_cmd, self._v_cmd])

    @staticmethod
    def _check_steer_jerk(steer_jerk_rate, steer_mode) -> None:
        if steer_jerk_rate is not None and steer_mode != "delta":
            raise ValueError(
                "steer_jerk_rate limits the change in a *rate* command and "
                "only means something in steer_mode='delta'; in absolute "
                "mode the action is an angle and the limit you want is "
                "steer_rate. Refusing rather than ignoring it.")

    def _corridor_ahead(self) -> float:
        """How far the corridor runs ahead, in metres, from the raw scan.

        The minimum of a +/-6 deg cone: the nearest thing the car would hit
        going straight on, which is what "how long is this straight" means
        to a vehicle. Taken from the full scan rather than the handful of
        beams the policy is given, for the same reason `aim_feature` is --
        the car's lidar publishes 1080 beams whatever the policy reads, so
        one scalar off all of them is what the ROS node would compute.
        """
        if self._scans is None:
            return self.corridor_hi
        cone = np.abs(self.scan_angles) <= np.radians(6.0)
        if not cone.any():
            return self.corridor_hi
        return float(np.min(self._scans[0][cone]))

    @property
    def steer_rate_command(self) -> float:
        """The rate the wheel was actually given, after `steer_jerk_rate`.

        Exposed because every acceleration number in this project was once
        measured on the request while the env was quietly delivering
        something else; the steering axis gets the accessor from the start.
        """
        return self._steer_rate_cmd

    @property
    def corridor_ahead(self) -> float:
        """Metres of clear corridor straight ahead, as the gate reads it."""
        return self._corridor_m

    def _pull(self, state):
        """Bring one step's worth of device arrays across, in two transfers.

        Indexing a JAX array (``state.x[0, 3]``) is a *traced operation* that
        dispatches to the device and syncs back; doing that once per field
        costs milliseconds per step and was, measured, the single most
        expensive thing in this adapter. Pull the whole array once, index the
        NumPy copy afterwards.
        """
        return np.asarray(state.x), np.asarray(state.scans)

    def _obs_from(self, x: np.ndarray, scans: np.ndarray) -> np.ndarray:
        parts = [np.clip(scans[0][self._idx] / BEAM_RANGE, 0.0, 1.0)]
        if self.observe_speed:
            parts.append(np.array([x[0, 3] / SPEED_MAX]))
        if self.aim_feature:
            from rtrrl_playground.envs.scripted import free_space_heading
            from rtrrl_playground.envs.track import HEADING_SCALE
            # Computed from the *raw* scan, not the handful of beams the policy
            # sees. That is not cheating and it is not privileged: the car's
            # lidar publishes 1080 beams whatever the policy is given, so
            # deriving one scalar from all of them is what the ROS node would
            # actually do. Computing it from 9 downsampled beams instead -- as
            # the first version of this did -- gives the direction a ~20 deg
            # quantisation and measures the downsampling rather than the idea.
            # Forward sector only. The scan spans 269 deg, so the furthest
            # free space in a corridor is frequently *behind* the car -- the
            # open track it has already driven. Aiming there is meaningless for
            # a vehicle with 22.9 deg of lock, and measured, the unrestricted
            # version saturated the feature at +/-1 most of the time. +/-90 deg
            # is also exactly the span `HEADING_SCALE` normalises, so a hard
            # left aim reads as -1 and a hard right as +1 rather than clipping.
            m = np.abs(self.scan_angles) <= (np.pi / 2)
            aim = free_space_heading(scans[0][m], self.scan_angles[m])
            parts.append(np.array([np.clip(aim / HEADING_SCALE, -1.0, 1.0)]))
        if self.drive_features:
            from rtrrl_playground.envs.track import drive_features as _df
            m = np.abs(self.scan_angles) <= (np.pi / 2)
            parts.append(_df(scans[0][m], self.scan_angles[m], BEAM_RANGE))
        if self.proprio:
            # body-frame specific force, which is what an IMU reads:
            #   ax = dvx/dt - r vy      ay = dvy/dt + r vx
            v, beta, r_ = float(x[0, 3]), float(x[0, 6]), float(x[0, 5])
            vx, vy = v * np.cos(beta), v * np.sin(beta)
            if self._prev_vxy is None:
                ax = ay = 0.0
            else:
                pvx, pvy = self._prev_vxy
                ax = (vx - pvx) / self.dt - r_ * vy
                ay = (vy - pvy) / self.dt + r_ * vx
            self._prev_vxy = (vx, vy)
            parts.append(np.array([
                v / SPEED_MAX,
                float(x[0, 2]) / STEER_MAX,
                np.clip(r_ / 3.0, -1.0, 1.0),
                np.clip(ax / 10.0, -1.0, 1.0),
                np.clip(ay / 10.0, -1.0, 1.0)]))
        if self.grip_feature:
            v, beta, r_ = float(x[0, 3]), float(x[0, 6]), float(x[0, 5])
            vx, vy = v * np.cos(beta), v * np.sin(beta)
            if self._prev_g is None:
                ax = ay = 0.0
            else:
                pvx, pvy = self._prev_g
                ax = (vx - pvx) / self.dt - r_ * vy
                ay = (vy - pvy) / self.dt + r_ * vx
            self._prev_g = (vx, vy)
            # 1.0 means the tyres are saturated; the budget is the measured
            # peak friction, not a generic constant
            parts.append(np.array([
                np.clip(np.hypot(ax, ay) / (self.grip * 9.81), 0.0, 2.0) / 2.0]))
        if self.lookahead is not None:
            from rtrrl_playground.envs.track import lookahead_features
            parts.append(lookahead_features(self.centreline, float(x[0, 0]),
                                            float(x[0, 1]), float(x[0, 4]),
                                            self.lookahead))
        if self.steer_feature:
            # the filtered command when there is one: the policy should see
            # the angle the servo was given, not the one it asked for
            # Last, so that turning it on does not move any existing feature:
            # a saved policy, a figure's axis labels and the feature indices
            # written down elsewhere all keep meaning what they meant. The
            # *commanded* setpoint, not `x[0, 2]`, which is where the servo
            # has got to -- the command is what the action adds to and the
            # only one of the two the car knows without a sensor.
            w = self._steer_out if self.steer_filter else self._steer_cmd
            parts.append(np.array([w / STEER_MAX]))
        if self.corridor_feature:
            # After `steer_feature`, so switching it on moves nothing that
            # came before it. Normalised on the gate's own range rather than
            # BEAM_RANGE: what the policy needs is where it sits between "a
            # hairpin" and "a straight", and on this track those are 3 m and
            # 6 m, not 0 and 10.
            cone = np.abs(self.scan_angles) <= np.radians(6.0)
            m_ = float(np.min(scans[0][cone])) if cone.any() else self.corridor_hi
            parts.append(np.array([np.clip(
                (m_ - self.corridor_lo)
                / max(self.corridor_hi - self.corridor_lo, 1e-9), 0.0, 1.0)]))
        if self.de_features:
            # Last again, for the same reason, and off the raw scan for the
            # same reason `aim_feature` is: the car's lidar publishes 1080
            # beams whatever the policy reads, so one scalar off all of them
            # is what the ROS node would compute. `_de_prev` is None on the
            # first step of an episode, which makes `d` and `e` read 0 rather
            # than a jump against the last episode's heading.
            from rtrrl_playground.envs.track import de_features as _def
            f, th = _def(scans[0], self.scan_angles, BEAM_RANGE,
                         spec=self.de_features, cone_deg=self.de_cone_deg,
                         prev_theta=self._de_prev, dt=self.dt,
                         margin=self.de_margin)
            self._de_prev = th
            parts.append(f)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def _reward(self, x: np.ndarray, x_prev: np.ndarray) -> float:
        """Progress along the centreline if there is one, else distance moved.

        Both are in units of "a full-speed step", so the scale matches
        ``lanekeep``. The distinction is not cosmetic: distance travelled
        rewards a fast lap of any loop inside a wide corridor, and going
        backwards earns the same as going forwards. Arc length along the
        centreline is signed and is what a lap time is made of.
        """
        if self.centreline is None:
            return float(np.linalg.norm(x[0, :2] - x_prev[0, :2])) / (SPEED_MAX * self.dt)
        s = self._arc(x[0, :2])
        if self._prev_s is None:
            self._prev_s = s
            return 0.0
        # shortest signed step around the loop, so crossing the start line
        # does not read as a lap of negative progress
        d = s - self._prev_s
        half = self.lap_length / 2.0
        d = d - self.lap_length if d > half else (d + self.lap_length if d < -half else d)
        self._prev_s = s
        return float(d) / (SPEED_MAX * self.dt)

    def _arc(self, xy) -> float:
        """Arc length of the nearest centreline point, in metres."""
        return self._nearest(xy)[0]

    def _nearest(self, xy) -> tuple[float, float]:
        """``(arc length, lateral distance)`` of the nearest centreline point.

        Distance to the nearest *sample*, not the true distance to the
        polyline, so it over-reads by up to half the sample spacing on a
        straight. Centrelines here are resampled to a fixed spacing
        (``bridges.mapimport._resample``), which bounds that error and keeps
        this one argmin rather than a projection onto every segment.
        """
        d2 = np.sum((self.centreline - xy) ** 2, axis=1)
        k = int(np.argmin(d2))
        return float(self._cl_s[k]), float(np.sqrt(d2[k])), k

    def _limit_at(self, k: int) -> float:
        """The half-width in force at centreline index ``k``."""
        return float(self.half_width[k % len(self.half_width)]
                     if self.half_width.size > 1 else self.half_width[0])

    def _pose_at(self, i: int):
        """Centreline point ``i``, facing along the track."""
        c = np.asarray(self.centreline, float)[:, :2]
        nxt = c[(i + 1) % len(c)]
        psi = float(np.arctan2(nxt[1] - c[i, 1], nxt[0] - c[i, 0]))
        return np.array([c[i, 0], c[i, 1], psi], dtype=float)

    def _sample_start(self):
        """Where the next episode begins: the line, a random point, or just
        before wherever the last one ended."""
        if self.centreline is None:
            return self.start_pose
        c = np.asarray(self.centreline, float)[:, :2]
        if (self.respawn_at_crash and self._last_end is not None
                and self._rs_rng.random() < self.respawn_crash_prob):
            # nearest centreline point to the crash, backed up along the
            # track by `respawn_backup` metres so the corner is approached
            # rather than started from inside
            k = int(np.argmin(((c - np.asarray(self._last_end)) ** 2).sum(1)))
            step = np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T)
            back, j = 0.0, k
            while back < self.respawn_backup:
                j = (j - 1) % len(c)
                back += float(step[j])
            return self._pose_at(j)
        if self.random_start:
            return self._pose_at(int(self._rs_rng.integers(len(c))))
        return self.start_pose

    # -- Env ---------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        jnp = self._jnp
        if seed is not None:
            self._key = self._jax.random.key(seed)
        poses = jnp.asarray(self._sample_start()).reshape(1, 3)
        _obs, self._state = self.env.reset(self._split(), poses)
        self._t = 0
        # Ramp mode starts with a 1 m/s *setpoint*, so the PID pulls the car
        # away from the line on its own and an untrained policy is always
        # moving. In accl mode the slot holds an acceleration, where the same
        # 1.0 would be a stray command and a fresh policy -- symmetric over
        # +/-accel_cmd, so mean zero -- never gets the car rolling at all.
        # Measured: 92 % of episodes ended `stalled`, i.e. the mode change
        # would have been scored on an exploration artefact rather than on
        # control authority. Both modes therefore get the same rolling start,
        # one as a setpoint and one as an actual speed.
        self._v_cmd = 0.0 if self.accel_mode == "accl" else 1.0
        if self.accel_mode == "accl" and self.rolling_start:
            self._state = self._state.replace(
                x=self._state.x.at[0, 3].set(float(self.rolling_start)))
        self._steer_cmd = 0.0
        self._steer_out = 0.0
        self._steer_move = 0.0
        self._steer_rate_cmd = 0.0
        self._steer_jerk = 0.0
        self._corridor_m = self.corridor_hi
        self._corridor_gate = 0.0
        self._de_prev = None
        self._accel_cmd = 0.0
        self._accel_jerk = 0.0
        self._prev_vxy = None
        self._prev_s = None
        self.history = []
        self._x, self._scans = self._pull(self._state)
        return self._obs_from(self._x, self._scans)

    def step(self, action):
        u = self._jnp.asarray(self._decode(action)).reshape(1, 2)
        x_prev = self._x
        # One key per agent step, reused across the sub-ticks. The key only
        # seeds the lidar noise, which is read once at the end, so splitting
        # per tick buys nothing and costs a dispatch each time.
        key = self._split()
        for _ in range(self.control_repeat):
            # step_env, not step: the simulator's step() auto-resets when every
            # agent is done, which would silently teleport the car mid-episode
            # and hand the agent a transition that never happened.
            _o, self._state, _r, _d, _i = self._step_env(key, self._state, u)
        self._x, self._scans = self._pull(self._state)
        crashed = bool(np.asarray(self._state.collisions)[0])
        # Kept as three named parts, not one float. The scalar is all the
        # agent ever sees, but a figure of "the reward" that cannot say how
        # much of it was progress and how much was a penalty cannot explain a
        # policy that drives slowly to avoid a cost -- which is the failure
        # these penalties are most likely to cause.
        r_progress = self._reward(self._x, x_prev)
        r_grip = 0.0
        if self.grip_penalty:
            xx = self._x[0]
            usage = abs(float(xx[3]) * float(xx[5])) / (self.grip * 9.81)
            r_grip = -self.grip_penalty * max(0.0, usage - self.grip_target)
        r_reverse = (-self.reverse_penalty
                     if self.reverse_penalty and self._steer_flip else 0.0)
        r_ajerk = -self.accel_jerk_penalty * self._accel_jerk
        r_smove = -self.steer_move_penalty * self._steer_move
        # free in a hairpin, full price down a straight; see the note on
        # `corridor_steer_penalty` for why the gate is not another flat cost
        r_corr = (-self.corridor_steer_penalty * self._corridor_gate
                  * self._steer_move)
        reward = r_progress + r_grip + r_reverse + r_ajerk + r_smove + r_corr
        self._t += 1
        x = self._x[0]
        self.history.append(dict(x=float(x[0]), y=float(x[1]), psi=float(x[4]),
                                 v=float(x[3]), d=0.0))
        # Off the track counts as a crash: same -1.0, same termination. The
        # info flag distinguishes them so a run can report which boundary it
        # actually hit -- on an imported map "left the corridor" is the one
        # that fires, and reading it as a collision would misattribute it.
        if self.half_width is not None:
            _s, lateral, k = self._nearest(x[:2])
            limit = self._limit_at(k)
        else:
            lateral, limit = float("nan"), float("nan")
        off_track = bool(self.half_width is not None and lateral > limit)
        if crashed or off_track:
            # remembered for the next reset; see `respawn_at_crash`
            self._last_end = (float(x[0]), float(x[1]))
            return (np.zeros(self.obs_dim), -1.0, True, False,
                    {"crashed": bool(crashed), "off_track": off_track,
                     "lateral": lateral, "r_progress": 0.0, "r_grip": 0.0,
                     "r_reverse": 0.0, "r_accel_jerk": 0.0,
                     "r_steer_move": 0.0, "r_corridor": 0.0,
                     "r_crash": -1.0})
        truncated = self._t >= self.max_steps
        return (self._obs_from(self._x, self._scans), float(reward), False, truncated,
                {"crashed": False, "off_track": False, "lateral": lateral,
                 "v": float(x[3]), "r_progress": float(r_progress),
                 "r_grip": float(r_grip), "r_reverse": float(r_reverse),
                 "r_accel_jerk": float(r_ajerk),
                 "r_steer_move": float(r_smove),
                 "r_corridor": float(r_corr),
                 "corridor_m": float(self._corridor_m),
                 "corridor_gate": float(self._corridor_gate),
                 "steer_rate_cmd": float(self._steer_rate_cmd),
                 "r_crash": 0.0})

    # -- pictures ----------------------------------------------------------
    def render_rollout(self, history=None, path: str = "rollout.png", title: str = ""):
        """Plot the driven line over the map's occupancy image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = history if history is not None else self.history
        fig, ax = plt.subplots(figsize=(7, 6))
        sim = getattr(self.env, "scan_sim", None)
        img = getattr(sim, "map_img", None)
        if img is not None:
            ax.imshow(np.asarray(img), cmap="gray", origin="lower")
            ax.set_title((title or f"{self.id} / {self.map_name}") + "  (map pixels)")
        elif hist:
            ax.set_aspect("equal")
            ax.set_title(title or f"{self.id} / {self.map_name}")
        if hist:
            xs = np.array([h["x"] for h in hist])
            ys = np.array([h["y"] for h in hist])
            vs = np.array([h["v"] for h in hist])
            res = getattr(sim, "map_resolution", None)
            org = getattr(sim, "origin", None)
            if img is not None and res and org is not None:
                xs = (xs - float(org[0])) / float(res)
                ys = (ys - float(org[1])) / float(res)
            sc = ax.scatter(xs, ys, c=vs, s=3, cmap="viridis")
            fig.colorbar(sc, ax=ax, label="speed [m/s]")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
