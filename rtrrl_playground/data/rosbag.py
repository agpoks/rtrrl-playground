"""Turn a ROS 2 bag of a real car into demonstrations this repo can learn from.

    from rtrrl_playground.data import BagDemonstrations
    demos = BagDemonstrations("/path/to/bag")          # a rosbag2 directory
    for obs, action, info in demos:
        agent.imitate(obs, action, info["reward"], info["terminated"], False)

**No ROS installation is needed.** Reading is done with ``rosbags``, which is
pure Python (``pip install rosbags``); the two message types an F1TENTH stack
uses that are not in the standard typestore -- ``ackermann_msgs`` and
``vesc_msgs`` -- are registered below from their ``.msg`` definitions.

## What it does, and the three judgement calls in it

A recording is not a dataset. Getting from one to the other means deciding
three things, and each of them is a place where a quiet mistake turns into a
policy that scores well on the recording and cannot drive:

**Which clock.** ``/scan`` arrives at ~40 Hz, ``/ackermann_cmd`` at ~38 Hz,
``/odom`` at ~50 Hz, none of them aligned. A sample is emitted **on each drive
command**, carrying the most recent scan and odometry *strictly older than it*.
Never a scan from after the command: that is the future leaking into the input,
and it makes a cloned policy look clairvoyant right up until it is deployed.

**Which beams.** A Hokuyo gives 1081 rays over 270 degrees. The agent here
takes nine over 120, so the raw scan is sampled at the nine angles nearest to
:data:`~rtrrl_playground.envs.lanekeep.BEAM_ANGLES` and normalised by the same
``BEAM_RANGE``. The point is that a real observation and a simulated one are
*the same vector*, so a policy cloned from a bag can be fine-tuned in the
simulator and vice versa.

**Which action.** The recorded command is continuous (a steering angle in
radians, a speed in m/s). The discrete agent needs one of nine. Steering maps
to left/straight/right through a deadband; throttle maps to the sign of the
*change* in commanded speed, since the environment's action is an acceleration.
``action_mode="continuous"`` skips all of that and hands back the raw pair.

Samples where the car is stationary are dropped by default (``min_speed``): a
recording of somebody standing still teaches a policy to stand still, and in a
localisation test bag that can be a third of the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rtrrl_playground.envs.lanekeep import (
    ACCEL_MAX, BEAM_ANGLES, BEAM_RANGE, SPEED_MAX, STEER_MAX, WHEELBASE,
)

# --- message definitions the standard typestore does not carry --------------
_ACKERMANN_DRIVE = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""
_ACKERMANN_DRIVE_STAMPED = """
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""


def _typestore():
    """A ROS 2 typestore with the F1TENTH message types registered."""
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading bags needs the pure-Python rosbags package (no ROS install "
            "required):\n    pip install rosbags"
        ) from exc
    ts = get_typestore(Stores.ROS2_HUMBLE)
    for text, name in ((_ACKERMANN_DRIVE, "ackermann_msgs/msg/AckermannDrive"),
                       (_ACKERMANN_DRIVE_STAMPED, "ackermann_msgs/msg/AckermannDriveStamped")):
        try:
            ts.register(get_types_from_msg(text, name))
        except Exception:  # already registered by a newer rosbags
            pass
    return ts


def _reader(path):
    from rosbags.highlevel import AnyReader

    return AnyReader([Path(path)], default_typestore=_typestore())


# --- inspection --------------------------------------------------------------
def bag_summary(path) -> dict:
    """Topics, counts and duration. Run this first on any unfamiliar recording."""
    with _reader(path) as r:
        topics = {c.topic: (c.msgtype, c.msgcount) for c in r.connections}
        return {"duration_s": (r.end_time - r.start_time) / 1e9,
                "topics": dict(sorted(topics.items(), key=lambda kv: -kv[1][1]))}


def load_map(path, topic: str = "/map"):
    """The occupancy grid, as ``(free, resolution, origin)``.

    ``free`` is a boolean array, ``True`` where the cell is known-free -- the
    same convention as :class:`~rtrrl_playground.envs.track.Track`'s rasterised
    bitmap, so a real map can be dropped straight into the simulator. Unknown
    cells (``-1``) count as blocked, which is the conservative reading and the
    one that stops an agent from driving confidently into unmapped space.
    """
    with _reader(path) as r:
        cons = [c for c in r.connections if c.topic == topic]
        if not cons:
            raise KeyError(f"no {topic} in this bag; try bag_summary()")
        for _c, _t, raw in r.messages(connections=cons):
            m = r.deserialize(raw, cons[0].msgtype)
            grid = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
            origin = np.array([m.info.origin.position.x, m.info.origin.position.y])
            return grid == 0, float(m.info.resolution), origin
    raise KeyError(f"{topic} present but empty")


def load_odometry(path, topic: str = "/odom"):
    """``(t, x, y, yaw, v)`` arrays from an ``nav_msgs/Odometry`` topic."""
    ts, xs, ys, yaws, vs = [], [], [], [], []
    with _reader(path) as r:
        cons = [c for c in r.connections if c.topic == topic]
        if not cons:
            raise KeyError(f"no {topic} in this bag; try bag_summary()")
        for _c, t, raw in r.messages(connections=cons):
            m = r.deserialize(raw, cons[0].msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            ts.append(t / 1e9)
            xs.append(p.x)
            ys.append(p.y)
            yaws.append(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                   1 - 2 * (q.y * q.y + q.z * q.z)))
            vs.append(m.twist.twist.linear.x)
    return (np.array(ts), np.array(xs), np.array(ys), np.array(yaws), np.array(vs))


# --- demonstrations ----------------------------------------------------------
class BagDemonstrations:
    """A stream of ``(observation, expert action, info)`` from a real recording.

    Iterating yields one sample per drive command, in recorded order. ``info``
    carries ``reward`` (distance travelled since the previous sample, in the
    same units as :class:`~rtrrl_playground.envs.lanekeep.LaneKeep`'s), ``speed``,
    ``terminated`` (true at a gap in the recording, so a caller can reset the
    agent's recurrent state rather than integrating across a discontinuity)
    and the raw command.
    """

    def __init__(self, path, scan_topic: str = "/scan",
                 cmd_topic: str = "/ackermann_cmd", odom_topic: str = "/odom",
                 action_mode: str = "discrete", observe_speed: bool = False,
                 min_speed: float = 0.3, gap_s: float = 0.5,
                 steer_deadband: float = 0.05, accel_deadband: float = 0.2,
                 max_range: float = BEAM_RANGE):
        self.path = Path(path)
        self.scan_topic, self.cmd_topic, self.odom_topic = scan_topic, cmd_topic, odom_topic
        self.action_mode = action_mode
        self.observe_speed = bool(observe_speed)
        self.min_speed, self.gap_s = float(min_speed), float(gap_s)
        self.steer_deadband, self.accel_deadband = float(steer_deadband), float(accel_deadband)
        self.max_range = float(max_range)
        self.n_beams = len(BEAM_ANGLES)
        self.obs_dim = self.n_beams + int(self.observe_speed)
        self._beam_idx = None
        self.stats: dict = {}

    # -- helpers ----------------------------------------------------------
    def _select_beams(self, scan) -> np.ndarray:
        """Indices of the raw rays nearest this repo's nine beam angles."""
        if self._beam_idx is None:
            angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
            self._beam_idx = np.array([int(np.argmin(np.abs(angles - a))) for a in BEAM_ANGLES])
            self.stats["beam_angles_deg"] = np.degrees(angles[self._beam_idx]).round(1).tolist()
        return self._beam_idx

    def _obs(self, scan, speed: float) -> np.ndarray:
        idx = self._select_beams(scan)
        r = np.asarray(scan.ranges, dtype=np.float64)[idx]
        # A lidar reports inf/nan for "nothing out there"; the simulator reports
        # max_range. Same meaning, and the network must see the same number.
        r = np.nan_to_num(r, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        beams = np.clip(r, 0.0, self.max_range) / self.max_range
        if self.observe_speed:
            return np.concatenate([beams, [speed / SPEED_MAX]])
        return beams

    def _action(self, cmd, prev_cmd_speed: float, dt: float):
        """Recorded command -> the agent's action."""
        steer, speed = float(cmd.steering_angle), float(cmd.speed)
        if self.action_mode == "continuous":
            accel = (speed - prev_cmd_speed) / max(dt, 1e-3)
            return np.array([np.clip(steer / STEER_MAX, -1, 1),
                             np.clip(accel / ACCEL_MAX, -1, 1)])
        s = 0 if abs(steer) < self.steer_deadband else (1 if steer > 0 else -1)
        accel = (speed - prev_cmd_speed) / max(dt, 1e-3)
        a = 0 if abs(accel) < self.accel_deadband else (1 if accel > 0 else -1)
        return 3 * (s + 1) + (a + 1)

    # -- iteration --------------------------------------------------------
    def __iter__(self):
        with _reader(self.path) as r:
            wanted = {self.scan_topic, self.cmd_topic, self.odom_topic}
            cons = [c for c in r.connections if c.topic in wanted]
            missing = wanted - {c.topic for c in cons}
            if missing:
                raise KeyError(f"bag is missing {sorted(missing)}; try bag_summary()")
            types = {c.topic: c.msgtype for c in cons}

            scan = odom = None
            prev_t = prev_speed = None
            prev_xy = None
            kept = dropped = 0
            for c, t_ns, raw in r.messages(connections=cons):
                t = t_ns / 1e9
                msg = r.deserialize(raw, types[c.topic])
                if c.topic == self.scan_topic:
                    scan = msg
                    continue
                if c.topic == self.odom_topic:
                    odom = msg
                    continue
                # --- a drive command: emit a sample, if we have the inputs ---
                if scan is None or odom is None:
                    continue
                speed = float(odom.twist.twist.linear.x)
                xy = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
                gap = prev_t is None or (t - prev_t) > self.gap_s
                dt = 0.05 if gap else t - prev_t
                if abs(speed) < self.min_speed:
                    dropped += 1
                    prev_t, prev_speed, prev_xy = t, float(msg.drive.speed), xy
                    continue
                action = self._action(msg.drive, prev_speed if prev_speed is not None
                                      else float(msg.drive.speed), dt)
                travelled = 0.0 if (gap or prev_xy is None) else float(np.linalg.norm(xy - prev_xy))
                info = {"t": t, "dt": dt, "speed": speed,
                        "reward": travelled / (SPEED_MAX * 0.05),
                        "terminated": bool(gap), "steering": float(msg.drive.steering_angle),
                        "cmd_speed": float(msg.drive.speed), "xy": xy}
                kept += 1
                prev_t, prev_speed, prev_xy = t, float(msg.drive.speed), xy
                yield self._obs(scan, speed), action, info
            self.stats.update(kept=kept, dropped_stationary=dropped)

    def to_arrays(self):
        """Materialise the whole recording as ``(obs, actions, infos)``.

        Convenient for a train/test split; do not use it on an hour-long bag
        unless you have checked how much memory that is.
        """
        obs, acts, infos = [], [], []
        for o, a, i in self:
            obs.append(o)
            acts.append(a)
            infos.append(i)
        return np.array(obs), np.array(acts), infos


# --- the track the recording was made on -------------------------------------
def _forward_only(xs, ys, vs, min_speed: float):
    """Keep the samples where the car was driving forwards, and only those.

    Two things have to go, and the second is the one that bites. Stationary
    samples are obvious -- a recording of somebody parked is not a lap.
    **Reversals** are not: ``|v| > min_speed`` happily keeps a car backing out
    of a corner, and a path that doubles back on itself has a *cusp*, which
    after resampling reads as a corner of near-zero radius. In the recording
    this was written against that single effect put 5% of the "track" below the
    car's minimum turning circle, and no amount of smoothing fixed it, because
    smoothing a cusp just moves it.
    """
    keep = vs > min_speed  # not abs(vs): reversing is not driving
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 3:
        return xs, ys
    # Then drop any step that turns back on the previous one by more than 90
    # degrees -- what is left of a reversal after the speed filter.
    d = np.diff(np.stack([xs, ys], axis=1), axis=0)
    n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-9]
    idx = np.where(n > 1e-9)[0]
    good = np.ones(len(idx) + 1, dtype=bool)
    dot = np.einsum("ij,ij->i", d[:-1], d[1:])
    back = dot < 0
    good[1:-1] &= ~back
    sel = np.concatenate([[0], idx + 1])[good]
    return xs[sel], ys[sel]


def _resample_closed(xs: np.ndarray, ys: np.ndarray, ds: float, smooth_m: float):
    """Uniform-arc-length resample of a closed path, smoothed over ``smooth_m``.

    Smoothing is specified in **metres of arc**, not in samples, because that
    is the quantity you actually have an intuition about: 1 m of a 40 m lap is
    a gentle tidy-up, and 15 m of it -- which is what a 75-sample box filter
    turned out to be -- shrinks the whole circuit and invents hairpins where
    the corners used to be.

    The filter is circular, so the point where the lap joins itself is not left
    as a corner; a kink there becomes a curvature spike and, downstream, one
    step of enormous reward once per lap.
    """
    pts = np.stack([xs, ys], axis=1)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6])
    pts = pts[keep]

    def resample(p, step):
        seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(s[-1] / step), 16)
        target = np.linspace(0.0, s[-1], n, endpoint=False)
        closed = np.vstack([p, p[:1]])
        return np.stack([np.interp(target, s, closed[:, 0]),
                         np.interp(target, s, closed[:, 1])], axis=1)

    out = resample(pts, ds)
    k = max(int(round(smooth_m / ds)), 1)
    if k > 1 and len(out) > 4 * k:
        kern = np.ones(k) / k
        pad = np.vstack([out[-k:], out, out[:k]])
        out = np.stack([np.convolve(pad[:, 0], kern, mode="same"),
                        np.convolve(pad[:, 1], kern, mode="same")], axis=1)[k:-k]
        out = resample(out, ds)  # smoothing shortens the path; restore uniform ds
    return out[:, 0], out[:, 1]


def _clearance(cx, cy, free, res, origin, max_m: float = 3.0, step: float = 0.05):
    """How far the walls are either side of each centreline point, from the map."""
    d = np.diff(np.vstack([np.stack([cx, cy], 1), np.stack([cx[:1], cy[:1]], 1)]), axis=0)
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    ny, nx = free.shape
    out = np.full(len(cx), max_m)
    for sign in (1.0, -1.0):
        dist = np.full(len(cx), max_m)
        alive = np.ones(len(cx), dtype=bool)
        for m in np.arange(step, max_m, step):
            px = cx + sign * nrm[:, 0] * m
            py = cy + sign * nrm[:, 1] * m
            i = ((px - origin[0]) / res).astype(np.int32)
            j = ((py - origin[1]) / res).astype(np.int32)
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            blocked = ~ok
            blocked[ok] |= ~free[j[ok], i[ok]]
            hit = alive & blocked
            dist[hit] = m
            alive &= ~hit
            if not alive.any():
                break
        out = np.minimum(out, dist)
    return out


POSE_TOPIC_CANDIDATES = ("/ekf/state", "/amcl_pose", "/pf/pose", "/odom")


def pick_pose_topic(path, free, res, origin, candidates=POSE_TOPIC_CANDIDATES):
    """Choose the pose topic that is actually in the **map** frame.

    This exists because of a bug that is easy to write and hard to see. A ROS
    stack publishes ``/odom`` in the odom frame and corrects it to the map frame
    through a ``map -> odom`` transform on ``/tf``; take ``/odom`` at face value
    and the trajectory drifts off the map -- in the recording this was written
    against, ``/odom`` reached x = 12.1 m on a map that ends at 7.7 m, so the
    "track" was built half outside the building.

    The test is blunt and works: whichever topic puts the most of its poses on
    a cell the map calls free is the one in the map frame.
    """
    best, best_score = None, -1.0
    scores = {}
    with _reader(path) as r:
        present = {c.topic for c in r.connections}
    for topic in candidates:
        if topic not in present:
            continue
        try:
            _t, x, y, _yaw, _v = load_odometry(path, topic)
        except Exception:
            continue
        i = ((x - origin[0]) / res).astype(np.int32)
        j = ((y - origin[1]) / res).astype(np.int32)
        ok = (i >= 0) & (i < free.shape[1]) & (j >= 0) & (j < free.shape[0])
        on_free = np.zeros(len(x), dtype=bool)
        on_free[ok] = free[j[ok], i[ok]]
        scores[topic] = float(on_free.mean())
        if scores[topic] > best_score:
            best, best_score = topic, scores[topic]
    if best is None:
        raise KeyError(f"none of {candidates} in this bag; try bag_summary()")
    return best, scores


def track_from_bag(path, half_width: float | None = None, ds: float = 0.2,
                   smooth_m: float = 1.0, map_topic: str = "/map",
                   odom_topic: str | None = None, min_speed: float = 0.3):
    """Build a :class:`~rtrrl_playground.envs.track.Track` from a recording.

    The **driven line becomes the centreline** -- resampled to uniform arc
    length, smoothed and closed -- and the recording's occupancy grid becomes
    the track bitmap, so the simulator's beams hit the walls the real lidar saw.

    Using the human's line rather than a geometric centre is deliberate: the
    progress reward then pays for following *the line that was driven*, which
    is a different objective from following the middle of the corridor and
    usually a better one. It also means a bad lap makes a bad track, so look at
    the picture before trusting it -- ``LaneKeep.render_rollout`` draws both.

    ``odom_topic=None`` (the default) picks the pose topic that is genuinely in
    the map frame -- see :func:`pick_pose_topic`, and read it before overriding.

    ``half_width`` defaults to the median measured clearance along the line.

    Returns ``(track, info)``.
    """
    from rtrrl_playground.envs.track import Track

    free, res, origin = load_map(path, map_topic)
    pose_scores = {}
    if odom_topic is None:
        odom_topic, pose_scores = pick_pose_topic(path, free, res, origin)
    _t, xs, ys, _yaw, vs = load_odometry(path, odom_topic)
    fx, fy = _forward_only(xs, ys, vs, min_speed)
    if len(fx) < 32:
        raise ValueError(f"only {len(fx)} forward-driving odometry samples -- "
                         "not enough of a lap to build a track from")
    cx, cy = _resample_closed(fx, fy, ds, smooth_m)
    clear = _clearance(cx, cy, free, res, origin)
    if half_width is None:
        # The median clearance, not the minimum. The minimum is wherever the
        # mapper put a stray occupied cell or the driver clipped an apex, and
        # taking it gives a corridor narrower than the car. The median is the
        # width of the corridor the lap was actually driven in; the tight spots
        # survive anyway, because the *bitmap* is the real map and a beam that
        # hits a real wall still ends there.
        half_width = float(np.clip(np.median(clear), 0.4, 2.0))

    track = Track(cx, cy, half_width=half_width, grid_res=res)
    # Replace the rasterised tube with the real map, resampled onto the Track's
    # own grid by nearest cell. The Track's bitmap and the real one do not share
    # an origin, and quietly assuming they do would put the walls in the wrong
    # place by up to half a track width.
    gx = track.origin[0] + np.arange(track.nx) * track.res
    gy = track.origin[1] + np.arange(track.ny) * track.res
    mi = ((gx - origin[0]) / res).astype(np.int32)
    mj = ((gy - origin[1]) / res).astype(np.int32)
    ok_i, ok_j = (mi >= 0) & (mi < free.shape[1]), (mj >= 0) & (mj < free.shape[0])
    real = np.zeros((track.ny, track.nx), dtype=bool)
    real[np.ix_(ok_j, ok_i)] = free[np.ix_(mj[ok_j], mi[ok_i])]
    # Keep only what is *both* mapped free and within the tube: the map includes
    # the whole building, and without the intersection the "track" is every
    # corridor the lidar ever saw.
    track.free &= real

    # Drivability, reported rather than assumed. A reconstructed centreline can
    # easily contain corners the car physically cannot take, and finding that
    # out from a flat learning curve is a bad afternoon.
    radius = 1.0 / np.maximum(np.abs(track.curvature), 1e-6)
    geo_min = WHEELBASE / np.tan(STEER_MAX)
    info = {"pose_topic": odom_topic, "pose_topic_scores": pose_scores,
            "length_m": track.length, "half_width": half_width,
            "clearance_min": float(clear.min()), "clearance_median": float(np.median(clear)),
            "radius_min": float(radius.min()), "radius_p5": float(np.percentile(radius, 5)),
            "radius_median": float(np.median(radius)),
            "frac_below_turning_circle": float(np.mean(radius < geo_min)),
            "car_min_radius": float(geo_min),
            "map_shape": free.shape, "map_res": res,
            "n_odom": int(len(xs)), "n_forward": int(len(fx))}
    return track, info
