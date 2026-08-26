"""A closed race track, and the geometry the two driving tasks need.

The track is a polyline centreline sampled from a closed parametric curve,
plus a constant half-width. The driving environments ask it three things:
*how far along am I*, *how far off the line am I*, and *how far to the wall
along this ray*. The first two come from a projection onto the polyline; the
third comes from a bitmap.

**Why a bitmap for the beams.** The boundary is implied by
``|d| <= half_width`` rather than stored as a polygon, so there is nothing to
intersect analytically, and projecting every sample point of every beam back
onto the polyline costs more than the network update it is supposed to feed.
So the track is rasterised once, at construction, into a boolean
"on-track" grid -- the union of discs of radius ``half_width`` centred on the
centreline samples, which at 20 cm sample spacing is within 5 mm of the true
tube -- and a beam becomes a strided lookup into that array. This is also
what the simulator you are heading for actually does: ``f1tenth_gym`` and
``scuderia_gym_jax`` both ray-march against an occupancy image rather than
against track geometry. The playground being bitmap-based here is not a
shortcut away from the real thing, it is a step towards it.

**Why the projection is windowed.** Searching the whole centreline for the
ego's own nearest sample is not merely slower, it is *wrong* on a track that
folds back near itself: the globally nearest sample can be on the neighbouring
lobe, and the car appears to teleport across the infield with a lap of
progress in one step. Searching only samples near where the car was last step
keeps the projection on the branch the car is actually driving.
"""

from __future__ import annotations

import numpy as np


class Track:
    """Closed centreline + constant half-width, with Frenet lookups and beams."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, half_width: float = 1.1,
                 grid_res: float = 0.05, margin: float = 1.0):
        self.center = np.stack([xs, ys], axis=1)  # (K, 2)
        self.K = len(self.center)
        self.cx, self.cy = np.ascontiguousarray(xs), np.ascontiguousarray(ys)
        d = np.diff(np.vstack([self.center, self.center[:1]]), axis=0)
        seg = np.linalg.norm(d, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])  # (K,) arc length
        self.length = float(seg.sum())
        self.ds = self.length / self.K
        self.tangent = d / seg[:, None]  # (K, 2)
        self.tx, self.ty = np.ascontiguousarray(self.tangent[:, 0]), np.ascontiguousarray(self.tangent[:, 1])
        self.normal = np.stack([-self.ty, self.tx], axis=1)  # left of travel is +d
        self.half_width = float(half_width)
        psi = np.unwrap(np.arctan2(self.ty, self.tx))
        self.heading = psi
        self.curvature = np.concatenate([np.diff(psi) / seg[:-1], [0.0]])
        self._build_grid(grid_res, margin)
        # Small per-shape caches. Every one of these arrays is rebuilt on every
        # observation otherwise, and at 20 Hz x a million training steps the
        # allocation dominates the actual geometry.
        self._offsets: dict[int, np.ndarray] = {}
        self._ray_t: dict[tuple, np.ndarray] = {}
        self._nidx: np.ndarray | None = None

    # -- rasterisation ----------------------------------------------------
    def _build_grid(self, res: float, margin: float) -> None:
        self.res = float(res)
        lo = self.center.min(axis=0) - (self.half_width + margin)
        hi = self.center.max(axis=0) + (self.half_width + margin)
        self.origin = lo
        self.nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
        self.ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
        free = np.zeros((self.ny, self.nx), dtype=bool)
        rr = int(np.ceil(self.half_width / res))
        oy, ox = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disc = (ox ** 2 + oy ** 2) * res ** 2 <= self.half_width ** 2
        oy, ox = oy[disc], ox[disc]
        ci = ((self.cx - lo[0]) / res).astype(np.int32)
        cj = ((self.cy - lo[1]) / res).astype(np.int32)
        jj = np.clip((cj[:, None] + oy[None, :]).ravel(), 0, self.ny - 1)
        ii = np.clip((ci[:, None] + ox[None, :]).ravel(), 0, self.nx - 1)
        free[jj, ii] = True
        self.free = free

    def grid_index(self, x, y):
        """World coordinates -> clipped ``(i, j)`` bitmap indices.

        Exposed because the safety filter needs *both* the occupancy bit and
        the nearest-centreline index for the same points, and converting the
        coordinates twice was, measured, a third of its cost.
        """
        i = ((x - self.origin[0]) / self.res).astype(np.int32).clip(0, self.nx - 1)
        j = ((y - self.origin[1]) / self.res).astype(np.int32).clip(0, self.ny - 1)
        return i, j

    def on_track(self, x, y) -> np.ndarray:
        """Bitmap membership test. Vectorised over any shape of ``x``, ``y``.

        The grid is built with a margin wider than any beam, so clipping to the
        array bounds can only ever clamp a point that was already far outside
        the track onto another cell that is also outside it."""
        i = ((x - self.origin[0]) / self.res).astype(np.int32).clip(0, self.nx - 1)
        j = ((y - self.origin[1]) / self.res).astype(np.int32).clip(0, self.ny - 1)
        return self.free[j, i]

    def nearest_index(self, x, y) -> np.ndarray:
        """Nearest centreline sample for points, in O(1). ``-1`` when far away.

        A cached lookup grid, built the same way as the occupancy bitmap: stamp
        a disc around each centreline sample and keep, per cell, the sample it
        is closest to. Cells further from the line than the disc radius keep
        ``-1``, which is a useful answer rather than a missing one -- anything
        that far out is off the track by construction.

        This exists for the predictive safety filter in
        :mod:`rtrrl_playground.safety`, which projects nine candidate
        trajectories over a 24-step horizon at every control tick. Done with a
        search over the centreline that is ~60k distance computations per tick
        and dominates everything; done with this it is an array index.
        """
        if self._nidx is None:
            self._build_nearest_grid()
        i = ((np.asarray(x) - self.origin[0]) / self.res).astype(np.int32).clip(0, self.nx - 1)
        j = ((np.asarray(y) - self.origin[1]) / self.res).astype(np.int32).clip(0, self.ny - 1)
        return self._nidx[j, i]

    def _build_nearest_grid(self, pad: float = 1.0) -> None:
        rr = int(np.ceil((self.half_width + pad) / self.res))
        oy, ox = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disc = (ox ** 2 + oy ** 2) * self.res ** 2 <= (self.half_width + pad) ** 2
        oy, ox = oy[disc], ox[disc]
        best_d2 = np.full((self.ny, self.nx), np.inf)
        best_k = np.full((self.ny, self.nx), -1, dtype=np.int32)
        ci = ((self.cx - self.origin[0]) / self.res).astype(np.int32)
        cj = ((self.cy - self.origin[1]) / self.res).astype(np.int32)
        for k in range(self.K):
            jj = np.clip(cj[k] + oy, 0, self.ny - 1)
            ii = np.clip(ci[k] + ox, 0, self.nx - 1)
            px = self.origin[0] + ii * self.res
            py = self.origin[1] + jj * self.res
            d2 = (px - self.cx[k]) ** 2 + (py - self.cy[k]) ** 2
            better = d2 < best_d2[jj, ii]
            best_d2[jj[better], ii[better]] = d2[better]
            best_k[jj[better], ii[better]] = k
        self._nidx = best_k

    # -- projection -------------------------------------------------------
    def nearest(self, x: float, y: float) -> int:
        """Globally nearest centreline sample. Only used at reset."""
        return int(np.argmin((self.cx - x) ** 2 + (self.cy - y) ** 2))

    def _window(self, k0: int, half: int) -> np.ndarray:
        off = self._offsets.get(half)
        if off is None:
            off = self._offsets[half] = np.arange(-half, half + 1)
        return (k0 + off) % self.K

    def project(self, px: np.ndarray, py: np.ndarray, k0: int, half: int = 24):
        """Project points onto the centreline within ``+/- half`` samples of ``k0``.

        Returns ``(s, d, k)``: arc length, signed lateral offset (left of the
        direction of travel positive), and the winning sample index.
        """
        idx = self._window(k0, half)  # (W,)
        dx = px[:, None] - self.cx[idx][None, :]
        dy = py[:, None] - self.cy[idx][None, :]
        k = idx[np.argmin(dx * dx + dy * dy, axis=1)]
        rx, ry = px - self.cx[k], py - self.cy[k]
        tx, ty = self.tx[k], self.ty[k]
        return (self.s[k] + rx * tx + ry * ty) % self.length, -rx * ty + ry * tx, k

    def frenet(self, x: float, y: float, k0: int | None = None, half: int = 24):
        """Scalar :meth:`project` -- flat arrays, no broadcasting, no wrappers.

        This is called once per environment step, so it is written out rather
        than routed through :meth:`project`: the two-line saving in source is
        not worth ~20 microseconds of NumPy dispatch on the hot path.
        """
        if k0 is None:
            k0, half = self.nearest(x, y), 1
        idx = self._window(k0, half)
        dx, dy = x - self.cx[idx], y - self.cy[idx]
        j = int(np.argmin(dx * dx + dy * dy))
        k = int(idx[j])
        rx, ry, tx, ty = dx[j], dy[j], self.tx[k], self.ty[k]
        return float((self.s[k] + rx * tx + ry * ty) % self.length), float(-rx * ty + ry * tx), k

    def ds_forward(self, s_new: float, s_old: float) -> float:
        """Signed progress from ``s_old`` to ``s_new``, correct across the line.

        A progress term that forgets this returns ``-length`` once per lap, and
        the agent learns that crossing the start/finish line is catastrophic.
        """
        delta = (s_new - s_old) % self.length
        return delta - self.length if delta > self.length / 2 else delta

    # -- sensing ----------------------------------------------------------
    def beam_ranges(self, x: float, y: float, psi: float, angles: np.ndarray,
                    max_range: float = 5.0, step: float = 0.15,
                    obstacles: np.ndarray | None = None, obs_radius: float = 0.25):
        """Ray-march ``angles`` (relative to ``psi``) from ``(x, y)``.

        Returns ``(ranges, is_obstacle)``: how far each beam gets before it
        leaves the track *or* touches one of ``obstacles`` ``(M, 2)``, and a
        flag saying which of the two stopped it. Beams that reach ``max_range``
        untroubled return ``max_range``.

        The obstacle flag is the only reason the overtaking agent can tell a
        wall from a car: without it, a car two metres ahead and a wall two
        metres ahead are the same observation, and no amount of recurrence
        recovers a distinction the sensor never made.
        """
        key = (max_range, step)
        t = self._ray_t.get(key)
        if t is None:
            n_t = max(int(max_range / step), 2)
            t = self._ray_t[key] = np.arange(1, n_t + 1) * step  # (T,)
        a = psi + angles
        px = x + np.cos(a)[:, None] * t[None, :]  # (B, T)
        py = y + np.sin(a)[:, None] * t[None, :]
        blocked = ~self.on_track(px, py)

        hit_obs = None
        if obstacles is not None and len(obstacles):
            dx = px[:, :, None] - obstacles[None, None, :, 0]
            dy = py[:, :, None] - obstacles[None, None, :, 1]
            hit_obs = ((dx * dx + dy * dy) < obs_radius ** 2).any(axis=2)
            blocked = blocked | hit_obs
        any_hit = blocked.any(axis=1)
        first = np.argmax(blocked, axis=1)
        ranges = np.where(any_hit, t[first], max_range)
        if hit_obs is None:
            return ranges, np.zeros(len(angles))
        flags = any_hit & hit_obs[np.arange(len(angles)), first]
        return ranges, flags.astype(np.float64)

    def curvature_preview(self, k: int, ahead_m: float = 6.0, n: int = 3) -> np.ndarray:
        """``n`` curvature samples spread over the next ``ahead_m`` metres."""
        stride = max(1, int(ahead_m / self.ds / n))
        return self.curvature[(k + stride * np.arange(1, n + 1)) % self.K]

    # -- factories --------------------------------------------------------
    @staticmethod
    def oval(length: float = 16.0, width: float = 5.0, half_width: float = 0.75,
             ds: float = 0.2) -> "Track":
        """A rounded rectangle: two straights joined by two constant-radius turns.

        The default is a ~27 m lap with 2.5 m turn radii -- sized so that the
        corner radius is *below* the radius a 1:10 car can hold at top speed
        (``v_max^2 / A_LAT_MAX`` = 2.7 m), which is what stops flat out from
        being the right answer everywhere.
        """
        r = width / 2.0
        straight = max(length - width, 1e-3) / 2.0
        n_s, n_c = max(int(straight / ds), 2), max(int(np.pi * r / ds), 4)
        xs, ys = [], []
        for i in range(n_s):  # bottom straight, +x
            xs.append(-straight / 2 + straight * i / n_s); ys.append(-r)
        for i in range(n_c):  # right-hand turn
            a = -np.pi / 2 + np.pi * i / n_c
            xs.append(straight / 2 + r * np.cos(a)); ys.append(r * np.sin(a))
        for i in range(n_s):  # top straight, -x
            xs.append(straight / 2 - straight * i / n_s); ys.append(r)
        for i in range(n_c):  # left-hand turn
            a = np.pi / 2 + np.pi * i / n_c
            xs.append(-straight / 2 + r * np.cos(a)); ys.append(r * np.sin(a))
        return Track(np.array(xs), np.array(ys), half_width)

    @staticmethod
    def curvy(scale: float = 5.0, lobe: float = 0.25, half_width: float = 0.75,
              ds: float = 0.2) -> "Track":
        """A three-lobed rosette, ``r(theta) = scale * (1 + lobe*cos(3 theta))``.

        Harder than the oval for the same reason a real circuit is harder than
        a skid pad: at ``lobe = 0.25`` the curvature changes sign three times a
        lap (about 1.9 m radius one way, 2.2 m the other), so a policy that has
        quietly learned "hold a little left lock" cannot survive, and neither
        can one constant speed. Unlike a figure-of-eight it never crosses
        itself, which keeps "am I on the track" a well-posed question.
        """
        n = max(int(2 * np.pi * scale * 1.1 / ds), 64)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = scale * (1 + lobe * np.cos(3 * th))
        return Track(r * np.cos(th), r * np.sin(th), half_width)


TRACKS = {"oval": Track.oval, "curvy": Track.curvy}
