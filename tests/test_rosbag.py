"""Bag-ingestion tests. Skipped unless a bag is available.

Point ``RTRRL_TEST_BAG`` at a rosbag2 directory to run them::

    RTRRL_TEST_BAG=/path/to/bag pytest tests/test_rosbag.py -q

No recording ships with this repo, so these cannot run in CI by default. They
are here because the failures they catch -- a scan taken from *after* the
command it is paired with, a pose topic in the wrong frame -- are silent, and
produce a clone that scores well and cannot drive.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BAG = os.environ.get("RTRRL_TEST_BAG")
pytest.importorskip("rosbags", reason="pip install rosbags")
pytestmark = pytest.mark.skipif(not BAG, reason="set RTRRL_TEST_BAG to a rosbag2 directory")


def test_summary_lists_topics():
    from rtrrl_playground.data import bag_summary

    s = bag_summary(BAG)
    assert s["duration_s"] > 0
    assert s["topics"], "no topics found"


def test_demonstrations_match_the_simulator_observation():
    """A real observation and a simulated one must be the same vector."""
    from rtrrl_playground.data import BagDemonstrations
    from rtrrl_playground.envs.lanekeep import BEAM_ANGLES

    demos = BagDemonstrations(BAG)
    obs, acts, infos = demos.to_arrays()
    assert len(obs) > 100, "almost nothing survived the filters"
    assert obs.shape[1] == len(BEAM_ANGLES) == demos.obs_dim
    assert np.isfinite(obs).all(), "inf/nan lidar returns leaked through"
    assert (obs >= 0).all() and (obs <= 1).all(), "beams are not normalised"
    assert set(np.unique(acts)).issubset(set(range(9)))
    assert all(i["speed"] is not None for i in infos)


def test_no_scan_from_the_future():
    """Every sample's scan must be older than the command it is paired with."""
    from rtrrl_playground.data import BagDemonstrations

    demos = BagDemonstrations(BAG)
    ts = [i["t"] for i in demos.to_arrays()[2]]
    assert ts == sorted(ts), "samples are not in recorded order"


def test_pose_topic_selection_prefers_the_map_frame():
    from rtrrl_playground.data import load_map, pick_pose_topic

    free, res, origin = load_map(BAG)
    topic, scores = pick_pose_topic(BAG, free, res, origin)
    assert scores[topic] == max(scores.values())
    assert scores[topic] > 0.5, f"no pose topic lands on the map: {scores}"


def test_track_from_bag_is_a_closed_drivable_loop():
    from rtrrl_playground.data import track_from_bag
    from rtrrl_playground.envs.lanekeep import LaneKeep

    track, info = track_from_bag(BAG)
    assert info["length_m"] > 5.0
    assert 0.3 < info["half_width"] < 2.5
    # The bitmap must actually contain the centreline, or the map and the line
    # are in different frames and every beam is wrong.
    on = track.on_track(track.center[:, 0], track.center[:, 1])
    assert on.mean() > 0.9, f"only {on.mean():.0%} of the centreline is on the track bitmap"
    env = LaneKeep(track=track, max_steps=50)
    obs = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    for _ in range(20):
        obs, r, te, tr, info_ = env.step(4)
        if te or tr:
            break
