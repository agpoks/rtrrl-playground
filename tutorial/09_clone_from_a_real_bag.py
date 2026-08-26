"""Lesson 9 -- clone a real driver from a ROS 2 recording, with the same rule.

    python tutorial/09_clone_from_a_real_bag.py --bag /path/to/rosbag2_dir
    python tutorial/09_clone_from_a_real_bag.py --bag ... --cells mlp ctrnn lrcu

No data ships with this repo; point ``--bag`` at any rosbag2 directory (or
``.db3``) that carries ``/scan``, a drive-command topic and odometry -- an
F1TENTH stack records all three by default. Reading needs the pure-Python
``rosbags`` package and **no ROS installation**::

    pip install rosbags
"""

# %% [markdown]
# # Lesson 9 — Clone a real driver
#
# Lesson 7 cloned a scripted controller, because a tutorial has to run on a
# laptop. This one does the same thing from an actual recording of an actual
# car, which is the offline half of the pipeline in Lemmel, Resch, Farsang,
# Hasani, Rus & Grosu ([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)) --
# behavioural cloning from demonstrations, then RTRRL fine-tuning on the
# vehicle. The cloning half runs on whatever you have already recorded; only
# the second half needs the car.
#
# The learning rule does not change. `agent.imitate()` is the same forward-mode
# RFLO gradient as the RL path, with the TD error swapped for
# `d log pi(a_expert)`. One pass, one update per step, constant memory —
# the same promise, on real data.

# %%
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground.data import BagDemonstrations, bag_summary, track_from_bag
from rtrrl_playground.utils.load import load_algo
from rtrrl_playground.spaces import Discrete

RTRRL = load_algo("rtrrl")

# %% [markdown]
# ## A recording is not a dataset
#
# Three decisions turn one into the other, and each is somewhere a quiet
# mistake produces a policy that scores beautifully on the recording and cannot
# drive. They are all in `rtrrl_playground/data/rosbag.py`, and worth reading
# before trusting any number below:
#
# * **Which clock.** `/scan` at ~40 Hz, `/ackermann_cmd` at ~38 Hz, `/odom` at
#   ~50 Hz, none aligned. A sample is emitted on each drive command with the
#   most recent scan *strictly older than it*. A scan from after the command is
#   the future leaking into the input, and it makes a clone look clairvoyant
#   right up until it is deployed.
# * **Which beams.** 1081 rays over 270° become the same nine over 120° the
#   simulator uses, normalised the same way — so a real observation and a
#   simulated one are literally the same vector.
# * **Which action.** A continuous steering angle and speed become one of nine
#   discrete actions, throttle from the *change* in commanded speed.

# %%
def summarise(bag):
    s = bag_summary(bag)
    print(f"  {Path(bag).name}: {s['duration_s']:.0f} s, {len(s['topics'])} topics")
    for t, (ty, n) in list(s["topics"].items())[:8]:
        print(f"    {n:>7}  {t:<32} {ty}")
    return s


def load(bag, **kw):
    demos = BagDemonstrations(bag, **kw)
    obs, acts, infos = demos.to_arrays()
    print(f"\n  {len(obs)} demonstration samples "
          f"({demos.stats['dropped_stationary']} dropped as stationary)")
    print(f"  beams at {demos.stats['beam_angles_deg']} degrees")
    u, c = np.unique(acts, return_counts=True)
    print(f"  action histogram: {dict(zip(u.tolist(), c.tolist()))}")
    print(f"  mean speed {np.mean([i['speed'] for i in infos]):.2f} m/s")
    return obs, acts, infos


# %% [markdown]
# ## Clone, and hold out the end of the recording
#
# The split is **by time**, not at random. Consecutive samples of a car driving
# are almost the same sample; shuffle them and the "held-out" set is the
# training set with extra steps, and the agreement number means nothing.
#
# The baseline to beat is *not* 1/9. It is the **majority action** — a driver
# holds one command for many frames at a time, so predicting the most common
# action is already a strong guess, and any honest claim has to clear it.

# %%
def clone_and_score(obs, acts, infos, cell, epochs=3, lr=1e-2, split=0.8, seed=0):
    n_train = int(len(obs) * split)
    agent = RTRRL(obs.shape[1], Discrete(9), cell=cell, seed=seed)
    agent.start(obs[0])
    for _ in range(epochs):
        agent.cell.reset_state()
        agent.start(obs[0])
        for t in range(n_train):
            nxt = obs[t + 1] if t + 1 < n_train else obs[t]
            if agent.imitate(nxt, int(acts[t]), infos[t]["reward"],
                             infos[t]["terminated"], False, lr=lr) is None:
                agent.start(nxt)

    # Held-out pass: no learning, but the recurrent state still has to be run
    # forward through the test segment -- a recurrent policy evaluated from a
    # zeroed state on frame one is not the policy that was trained.
    agent.cell.reset_state()
    agent.start(obs[n_train])
    correct = 0
    for t in range(n_train, len(obs) - 1):
        correct += int(agent.a == int(acts[t]))
        agent.h = agent.cell.step(agent._input(obs[t + 1], int(acts[t]), infos[t]["reward"]))
        agent.a, agent.cache = agent.actor.act(agent.h, agent.rng)
    n_test = len(obs) - 1 - n_train
    return correct / max(n_test, 1), n_test


# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="a rosbag2 directory or .db3 file")
    ap.add_argument("--cells", nargs="+", default=["mlp", "ctrnn", "lrcu"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--cmd-topic", default="/ackermann_cmd")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true", help="skip rebuilding the track")
    args = ap.parse_args(argv)

    summarise(args.bag)
    obs, acts, infos = load(args.bag, scan_topic=args.scan_topic,
                            cmd_topic=args.cmd_topic, odom_topic=args.odom_topic)

    n_train = int(len(obs) * 0.8)
    test_acts = acts[n_train:len(obs) - 1]
    majority = float(np.mean(test_acts == np.bincount(acts[:n_train].astype(int)).argmax()))
    print(f"\n  held-out agreement, {len(test_acts)} frames "
          f"(majority-action baseline {majority:.1%}):")
    for cell in args.cells:
        agree, n_test = clone_and_score(obs, acts, infos, cell,
                                        epochs=args.epochs, seed=args.seed)
        flag = "  <- beats the baseline" if agree > majority else ""
        print(f"    {cell:<6} {agree:6.1%}{flag}")

    if not args.no_track:
        rebuild_track(args.bag)
    return obs, acts


# %% [markdown]
# ## The track the recording was made on
#
# The bag also carries the map the car localised against and the pose it
# reported, so the circuit can be rebuilt and driven in the simulator: the
# **human's line becomes the centreline** and the **recorded occupancy grid
# becomes the wall bitmap**, so the simulated beams hit the walls the real
# lidar saw.
#
# Look at the drivability report before believing the track. A line
# reconstructed from a slow, noisy recording can contain corners tighter than
# the car's turning circle — the human at walking pace could shuffle round
# them, and a 0.33 m wheelbase at 0.4 rad of lock cannot. That is a property of
# the recording, not a bug in the reconstruction, and the numbers below say
# which you are looking at.

# %%
def rebuild_track(bag, out=None):
    try:
        track, info = track_from_bag(bag)
    except Exception as exc:
        print(f"\n  could not rebuild the track: {type(exc).__name__}: {exc}")
        return None
    print(f"\n  track from {info['pose_topic']} (frame check: "
          + ", ".join(f"{k} {v:.0%}" for k, v in info["pose_topic_scores"].items()) + ")")
    print(f"    {info['length_m']:.1f} m lap, half-width {info['half_width']:.2f} m, "
          f"map {info['map_shape'][1]}x{info['map_shape'][0]} @ {info['map_res']:.2f} m/px")
    print(f"    corner radius: min {info['radius_min']:.2f} m, "
          f"5th pct {info['radius_p5']:.2f} m, median {info['radius_median']:.2f} m")
    print(f"    {info['frac_below_turning_circle']:.1%} of the lap is tighter than the "
          f"car's {info['car_min_radius']:.2f} m turning circle"
          + ("  <- this track is not fully drivable by the kinematic model"
             if info["frac_below_turning_circle"] > 0.01 else ""))

    from rtrrl_playground.envs.lanekeep import LaneKeep
    env = LaneKeep(track=track, max_steps=800)
    path = out or (ROOT / "runs" / "lesson09_bag_track.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    env.reset(seed=0)
    env.render_rollout([], str(path), title=f"track rebuilt from {Path(bag).name}")
    print(f"    wrote {path}")
    return track


if __name__ == "__main__":
    main()

# %% [markdown]
# ## What to take from this
#
# * **The cloning half of a real deployment needs nothing but a recording.** No
#   simulator, no reward function, no car — and with `imitate()` it uses the
#   same constant-memory forward-mode gradient as the RL half, so nothing about
#   the agent changes between the two phases.
# * **Check the agreement against the majority action, not against chance.** A
#   human holds a command for many frames; a clone that has learned only "keep
#   doing what you were doing" can look like 60% accuracy.
# * **Whether memory helps here is an empirical question with a real answer.**
#   The `mlp` row is the same clone without a recurrent state. If it matches the
#   recurrent rows, the human's next action was predictable from the current
#   scan alone, and the recording is not evidence for or against memory.
# * **A track rebuilt from a slow recording is not a racing line.** Use the
#   drivability report; a bag recorded at walking pace during a localisation
#   test produces a noisier centreline than one recorded on a flying lap.
