# Real recordings

Everything else in this repo is simulated. `rtrrl_playground/data/rosbag.py` is
the seam where it stops being: it turns a ROS 2 recording of an actual car into
the two things the offline half of the pipeline needs — a stream of
`(observation, expert action)` pairs, and the track the recording was made on.

```bash
pip install -e ".[bags]"     # rosbags: pure Python, no ROS installation
python tutorial/09_clone_from_a_real_bag.py --bag /path/to/rosbag2_dir
```

The bag needs `/scan`, a drive-command topic and odometry; an F1TENTH stack
records all three by default. `bag_summary(path)` prints what is actually in
one, and is the right first move on any unfamiliar recording.

## A recording is not a dataset

Three decisions turn one into the other. Each is a place where a quiet mistake
produces a policy that scores beautifully on the recording and cannot drive.

**Which clock.** `/scan` arrives at ~40 Hz, `/ackermann_cmd` at ~38 Hz, `/odom`
at ~50 Hz, none aligned. A sample is emitted on each drive command carrying the
most recent scan *strictly older than it*. A scan from after the command is the
future leaking into the input, and it makes a clone look clairvoyant right up
until it is deployed.

**Which beams.** A Hokuyo gives 1081 rays over 270°. The agent takes nine over
120°, so the raw scan is sampled at the nine angles nearest the simulator's own
and normalised by the same range. The point is that a real observation and a
simulated one are *literally the same vector* — so a policy cloned from a bag
can be fine-tuned in the simulator, and vice versa.

**Which action.** The recorded command is continuous (steering angle in
radians, speed in m/s); the discrete agent needs one of nine. Steering maps
through a deadband; throttle maps to the sign of the *change* in commanded
speed, because the environment's action is an acceleration.
`action_mode="continuous"` skips all of it.

Stationary samples are dropped by default. A recording of somebody parked
teaches a policy to park, and in a localisation-test bag that can be a third of
the file.

## Rebuilding the circuit

`track_from_bag()` uses the recording twice over: the **driven line becomes the
centreline**, resampled and smoothed, and the recorded **occupancy grid becomes
the wall bitmap**. The progress reward then pays for following the line that
was actually driven, which is a different objective from following the middle
of the corridor — usually a better one, and worth knowing you have chosen.

Two failure modes are handled explicitly rather than silently:

**The pose topic must be in the map frame.** A ROS stack publishes `/odom` in
the odom frame and corrects it to the map frame through a `map -> odom`
transform on `/tf`. Take `/odom` at face value and the trajectory drifts off
the map — in the recording this was written against, `/odom` reached x = 12.1 m
on a map that ends at 7.7 m, so the "track" came out half outside the building.
`pick_pose_topic()` scores each candidate by how much of it lands on a cell the
map calls free, and picks the winner; `/ekf/state` scored 97%, `/odom` 87%.

**Reversals make cusps.** Filtering on `|v| > min_speed` keeps a car backing
out of a corner, and a path that doubles back has a cusp, which after
resampling reads as a corner of near-zero radius. The filter is on `v >
min_speed`, plus a check for steps that turn back on the previous one.

## Check the drivability report

`track_from_bag` returns an `info` dict with the reconstructed corner radii
against the car's own turning circle. On a bag recorded at walking pace during
a localisation test, the result was:

```
49.9 m lap, half-width 0.50 m, map 325x231 @ 0.05 m/px
corner radius: min 0.08 m, 5th pct 0.74 m, median 4.76 m
5.6% of the lap is tighter than the car's 0.78 m turning circle
```

That last line is not a bug in the reconstruction. A human at 0.18 m/s average
can shuffle round a corner that a 0.33 m wheelbase at 0.4 rad of lock cannot
take at speed, and localisation noise at walking pace inflates curvature
further. A bag recorded on a flying lap gives a much cleaner line. **Look at the
rendered picture before trusting the track** — the plot draws the reconstructed
tube over the real map, and a bad lap is obvious in it.

## Measured

254 s F1TENTH recording, 6375 usable demonstrations, held-out final 20% by
time (not shuffled — consecutive frames of a driving car are almost the same
frame, and a random split makes the test set into the training set):

| clone | held-out agreement with the human's next command |
|---|---|
| RTRRL / `ctrnn` | **61.5%** |
| memoryless (`mlp`) | 57.1% |
| RTRRL / `lrcu` | 55.4% |
| majority action | 31.6% |

The baseline to beat is the majority action, not 1/9. A driver holds one
command for many frames, so a clone that has learned only "keep doing what you
were doing" already looks good. The recurrent clone beats the memoryless one by
4 points — the honest size of what memory buys at predicting a human here, on
this recording.

`imitate()` uses the same forward-mode RFLO gradient as the RL path, with the
TD error swapped for `d log pi(a_expert)`. One pass, one update per step,
constant memory — nothing about the agent changes between cloning and
fine-tuning, which is the whole point of doing the offline half this way.
