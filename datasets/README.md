# Data

**Nothing here downloads anything, and nothing ships with the repo.** That is
not an omission — it is what this playground is. Three of its four environments
generate their own data by being simulated, and the fourth kind of data (real
recordings) is yours and cannot be redistributed.

This file exists so that "where does the data come from" has an answer in one
place, the same way the other playgrounds in this family have a `datasets/`.

## 1. The environments generate their own

| environment | what produces the data | size on disk |
|---|---|---|
| `memory-chain` | a bit, a clock and a query flag | 0 |
| `cartpole-vel` | textbook CartPole dynamics, velocities masked | 0 |
| `lanekeep` | a kinematic bicycle on a rasterised oval | 0 |
| `overtake` | the same, plus two constant-speed cars | 0 |

All four are a few hundred lines of NumPy in
[`rtrrl_playground/envs/`](../rtrrl_playground/envs). There is no dataset to
version, no download to fail, and no licence to check. Every number in
[`docs/source/benchmark_results.md`](../docs/source/benchmark_results.md) is
reproducible from a seed.

This is a real difference from the supervised playgrounds in this family
(`liquid-nn-playground` and the rest download MNIST, ETTh1, Speech Commands).
An RL agent's data is a consequence of its own behaviour, so it cannot be
shipped: two agents on the same environment never see the same data.

## 2. Real recordings — you supply them

[`rtrrl_playground/data/rosbag.py`](../rtrrl_playground/data/rosbag.py) reads a
ROS 2 bag of an actual car into demonstrations, and rebuilds the circuit it was
recorded on. See [`docs/source/real_data.md`](../docs/source/real_data.md).

```bash
pip install -e ".[bags]"        # rosbags; pure Python, no ROS installation
python -c "from rtrrl_playground.data import bag_summary; print(bag_summary('/path/to/bag'))"
python tutorial/09_clone_from_a_real_bag.py --bag /path/to/bag
```

**What a usable bag needs**, and the reason each one is needed:

| topic | type | why |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | becomes the nine-beam observation |
| `/ackermann_cmd` (or `/teleop`) | `ackermann_msgs/AckermannDriveStamped` | the expert action to clone |
| `/odom` | `nav_msgs/Odometry` | speed, and the progress reward |
| `/map` | `nav_msgs/OccupancyGrid` | *optional*: rebuilds the track |
| a **map-frame** pose | `nav_msgs/Odometry` | *optional*: the driven centreline |

An F1TENTH stack records all of these by default. The last row is the one that
catches people: `/odom` is in the odom frame, not the map frame, and
`pick_pose_topic()` exists because taking it at face value builds a track half
outside the building.

`bag_summary(path)` prints what is actually in a recording, and is the right
first move on an unfamiliar one.

## 3. The maps in `scuderia_gym_jax`

[`tutorial/08`](../tutorial/08_to_scuderia_gym_jax.py) drives on the occupancy
maps that ship with
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax) — `berlin`,
`skirk`, `vegas`, `stata_basement` (`.png`) and `levine` (`.pgm`). They come
with that package; nothing is copied here.

```bash
pip install jax chex
PYTHONPATH=/path/to/scuderia_gym_jax python tutorial/08_to_scuderia_gym_jax.py
```

## 4. What is written *out*

Everything a run produces — learning curves, rendered episodes, sweep JSON —
goes to `runs/`, which is gitignored. Benchmark tables go to
`benchmarks/results/`. Neither is data you need to fetch; both are data you can
delete.
