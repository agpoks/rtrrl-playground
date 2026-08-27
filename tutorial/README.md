# The tutorial

Eight lessons, in order. Each one is a plain Python file you can run, and a
notebook generated from it (`python scripts/make_notebooks.py`). Nothing is
imported from a lesson into another lesson — read them in order, or drop into
the one you want.

| # | file | what it establishes | runtime |
|---|---|---|---|
| 1 | [`01_the_problem.py`](01_the_problem.py) | Partial observability, measured: a memoryless agent on a task where the state it needs is not in the observation. | ~3 min |
| 2 | [`02_gradients_online.py`](02_gradients_online.py) | Five ways to get a recurrent gradient — RTRL, UORO, SnAp-1, RFLO, truncated BPTT — graded against the exact answer, in alignment, memory and microseconds. | ~4 min |
| 3 | [`03_traces.py`](03_traces.py) | Eligibility traces: how one scalar credits an action ten steps back, accumulating vs Dutch, and what λ actually buys. | seconds |
| 4 | [`04_rtrrl_from_scratch.py`](04_rtrrl_from_scratch.py) | **The whole algorithm in one file**, about a hundred lines of NumPy, importing nothing from this repo but an environment. If you read one file, read this one. | ~2 min |
| 5 | [`05_learn_to_drive.py`](05_learn_to_drive.py) | RTRRL on a 1:10 car with no speedometer, all four cells side by side against a scripted wall-follower. | ~10 min |
| 6 | [`06_learn_to_overtake.py`](06_learn_to_overtake.py) | The same car plus traffic whose speed is never observed — where the memory stops being a teaching device. | ~20 min |
| 7 | [`07_finetune_a_controller.py`](07_finetune_a_controller.py) | Clone a controller offline, then improve it online while driving — the deployment story from Lemmel et al. (2026), in miniature. | ~15 min |
| 8 | [`08_to_scuderia_gym_jax.py`](08_to_scuderia_gym_jax.py) | Swap the toy vehicle for `scuderia_gym_jax`'s real ST/STD models, and what changes when you do. | varies |
| 9 | [`09_clone_from_a_real_bag.py`](09_clone_from_a_real_bag.py) | Clone a real driver from a ROS 2 recording, and rebuild the circuit it was recorded on. Needs `--bag`; no data ships here. | ~5 min |
| 10 | [`10_safety_filter.py`](10_safety_filter.py) | A predictive safety filter from scratch: never leave the track while learning, and what that costs. | ~10 min |
| 11 | [`11_sim_to_real.py`](11_sim_to_real.py) | Train in simulation, deploy on a vehicle the simulator was wrong about, close the gap online. | ~15 min |

Every lesson takes `--help`. Runtimes are for a laptop CPU at the default
`--steps`; all of them take a smaller number.

## The shape of the argument

1. You cannot see the whole state (1).
2. So the policy needs memory, which means training a recurrent network —
   and the standard way to do that does not fit on a vehicle (2).
3. Gradients can travel forwards instead, and reward can travel backwards
   through a trace (2, 3).
4. Put those together and you have RTRRL (4).
5. It drives (5), it overtakes (6), and the way you would actually deploy it
   is to fine-tune something that already works (7).
6. Then move it onto real vehicle dynamics (8), and onto real recorded data (9).
7. And if it is going to learn on a real vehicle, it must not crash it while
   learning (10).
8. Which is the whole point: a policy trained in a simulator meets a vehicle
   the simulator was wrong about, and fixes itself while driving (11).
