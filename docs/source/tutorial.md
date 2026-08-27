# The tutorial

Eight lessons, in order. Each is a plain Python file you can run and a notebook
generated from it (`python scripts/make_notebooks.py`). Nothing is imported
from one lesson into another.

| # | lesson | what it establishes |
|---|---|---|
| 1 | `01_the_problem.py` | Partial observability, measured: a memoryless agent on tasks where the state it needs is not in the observation. |
| 2 | `02_gradients_online.py` | Five ways to get a recurrent gradient, graded against the exact answer in alignment, memory and microseconds. |
| 3 | `03_traces.py` | Eligibility traces: how one scalar credits an action ten steps back, at one number per parameter. |
| 4 | `04_rtrrl_from_scratch.py` | **The whole algorithm in one file**, ~100 lines of NumPy, importing nothing from this repo but an environment. |
| 5 | `05_learn_to_drive.py` | RTRRL on a 1:10 car with no speedometer and a grip level that changes every episode. |
| 6 | `06_learn_to_overtake.py` | The same car plus traffic whose speed is never observed. |
| 7 | `07_finetune_a_controller.py` | Clone offline, improve online while driving -- Lemmel et al. (2026) in miniature. |
| 8 | `08_to_scuderia_gym_jax.py` | Swap the toy vehicle for real ST/STD models. |
| 9 | `09_clone_from_a_real_bag.py` | Clone a real driver from a ROS 2 recording. |
| 10 | `10_safety_filter.py` | A predictive safety filter: never crash while learning. |
| 11 | `11_sim_to_real.py` | Deploy on a vehicle the simulator was wrong about, and adapt online. |
| 12 | `12_cells_from_scratch.py` | LiGRU, LRCU and LiquidGRU from scratch: equations, papers, derivatives, checks. |

## The shape of the argument

1. You cannot see the whole state (1).
2. So the policy needs memory, which means training a recurrent network -- and
   the standard way to do that wants the whole episode in memory and cannot
   produce an update until the episode is over (2).
3. Gradients can travel forwards instead, and reward can travel backwards
   through a trace (2, 3).
4. Put those together and you have RTRRL (4).
5. It drives (5), it overtakes (6), and the way you would actually deploy it is
   to fine-tune something that already works (7).
6. Then move it onto real vehicle dynamics (8).

## If you only run one

Lesson 4. Every other file in the repo is a generalisation of it.

## Two things the lessons deliberately show going wrong

**Lesson 4's "things to try" are failure modes, not decorations.** Raising the
actor learning rate to the paper's value collapses the policy entropy inside a
few thousand steps, for a reason worth understanding: the critic starts at
zero, so the early TD error is persistently positive, and a positive TD error
reinforces whatever the policy already prefers regardless of whether it was any
good. Two-timescale -- critic fast, actor slow -- is the difference between
learning and not.

**Lesson 2's UORO result looks like a bug and is not.** One UORO sample scores
0.1 against the exact gradient, far worse than RFLO's 0.7. Averaging a few
hundred samples of the *same* gradient takes it to nearly 1, which RFLO never
does at any number of samples. Unbiased-but-noisy versus biased-but-quiet is a
real choice and the lesson measures both sides of it rather than picking one.
