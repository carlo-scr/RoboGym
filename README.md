# RoboGym

A MuJoCo benchmark of **xArm 7 manipulation tasks with labelled failures, out-of-distribution shifts,
test-time regimes and exact counterfactual branching**, built as the lab's common data source for
world-model failure detection, latent steering, test-time adaptation and long-horizon manipulation
research. It generalises [RobotPluggingSim](https://github.com/carlo-scr/RobotPluggingSim) (WR1 cable
plugging) to the lab's two UFACTORY xArm 7 and to three task families.

| cube stacking | oversized LEGO | Ethernet plugging |
|---|---|---|
| ![](media/hero_cube_stack.png) | ![](media/hero_lego.png) | ![](media/hero_cable_plug.png) |

* **Robot**: one or two UFACTORY xArm 7 with the xArm gripper, from MuJoCo Menagerie (measured
  kinematics / inertia / servo gains / gripper linkage), on a lab bench with a wrist camera per arm and
  three fixed cameras. Cartesian 20 Hz control with a compliant insertion mode and a wrist force-torque
  sensor in the proprioceptive state.
* **Tasks** (each with a scripted, closed-loop expert): `cube_stack` (2-4 cubes by size or pattern),
  `lego` (2-4 oversized bricks pressed onto a baseplate following a random stable model), `cable_plug`
  (1-2 RJ45 patch cables into LED-marked ports of a 24-port panel). The horizon is a variant knob.
* **Failures**: 6 generic injected kinds (drop, weak grip, misgrasp, premature release, bump, stall) +
  3 per task (misplace / wrong cube / low carry; misalign / partial press / wrong slot; misalign /
  partial insert / wrong port), labelled separately from the *physical* outcome and with onset times.
* **OOD shifts**: 20 generic (lighting, appearance, sensor, physics) + task-specific ones (cube size,
  tight fit, stiff snap, plate shift, dense ports, LED distractors, panel offset, cable colours...),
  each with a severity.
* **Test-time regimes**: physics knobs (friction, mass, arm gains, gripper force, actuation delay,
  proprio noise, ...) that can be switched on mid-episode and scheduled; TTA stream protocols
  (`sudden`, `gradual`, `recurring`, `mixed`, `in_episode`) with ground-truth regime labels per step.
* **Exact snapshots**: `get_state` / `set_state` capture physics + controller + latch state + regime;
  restore-then-step is bit-exact, so K candidate action chunks can be executed from the same state.
* **Datasets** in the layout of the lab's real-robot data (`episode.parquet`, one mp4 per camera,
  `annotations.json`, `feature_report.json`) plus `sim_states.npz` for counterfactual restarts; a
  resumable multiprocess generator; a Gymnasium wrapper.

Everything not measured on the real cell is a strong placeholder assumption in
[robogym/config.py](robogym/config.py), listed in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Quick start

```bash
git clone https://github.com/carlo-scr/RoboGym.git && cd RoboGym
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,notebooks]"
.venv/bin/pytest -q tests                                    # ~6 min: all tasks, restore exactness, export
.venv/bin/python notebooks/build.py --execute                # tour / failures & shifts / data & hooks
.venv/bin/python scripts/generate_dataset.py --task lego --out data/lego_v0 --n 50 --workers 8
.venv/bin/python scripts/benchmark_experts.py --tasks cube_stack lego cable_plug --n 30
```

```python
from robogym.env import RoboGymEnv
env = RoboGymEnv("lego")                       # or EnvCfg / make_cfg("lego", **{"robot.n_arms": 2, "task.bricks": ["2x2", "2x2"]})
obs = env.reset(seed=0, shifts=["slippery"], severity=0.7)
ex = env.task.make_expert(env, seed=0)         # privileged "teleoperator"
while not ex.done:
    obs = env.step(ex.act())                   # 8-D per arm; obs 28-D per arm
    info = env.info()                          # stage, progress, success, regime
img = env.render("right_wrist")
snap = env.get_state(); env.set_state(snap)    # exact restore
env.set_regime({"pad_friction": 0.3})          # online test-time shift
```

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The research-facing API (branching, monitor labels,
streams, feature columns) is mapped to each research thread in
[docs/RESEARCH_HOOKS.md](docs/RESEARCH_HOOKS.md).

| | |
|---|---|
| [robogym/env.py](robogym/env.py) | `RoboGymEnv`: IK controller, compliant mode, observe / privileged / info, render, snapshots, regimes |
| [robogym/tasks/](robogym/tasks/) | task interface + the three tasks (model fragments, scenario sampling, labels, outcome classification) |
| [robogym/experts/](robogym/experts/) | generator-based experts with Cartesian primitives and failure hooks |
| [robogym/shifts.py](robogym/shifts.py), [failures.py](robogym/failures.py), [streams.py](robogym/streams.py) | OOD shifts, injected failures, TTA protocols |
| [robogym/datagen.py](robogym/datagen.py), [parallel.py](robogym/parallel.py), [research.py](robogym/research.py) | episodes, export, resumable pool, branching / monitor labels |
| [scripts/](scripts/) | dataset generation, expert benchmark, gallery / videos |
| [notebooks/](notebooks/) | `00_tour`, `01_failures_shifts_regimes`, `02_data_and_research_hooks` (sources in `notebooks/src`, Colab-ready setup cell) |

## Status (v0.1)

Scripted expert success (`scripts/benchmark_experts.py --n 40`, laptop, physics only; shifts at severity 0.7):

| task | horizon | nominal | slippery | heavy object | weak gripper | actuation delay | arm gain drift |
|---|---|---|---|---|---|---|---|
| cube_stack | 1 / 2 / 3 cubes to place | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.91 |
| lego | 2 / 3 / 4 bricks | 1.00 / 1.00 / 0.56 | 1.00 / 0.86 / 0.36 | 1.00 / 0.95 / 0.55 | 1.00 / 0.91 / 0.55 | 1.00 / 0.91 / 0.55 | 0.71 / 0.45 / 0.09 |
| cable_plug | 1 / 2 cables | 0.82 / 0.61 | 0.87 / 0.52 | 0.93 / 0.52 | 0.87 / 0.52 | 0.73 / 0.32 | 0.67 / 0.44 |

Reading the table: cube stacking is solved by the expert under every shift; the LEGO expert is reliable
up to 3 bricks and the 4-brick builds (which include the 2x4) still jam or drop about half the time; the
cable expert jams at the jack lead-in in ~15 % of single-cable episodes (perception noise 0.3 mm against
a 0.8 mm half-clearance) and errors compound over two cables. Arm-gain drift is the most hurtful regime
(force-limited pushes become too weak for the detents), actuation delay the second: both are meant to be
hurtful test-time shifts. Failure modes are labelled (`jammed`, `not_seated`, `dropped`, `incomplete`,
`misaligned`, ...), so these episodes are usable data, not waste. Throughput on an M-series laptop core:
~1-2 s per physics-only episode, ~3x more with two 320x240 cameras.

## Media

`scripts/render_gallery.py --out media --video` renders the hero shots above, all cameras per task,
an OOD gallery per task and one expert episode per task as a 2x2 mosaic video (media/ is gitignored).

## Roadmap

1. Real cell numbers: arm mounting, gripper force setting, camera poses / intrinsics, bench, object sizes.
2. Bimanual experts (hand-over, holding the panel / baseplate with the second arm).
3. Disassembly tasks (unplug, pull bricks apart): replace the welds with breakable constraints.
4. Image-based policies + the RobotPluggingSim learners (Diffusion Policy, latent WM monitor, steering)
   on these datasets; LeRobot-format export.
5. MJX / MuJoCo Warp batch stepping for GPU counterfactual scoring.

## License

MIT. The xArm 7 meshes are BSD-3-Clause (UFACTORY Inc., via MuJoCo Menagerie), see
[robogym/assets/xarm7/LICENSE](robogym/assets/xarm7/LICENSE).
