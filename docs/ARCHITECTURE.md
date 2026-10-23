# Architecture

```
robogym/
  config.py        every assumption as dataclass fields; EnvCfg(task=..., robot, camera, bench, sim)
  scene.py         bench / background / cameras / distractors + build_mjcf(cfg, task fragments)
  robot/xarm7.py   per-arm MJCF from the Menagerie numbers (+ gripper variants, TCP / tip / F-T sites,
                   wrist camera, finger-arc kinematics used to plan grasp heights)
  env.py           RoboGymEnv: IK controller, compliant mode, gripper, observe / privileged / info,
                   render, exact get_state / set_state, online regimes, action delay, bumps
  scenario.py      Scenario dataclass (generic randomisation + regime schedule + task dict), sample_scenario
  shifts.py        OOD shift families (lighting, appearance, sensor, physics -> regime) with severities
  failures.py      FailureSpec + generic injected failures (task kinds live in the tasks)
  tasks/base.py    Task interface + registry
  tasks/cube_stack.py, lego.py, cable_plug.py
  experts/base.py  generator-based privileged expert with Cartesian primitives and failure hooks
  experts/cube_stack.py, lego.py, cable_plug.py
  datagen.py       EpisodeSpec, make_plan, run_episode (labels, DART), write / load / restore
  streams.py       TTA protocols (sudden / gradual / recurring / mixed / in_episode) + regime labels
  research.py      branching, snapshots, closed-loop policy runner, monitor labels, seeding
  parallel.py      resumable process pool (one env per worker, shards on disk)
  gym_wrapper.py   Gymnasium Env (dict obs, Box action, progress reward)
  tools.py         image grids, inline video, mosaic, run_until / finish / expert_at
  textures.py      procedural PNG textures (wood, mat, floor, labels, switch front)
scripts/           generate_dataset.py, benchmark_experts.py, render_gallery.py
notebooks/         percent-format sources in notebooks/src, built with notebooks/build.py (Colab bootstrap cell)
tests/             smoke (per task), shifts / failures / regimes / streams, research hooks, gym wrapper
```

## Control loop

One control step (20 Hz) = 25 physics sub-steps (2 ms). Each sub-step: `mj_step1` (fresh kinematics),
external wrenches and task detent forces (`Task.pre_step`), the IK controller (damped least squares on
the TCP, or on the remote-compliance point in compliant mode, with per-axis gains in the TCP frame),
`mj_step2`, then `Task.post_step` (latch / snap checks). Because the controller only reads the current
state and the fixed per-step target, `set_state` + `step(a)` reproduces a recording bit for bit.

## Adding a task

Subclass `tasks.base.Task`, decorate with `@register`, implement `xml`, `bind`, `sample`, `apply`,
`reset_state`, `privileged`, `stages / stage / progress / success`, `task_features / feature_names`,
`outcome`, `make_expert` and (optionally) `sample_failure`, `apply_regime`, `pre_step / post_step`,
`get_aux / set_aux` for latch bookkeeping, `record_columns`, `footprint_xy`. Add a `TaskCfg` dataclass
in `config.py` and register it in `TASK_CFGS`. The expert subclasses `experts.base.Expert` and
implements `plan()` as a generator over the primitives (`move`, `grasp`, `carry`, `set_grip`, `hold`).
