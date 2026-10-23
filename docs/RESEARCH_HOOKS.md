# Research hooks

What the simulator provides for each research thread, and the API that gives it to you. Everything
below is deterministic given (EnvCfg, Scenario / EpisodeSpec, seed) and runs in plain MuJoCo.

## 1. World-model failure detection (success / known failure / OOD)

| Need | Where |
|---|---|
| Labelled episodes in the lab's real-data layout (`episode.parquet` + one mp4 per camera + `annotations.json` + `feature_report.json`) | `robogym.datagen.run_episode / write_episode`, `scripts/generate_dataset.py` |
| Three categories with separate *injected* and *physical* labels | `EpisodeSpec.category` (`nominal`, `known_failure`, `ood`, `regime_shift`); `annotations.injected_failure` vs `annotations.failure_mode` |
| Failure onset time, drop times, stage timeline | `annotations.injection_time_s`, `drop_times_s`, `stage_times_s`, `phase_events` |
| Per-step dense labels (stage, progress, held object, regime id, task residuals) | parquet columns `priv.*`; `research.monitor_labels(df, ann, lead_s)` adds `time_to_failure`, `pre_failure`, `is_ood` |
| OOD families with severities (lighting / appearance / sensor / physics + task-specific) | `robogym.shifts.SHIFT_GROUPS`, `Task.shift_types`, `EpisodeSpec.shifts`, `shift_severity` |
| Wrist force-torque in the proprioceptive state | state dims `*.f{xyz}`, `*.t{xyz}` (6 of the 28 per arm) |

Calibration splits: `nominal` episodes for the success world model and conformal calibration;
`known_failure` for the failure world model / classifier; `ood` strictly held out.

## 2. Latent steering (best-of-K over policy chunks, RQ1 oracle ranking)

| Need | Where |
|---|---|
| Exact snapshot / restore of physics + controller + latch state + regime | `env.get_state()`, `env.set_state()`; `datagen.restore(env, episode, step)` from a recording (`sim_states.npz`) |
| Execute K candidate chunks from one state and finish each branch with a continuation policy | `research.branch_outcomes(env, state, chunks, continuation)` -> success / progress per branch |
| Snapshot points along a rollout | `research.capture_states(env, actor, at_steps)`, `tools.run_until(env, expert, phase, subgoal)` |
| Reproducible per-(episode, replan, candidate) noise so that candidate 0 with lambda = 0 is the base policy bit for bit | `research.seed_for(*parts)` |
| Chunk semantics compatible with RobotPluggingSim's learners | same 8-D-per-arm Cartesian action, deltas applied to the controller target (`action.ee_target` column gives the absolute targets) |
| Privileged low-dim features for a state policy / value head | `Task.task_features`, parquet column `task.features`, names in `feature_report.task_feature_names` |

The `restore -> step` path is bit-exact (see `tests/test_smoke.py::test_restore_replay_exact`), so
differences between branches are caused by the chunks alone.

## 3. Test-time adaptation / training (TTA / TTT)

| Need | Where |
|---|---|
| Physics regimes that can change *online* | `env.set_regime({...})`; keys in `scenario.REGIME_KEYS` (pad friction, object mass / friction, arm gains, gripper force, actuation delay, proprio noise, bench friction, cable mass / stiffness, gravity) |
| Within-episode schedules and per-step regime labels | `EpisodeSpec.regime_schedule = [[t_s, {...}], ...]`; parquet `priv.regime_id`; `streams.step_regime_labels` |
| Standard continual-TTA protocols over episode streams | `streams.make_stream(name, n, ...)` with `sudden`, `gradual`, `recurring`, `mixed`, `in_episode`; `regime_table` for ground truth |
| Same shift expressed as OOD data *and* as an online regime | physics shifts write `Scenario.regime`; `streams.REGIME_FAMILIES` |
| Snapshots that include the regime | `get_state()["regime"]` |

## 4. Long-horizon tasks

| Need | Where |
|---|---|
| Horizon as a variant knob | `EpisodeSpec.variant = {"n": k}` (cubes / bricks / cables); `annotations.horizon` |
| Subgoal structure | `Task.stages()` (e.g. `reach_1, grasp_1, carry_1, place_1, ...`), `priv.stage`, `priv.subgoal` |
| Dense monotone progress | `priv.progress` in [0, 1] (seated / snapped / latched count + partial credit) |
| Goal observable from pixels | cube order by size, LEGO ghost of the target model (`render(..., ghost=True)`), blue LED on the target port |
| Dependencies between subgoals | LEGO goals are stable, top-down-insertable stackings: later bricks need earlier ones seated |
| Timeout-type failures | `stall` injected failure; `episode_max_s` scales with horizon |

## 5. Scripted experts as data source

`Task.make_expert(env, failure, seed)` returns a generator-based privileged expert (`experts/base.py`)
with closed-loop re-measurement, force-limited pushes, retries, and the injected-failure hooks.
DART-style noisy demos: `EpisodeSpec.dart = sigma` (executed action perturbed in free-space phases,
label = clean expert action in `action.expert`).

## 6. Scaling

`robogym.parallel.run_jobs` is a resumable, time-budgeted process pool (one env per worker, shards on
disk). Physics-only episodes run at ~20-40x real time per core on a laptop; rendering 2 cameras at
320x240 costs ~1.5x more. On a headless GPU node set `MUJOCO_GL=egl` (`scripts/generate_dataset.py`
does this for you under `parallel._init_worker`).
