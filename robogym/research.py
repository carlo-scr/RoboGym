"""Research hooks: the pieces the lab's experiment code needs from the simulator.

* Counterfactual branching: `capture_states` saves exact snapshots along a rollout; `branch_outcomes`
  executes K candidate action chunks from one snapshot and finishes each branch with a continuation
  policy, returning per-branch outcome / progress (the RQ1 "oracle" test for latent steering).
* Closed-loop policy runner: `rollout_policy` runs any `policy(obs_dict) -> action` with recording.
* Monitor labels: `monitor_labels` derives per-step supervision for failure monitors from a recorded
  episode (time-to-failure, pre-failure window, regime id, stage / progress).
* Seeding: `seed_for(*parts)` gives reproducible per-(episode, replan, candidate) seeds so that
  "candidate 0 with lambda = 0" is bit-identical to the base policy.
"""
from __future__ import annotations

import numpy as np

from .env import RoboGymEnv


def seed_for(*parts) -> int:
    return int(np.random.SeedSequence([int(p) & 0xFFFFFFFF for p in parts]).generate_state(1)[0])


class ObsDict:
    """Everything a policy may want at a step, computed lazily."""

    def __init__(self, env: RoboGymEnv, cams=()):
        self.env = env
        self.cams = list(cams)
        self.state = env.observe()
        self.pr = env.privileged()
        self.features = env.task.task_features(env, self.pr)
        self.info = env.info(self.pr)
        self._images = None

    @property
    def images(self):
        if self._images is None:
            self._images = {c: self.env.render(c) for c in self.cams}
        return self._images


def rollout_policy(env: RoboGymEnv, policy, scenario=None, seed=None, max_s=None, cams=(), record_cams=(), snapshot_every=0, on_step=None):
    """Closed-loop rollout of `policy(obs: ObsDict) -> action (n_arms * 8)`. Returns a dict with the
    per-step record, success / outcome and optional snapshots (every `snapshot_every` steps)."""
    if scenario is not None or seed is not None:
        env.reset(scenario, seed=seed)
    max_s = max_s or env.episode_max_s
    rec = dict(t=[], state=[], action=[], features=[], stage=[], progress=[], frames={c: [] for c in record_cams}, snapshots=[])
    while env.time < max_s and not env.unstable:
        obs = ObsDict(env, cams)
        if snapshot_every and env.step_count % snapshot_every == 0:
            rec["snapshots"].append((env.step_count, env.get_state()))
        for c in record_cams:
            rec["frames"][c].append(env.render(c))
        a = env.clip_action(policy(obs))
        rec["t"].append(env.time)
        rec["state"].append(obs.state)
        rec["action"].append(a)
        rec["features"].append(obs.features)
        rec["stage"].append(obs.info["stage"])
        rec["progress"].append(obs.info["progress"])
        if on_step:
            on_step(env, obs, a)
        env.step(a)
        if obs.info["success"]:
            break
    info = env.info()
    rec.update(success=info["success"], progress_final=info["progress"], unstable=env.unstable, duration_s=env.time)
    return rec


def capture_states(env: RoboGymEnv, actor, at_steps, max_s=None):
    """Run `actor()` (-> action) and snapshot the env at the given step counts. Returns {step: state}."""
    max_s = max_s or env.episode_max_s
    want = set(at_steps)
    out = {}
    while env.time < max_s and not env.unstable and want:
        if env.step_count in want:
            out[env.step_count] = env.get_state()
            want.discard(env.step_count)
        env.step(actor())
    return out


def branch_outcomes(env: RoboGymEnv, state, chunks, continuation, max_s=None, settle_s=0.0):
    """Execute each candidate chunk [K, H, A] from the same snapshot, then let `continuation(env) ->
    action` finish the episode. Returns per-branch dicts (success, progress, outcome stage, duration).
    Restoring is exact, so differences between branches are caused by the chunks alone."""
    results = []
    for chunk in chunks:
        env.set_state(state)
        for a in chunk:
            env.step(a)
            if env.unstable:
                break
        max_t = max_s or env.episode_max_s
        while not env.unstable and env.time < max_t:
            info = env.info()
            if info["success"]:
                break
            env.step(continuation(env))
        info = env.info()
        results.append(dict(success=bool(info["success"]), progress=float(info["progress"]), stage=info["stage_name"], duration_s=env.time, unstable=env.unstable))
    return results


def monitor_labels(df, annotations, lead_s=1.0, dt=None):
    """Per-step labels for failure monitors from a recorded episode:
      stage, progress, regime_id, time_to_failure (s, inf for successes), pre_failure (within lead_s of the
      failure onset), onset_s (the injection / drop time, or the time the physical failure manifests).
    The physical onset is the first drop time if any, else the injection time, else the end of the episode
    for failed episodes without a known onset."""
    from .streams import step_regime_labels

    t = df["timestamp"].values.astype(np.float64) * 1e-9
    onset = None
    if annotations["outcome"] == "failure":
        if annotations.get("drop_times_s"):
            onset = annotations["drop_times_s"][0]
        elif annotations.get("injection_time_s") is not None:
            onset = annotations["injection_time_s"]
        else:
            onset = t[-1]
    ttf = np.full(len(t), np.inf) if onset is None else np.maximum(onset - t, 0.0)
    return dict(
        stage=df["priv.stage"].values.astype(int),
        progress=df["priv.progress"].values.astype(np.float32),
        regime_id=step_regime_labels(df, annotations),
        time_to_failure=ttf.astype(np.float32),
        pre_failure=(ttf <= lead_s) if onset is not None else np.zeros(len(t), bool),
        onset_s=onset,
        is_ood=len(annotations.get("shifts", [])) > 0,
        category=annotations["category"],
    )


def stage_progress_matrix(df):
    """(n_stages, 2) first and last time each stage was active, for timeline plots."""
    st = df["priv.stage"].values
    t = df["timestamp"].values.astype(np.float64) * 1e-9
    out = {}
    for s, ti in zip(st, t):
        a, b = out.get(s, (ti, ti))
        out[s] = (min(a, ti), max(b, ti))
    return out
