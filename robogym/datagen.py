"""Episode rollout, labelling and export.

Each episode is one folder in the layout the lab's real-robot loaders (GaugeFailClassification
`Trajectory`) expect, extended with privileged labels and exact simulator snapshots:

    ep_000123/
      episode.parquet      timestamp (int64 ns), observation.state, action, action.expert, action.ee_target,
                           task.features, priv.* (phase, subgoal, stage, progress, grasped, regime, task columns)
      <camera>.mp4         one video per recorded camera (control-rate fps)
      annotations.json     labels: category, outcome, failure mode, injected failure, shifts, regime schedule,
                           events + times, stage timeline, goal, scenario
      feature_report.json  names of every state / action / feature dimension, cameras, fps, full EnvCfg
      sim_states.npz       full simulator snapshot per step -> exact counterfactual restarts
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import EnvCfg
from .env import ACT_DIM, RoboGymEnv, action_names, state_names
from .failures import FailureSpec, all_failure_types, sample_failure
from .scenario import Scenario, sample_scenario

SIM_VERSION = "0.1.0"
DART_PHASES = ("reach", "lift", "carry", "retreat", "home")  # free-space phases only
CATEGORIES = ("nominal", "known_failure", "ood", "regime_shift")


@dataclass
class EpisodeSpec:
    seed: int
    category: str = "nominal"  # nominal | known_failure | ood | regime_shift
    failure: str | None = None
    severity: float = 0.0
    shifts: list = field(default_factory=list)
    shift_severity: float = 0.5
    variant: dict = field(default_factory=dict)  # task variant knobs, e.g. {"n": 3}
    regime_schedule: list = field(default_factory=list)  # [[t_s, {regime overrides}], ...] inside the episode
    dart: float = 0.0  # DART noise (m / step on the active arm's translation; rotation 4x in rad)
    tag: str = ""  # free label (stream id, condition name, ...)
    recover: bool = False  # the expert notices a failed subgoal and retries it (recovery demonstrations)

    def to_dict(self):
        return dict(seed=self.seed, category=self.category, failure=self.failure, severity=self.severity, shifts=list(self.shifts),
                    shift_severity=self.shift_severity, variant=dict(self.variant), regime_schedule=list(self.regime_schedule), dart=self.dart, tag=self.tag,
                    recover=self.recover)

    @staticmethod
    def from_dict(d):
        return EpisodeSpec(**{k: v for k, v in d.items() if k in EpisodeSpec.__dataclass_fields__})


def make_plan(task, n: int, seed0: int = 0, mix=(("nominal", 0.6), ("known_failure", 0.2), ("ood", 0.15), ("regime_shift", 0.05)),
              failure_types=None, shift_types=None, variants=None, dart=0.0):
    """EpisodeSpecs with the requested category mix. Failures / shifts are drawn uniformly, severities
    uniformly in [0, 1]; OOD episodes get one or two shifts; regime_shift episodes switch a physics regime
    on at a random time. `variants`: list of variant dicts to cycle through (e.g. horizons)."""
    from . import shifts as S

    rng = np.random.default_rng(seed0)
    failure_types = list(failure_types or all_failure_types(task))
    shift_types = list(shift_types or (S.ALL_SHIFTS + tuple(task.shift_types)))
    cats, probs = zip(*mix)
    plan = []
    for i in range(n):
        cat = str(rng.choice(cats, p=np.asarray(probs) / np.sum(probs)))
        sp = EpisodeSpec(seed=seed0 + i, category=cat, dart=dart)
        if variants:
            sp.variant = dict(variants[i % len(variants)])
        if cat == "known_failure":
            sp.failure = str(rng.choice(failure_types))
            sp.severity = float(rng.uniform(0, 1))
        elif cat == "ood":
            sp.shifts = [str(x) for x in rng.choice(shift_types, size=int(rng.integers(1, 3)), replace=False)]
            sp.shift_severity = float(rng.uniform(0.2, 1.0))
        elif cat == "regime_shift":
            from .streams import random_regime

            sp.regime_schedule = [[float(rng.uniform(1.0, 6.0)), random_regime(rng, float(rng.uniform(0.3, 1.0)))]]
        plan.append(sp)
    return plan


def build_scenario(env: RoboGymEnv, spec: EpisodeSpec) -> Scenario:
    sc = sample_scenario(env.cfg, spec.seed, shifts=spec.shifts, severity=spec.shift_severity, variant=spec.variant, task=env.task)
    sc.regime_schedule = [list(x) for x in spec.regime_schedule]
    return sc


def build_episode(env: RoboGymEnv, spec: EpisodeSpec):
    """Reset env for a spec; returns (scenario, failure, expert)."""
    sc = build_scenario(env, spec)
    env.reset(sc)
    failure = None
    if spec.failure:
        failure = sample_failure(spec.failure, spec.severity, np.random.default_rng([spec.seed, 104729]), env.task, sc.task)
    expert = env.task.make_expert(env, failure=failure, seed=spec.seed, recover=spec.recover)
    return sc, failure, expert


def _ee_target(env: RoboGymEnv):
    return np.concatenate([np.concatenate([env.targets[s][0], env.targets[s][1], [env.grip_cmd[s]]]) for s in env.arms]).astype(np.float32)


def run_episode(env: RoboGymEnv, spec: EpisodeSpec, cameras=None, save_states=True, max_s=None, policy=None, render_ghost=False):
    """Roll out the scripted expert (or `policy(env) -> action`, with the expert only providing labels) for
    one spec and return everything needed to write the episode."""
    sc, failure, ex = build_episode(env, spec)
    task = env.task
    cameras = list(env.cfg.camera.record if cameras is None else cameras)
    max_s = max_s or env.episode_max_s * (1.6 if spec.recover else 1.0)  # recoveries replay a subgoal
    cols = ("t", "state", "action", "action_expert", "ee_target", "features", "phase", "subgoal", "stage", "progress", "grasped_k", "regime_id")
    rec = {k: [] for k in cols}
    extra = {}
    frames = {c: [] for c in cameras}
    phys, aux, act64 = [], [], []
    stage_times, drops, drop_times = {}, 0, []
    ex_arms = tuple(getattr(ex, "arms", (ex.arm,)))
    held_k = {a: -1 for a in ex_arms}
    ungrasped_n = {a: 0 for a in ex_arms}
    holding_of = getattr(ex, "holding_of", lambda a: ex.holding)
    while not ex.done and env.time < max_s and not env.unstable:
        obs = env.observe()
        for c in cameras:
            frames[c].append(env.render(c, ghost=render_ghost))
        pr = env.privileged()
        info = env.info(pr)
        feats = task.task_features(env, pr)
        target_before = _ee_target(env)
        phase, subgoal = ex.phase, ex.subgoal
        a_exp = env.clip_action(ex.act())
        a = a_exp
        if policy is not None:
            a = env.clip_action(policy(env))
        elif spec.dart > 0 and phase in DART_PHASES:
            # DART (Laskey et al. 2017): execute a perturbed action, record the expert's clean action as the label.
            rng_d = np.random.default_rng((spec.seed * 1000003 + env.step_count) % (1 << 32))
            a = a_exp.copy()
            for arm in ex_arms:
                i = ACT_DIM * env.arms.index(arm)
                a[i : i + 3] += rng_d.normal(0, spec.dart, 3)
                a[i + 3 : i + 6] += rng_d.normal(0, 4 * spec.dart, 3)
            a = env.clip_action(a)
        if save_states:
            st = env.get_state()  # after the expert acted (it may switch modes), right before stepping
            phys.append(st["physics"])
            aux.append(st["aux"])
            act64.append(a)
        rec["t"].append(env.time)
        rec["state"].append(obs)
        rec["action"].append(a.astype(np.float32))
        rec["action_expert"].append(a_exp.astype(np.float32))
        rec["ee_target"].append(target_before)
        rec["features"].append(feats)
        rec["phase"].append(phase)
        rec["subgoal"].append(subgoal)
        rec["stage"].append(info["stage"])
        rec["progress"].append(info["progress"])
        rec["regime_id"].append(env._sched_i)
        for k2, v in getattr(task, "record_columns", lambda e, p: {})(env, pr).items():
            extra.setdefault(k2, []).append(v)
        stage_times.setdefault(info["stage_name"], round(env.time, 4))
        # which task object each expert arm holds (-1: none); drop = an arm loses an object while its
        # expert still intends to hold it (outside the release / retreat phases)
        gks = {a: -1 for a in ex_arms}
        for j, g in enumerate(pr["grasped"]):
            if g in gks:
                gks[g] = j
        rec["grasped_k"].append(gks[ex_arms[0]])
        if len(ex_arms) > 1:
            extra.setdefault("grasped_k_left", []).append(gks[ex_arms[1]])
        for a_ in ex_arms:
            ph = phase.split(f"{a_[0]}:")[-1].split("|")[0] if "|" in phase else phase
            if gks[a_] >= 0:
                held_k[a_], ungrasped_n[a_] = gks[a_], 0
            elif held_k[a_] >= 0 and holding_of(a_) and ph not in ("release", "retreat", "clear", "home", "done", "wait", "notice"):
                ungrasped_n[a_] += 1
                if ungrasped_n[a_] == 3:
                    drops += 1
                    drop_times.append(round(env.time - 2 * env.dt, 4))
                    held_k[a_] = -1
            elif not holding_of(a_):
                held_k[a_], ungrasped_n[a_] = -1, 0
        env.step(a)
    pr = env.privileged()
    info = env.info(pr)
    grasped_all = rec["grasped_k"] + extra.get("grasped_k_left", [])
    record = dict(grasped_k=grasped_all, drops=drops, drop_times=drop_times, n_seated=extra.get("n_seated", []),
                  max_depth={int(k2.split("_")[-1]): float(np.max(v)) for k2, v in extra.items() if k2.startswith("depth_")} if extra else {})
    mode = task.outcome(env, record, ex)
    success = mode == "none"
    ann = dict(
        sim_version=SIM_VERSION,
        task=task.name,
        spec=spec.to_dict(),
        category=spec.category,
        outcome="success" if success else "failure",
        failure_mode=mode,
        injected_failure=None if failure is None else failure.to_dict(),
        shifts=list(spec.shifts),
        shift_severity=spec.shift_severity,
        regime_schedule=[list(x) for x in spec.regime_schedule],
        variant=dict(spec.variant),
        horizon=int(task.horizon(sc.task)),
        goal=task.goal_description(sc.task),
        final_progress=float(info["progress"]),
        final_stage=info["stage_name"],
        active_arm=ex.arm,  # "both" for the bimanual expert
        retries=int(ex.retries),
        drops=drops,
        drop_times_s=drop_times,
        injection_time_s=_injection_time(ex, failure),
        recoveries=_recoveries(ex.events),
        phase_events=ex.events,
        stage_times_s=stage_times,
        duration_s=round(env.time, 4),
        n_steps=len(rec["t"]),
        unstable=bool(env.unstable),
        scenario=sc.to_dict(),
    )
    data = {k: np.asarray(v) for k, v in rec.items()}
    data.update({k: np.asarray(v) for k, v in extra.items()})
    return dict(annotations=ann, data=data, frames=frames, physics=np.asarray(phys), aux=np.asarray(aux), actions64=np.asarray(act64), extra_cols=list(extra))


def _recoveries(events):
    """[{arm, subgoal, attempt, t_start, t_end, success}] from recover_start / recover_end events
    (events are "arm:name" in bimanual runs; name = recover_start:k:attempt / recover_end:k:ok|fail)."""
    out, open_ = [], {}
    for name, t in events:
        parts = name.split(":")
        arm = None
        if parts[0] in ("right", "left"):
            arm, parts = parts[0], parts[1:]
        if parts[0] == "recover_start":
            open_[(arm, int(parts[1]))] = dict(arm=arm, subgoal=int(parts[1]), attempt=int(parts[2]), t_start=t, t_end=None, success=None)
        elif parts[0] == "recover_end":
            r = open_.pop((arm, int(parts[1])), dict(arm=arm, subgoal=int(parts[1]), attempt=0, t_start=None))
            r.update(t_end=t, success=parts[2] == "ok")
            out.append(r)
    out += list(open_.values())
    return out


def _injection_time(ex, failure: FailureSpec | None):
    """When the injected perturbation first acts on the rollout (None for failures that never triggered)."""
    if failure is None:
        return None
    ev = {}
    for name, t in ex.events:
        ev.setdefault(name, t)
    k = failure.kind
    if k in ("bump", "stall", "weak_grip", "misalign", "misplace", "wrong_slot", "wrong_cube", "wrong_port", "low_carry"):
        return ev.get(k if k != "stall" else "stall")
    if k == "drop":
        return ev.get("drop_open")
    if k == "misgrasp":
        return ev.get("descend")
    if k in ("premature_release", "partial_insert", "partial_press"):
        return ev.get("release")
    return None


def write_episode(ep: dict, out_dir: str | Path, fps: float, cfg: EnvCfg):
    import imageio.v2 as imageio
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = ep["data"]
    n = len(d["t"])
    cols = {
        "timestamp": np.round(d["t"] * 1e9).astype(np.int64),
        "frame_index": np.arange(n),
        "observation.state": list(d["state"]),
        "action": list(d["action"]),
        "action.ee_target": list(d["ee_target"]),
        "action.expert": list(d["action_expert"]),
        "task.features": list(d["features"].astype(np.float32)),
        "priv.phase": d["phase"],
        "priv.subgoal": d["subgoal"].astype(np.int16),
        "priv.stage": d["stage"].astype(np.int16),
        "priv.progress": d["progress"].astype(np.float32),
        "priv.grasped_k": d["grasped_k"].astype(np.int8),
        "priv.regime_id": d["regime_id"].astype(np.int8),
    }
    for k in ep.get("extra_cols", []):
        cols[f"priv.{k}"] = np.asarray(d[k], np.float32)
    pd.DataFrame(cols).to_parquet(out / "episode.parquet", engine="pyarrow")
    for cam, fr in ep["frames"].items():
        if not fr:
            continue
        w = imageio.get_writer(out / f"{cam}.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
        for f in fr:
            w.append_data(f)
        w.close()
    if len(ep["physics"]):
        np.savez_compressed(out / "sim_states.npz", physics=ep["physics"], aux=ep["aux"], actions=ep["actions64"])
    with open(out / "annotations.json", "w") as f:
        json.dump(ep["annotations"], f, indent=1, default=_json_default)
    fr = dict(
        sim_version=SIM_VERSION,
        task=ep["annotations"]["task"],
        fps=fps,
        observation_state_names=state_names(cfg),
        action_names=action_names(cfg),
        ee_target_names=[f"{s}.{n}" for s in cfg.robot.arms for n in ("x", "y", "z", "qw", "qx", "qy", "qz", "grip")],
        task_feature_names=_feature_names(cfg),
        cameras=list(ep["frames"].keys()),
        image_shape=[cfg.camera.height, cfg.camera.width, 3],
        action_semantics="per arm: TCP delta (world, m), rotation-vector delta (world, rad), grip in [0,1] (1 = closed), compliant in {0,1}; deltas apply to the controller target",
        env_cfg=cfg.to_dict(),
    )
    with open(out / "feature_report.json", "w") as f:
        json.dump(fr, f, indent=1, default=_json_default)
    return out


def _feature_names(cfg):
    from .tasks.base import get_task_class

    return get_task_class(cfg.task_name)(cfg).feature_names()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def summary_row(ann: dict, path: str):
    inj = ann["injected_failure"] or {}
    return dict(
        episode=os.path.basename(path), path=path, task=ann["task"], seed=ann["spec"]["seed"], category=ann["category"], tag=ann["spec"].get("tag", ""),
        outcome=ann["outcome"], failure_mode=ann["failure_mode"], injected=inj.get("kind"), severity=inj.get("severity"),
        shifts="+".join(ann["shifts"]), shift_severity=ann.get("shift_severity"), regime_shift=len(ann.get("regime_schedule", [])) > 0,
        recover=ann["spec"].get("recover", False), recoveries=len(ann.get("recoveries", [])), recovered=any(r.get("success") for r in ann.get("recoveries", [])),
        horizon=ann["horizon"], final_progress=ann["final_progress"], active_arm=ann["active_arm"], retries=ann["retries"], drops=ann["drops"],
        duration_s=ann["duration_s"], n_steps=ann["n_steps"], unstable=ann["unstable"],
    )


# ---------------------------------------------------------------------- loading / counterfactual restarts
def load_episode(path: str | Path, load_frames=False):
    import pandas as pd

    path = Path(path)
    out = dict(df=pd.read_parquet(path / "episode.parquet"), annotations=json.load(open(path / "annotations.json")),
               features=json.load(open(path / "feature_report.json")))
    if (path / "sim_states.npz").exists():
        z = np.load(path / "sim_states.npz")
        out["physics"], out["aux"], out["actions64"] = z["physics"], z["aux"], z["actions"]
    if load_frames:
        import imageio.v2 as imageio

        out["frames"] = {c: np.stack(imageio.mimread(path / f"{c}.mp4", memtest=False)) for c in out["features"]["cameras"]}
    return out


def restore(env: RoboGymEnv, episode: dict, step: int):
    """Put `env` into the exact simulator state of `episode` at `step` (for counterfactual rollouts).
    The env must have been built with the same EnvCfg."""
    env.reset(Scenario.from_dict(episode["annotations"]["scenario"]))
    env.set_state(dict(physics=episode["physics"][step], aux=episode["aux"][step], regime=episode["annotations"]["scenario"]["regime"]))
    return env


def list_episodes(root):
    """Concatenate all index_*.parquet files of a dataset folder."""
    import pandas as pd

    root = Path(root)
    idx = [pd.read_parquet(p) for p in sorted(root.glob("index_*.parquet"))]
    if not idx:
        raise FileNotFoundError(f"no index_*.parquet in {root}")
    return pd.concat(idx, ignore_index=True).sort_values(["task", "seed"]).reset_index(drop=True)
