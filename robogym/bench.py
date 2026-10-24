"""The benchmark protocol: versioned seed splits, named shift suites and regime conditions, metrics and a
results schema, so that policies, monitors and adaptation methods are compared the same way.

Seeds (per task, version 1):
  train     0 .. 1999        nominal demonstrations (and failure / OOD episodes for failure models)
  cal       2000 .. 2499     nominal episodes for conformal calibration
  test      3000 .. 3499     nominal evaluation
  shift     4000 .. 4499     evaluation under shifts (same seeds for every suite / severity -> paired comparisons)
  stream    5000 ..          test-time-adaptation streams
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .datagen import EpisodeSpec
from .streams import REGIME_FAMILIES, make_stream, regime

VERSION = 1
SPLITS = {"train": (0, 2000), "cal": (2000, 2500), "test": (3000, 3500), "shift": (4000, 4500), "stream": (5000, 6000)}
SEVERITIES = (0.3, 0.6, 0.9)
# Shift suites (names from robogym.shifts + Task.shift_types)
SUITES = {
    "visual": ("lighting_dim", "lighting_harsh", "lighting_colored", "lighting_flicker", "bench_color", "object_color", "clutter",
               "camera_shift", "camera_noise", "motion_blur", "overexposure", "lens_occlusion"),
    "physics": ("slippery", "heavy_object", "arm_gain_drift", "weak_gripper", "action_delay", "sticky_bench", "obs_noise"),
    "task": {"cube_stack": ("cube_size", "object_texture"), "lego": ("tight_fit", "stiff_snap", "plate_shift"),
             "cable_plug": ("dense_ports", "led_distractors", "panel_offset", "heavy_cable", "stiff_cable", "distractor_cable")},
}
# Named regime conditions (online physics), as in the plugging-sim pilot
CONDITIONS = {"nominal": {}} | {f: regime(f, 0.7) for f in REGIME_FAMILIES}
HORIZONS = {"cube_stack": (2, 3, 4), "lego": (2, 3, 4), "cable_plug": (1, 2)}


def seeds(split: str, n: int | None = None, offset: int = 0):
    a, b = SPLITS[split]
    s = list(range(a + offset, b))
    return s if n is None else s[:n]


def split_specs(task: str, split: str, n: int | None = None, horizon: int | None = None, dart: float = 0.0, mix=None):
    """Nominal EpisodeSpecs of a split (train: optionally a category mix via `mix`, see datagen.make_plan)."""
    from .config import make_cfg
    from .datagen import make_plan
    from .tasks.base import get_task_class

    variant = {} if horizon is None else {"n": horizon}
    if mix is not None:
        t = get_task_class(task)(make_cfg(task))
        plan = make_plan(t, len(seeds(split, n)), seed0=SPLITS[split][0], mix=mix, dart=dart)
        for sp in plan:
            sp.variant = dict(variant)
        return plan
    return [EpisodeSpec(seed=s, variant=dict(variant), dart=dart, tag=f"{split}") for s in seeds(split, n)]


def suite_specs(task: str, suite: str, n: int = 50, severities=SEVERITIES, horizon: int | None = None):
    """Paired shift evaluation: the same `shift` seeds under every shift of the suite at every severity."""
    names = SUITES[suite][task] if isinstance(SUITES[suite], dict) else SUITES[suite]
    variant = {} if horizon is None else {"n": horizon}
    out = []
    for sh in names:
        for sev in severities:
            out += [EpisodeSpec(seed=s, category="ood", shifts=[sh], shift_severity=sev, variant=dict(variant), tag=f"{suite}:{sh}:{sev}") for s in seeds("shift", n)]
    return out


def condition_specs(task: str, n: int = 50, horizon: int | None = None, conditions=None):
    """The same `shift` seeds under each named regime condition (applied from t = 0)."""
    variant = {} if horizon is None else {"n": horizon}
    out = []
    for name, reg in (conditions or CONDITIONS).items():
        sched = [[0.0, reg]] if reg else []
        out += [EpisodeSpec(seed=s, category="regime_shift" if reg else "nominal", regime_schedule=sched, variant=dict(variant), tag=f"cond:{name}") for s in seeds("shift", n)]
    return out


def stream_specs(task: str, protocol: str, n: int = 40, **kw):
    return make_stream(protocol, n, seed0=SPLITS["stream"][0], **kw)


# ---------------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial rate."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def rate(flags):
    f = np.asarray(flags, bool)
    p, lo, hi = wilson(int(f.sum()), len(f))
    return dict(rate=p, lo=lo, hi=hi, n=int(len(f)))


def paired_delta(a_flags, b_flags):
    """Difference of paired success rates (same seeds) with a normal-approximation CI and the paired
    harm rate (b failed where a succeeded)."""
    a, b = np.asarray(a_flags, bool), np.asarray(b_flags, bool)
    d = b.astype(float) - a.astype(float)
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return dict(delta=float(d.mean()), lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se), harm=float((a & ~b).mean()), help=float((~a & b).mean()), n=int(len(d)))


def detection_metrics(nominal_scores, positive_scores, flagged_nominal, flagged_positive, leads=()):
    from .learn.monitor import auroc

    leads = np.asarray([x for x in leads if x is not None], float)
    return dict(auroc=auroc(nominal_scores, positive_scores), false_alarm=rate(flagged_nominal), tpr=rate(flagged_positive),
                median_lead_s=float(np.median(leads)) if len(leads) else float("nan"), warned_before=float((leads > 0).mean()) if len(leads) else float("nan"))


# ---------------------------------------------------------------------- results
@dataclass
class Results:
    name: str
    task: str
    method: str
    config: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)  # dicts: {"split"/"suite"/"condition": ..., "metric": ..., ...}
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    protocol_version: int = VERSION

    def add(self, **row):
        self.entries.append(row)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.__dict__, open(path, "w"), indent=1, default=_json)
        return path

    @staticmethod
    def load(path):
        d = json.load(open(path))
        return Results(**d)

    def table(self):
        import pandas as pd

        return pd.DataFrame(self.entries)


def _json(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def evaluate_actor(env, act_fn, specs, max_s=None, log=None, settle_s=1.0, reset_fn=None):
    """Run `act_fn(env) -> action` on every spec (same seeds -> paired across conditions). Returns a DataFrame
    with success / progress / duration / tag per episode. `reset_fn(env, spec)` is called after env.reset."""
    import pandas as pd

    from .datagen import build_scenario

    rows = []
    for i, sp in enumerate(specs):
        sc = build_scenario(env, sp)
        env.reset(sc)
        if reset_fn:
            reset_fn(env, sp)
        lim = max_s or env.episode_max_s
        t_ok = None
        while env.time < lim and not env.unstable:
            env.step(act_fn(env))
            if t_ok is None and env.info()["success"]:
                t_ok = env.time
            if t_ok is not None and env.time - t_ok > settle_s:
                break
        info = env.info()
        rows.append(dict(seed=sp.seed, tag=sp.tag, category=sp.category, shifts="+".join(sp.shifts), severity=sp.shift_severity if sp.shifts else 0.0,
                         success=bool(info["success"]), progress=float(info["progress"]), duration=env.time, unstable=env.unstable, horizon=env.task.horizon(sc.task)))
        if log and (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(specs)} episodes, success so far {np.mean([r['success'] for r in rows]):.2f}")
    return pd.DataFrame(rows)


def summarize_eval(df):
    """Success rate (+ Wilson CI) and mean progress per tag."""
    import pandas as pd

    out = []
    for tag, g in df.groupby("tag"):
        r = rate(g.success.values)
        out.append(dict(tag=tag, n=r["n"], success=r["rate"], lo=r["lo"], hi=r["hi"], progress=float(g.progress.mean()), duration=float(g.duration.mean())))
    return pd.DataFrame(out).set_index("tag")
