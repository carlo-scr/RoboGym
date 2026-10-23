"""Test-time adaptation protocols: sequences of episodes (and within-episode schedules) whose physics
regime changes over time, with ground-truth regime labels for every step.

Protocols (named after the continual-TTA literature):
  sudden      nominal for `n0` episodes, then one shifted regime for the rest
  gradual     the shift's severity ramps up linearly over the stream
  recurring   A B A B ... (regime switches every `period` episodes)
  mixed       a random regime per block of `period` episodes (incl. nominal blocks)
  in_episode  every episode starts nominal and switches regime at a random time inside the episode
Each returns a list of EpisodeSpec with `tag=f"{name}:{regime_id}"` and `regime_schedule` set for
in-episode changes; `regime_table(specs)` recovers the ground-truth regime per episode.
"""
from __future__ import annotations

import numpy as np

from .datagen import EpisodeSpec
from .shifts import lerp

REGIME_FAMILIES = {
    "slippery": lambda s: {"pad_friction": lerp(0.5, 0.15, s)},
    "heavy": lambda s: {"object_mass": lerp(2.0, 5.0, s)},
    "arm_gain": lambda s: {"arm_gain": lerp(0.6, 0.25, s)},
    "weak_gripper": lambda s: {"grip_force": lerp(0.45, 0.15, s)},
    "delay": lambda s: {"action_delay": int(round(lerp(1, 5, s)))},
    "sticky": lambda s: {"bench_friction": lerp(1.8, 4.0, s)},
    "noisy_obs": lambda s: {"obs_noise": lerp(0.002, 0.02, s)},
    "heavy_cable": lambda s: {"cable_mass": lerp(2.5, 4.0, s)},
    "stiff_cable": lambda s: {"cable_stiffness": lerp(3.0, 6.0, s)},
}


def regime(family: str, severity: float) -> dict:
    return REGIME_FAMILIES[family](float(np.clip(severity, 0, 1)))


def random_regime(rng: np.random.Generator, severity: float, families=None) -> dict:
    fam = str(rng.choice(list(families or REGIME_FAMILIES)))
    return regime(fam, severity)


def make_stream(name: str, n: int, seed0: int = 0, families=("slippery", "heavy", "arm_gain", "weak_gripper", "delay"), severity: float = 0.7,
                n0: int | None = None, period: int = 5, variant: dict | None = None, dart: float = 0.0, onset_s=(1.0, 6.0)):
    """A list of EpisodeSpec implementing one TTA protocol."""
    rng = np.random.default_rng([seed0, 4242])
    families = list(families)
    specs = []
    n0 = n // 4 if n0 is None else n0
    for i in range(n):
        sp = EpisodeSpec(seed=seed0 + i, category="regime_shift", variant=dict(variant or {}), dart=dart)
        rid, sched = 0, []
        if name == "sudden":
            if i >= n0:
                rid = 1
                sched = [[0.0, regime(families[0], severity)]]
        elif name == "gradual":
            s = severity * i / max(1, n - 1)
            rid = int(np.ceil(4 * i / max(1, n - 1)))
            if s > 0:
                sched = [[0.0, regime(families[0], s)]]
        elif name == "recurring":
            block = i // period
            rid = block % 2
            if rid:
                sched = [[0.0, regime(families[0], severity)]]
        elif name == "mixed":
            block = i // period
            brng = np.random.default_rng([seed0, 99, block])
            fam_i = int(brng.integers(0, len(families) + 1))  # == len(families): nominal block
            rid = fam_i
            if fam_i < len(families):
                sched = [[0.0, regime(families[fam_i], severity)]]
        elif name == "in_episode":
            fam_i = int(rng.integers(0, len(families)))
            rid = fam_i + 1
            sched = [[float(rng.uniform(*onset_s)), regime(families[fam_i], severity)]]
        else:
            raise ValueError(f"unknown protocol {name!r}")
        sp.regime_schedule = sched
        sp.category = "nominal" if not sched else "regime_shift"
        sp.tag = f"{name}:{rid}"
        specs.append(sp)
    return specs


def regime_table(specs):
    """(episode index, regime id, schedule) for a stream."""
    return [(i, int(sp.tag.split(":")[1]), sp.regime_schedule) for i, sp in enumerate(specs)]


def step_regime_labels(df, annotations):
    """Per-step regime id for a recorded episode: 0 before the first scheduled change, k after the k-th."""
    sched = annotations.get("regime_schedule", [])
    t = df["timestamp"].values.astype(np.float64) * 1e-9
    rid = np.zeros(len(t), int)
    for k, (t0, _) in enumerate(sched, start=1):
        rid[t >= t0 - 1e-9] = k
    return rid
