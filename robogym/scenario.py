"""Per-episode randomisation: nominal domain randomisation, OOD shifts and test-time regimes.

A `Scenario` (+ EnvCfg) fully determines an episode's initial conditions and its scheduled regime
changes, so any episode can be rebuilt exactly from its annotations.json. The env applies the generic
fields; `Scenario.task` is the task's own dict (see tasks/*.py).

Physics-type shifts are expressed as a *regime*, a dict of multiplicative scales / knobs that can also be
changed online (`env.set_regime`) and scheduled within an episode (`regime_schedule`), which is what the
test-time-adaptation protocols in streams.py build on.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .config import EnvCfg

BASE_LIGHTS = {"key": 0.6, "fill": 0.25, "ceil_a": 0.2, "ceil_b": 0.2, "spot": 0.0, "back": 0.0}
REGIME_KEYS = ("pad_friction", "bench_friction", "object_friction", "object_mass", "arm_gain", "grip_force",
               "action_delay", "obs_noise", "cable_mass", "cable_stiffness", "gravity")
NOMINAL_REGIME = {k: 1.0 for k in REGIME_KEYS} | {"action_delay": 0, "obs_noise": 0.0}


@dataclass
class Scenario:
    seed: int = 0
    shifts: list = field(default_factory=list)  # OOD shift names (see shifts.py)
    severity: float = 0.5  # severity used for the shifts in [0, 1]
    variant: dict = field(default_factory=dict)  # task variant knobs (n_objects, pattern, ...)
    # Appearance / lighting
    lights: dict = field(default_factory=lambda: {k: [v] * 3 for k, v in BASE_LIGHTS.items()})
    headlight: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])  # [scale, r, g, b]
    flicker: list = field(default_factory=list)  # [amplitude, Hz, phase] or []
    bench_rgba: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])  # tint on the bench texture
    mat_rgba: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    distractors: list = field(default_factory=list)  # [[index, x, y, yaw], ...]
    # Sensors
    cam_jitter: dict = field(default_factory=dict)  # camera -> [dx, dy, dz, rx, ry, rz]
    image_fx: dict = field(default_factory=dict)  # see env.apply_image_fx
    # Robot
    home_noise: list = field(default_factory=list)  # 7 * n_arms joint offsets
    # Physics regime at t = 0 and scheduled changes [[t_s, {regime overrides}], ...]
    regime: dict = field(default_factory=lambda: dict(NOMINAL_REGIME))
    regime_schedule: list = field(default_factory=list)
    # Task-specific part
    task: dict = field(default_factory=dict)

    def to_dict(self):
        return _jsonable(dataclasses.asdict(self))

    @staticmethod
    def from_dict(d):
        known = {f.name for f in dataclasses.fields(Scenario)}
        sc = Scenario(**{k: v for k, v in d.items() if k in known})
        sc.regime = dict(NOMINAL_REGIME) | dict(sc.regime)
        return sc

    def regime_at(self, t: float) -> dict:
        r = dict(self.regime)
        for t0, over in self.regime_schedule:
            if t >= t0:
                r.update(over)
        return r


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def sample_scenario(cfg: EnvCfg, seed: int, shifts=(), severity: float = 0.5, variant: dict | None = None, task=None) -> Scenario:
    """Nominal randomisation + the requested OOD shifts (names, see shifts.SHIFT_GROUPS) at `severity`."""
    from . import shifts as S
    from .tasks.base import get_task_class

    rng = np.random.default_rng([int(seed), 12345])
    shifts = [S.ALIASES.get(s, s) for s in shifts]
    for s in shifts:
        if s not in S.ALL_SHIFTS and s not in (task or get_task_class(cfg.task_name)(cfg)).shift_types:
            raise ValueError(f"unknown shift {s!r}; generic: {S.ALL_SHIFTS}")
    sc = Scenario(seed=int(seed), shifts=list(shifts), severity=float(severity), variant=dict(variant or {}))
    cams = cfg.camera.all_cameras(cfg.robot)
    # ---- nominal randomisation (in distribution)
    tint = rng.uniform(0.95, 1.0, 3)
    sc.lights = {k: (v * rng.uniform(0.8, 1.2) * tint).tolist() for k, v in BASE_LIGHTS.items()}
    sc.bench_rgba = [*(rng.uniform(0.9, 1.05, 3)).tolist(), 1.0]
    sc.cam_jitter = {c: np.concatenate([rng.normal(0, 0.0015, 3), rng.normal(0, 0.012, 3)]).tolist() for c in cams}
    sc.image_fx = {"noise": float(rng.uniform(0, 2.0)), "gain": float(rng.uniform(0.95, 1.05))}
    sc.home_noise = rng.normal(0, 0.02, 7 * cfg.robot.n_arms).tolist()
    sc.regime = dict(NOMINAL_REGIME)
    sc.regime.update(pad_friction=float(rng.uniform(0.9, 1.1)), object_mass=float(rng.uniform(0.9, 1.1)),
                     bench_friction=float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.25:
        sc.distractors = [S.distractor_item(cfg, rng, 0, [], 0.0)]
    # ---- generic OOD shifts
    S.apply_shifts(cfg, sc, rng, shifts, severity)
    # ---- task part (placement, goal, task-specific shifts; keeps distractors clear of the objects)
    task = task or get_task_class(cfg.task_name)(cfg)
    sc.task = task.sample(np.random.default_rng([int(seed), 777]), shifts, sc.variant)
    sc.task["severity"] = float(severity)
    if sc.distractors:
        sc.distractors = S.relocate_distractors(cfg, rng, sc, task)
    return sc
