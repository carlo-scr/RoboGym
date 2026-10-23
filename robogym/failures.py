"""Injected failures: the scripted expert reads a FailureSpec and corrupts its own behaviour.

`severity` in [0, 1] scales each perturbation. The *injected* failure and the *physical outcome* are
labelled separately (a mild misplacement can still succeed; such episodes stay labelled success).
Generic kinds apply to every pick-and-place-style task; tasks add their own in `Task.failure_types`
and draw their parameters in `Task.sample_failure`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

GENERIC_FAILURES = (
    "drop",  # gripper opens briefly while carrying -> object dropped
    "weak_grip",  # too little grip force -> object slips / rotates in hand
    "misgrasp",  # grasp point offset along the object -> unstable grasp
    "premature_release",  # lets go above the target -> object bounces / tilts
    "bump",  # external knock on the gripper (collision with the environment / the other arm)
    "stall",  # operator freezes for several seconds mid-task (timeouts on long horizons)
)


@dataclass
class FailureSpec:
    kind: str
    severity: float = 0.5
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d):
        return None if d is None else FailureSpec(d["kind"], d.get("severity", 0.5), dict(d.get("params", {})))


def sample_failure(kind: str, severity: float, rng: np.random.Generator, task, sc: dict) -> FailureSpec:
    """Concrete perturbation parameters for a failure kind at a given severity."""
    s = float(np.clip(severity, 0, 1))
    if kind not in GENERIC_FAILURES and kind not in task.failure_types:
        raise ValueError(f"unknown failure {kind!r} for task {task.name}; choose from {GENERIC_FAILURES + tuple(task.failure_types)}")
    n_sub = max(1, task.horizon(sc))
    p = {"subgoal": int(rng.integers(0, n_sub))}  # which subgoal (object) the failure hits
    if kind == "drop":
        p["at_frac"] = float(rng.uniform(0.15, 0.85))  # fraction of the carry phase
        p["open_s"] = 0.4 + 0.5 * s  # the rate-limited gripper opens fully in ~0.6 s
    elif kind == "weak_grip":
        p["force_frac"] = float(0.3 * (1 - s) ** 1.5 + 0.04)
    elif kind == "misgrasp":
        p["offset"] = float(rng.choice([-1, 1]) * (0.008 + 0.014 * s))  # along the object's grasp axis
    elif kind == "premature_release":
        p["height"] = float(0.02 + 0.06 * s)
    elif kind == "bump":
        d = rng.normal(0, 1, 3)
        d[2] = abs(d[2]) * 0.3
        tq = rng.normal(0, 1, 3)
        p["force"] = (d / (np.linalg.norm(d) + 1e-9) * (25.0 + 175.0 * s)).tolist()
        p["torque"] = (tq / (np.linalg.norm(tq) + 1e-9) * (1.0 + 8.0 * s)).tolist()
        p["duration_s"] = 0.05 + 0.10 * s
        p["phase"] = str(rng.choice(["carry", "place", "descend"]))
        p["at_frac"] = float(rng.uniform(0.2, 0.8))
    elif kind == "stall":
        p["phase"] = str(rng.choice(["carry", "descend"]))
        p["at_frac"] = float(rng.uniform(0.2, 0.8))
        p["duration_s"] = float(1.5 + 6.0 * s)
    else:
        p.update(task.sample_failure(kind, s, rng, sc))
    return FailureSpec(kind, s, p)


def all_failure_types(task) -> tuple:
    return GENERIC_FAILURES + tuple(task.failure_types)
