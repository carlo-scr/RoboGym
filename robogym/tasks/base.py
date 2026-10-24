"""Task interface. A Task contributes bodies to the MJCF, samples/applies its part of the Scenario,
exposes privileged state, stage / progress / success, labels the physical outcome of an episode, and
builds its scripted expert. The env owns the robot, bench, cameras, shifts that are task-agnostic,
snapshots and regimes.

Long-horizon structure: a task instance has a list of *subgoals* (`stages()`), and `progress()` is a
monotone-ish scalar in [0, 1] over them, so monitors / world models get dense labels and curricula can
scale the horizon (`horizon()` = number of subgoals, e.g. number of cubes to stack).
"""
from __future__ import annotations

import numpy as np


class Task:
    name: str = "task"
    failure_types: tuple = ()  # task-specific injected failures (see failures.py)
    shift_types: tuple = ()  # task-specific OOD shifts (see shifts.py for the shared ones)
    outcome_modes: tuple = ("none",)  # physical failure modes this task labels
    tool_tip = None  # remote-compliance point in the TCP frame (None: cfg.robot.gripper.tool_tip)
    compliant_gains = None  # per-axis IK gains in the TCP frame for the compliant mode (None: cfg default)

    def __init__(self, cfg):
        self.cfg = cfg  # EnvCfg
        self.tc = cfg.task

    # ---- model
    def xml(self) -> tuple[str, list, str, str, str]:
        """(worldbody xml, asset xml list, equality xml, contact xml, defaults xml)."""
        raise NotImplementedError

    def bind(self, env):
        """Cache model ids after compilation."""
        raise NotImplementedError

    # ---- scenario
    def sample(self, rng: np.random.Generator, shifts: list, variant: dict | None = None) -> dict:
        """Task part of the Scenario (JSON-serialisable dict)."""
        raise NotImplementedError

    def apply(self, env, sc: dict):
        """Set model parameters (colours, masses, static poses) for the task bodies at reset."""
        raise NotImplementedError

    def reset_state(self, env, sc: dict):
        """Set qpos of the task's free bodies at reset (before the settle phase)."""
        raise NotImplementedError

    def after_settle(self, env):
        """Called once after the settle phase (e.g. record resting poses / goal poses)."""

    def apply_regime(self, env, regime: dict):
        """Task-specific online physics changes (object mass / friction scales...)."""

    def default_episode_s(self, sc: dict) -> float:
        return 20.0

    # ---- per-step
    def pre_step(self, env):
        """Called at every physics sub-step after mj_step1 (e.g. detent forces)."""

    def post_step(self, env):
        """Called after every physics sub-step (e.g. latch / snap checks)."""

    def privileged(self, env) -> dict:
        raise NotImplementedError

    def stages(self) -> tuple:
        return ("done",)

    def stage(self, env, pr: dict) -> int:
        return 0

    def progress(self, env, pr: dict) -> float:
        return 0.0

    def success(self, env, pr: dict) -> bool:
        return False

    def horizon(self, sc: dict) -> int:
        return 1

    def subgoal_done(self, env, k: int) -> bool:
        """Subgoal k (cube k placed / brick k snapped at its goal / cable k latched) physically achieved."""
        return False

    def task_features(self, env, pr: dict) -> np.ndarray:
        """Privileged low-dim features for state-based policies (fixed size, see feature_names)."""
        return np.zeros(0, np.float32)

    def feature_names(self) -> list:
        return []

    # ---- labels
    def outcome(self, env, record: dict, expert) -> str:
        """Physical failure mode of a finished episode ("none" = success)."""
        return "none" if self.success(env, self.privileged(env)) else "other"

    def goal_description(self, sc: dict) -> str:
        return self.name

    def record_columns(self, env, pr: dict) -> dict:
        """Extra per-step scalar labels written as priv.<name> columns (task-specific)."""
        return {}

    # ---- snapshots
    def aux_size(self) -> int:
        return 0

    def get_aux(self, env) -> np.ndarray:
        return np.zeros(0)

    def set_aux(self, env, aux: np.ndarray):
        pass

    # ---- expert / failures
    def make_expert(self, env, failure=None, seed: int = 0, noise: bool = True):
        raise NotImplementedError

    def sample_failure(self, kind: str, severity: float, rng: np.random.Generator, sc: dict) -> dict:
        """Concrete perturbation parameters for an injected failure of this task."""
        return {}


_REGISTRY: dict = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_task_class(name: str):
    if name not in _REGISTRY:
        import importlib

        for mod in ("cube_stack", "lego", "cable_plug"):
            try:
                importlib.import_module(f"{__package__}.{mod}")
            except ModuleNotFoundError as e:  # a task module that does not exist yet
                if mod not in str(e):
                    raise
    return _REGISTRY[name]


def task_names() -> tuple:
    try:
        get_task_class("")
    except KeyError:
        pass
    return tuple(sorted(_REGISTRY))
