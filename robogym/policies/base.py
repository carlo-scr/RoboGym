"""One interface for every controller that can drive a RoboGymEnv, so the benchmark compares them the
same way: scripted experts, learned chunk policies (Diffusion Policy / ACT / MLP), off-the-shelf
checkpoints behind an adapter, and the hierarchical planner + executor policies.

    policy.reset(env, seed)        after env.reset(); seed = the episode seed (candidate / noise seeding)
    a = policy.act(env)            one (8 x n_arms) action per control step
    policy.finished                the policy has nothing more to do (evaluation may stop early)
    policy.info()                  per-episode diagnostics merged into the evaluation row

`privileged` says whether the policy reads simulator ground truth (expert, oracle planner, state-based
learners that use task features); `needs_render` whether the env must be built with rendering.
"""
from __future__ import annotations

import numpy as np

from ..env import ACT_DIM


class Policy:
    name = "policy"
    privileged = False
    needs_render = False

    def reset(self, env, seed: int | None = None):
        pass

    def act(self, env) -> np.ndarray:
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        return False

    def finalize(self, env):
        """Called once when the evaluator stops the episode (success, time limit, `finished`): close any
        bookkeeping that depends on the final state before `info()` is read."""

    def info(self) -> dict:
        return {}

    def close(self):
        pass

    @staticmethod
    def hold(env) -> np.ndarray:
        """The no-op action: keep every arm's target and grip command."""
        a = np.zeros((len(env.arms), ACT_DIM))
        for i, s in enumerate(env.arms):
            a[i, 6] = env.grip_cmd[s]
        return a.reshape(-1)

    def __repr__(self):
        return self.name


class FunctionPolicy(Policy):
    """Any `fn(env) -> action` (or `fn(obs: research.ObsDict) -> action` with obs_dict=True)."""

    def __init__(self, fn, name="function", obs_dict=False, cams=(), privileged=False, needs_render=False):
        self.fn, self.name, self.obs_dict, self.cams = fn, name, obs_dict, tuple(cams)
        self.privileged, self.needs_render = privileged, needs_render or bool(cams)

    def act(self, env):
        if self.obs_dict:
            from ..research import ObsDict

            return self.fn(ObsDict(env, self.cams))
        return self.fn(env)


class HoldPolicy(Policy):
    """Does nothing: the floor of every metric."""

    name = "hold"

    def act(self, env):
        return self.hold(env)


class RandomPolicy(Policy):
    """Uniform random Cartesian deltas (scaled), a random-walk grip command, no compliance."""

    name = "random"

    def __init__(self, scale=0.5, seed=0):
        self.scale, self.seed0 = scale, seed

    def reset(self, env, seed=None):
        self.rng = np.random.default_rng([self.seed0, seed or 0])
        self.grip = {s: 0.0 for s in env.arms}

    def act(self, env):
        cfg = env.cfg.sim
        a = np.zeros((len(env.arms), ACT_DIM))
        for i, s in enumerate(env.arms):
            a[i, :3] = self.rng.uniform(-1, 1, 3) * cfg.max_dpos * self.scale
            a[i, 3:6] = self.rng.uniform(-1, 1, 3) * cfg.max_drot * self.scale
            self.grip[s] = float(np.clip(self.grip[s] + self.rng.normal(0, 0.1), 0, 1))
            a[i, 6] = self.grip[s]
        return a.reshape(-1)


class ReplayPolicy(Policy):
    """Open-loop replay of a recorded episode's actions (`episode.parquet`), then hold. Replayed on the
    recording's own seed it reproduces the recording bit for bit; on any other seed it shows how far
    pure open-loop execution gets (no feedback at all)."""

    name = "replay"

    def __init__(self, episode_dir, column="action"):
        import pandas as pd

        self.actions = np.stack(pd.read_parquet(f"{episode_dir}/episode.parquet")[column].values).astype(np.float64)
        self.i = 0

    def reset(self, env, seed=None):
        self.i = 0

    def act(self, env):
        if self.i >= len(self.actions):
            return self.hold(env)
        a = self.actions[self.i]
        self.i += 1
        return a

    @property
    def finished(self):
        return self.i >= len(self.actions)
