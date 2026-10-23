"""Bimanual expert: a right and a left single-arm expert run concurrently, each on the objects on its
side of the bench, taking turns at the shared structure (stack / baseplate / panel) through a
Coordinator. Phases, events and retries of both arms are exposed for labelling."""
from __future__ import annotations

import numpy as np

from ..env import ACT_DIM
from .base import Coordinator


def assign_by_side(env, subgoal_xy: dict) -> dict:
    """arm -> subgoal indices, by the y side of each subgoal's object (right arm: y < 0)."""
    out = {arm: [] for arm in env.arms}
    for k, xy in subgoal_xy.items():
        out["right" if xy[1] < 0 else "left"].append(int(k))
    return out


class DualArmExpert:
    name = "dual_arm_expert"

    def __init__(self, env, make_sub, subgoal_xy: dict, failure=None, seed: int = 0, noise: bool = True, ordered: bool = True):
        self.env = env
        self.arms = tuple(env.arms)
        self.coord = Coordinator(ordered=ordered, subgoals=subgoal_xy.keys())
        self.assign = assign_by_side(env, subgoal_xy)
        f_sub = failure.params.get("subgoal", 0) if failure is not None else None
        self.subs = {}
        for i, arm in enumerate(self.arms):
            f = failure if (failure is not None and f_sub in self.assign[arm]) else None
            self.subs[arm] = make_sub(env, arm=arm, subgoals=self.assign[arm], coord=self.coord, failure=f, seed=seed + 31 * i, noise=noise)
        self.failure = failure
        self.i_arm = 0
        self.arm = "both"

    @property
    def done(self):
        return all(ex.done for ex in self.subs.values())

    @property
    def phase(self):
        return "|".join(f"{a[0]}:{ex.phase}" for a, ex in self.subs.items())

    @property
    def subgoal(self):
        return len(self.coord.done)

    @property
    def events(self):
        ev = [(f"{a}:{n}", t) for a, ex in self.subs.items() for n, t in ex.events]
        return sorted(ev, key=lambda e: e[1])

    @property
    def retries(self):
        return sum(ex.retries for ex in self.subs.values())

    @property
    def holding(self):
        return any(ex.holding for ex in self.subs.values())

    def holding_of(self, arm):
        return self.subs[arm].holding

    def grip_of(self, arm):
        return self.subs[arm].grip

    def act(self):
        a = np.zeros((len(self.arms), ACT_DIM))
        for i, arm in enumerate(self.arms):
            ex = self.subs[arm]
            a[i] = ex.act().reshape(len(self.arms), ACT_DIM)[ex.i_arm]
        return a.reshape(-1)
