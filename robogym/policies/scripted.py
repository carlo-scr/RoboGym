"""The scripted experts as policies: the privileged upper bound, and degraded versions of it that make
*execution* quality a knob (actuation noise on every phase, injected failures with a chosen
probability, no recovery) while the plan stays the expert's own."""
from __future__ import annotations

import numpy as np

from ..env import ACT_DIM
from ..failures import sample_failure
from .base import Policy


def add_exec_noise(a, env, rng, sigma):
    """Gaussian perturbation of the executed translation (sigma m/step; rotation 4 sigma rad/step) on every
    arm. Unlike DART it also hits contact phases, so it degrades grasps and insertions, not just transport."""
    if sigma <= 0:
        return a
    a = np.asarray(a, float).reshape(len(env.arms), ACT_DIM).copy()
    a[:, :3] += rng.normal(0, sigma, (len(env.arms), 3))
    a[:, 3:6] += rng.normal(0, 4 * sigma, (len(env.arms), 3))
    return a.reshape(-1)


class ExpertPolicy(Policy):
    """`env.task.make_expert` behind the Policy interface.

    exec_noise: actuation noise (m/step) on the executed action; failure / severity / p_failure: inject
    the named failure kind (see robogym.failures) with probability p_failure per episode; recover: the
    expert notices a failed subgoal and retries it; noise: the expert's own operator variability."""

    name = "expert"
    privileged = True

    def __init__(self, exec_noise=0.0, failure=None, severity=0.5, p_failure=1.0, recover=False, noise=True, bimanual=None):
        self.exec_noise, self.failure, self.severity, self.p_failure = float(exec_noise), failure, float(severity), float(p_failure)
        self.recover, self.noise, self.bimanual = bool(recover), bool(noise), bimanual
        self.ex = None
        self.name = "expert" + (f"_noise{exec_noise}" if exec_noise else "") + (f"_{failure}" if failure else "") + ("_recover" if recover else "")

    def reset(self, env, seed=None):
        seed = int(seed or 0)
        self.rng = np.random.default_rng([seed, 90001])
        fs = None
        if self.failure and self.rng.random() < self.p_failure:
            fs = sample_failure(self.failure, self.severity, np.random.default_rng([seed, 104729]), env.task, env.scenario.task)
        self.fs = fs
        self.ex = env.task.make_expert(env, failure=fs, seed=seed, noise=self.noise, bimanual=self.bimanual, recover=self.recover)

    def act(self, env):
        return add_exec_noise(self.ex.act(), env, self.rng, self.exec_noise)

    @property
    def finished(self):
        return self.ex is not None and self.ex.done

    def info(self):
        return dict(expert_retries=int(self.ex.retries), injected=None if self.fs is None else self.fs.to_dict(),
                    expert_events=[list(e) for e in self.ex.events][-40:])
