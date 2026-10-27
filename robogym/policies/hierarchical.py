"""Planner + executor policies: the long-horizon problem split into *which subgoal next* and *how to
achieve it*, so each half can be swapped and scored on its own.

Executors run one planned subgoal at a time:
  ScriptedExecutor   the task expert's own subgoal routine (privileged), optionally with actuation noise
                     and injected failures with a per-subgoal probability: execution quality as a knob
  LearnedExecutor    a goal-conditioned chunk policy (DP / ACT / MLP) whose task-feature slots are filled
                     with the *planned* subgoal (object pose, goal pose, relative vectors) instead of the
                     env's own current subgoal, so one checkpoint trained on expert demonstrations serves
                     as the skill under any planner

HierarchicalPolicy advances when the executor reports done (scripted) or the subgoal has been achieved
for `settle_s` (learned), gives up on a subgoal after `budget_s` and then skips / retries / stops
(`on_fail`), and logs every subgoal execution (`info()["hier_log"]`) for the plan / execution split in
robogym.lh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..failures import GENERIC_FAILURES, sample_failure
from .base import Policy
from .planning import OraclePlanner, adapter_for, plan_diff
from .scripted import add_exec_noise


class Executor:
    name = "executor"
    privileged = False
    needs_render = False

    def reset(self, env, seed=None):
        pass

    def start(self, env, k, sg, plan):
        pass

    def act(self, env):
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return False

    def finish(self, env):
        """Optional generator run after the last subgoal (e.g. go home)."""
        return None


class ScriptedExecutor(Executor):
    name = "scripted"
    privileged = True

    def __init__(self, exec_noise=0.0, noise=True, failure=None, severity=0.5, p_failure=0.0, max_retries=None):
        """failure: an injected failure kind, or "any" (one of the generic kinds), injected on a subgoal
        with probability p_failure (at `severity`); exec_noise: actuation noise on every executed action."""
        self.exec_noise, self.noise, self.failure, self.severity, self.p_failure = float(exec_noise), bool(noise), failure, float(severity), float(p_failure)
        self.max_retries = max_retries
        self.name = "scripted" + (f"_noise{exec_noise}" if exec_noise else "") + (f"_{failure}{p_failure}" if failure and p_failure else "")
        self.gen, self._done, self._weak = None, False, False
        self.injected = []

    def reset(self, env, seed=None):
        seed = int(seed or 0)
        self.ex = env.task.make_expert(env, failure=None, seed=seed, noise=self.noise, bimanual=False)
        if self.max_retries is not None and hasattr(self.ex, "MAX_RETRIES"):
            self.ex.MAX_RETRIES = int(self.max_retries)
        self.rng = np.random.default_rng([seed, 31337])
        self.adapter = adapter_for(env)
        self.gen, self._done, self._weak, self.injected = None, False, False, []

    def start(self, env, k, sg, plan):
        ex = self.ex
        ex._bumped = ex._stalled = ex._dropped = False
        ex.recovering = False
        ex.failure = None
        if self._weak:  # a weak-grip injection changed the regime: restore it for the next subgoal
            env.set_regime({"grip_force": env.scenario.regime.get("grip_force", 1.0)})
            self._weak = False
        if self.failure and self.rng.random() < self.p_failure:
            kind = str(self.rng.choice(GENERIC_FAILURES)) if self.failure == "any" else self.failure
            fs = sample_failure(kind, self.severity, self.rng, env.task, env.scenario.task)
            fs.params["subgoal"] = self.adapter.expert_subgoal_id(k, sg)
            ex.failure = fs
            self._weak = kind == "weak_grip"
            self.injected.append(dict(k=int(k), kind=kind, t=round(env.time, 3)))
        self.gen = self.adapter.expert_run(ex, k, sg, plan)
        self._done = False

    def act(self, env):
        a = None
        if self.gen is not None and not self._done:
            try:
                a = next(self.gen)
            except StopIteration:
                self._done = True
        if a is None:
            a = self.ex._action()
        return add_exec_noise(a, env, self.rng, self.exec_noise)

    @property
    def done(self):
        return self._done

    def finish(self, env):
        self.ex.failure = None
        return self.ex.go_home()


class LearnedExecutor(Executor):
    name = "learned"

    def __init__(self, path_or_model, device="cpu", name=None):
        from ..learn.policy import load_policy

        self.pol = load_policy(path_or_model, device=device) if isinstance(path_or_model, (str, Path)) else path_or_model
        self.device = device
        cfg = self.pol.cfg
        self.needs_render, self.privileged = cfg.obs_mode == "image", cfg.obs_mode == "state"
        self.name = name or f"{cfg.arch}_{cfg.obs_mode}"
        self.runner, self._env_id = None, None
        self.k, self.sg, self.plan = 0, None, []

    def reset(self, env, seed=None):
        from ..learn.policy import PolicyRunner

        self.adapter = adapter_for(env)
        if self.runner is None or self._env_id != id(env):
            self.runner = PolicyRunner(self.pol, env, device=self.device, seed=int(seed or 0), features=self._features)
            self._env_id = id(env)
        self.runner.reset(seed=int(seed or 0))

    def _features(self, env):
        return self.adapter.features(env, env.privileged(), self.k, self.sg, self.plan)

    def start(self, env, k, sg, plan):
        self.k, self.sg, self.plan = int(k), sg, list(plan)
        self.runner.queue = []  # replan at a subgoal switch

    def act(self, env):
        return self.runner.act()


class HierarchicalPolicy(Policy):
    def __init__(self, planner=None, executor=None, budget_s=None, on_fail="skip", max_retries=1, settle_s=0.5, go_home=True, name=None):
        self.planner = planner or OraclePlanner()
        self.executor = executor or ScriptedExecutor()
        self.budget_s, self.on_fail, self.max_retries, self.settle_s, self.go_home = budget_s, on_fail, int(max_retries), float(settle_s), bool(go_home)
        self.privileged = self.planner.privileged or self.executor.privileged
        self.needs_render = getattr(self.executor, "needs_render", False)
        opts = ("" if on_fail == "skip" else f":{on_fail}" + (f"{max_retries}" if on_fail == "retry" and max_retries != 1 else "")) + (f":budget{budget_s}" if budget_s else "")
        self.name = name or f"hier[{self.planner.name}+{self.executor.name}]{opts}"
        self.plan, self.log, self._finished = [], [], True

    def reset(self, env, seed=None):
        self.adapter = adapter_for(env)
        self.planner.reset(env, seed)
        self.executor.reset(env, seed)
        self.plan = list(self.planner.plan(env, seed))
        self.oracle = self.adapter.oracle(env)
        self.plan_valid, self.plan_reason = self.adapter.valid(env, self.plan)
        self.diff = plan_diff(self.plan, self.oracle, self.adapter.ordered)
        self.log, self.i, self.attempt, self._finished, self._fin_gen = [], 0, 0, False, None
        self.budget = self.budget_s or 1.5 * env.episode_max_s / max(1, len(self.oracle))
        self._start(env)

    def _start(self, env):
        if self.i >= len(self.plan):
            self._fin_gen = self.executor.finish(env) if self.go_home else None
            self._finished = self._fin_gen is None
            return
        sg = self.plan[self.i]
        self.executor.start(env, self.i, sg, self.plan)
        self.t0, self._t_ok = env.time, None
        self.log.append(dict(k=self.i, obj=int(sg.obj), goal=sg.to_json()["goal"], label=self.adapter.describe(env, sg), attempt=self.attempt,
                             t_start=round(env.time, 3), t_end=None, achieved=False, timeout=False))

    def act(self, env):
        if self._fin_gen is not None:
            try:
                return next(self._fin_gen)
            except StopIteration:
                self._fin_gen, self._finished = None, True
        if self._finished:
            return self.hold(env)
        sg = self.plan[self.i]
        ok = self.adapter.achieved(env, sg, env.privileged())
        if ok and self._t_ok is None:
            self._t_ok = env.time
        elif not ok:
            self._t_ok = None
        settled = self._t_ok is not None and env.time - self._t_ok >= self.settle_s
        done = self.executor.done or (settled and not self.executor.privileged)
        timeout = env.time - self.t0 > self.budget
        if done or timeout:
            rec = self.log[-1]
            rec.update(t_end=round(env.time, 3), achieved=bool(ok), timeout=bool(timeout and not done))
            if ok:
                self.i, self.attempt = self.i + 1, 0
            elif self.on_fail == "retry" and self.attempt < self.max_retries:
                self.attempt += 1
            elif self.on_fail == "stop":
                self._finished = True
                return self.hold(env)
            else:
                self.i, self.attempt = self.i + 1, 0
            self._start(env)
            if self._fin_gen is not None or self._finished:
                return self.act(env)
        return self.executor.act(env)

    @property
    def finished(self):
        return self._finished

    def finalize(self, env):
        """Close the subgoal still in flight when the episode ends (e.g. success detected mid-retreat)."""
        if self.log and self.log[-1]["t_end"] is None and self.i < len(self.plan):
            self.log[-1].update(t_end=round(env.time, 3), achieved=bool(self.adapter.achieved(env, self.plan[self.i], env.privileged())), in_flight=True)

    def info(self):
        n_ok = sum(1 for r in self.log if r["achieved"])
        return dict(plan=[s.to_json() for s in self.plan], plan_labels=[r["label"] for r in self.log[: len(self.plan)]], plan_valid=bool(self.plan_valid),
                    plan_reason=self.plan_reason, plan_diff=self.diff, n_plan=len(self.plan), hier_log=self.log, n_subgoals_ok=n_ok,
                    injected=getattr(self.executor, "injected", []), planner_info=self.planner.info())
