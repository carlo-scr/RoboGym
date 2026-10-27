"""Learned and off-the-shelf policies behind the Policy interface.

* ChunkPolicy: any robogym.learn checkpoint (Diffusion Policy, ACT, MLP; state or image observations)
  executed receding-horizon by PolicyRunner. `K > 1` with a scorer gives best-of-K steering.
* LeRobotPolicy: a LeRobot checkpoint (ACT / Diffusion / VQ-BeT / SmolVLA / pi0) trained on an exported
  RoboGym dataset (robogym.learn.lerobot_io), run through the checkpoint's own processors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..env import ACT_DIM
from .base import Policy


class ChunkPolicy(Policy):
    name = "chunk"

    def __init__(self, path_or_model, device="cpu", name=None, K=1, scorer=None, features=None):
        from ..learn.policy import load_policy

        self.pol = load_policy(path_or_model, device=device) if isinstance(path_or_model, (str, Path)) else path_or_model
        self.device, self.K, self.scorer, self.features = device, int(K), scorer, features
        cfg = self.pol.cfg
        self.needs_render = cfg.obs_mode == "image"
        self.privileged = cfg.obs_mode == "state"  # task features are simulator ground truth
        base = f"{cfg.arch}_{cfg.obs_mode}"
        run = Path(path_or_model).parent.name if isinstance(path_or_model, (str, Path)) else ""
        self.name = name or (base if run in ("", base) else f"{base}_{run}")  # runs/bench/<task>/act_state/policy.pt -> act_state
        self.runner, self._env_id = None, None

    def reset(self, env, seed=None):
        from ..learn.policy import PolicyRunner

        if self.runner is None or self._env_id != id(env):
            self.runner = PolicyRunner(self.pol, env, device=self.device, seed=int(seed or 0), features=self.features)
            self._env_id = id(env)
        self.runner.reset(seed=int(seed or 0))

    def act(self, env):
        r = self.runner
        if self.K <= 1 or self.scorer is None:
            return r.act()
        r.observe()
        if not r.queue:
            chunks = r.sample_chunks(self.K)
            sc = self.scorer.score(None, None, np.stack([r.to_env_actions(c) for c in chunks]))
            r.set_plan(chunks[int(np.argmin(sc))])
        return r._env_action(r.queue.pop(0))


class LeRobotPolicy(Policy):
    """A LeRobot checkpoint (ACT / Diffusion / VQ-BeT / SmolVLA / pi0, trained with
    robogym.learn.lerobot_io) run closed loop: the observation is built the way the export wrote it
    (robot state, the cameras at the training resolution, the privileged environment state if the
    variant had it, the goal text), passed through the checkpoint's own pre-processor, `select_action`,
    and post-processor. Actions are the env's delta actions (`action_scale` rescales them)."""

    name = "lerobot"

    def __init__(self, ckpt, device="cpu", action_scale=1.0, name=None, task_text=None):
        from ..learn.lerobot_io import load_lerobot

        self.policy, self.pre, self.post, self.export = load_lerobot(ckpt, device=device)
        self.device, self.action_scale, self.task_text = device, float(action_scale), task_text
        self.cams = tuple(self.export.get("cams", ()))
        self.env_state = bool(self.export.get("env_state", False))
        self.hw = self.export.get("hw")
        self.needs_render = len(self.cams) > 0
        self.privileged = self.env_state
        kind = getattr(getattr(self.policy, "config", None), "type", "policy")
        run = Path(ckpt).name if Path(ckpt).name not in ("pretrained_model", "last") else Path(ckpt).parents[2].name
        self.name = name or (f"lerobot_{kind}" if run == kind else f"lerobot_{kind}_{run}")

    def reset(self, env, seed=None):
        self.policy.reset()
        self.text = self.task_text or env.task.goal_description(env.scenario.task)

    def info(self):
        return dict(lerobot_type=getattr(getattr(self.policy, "config", None), "type", None), cams=list(self.cams), env_state=self.env_state)

    def _obs(self, env):
        import torch

        obs = {"observation.state": torch.as_tensor(env.observe(), dtype=torch.float32)[None]}
        if self.env_state:
            obs["observation.environment_state"] = torch.as_tensor(env.task_features(), dtype=torch.float32)[None]
        for c in self.cams:
            img = env.render(c, int(self.hw[1]), int(self.hw[0])) if self.hw else env.render(c)
            obs[f"observation.images.{c}"] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float() / 255.0
        obs["task"] = [self.text]
        return obs

    def act(self, env):
        import torch

        obs = self.pre(self._obs(env))
        with torch.inference_mode():
            a = self.policy.select_action(obs)
        a = self.post(a)
        a = np.asarray(a.detach().cpu().numpy(), np.float64).reshape(-1) * self.action_scale
        if a.size != ACT_DIM * len(env.arms):
            raise ValueError(f"LeRobot policy returned {a.size} dims, expected {ACT_DIM * len(env.arms)}")
        return a
