"""Diffusion Policy (Chi et al. 2023): 1D-conv U-Net over action chunks with FiLM conditioning, DDPM
training, DDIM sampling. Ported from RobotPluggingSim and generalised to n arms and any task-feature
size.

Observation modes: "state" = proprio (28 n) + privileged task features (F); "image" = proprio + camera
frames (small CNN with spatial softmax per camera). Action mode "rel" (default): chunks of controller
targets relative to the current controller target, per arm [dpos(3), rotvec(3), grip, compliant] (8 n),
converted to env delta actions at execution time; the current target is part of the observation. "abs"
predicts absolute targets [pos(3), rot6d(6), grip, compliant] (11 n); "delta" predicts env actions.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import geometry as G
from ..env import ACT_DIM, STATE_PER_ARM
from .data import Normalizer


def quat_to_rot6(q):
    q = np.asarray(q, np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    return np.concatenate([c0, c1], -1).astype(np.float32)


def rot6_to_quat(r6):
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    x = a / (np.linalg.norm(a) + 1e-9)
    b = b - (x @ b) * x
    y = b / (np.linalg.norm(b) + 1e-9)
    return G.mat_to_quat(np.stack([x, y, np.cross(x, y)], 1))


def binary_dims(mode, n_arms):
    per = {"delta": (6, 7, 8), "rel": (6, 7, 8), "abs": (9, 10, 11)}[mode]
    return [per[2] * i + per[0] for i in range(n_arms)] + [per[2] * i + per[1] for i in range(n_arms)]


def intended_targets(ee_target, actions, n_arms):
    """Controller targets the actions aim for: pos [T,n,3], quat [T,n,4], grip/compliant [T,n,2]."""
    T = len(actions)
    P, Q, B = np.zeros((T, n_arms, 3)), np.zeros((T, n_arms, 4)), np.zeros((T, n_arms, 2))
    for t in range(T):
        for i in range(n_arms):
            e, a = ee_target[t, 8 * i : 8 * i + 8], actions[t, 8 * i : 8 * i + 8]
            P[t, i] = e[:3] + a[:3]
            Q[t, i] = G.quat_mul(G.quat_from_rotvec(a[3:6]), e[3:7])
            B[t, i] = a[6], a[7]
    return P, Q, B


def targets_after(ee_target, actions, n_arms):
    P, Q, B = intended_targets(ee_target, actions, n_arms)
    out = np.zeros((len(actions), 11 * n_arms), np.float32)
    for t in range(len(actions)):
        for i in range(n_arms):
            out[t, 11 * i : 11 * i + 11] = np.concatenate([P[t, i], quat_to_rot6(Q[t, i]), B[t, i]])
    return out


def rel_chunk_labels(ee_target, actions, horizon, n_arms):
    """[T, H, 8 n]: targets t..t+H-1 relative to the controller target at t (per arm dpos, rotvec, grip, compliant)."""
    P, Q, B = intended_targets(ee_target, actions, n_arms)
    T = len(actions)
    out = np.zeros((T, horizon, 8 * n_arms), np.float32)
    for t in range(T):
        for k in range(horizon):
            j = min(t + k, T - 1)
            for i in range(n_arms):
                ref_p, ref_q = ee_target[t, 8 * i : 8 * i + 3], ee_target[t, 8 * i + 3 : 8 * i + 7]
                out[t, k, 8 * i : 8 * i + 3] = P[j, i] - ref_p
                out[t, k, 8 * i + 3 : 8 * i + 6] = G.quat_error(Q[j, i], ref_q)
                out[t, k, 8 * i + 6 : 8 * i + 8] = B[j, i]
    return out


def rel_to_abs(chunk, targets, arms):
    out = np.zeros((len(chunk), 11 * len(arms)), np.float32)
    for i, s in enumerate(arms):
        p0, q0 = targets[s]
        for k, x in enumerate(chunk):
            y = x[8 * i : 8 * i + 8]
            q = G.quat_mul(G.quat_from_rotvec(y[3:6]), q0)
            out[k, 11 * i : 11 * i + 11] = np.concatenate([p0 + y[:3], quat_to_rot6(q), y[6:8]])
    return out


def target_obs(ee_target, n_arms):
    """[T,8n] (pos, quat, grip per arm) -> [T,10n] (pos, rot6, grip per arm)."""
    parts = []
    for i in range(n_arms):
        e = ee_target[:, 8 * i : 8 * i + 8]
        parts += [e[:, :3], quat_to_rot6(e[:, 3:7]), e[:, 7:8]]
    return np.concatenate(parts, 1).astype(np.float32)


def env_target_obs(env):
    e = np.concatenate([np.concatenate([env.targets[s][0], env.targets[s][1], [env.grip_cmd[s]]]) for s in env.arms])
    return target_obs(e[None], len(env.arms))[0]


def abs_to_env_action(abs_a, targets, arms):
    a = np.zeros(ACT_DIM * len(arms))
    for i, s in enumerate(arms):
        x = abs_a[11 * i : 11 * i + 11]
        p0, q0 = targets[s]
        a[8 * i : 8 * i + 3] = x[:3] - p0
        a[8 * i + 3 : 8 * i + 6] = G.quat_error(rot6_to_quat(x[3:9]), q0)
        a[8 * i + 6], a[8 * i + 7] = x[9], x[10]
    return a


def abs_chunk_to_deltas(chunk, targets, arms):
    tg = {s: (np.asarray(targets[s][0]).copy(), np.asarray(targets[s][1]).copy()) for s in arms}
    out = []
    for x in chunk:
        out.append(abs_to_env_action(x, tg, arms))
        for i, s in enumerate(arms):
            y = x[11 * i : 11 * i + 11]
            tg[s] = (y[:3].copy(), rot6_to_quat(y[3:9]))
    return np.asarray(out)


@dataclass
class PolicyCfg:
    arch: str = "dp"  # "dp" (Diffusion Policy) | "act" (Action Chunking Transformer) | "mlp" (chunked behaviour cloning)
    obs_mode: str = "state"  # "state" | "image"
    action_mode: str = "rel"  # "rel" | "abs" | "delta"
    n_arms: int = 1
    task_dim: int = 0  # privileged task-feature dim (state mode; read from the data)
    cams: tuple = ("right_wrist", "front")
    img_hw: tuple = (72, 96)  # policy camera size (must match the frames fed at training)
    n_obs: int = 2
    horizon: int = 16
    n_act: int = 8
    down_dims: tuple = (64, 128, 256)
    cond_dim: int = 256
    img_feat: int = 64
    diffusion_steps: int = 100
    prediction: str = "sample"
    ddim_steps: int = 10
    ddim_eta: float = 1.0
    # ACT / MLP (robogym.learn.archs)
    d_model: int = 256
    n_heads: int = 8
    n_enc: int = 3
    n_dec: int = 3
    latent: int = 32
    kl_weight: float = 10.0
    z_sample: bool = False  # ACT: draw the style latent from the prior per candidate (else z = 0, deterministic)
    mlp_hidden: int = 1024

    @property
    def state_dim(self):
        return STATE_PER_ARM * self.n_arms

    @property
    def act_dim(self):
        return {"abs": 11, "rel": 8, "delta": 8}[self.action_mode] * self.n_arms

    @property
    def tgt_dim(self):
        return 10 * self.n_arms


# ---------------------------------------------------------------------- networks
class SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        e = t[:, None].float() * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


class Conv1dBlock(nn.Module):
    def __init__(self, i, o, k=5, g=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(g, o), nn.Mish())

    def forward(self, x):
        return self.net(x)


class CondResBlock(nn.Module):
    def __init__(self, i, o, cond, k=5):
        super().__init__()
        self.b1, self.b2 = Conv1dBlock(i, o, k), Conv1dBlock(o, o, k)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, 2 * o))
        self.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x)
        s, b = self.film(c).unsqueeze(-1).chunk(2, 1)
        h = self.b2(h * (1 + s) + b)
        return h + self.res(x)


class CondUnet1D(nn.Module):
    def __init__(self, act_dim, cond_dim, dims=(64, 128, 256)):
        super().__init__()
        t_dim = 64
        self.t_emb = nn.Sequential(SinPosEmb(t_dim), nn.Linear(t_dim, 4 * t_dim), nn.Mish(), nn.Linear(4 * t_dim, t_dim))
        c = cond_dim + t_dim
        ch = [act_dim] + list(dims)
        self.down = nn.ModuleList()
        for j, (i, o) in enumerate(zip(ch[:-1], ch[1:])):
            last = j == len(dims) - 1
            self.down.append(nn.ModuleList([CondResBlock(i, o, c), CondResBlock(o, o, c), nn.Conv1d(o, o, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondResBlock(dims[-1], dims[-1], c), CondResBlock(dims[-1], dims[-1], c)])
        self.up = nn.ModuleList()
        rev = list(reversed(dims))
        for i, o in zip(rev[:-1], rev[1:]):
            self.up.append(nn.ModuleList([CondResBlock(2 * i, o, c), CondResBlock(o, o, c), nn.ConvTranspose1d(o, o, 4, 2, 1)]))
        self.final = nn.Sequential(Conv1dBlock(dims[0], dims[0]), nn.Conv1d(dims[0], act_dim, 1))

    def forward(self, x, t, cond):
        c = torch.cat([cond, self.t_emb(t)], -1)
        h = x.transpose(1, 2)
        skips = []
        for r1, r2, ds in self.down:
            h = r2(r1(h, c), c)
            skips.append(h)
            h = ds(h)
        for m in self.mid:
            h = m(h, c)
        for r1, r2, us in self.up:
            h = torch.cat([h, skips.pop()], 1)
            h = us(r2(r1(h, c), c))
        return self.final(h).transpose(1, 2)


class SpatialSoftmax(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        p = F.softmax(x.flatten(2), -1).view(B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)


class ImageEncoder(nn.Module):
    def __init__(self, n_cams, feat=64):
        super().__init__()
        g = lambda c: nn.GroupNorm(8, c)  # noqa: E731
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), g(32), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), g(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), g(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.ss = SpatialSoftmax()
        self.proj = nn.Linear(128, feat)
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, feat))

    def forward(self, imgs):
        B, N = imgs.shape[:2]
        f = self.proj(self.ss(self.cnn(imgs.flatten(0, 1) * 2 - 1))).view(B, N, -1) + self.cam_emb
        return f.flatten(1)


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        low = cfg.state_dim + (cfg.task_dim if cfg.obs_mode == "state" else 0) + (cfg.tgt_dim if cfg.action_mode in ("abs", "rel") else 0)
        self.low_dim = low
        self.act_dim = cfg.act_dim
        img = 0
        if cfg.obs_mode == "image":
            self.img_enc = ImageEncoder(len(cfg.cams), cfg.img_feat)
            img = len(cfg.cams) * cfg.img_feat
        self.obs_mlp = nn.Sequential(nn.Linear(cfg.n_obs * (low + img), cfg.cond_dim), nn.Mish(), nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.net = CondUnet1D(self.act_dim, cfg.cond_dim, cfg.down_dims)
        T = cfg.diffusion_steps
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = torch.clamp(1 - ac[1:] / ac[:-1], max=0.999)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0))
        self.obs_norm = Normalizer()
        self.act_norm = Normalizer()

    def cond(self, low, imgs=None):
        B = low.shape[0]
        parts = [low]
        if self.cfg.obs_mode == "image":
            parts.append(self.img_enc(imgs.flatten(0, 1)).view(B, self.cfg.n_obs, -1))
        return self.obs_mlp(torch.cat(parts, -1).flatten(1))

    def loss(self, low, act, imgs=None):
        B = act.shape[0]
        c = self.cond(low, imgs)
        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=act.device)
        eps = torch.randn_like(act)
        a = self.alphas_cumprod[t].view(B, 1, 1)
        x = a.sqrt() * act + (1 - a).sqrt() * eps
        return F.mse_loss(self.net(x, t, c), act if self.cfg.prediction == "sample" else eps)

    @torch.no_grad()
    def sample(self, low, imgs=None, K=1, steps=None, eta=None, generator=None, noise=None):
        """-> normalised action chunks [B, K, horizon, act_dim]. `noise` [B*K, horizon, act_dim] fixes the
        initial latent (per-candidate seeding for steering experiments)."""
        cfg = self.cfg
        steps, eta = steps or cfg.ddim_steps, cfg.ddim_eta if eta is None else eta
        B = low.shape[0]
        c = self.cond(low, imgs).repeat_interleave(K, 0)
        x = torch.randn((B * K, cfg.horizon, self.act_dim), device=low.device, generator=generator) if noise is None else noise.clone()
        ts = torch.linspace(cfg.diffusion_steps - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(ts):
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=low.device)
            out = self.net(x, torch.full((B * K,), t, device=low.device), c)
            if cfg.prediction == "sample":
                x0 = out.clamp(-1.5, 1.5)
                eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            else:
                eps = out
                x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            sig = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sig**2).clamp(min=0).sqrt() * eps
            if i + 1 < len(ts) and eta > 0:
                x = x + sig * torch.randn(x.shape, device=x.device, generator=generator)
        return x.view(B, K, cfg.horizon, self.act_dim)


def make_model(cfg: PolicyCfg) -> nn.Module:
    """The chunk policy for cfg.arch; every architecture shares the DiffusionPolicy interface
    (cfg, obs_norm, act_norm, loss(low, act, imgs), sample(low, imgs, K, ...) -> [B, K, horizon, act_dim])."""
    if cfg.arch == "dp":
        return DiffusionPolicy(cfg)
    from .archs import ACTPolicy, MLPChunkPolicy

    return {"act": ACTPolicy, "mlp": MLPChunkPolicy}[cfg.arch](cfg)


# ---------------------------------------------------------------------- data
def make_windows(episodes, cfg: PolicyCfg):
    lows, acts, idx = [], [], []
    n = cfg.n_arms
    for e, ep in enumerate(episodes):
        parts = [ep["state"]] + ([ep["task"]] if cfg.obs_mode == "state" else [])
        acts_lbl = ep["action_expert"] if ep.get("action_expert") is not None else ep["action"]
        if cfg.action_mode == "rel":
            parts.append(target_obs(ep["ee_target"], n))
            low = np.concatenate(parts, 1)
            R = rel_chunk_labels(ep["ee_target"], acts_lbl, cfg.horizon, n)
            for t in range(len(R)):
                o = [max(0, t - cfg.n_obs + 1 + i) for i in range(cfg.n_obs)]
                lows.append(low[o])
                acts.append(R[t])
                idx.append((e, o))
            continue
        if cfg.action_mode == "abs":
            parts.append(target_obs(ep["ee_target"], n))
            A = targets_after(ep["ee_target"], acts_lbl, n)
            pad = np.repeat(A[-1:], cfg.horizon, 0)
        else:
            A = acts_lbl
            pad = np.zeros((cfg.horizon, A.shape[1]), np.float32)
            bd = binary_dims("delta", n)
            pad[:, bd] = A[-1, bd]
        low = np.concatenate(parts, 1)
        Ap = np.concatenate([A, pad], 0)
        for t in range(len(A)):
            o = [max(0, t - cfg.n_obs + 1 + i) for i in range(cfg.n_obs)]
            lows.append(low[o])
            acts.append(Ap[t : t + cfg.horizon])
            idx.append((e, o))
    return np.asarray(lows, np.float32), np.asarray(acts, np.float32), idx


def train_policy(episodes, cfg: PolicyCfg, steps=3000, bs=256, lr=3e-4, device="cpu", log_every=250, log=print, seed=0, ckpt_dir=None, ckpt_every=1000,
                 budget_s=None):
    """ckpt_dir: write `ckpt.pt` every ckpt_every steps and resume from it (same `steps`) if it exists;
    budget_s: stop after the next checkpoint once this much wall time has passed (the returned history is then
    shorter than `steps`: call again to continue). Interrupted runs lose at most ckpt_every steps."""
    from .ckpt import load_ckpt, rng_state, save_ckpt, set_rng_state

    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg.obs_mode == "state" and cfg.task_dim == 0:
        cfg.task_dim = episodes[0]["task"].shape[1]
    lows, acts, idx = make_windows(episodes, cfg)
    pol = make_model(cfg)
    pol.obs_norm.fit(lows)
    pol.act_norm.fit(acts, robust=cfg.action_mode == "rel")
    L = torch.as_tensor(pol.obs_norm.norm(lows), device=device)
    A = torch.as_tensor(pol.act_norm.norm(acts), device=device)
    frames = None
    if cfg.obs_mode == "image":
        frames = [{c: torch.as_tensor(ep["frames"][c]) for c in cfg.cams} for ep in episodes]
    pol = pol.to(device)
    ema = copy.deepcopy(pol).eval()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    hist, t0 = [], time.time()
    n = len(L)
    p_ema, p_pol = list(ema.parameters()), list(pol.parameters())
    start = 0
    ck = Path(ckpt_dir) / "ckpt.pt" if ckpt_dir else None
    st = load_ckpt(ck, device) if ck else None
    if st is not None and st["steps"] == steps:
        pol.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        hist = [torch.tensor(h, device=device) for h in st["hist"]]
        start = int(st["step"])
        set_rng_state(st.get("rng"))
        if log:
            log(f"[policy] resumed from {ck} at step {start}/{steps}")
    for it in range(start, steps):
        j = torch.randint(0, n, (bs,), device=device) if frames is None else np.random.randint(0, n, bs)
        imgs = None
        if frames is not None:
            imgs = torch.stack([torch.stack([torch.stack([frames[idx[k][0]][c][o] for c in cfg.cams]) for o in idx[k][1]]) for k in j])
            imgs = (imgs.permute(0, 1, 2, 5, 3, 4).float() / 255).to(device)
        loss = pol.loss(L[j], A[j], imgs)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        d = min(0.999, (1 + it) / (10 + it))
        with torch.no_grad():
            torch._foreach_mul_(p_ema, d)
            torch._foreach_add_(p_ema, p_pol, alpha=1 - d)
        hist.append(loss.detach())
        if log and (it % log_every == 0 or it == steps - 1):
            log(f"[policy] step {it:5d}  loss {torch.stack(hist[-log_every:]).mean().item():.4f}  ({time.time() - t0:.0f}s)")
        if ck is not None and ((it + 1) % ckpt_every == 0 or it == steps - 1):
            save_ckpt(ck, dict(step=it + 1, steps=steps, model=pol.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                               hist=[float(h) for h in hist], rng=rng_state(), cfg=asdict(cfg)))
            if budget_s is not None and time.time() - t0 > budget_s and it < steps - 1:
                if log:
                    log(f"[policy] time budget reached at step {it + 1}/{steps}; checkpoint written, call again to continue")
                break
    ema.obs_norm, ema.act_norm = pol.obs_norm, pol.act_norm
    return ema.eval(), torch.stack(hist).cpu().numpy().tolist()


def save_policy(pol: DiffusionPolicy, path):
    torch.save(dict(cfg=asdict(pol.cfg), state=pol.state_dict(), obs_norm=pol.obs_norm.state_dict(), act_norm=pol.act_norm.state_dict()), path)


def load_policy(path, device="cpu") -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    pol = make_model(cfg)
    pol.load_state_dict(ck["state"])
    pol.obs_norm, pol.act_norm = Normalizer.from_state(ck["obs_norm"]), Normalizer.from_state(ck["act_norm"])
    return pol.to(device).eval()


# ---------------------------------------------------------------------- closed loop
class PolicyRunner:
    """Receding-horizon execution of a DiffusionPolicy in a RoboGymEnv (keeps the obs history).
    Candidate chunks are seeded per (episode, replan, candidate) through `seed_for` so that candidate 0
    is identical for the base policy and any steering method (see robogym.research.seed_for)."""

    def __init__(self, pol, env, device="cpu", seed=0, features=None):
        """`features(env) -> np.ndarray` replaces env.task_features() in the observation (state mode): a
        hierarchical executor passes the *planned* subgoal's features instead of the env's own."""
        self.pol, self.env, self.device, self.seed = pol, env, device, seed
        self.features = features
        self.arms = tuple(env.arms)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self, seed=None):
        self.hist_low, self.hist_img, self.queue, self.n_replan = [], [], [], 0
        if seed is not None:
            self.seed = seed
            self.gen.manual_seed(seed)

    def _obs(self):
        env, cfg = self.env, self.pol.cfg
        parts = [env.observe()] + ([self.features(env) if self.features else env.task_features()] if cfg.obs_mode == "state" else [])
        if cfg.action_mode in ("abs", "rel"):
            parts.append(env_target_obs(env))
        self.hist_low.append(np.concatenate(parts))
        if cfg.obs_mode == "image":
            self.hist_img.append(np.stack([env.render(c, cfg.img_hw[1], cfg.img_hw[0]) for c in cfg.cams]))
        self.hist_low, self.hist_img = self.hist_low[-cfg.n_obs :], self.hist_img[-cfg.n_obs :]

    def _tensors(self):
        cfg = self.pol.cfg
        H = [self.hist_low[0]] * (cfg.n_obs - len(self.hist_low)) + self.hist_low
        low = torch.as_tensor(self.pol.obs_norm.norm(np.stack(H))[None], device=self.device)
        imgs = None
        if cfg.obs_mode == "image":
            hist = [self.hist_img[0]] * (cfg.n_obs - len(self.hist_img)) + self.hist_img
            imgs = torch.as_tensor(np.stack(hist)[None]).permute(0, 1, 2, 5, 3, 4).float().to(self.device) / 255
        return low, imgs

    def observe(self):
        self._obs()

    def sample_chunks(self, K, seeded=True):
        """K candidate chunks (denormalised, binary dims thresholded) in absolute target format
        [K, horizon, 11 n] (rel / abs) or env actions [K, horizon, 8 n] (delta)."""
        from ..research import seed_for

        low, imgs = self._tensors()
        noise = None
        if seeded:  # candidate k of replan r of this episode always draws the same initial latent
            cfg = self.pol.cfg
            noise = torch.stack([torch.randn((cfg.horizon, self.pol.act_dim), generator=torch.Generator().manual_seed(seed_for(self.seed, self.n_replan, k))) for k in range(K)]).to(self.device)
            self.gen.manual_seed(seed_for(self.seed, self.n_replan, 999))
        x = self.pol.sample(low, imgs, K=K, generator=self.gen, noise=noise)[0].cpu().numpy()
        mode = self.pol.cfg.action_mode
        a = self.pol.act_norm.denorm(x)
        bd = binary_dims(mode, self.pol.cfg.n_arms)
        a[..., bd] = (a[..., bd] > 0.5).astype(np.float32)
        if mode == "rel":
            a = np.stack([rel_to_abs(c, self.env.targets, self.arms) for c in a])
        self.n_replan += 1
        return a

    def to_env_actions(self, chunk):
        return abs_chunk_to_deltas(chunk, self.env.targets, self.arms) if self.pol.cfg.action_mode in ("abs", "rel") else np.asarray(chunk)

    def _env_action(self, x):
        return abs_to_env_action(x, self.env.targets, self.arms) if self.pol.cfg.action_mode in ("abs", "rel") else x

    def act(self):
        self._obs()
        if not self.queue:
            self.queue = list(self.sample_chunks(1)[0][: self.pol.cfg.n_act])
        return self._env_action(self.queue.pop(0))

    def set_plan(self, chunk):
        self.queue = list(chunk[: self.pol.cfg.n_act])


def rollout(env, runner: PolicyRunner, scenario=None, seed=None, max_s=None, settle_s=1.0, cams=(), fps_stride=1):
    """Run the policy until success (+ settle) or timeout. -> dict(success, progress, duration, frames)."""
    env.reset(scenario, seed=seed)
    runner.reset(seed=seed if seed is not None else runner.seed)
    max_s = max_s or env.episode_max_s
    frames = {c: [] for c in cams}
    t_ok = None
    while env.time < max_s and not env.unstable:
        if cams and env.step_count % fps_stride == 0:
            for c in cams:
                frames[c].append(env.render(c))
        env.step(runner.act())
        if t_ok is None and env.info()["success"]:
            t_ok = env.time
        if t_ok is not None and env.time - t_ok > settle_s:
            break
    info = env.info()
    return dict(success=bool(info["success"]), progress=float(info["progress"]), stage=info["stage_name"], duration=env.time, t_success=t_ok, frames=frames,
                unstable=env.unstable)
