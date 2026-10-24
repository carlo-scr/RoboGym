"""Inference-time steering of a frozen policy with a latent world model, and the RQ1 oracle ranking.

At each replanning point: sample K candidate chunks from the policy (seeded per replan and candidate, so
candidate 0 is the base policy's choice), imagine each with the WM from the current latent context,
score it (lower = better), execute the selected one; abstain when no candidate passes the threshold.
The oracle ranking executes every candidate from the restored simulator state and lets the base policy
finish, giving ground truth for how well a score ranks candidates.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..env import ACT_DIM
from .wm import LatentWM, wm_features


class FeatureTracker:
    """Renders the WM cameras, encodes them and keeps the last H (x, action) pairs."""

    def __init__(self, encoder, cams, n_arms=1, H=2, hw=(72, 96)):
        self.enc, self.cams, self.n_arms, self.H, self.hw = encoder, list(cams), n_arms, H, hw
        self.reset()

    def reset(self):
        self.xs, self.acts = [], []

    def observe(self, env):
        frames = np.stack([env.render(c, self.hw[1], self.hw[0]) for c in self.cams])
        z = self.enc.encode(frames).reshape(1, -1)
        x = wm_features(z, env.observe()[None], self.n_arms)[0]
        self.xs.append(x)
        self.xs = self.xs[-self.H :]
        return x

    def record_action(self, a):
        self.acts.append(np.asarray(a, np.float32))
        self.acts = self.acts[-self.H :]

    def context(self):
        xs = [self.xs[0]] * (self.H - len(self.xs)) + self.xs
        prev = [np.zeros(ACT_DIM * self.n_arms, np.float32)] + self.acts
        ha = prev[-self.H :]
        ha = [ha[0]] * (self.H - len(ha)) + ha
        return np.stack(xs), np.stack(ha)

    def snapshot(self):
        return copy.deepcopy((self.xs, self.acts))

    def restore(self, snap):
        self.xs, self.acts = copy.deepcopy(snap)


class RandomScorer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def score(self, hx, ha, chunks):
        return self.rng.random(len(chunks))


class WMScorer:
    """u(A) = max over the imagined chunk of the WM's predicted std (alea) or ensemble disagreement (epi)."""

    def __init__(self, wm: LatentWM, mode="alea", name=None, device="cpu"):
        self.wm, self.mode, self.device = wm, mode, device
        self.name = name or f"wm_{mode}"

    def per_step(self, hx, ha, chunks):
        K = len(chunks)
        t = lambda v: torch.as_tensor(np.asarray(v, np.float32), device=self.device)  # noqa: E731
        alea, epi = self.wm.imagine(t(hx)[None].repeat(K, 1, 1), t(ha)[None].repeat(K, 1, 1), t(chunks))
        return (alea if self.mode == "alea" else epi).cpu().numpy()

    def score(self, hx, ha, chunks):
        return self.per_step(hx, ha, chunks).max(-1)


class ContrastScorer:
    name = "contrast"

    def __init__(self, s: WMScorer, f: WMScorer, beta=1.0, kappa=np.inf):
        self.s, self.f, self.beta, self.kappa = s, f, beta, kappa

    def score(self, hx, ha, chunks):
        return self.s.score(hx, ha, chunks) - self.beta * np.minimum(self.f.score(hx, ha, chunks), self.kappa)


class ProgressScorer:
    """Privileged oracle-ish scorer for sanity checks: imagined progress is not available, so it ranks
    candidates by the executed-motion heuristic 'bigger step towards the goal' (task features: goal - obj)."""

    name = "progress_heuristic"

    def __init__(self, env):
        self.env = env

    def score(self, hx, ha, chunks):
        return -np.linalg.norm(np.asarray(chunks)[:, :, :3].sum(1), axis=-1)


def select(scores, lam=np.inf, eta=np.inf, rng=None):
    """Tilted selection: lam=0 -> candidate 0 (base policy), lam=inf -> best-of-K; scores > eta are not
    certified; none certified -> -1 (abstain)."""
    rng = rng or np.random.default_rng()
    scores = np.asarray(scores, float)
    ok = scores <= eta
    if not ok.any():
        return -1
    if lam == 0:
        return int(np.flatnonzero(ok)[0])
    z = (scores - scores.mean()) / (scores.std() + 1e-8)
    if np.isinf(lam):
        return int(np.argmin(np.where(ok, z, np.inf)))
    logits = np.where(ok, -lam * z, -np.inf)
    p = np.exp(logits - logits.max())
    return int(rng.choice(len(scores), p=p / p.sum()))


def steered_episode(env, runner, tracker, scorer, scenario=None, seed=0, K=8, lam=np.inf, eta=np.inf, max_s=None, settle_s=1.0, record=()):
    """One episode with steering at every replan. -> dict(success, progress, abstained, decisions, scores)."""
    rng = np.random.default_rng(seed)
    env.reset(scenario, seed=seed if scenario is None else None)
    runner.reset(seed=seed)
    tracker.reset()
    max_s = max_s or env.episode_max_s
    frames = {c: [] for c in record}
    n_dec, t_ok, abst, chosen = 0, None, False, []
    while env.time < max_s and not env.unstable:
        if len(runner.queue) < tracker.H or record:
            tracker.observe(env)
        runner.observe()
        for c in record:
            frames[c].append(env.render(c))
        if not runner.queue:
            chunks = runner.sample_chunks(K)
            if K == 1 and np.isinf(eta):
                k, sc = 0, np.zeros(1)
            else:
                hx, ha = tracker.context()
                sc = scorer.score(hx, ha, np.stack([runner.to_env_actions(c) for c in chunks]))
                k = select(sc, lam, eta, rng)
            n_dec += 1
            if k < 0:
                abst = True
                break
            chosen.append(float(sc[k]))
            runner.set_plan(chunks[k])
        a = runner._env_action(runner.queue.pop(0))
        tracker.record_action(a)
        env.step(a)
        if t_ok is None and env.info()["success"]:
            t_ok = env.time
        if t_ok is not None and env.time - t_ok > settle_s:
            break
    info = env.info()
    return dict(success=bool(info["success"]), progress=float(info["progress"]), abstained=abst, decisions=n_dec, duration=env.time, scores=chosen, frames=frames)


# ---------------------------------------------------------------------- RQ1 oracle ranking
def capture_branches(env, runner, tracker, seeds, at_replans=(2, 5, 8), max_s=None):
    """Run the base policy and capture full (sim, policy history, WM context) snapshots at given replan indices."""
    out = []
    for i, seed in enumerate(seeds):
        r_at = at_replans[i % len(at_replans)]
        env.reset(seed=seed)
        runner.reset(seed=seed)
        if tracker is not None:
            tracker.reset()
        n_plan = 0
        lim = max_s or env.episode_max_s
        while env.time < lim and not env.unstable and not env.info()["success"]:
            if tracker is not None and len(runner.queue) < tracker.H:
                tracker.observe(env)
            runner.observe()
            if not runner.queue:
                if n_plan == r_at:
                    out.append(dict(scenario=env.scenario, seed=seed, sim=env.get_state(), runner=copy.deepcopy((runner.hist_low, runner.hist_img, runner.n_replan)),
                                    tracker=tracker.snapshot() if tracker is not None else None, replan=r_at))
                    break
                runner.set_plan(runner.sample_chunks(1)[0])
                n_plan += 1
            a = runner._env_action(runner.queue.pop(0))
            if tracker is not None:
                tracker.record_action(a)
            env.step(a)
    return out


def _restore(env, runner, tracker, b):
    env.reset(b["scenario"])
    env.set_state(b["sim"])
    runner.reset(seed=b["seed"])
    runner.hist_low, runner.hist_img, runner.n_replan = copy.deepcopy(b["runner"])
    if tracker is not None and b["tracker"] is not None:
        tracker.restore(b["tracker"])


def finish_with_base(env, runner, max_s=None, settle_s=1.0, observed=True):
    """Continue with the base policy; returns (success, progress)."""
    t_ok = None
    first = observed and bool(runner.queue)
    lim = max_s or env.episode_max_s
    while env.time < lim and not env.unstable:
        if first:
            env.step(runner._env_action(runner.queue.pop(0)))
            first = False
        else:
            env.step(runner.act())
        if t_ok is None and env.info()["success"]:
            t_ok = env.time
        if t_ok is not None and env.time - t_ok > settle_s:
            break
    info = env.info()
    return bool(info["success"]), float(info["progress"])


def rq1_ranking(env, runner, tracker, scorers, branches, K=8, max_s=None, log=print):
    """For each branch state: sample K chunks, score with every scorer, execute each candidate from the
    restored state and let the base policy finish. -> (succ [N,K], progress [N,K], scores {name: [N,K]})"""
    N = len(branches)
    succ, prog = np.zeros((N, K), bool), np.zeros((N, K))
    scores = {s.name: np.zeros((N, K)) for s in scorers}
    for i, b in enumerate(branches):
        _restore(env, runner, tracker, b)
        chunks = runner.sample_chunks(K)
        env_chunks = np.stack([runner.to_env_actions(c) for c in chunks])
        hx, ha = tracker.context() if tracker is not None else (None, None)
        for s in scorers:
            scores[s.name][i] = s.score(hx, ha, env_chunks)
        for k in range(K):
            _restore(env, runner, tracker, b)
            runner.n_replan += 1  # the candidates were drawn at this replan; the continuation starts at the next
            runner.set_plan(chunks[k])
            succ[i, k], prog[i, k] = finish_with_base(env, runner, max_s)
        if log:
            log(f"  branch {i + 1}/{N}: {succ[i].sum()}/{K} candidates succeed, progress {prog[i].mean():.2f}")
    return succ, prog, scores


def rq1_metrics(succ, scores, prog=None):
    from .monitor import auroc

    mixed = succ.any(1) & ~succ.all(1)
    res = dict(n_states=len(succ), base_success=float(succ.mean()), frac_mixed=float(mixed.mean()), oracle_success=float(succ.any(1).mean()))
    rows = []
    for n, s in scores.items():
        r = dict(scorer=n)
        if 0 < succ.mean() < 1:
            r["auroc_fail_pooled"] = auroc(s[succ], s[~succ])
        if mixed.any():
            best = s[mixed].argmin(1)
            r["p_success_best"] = float(succ[mixed][np.arange(mixed.sum()), best].mean())
            r["p_success_mean"] = float(succ[mixed].mean())
            r["steering_gain"] = r["p_success_best"] - r["p_success_mean"]
            r["auroc_fail_within"] = float(np.nanmean([auroc(s[i][succ[i]], s[i][~succ[i]]) for i in np.flatnonzero(mixed)]))
        if prog is not None:
            r["spearman_progress"] = float(np.nanmean([_spearman(-s[i], prog[i]) for i in range(len(s))]))
        rows.append(r)
    return res, rows


def _spearman(a, b):
    import pandas as pd

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


# ---------------------------------------------------------------------- Learn-then-Test
def binom_pvalue(n_losses, n, level):
    from scipy.stats import binom

    return float(binom.cdf(n_losses, n, level))


def ltt_calibrate(lams, harm: dict, abst: dict, eps=0.05, alpha=0.1, delta=0.1):
    """Fixed-sequence LTT over increasing steering strength: certify the largest lam with paired-harm
    rate <= eps and false-abstention rate <= alpha (FWER <= delta)."""
    best, table = None, []
    for lam in lams:
        h, a = np.asarray(harm[lam]), np.asarray(abst[lam])
        p = max(binom_pvalue(int(h.sum()), len(h), eps), binom_pvalue(int(a.sum()), len(a), alpha))
        table.append(dict(lam=lam, harm=h.mean(), abst=a.mean(), p=p))
        if p > delta:
            break
        best = lam
    return dict(lam=0.0 if best is None else best, certified=best is not None, table=table)
