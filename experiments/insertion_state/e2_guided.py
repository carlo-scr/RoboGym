"""E2: does the contact state guide the plug in? Retrying insertion controllers that share one in/out
detector (logistic model from E1, 100 ms after contact) and the same "partial -> push deeper" rule, and
differ in what they do after a miss:
  blind    one push, no retry (E1's controller)
  spiral   sunflower search around the first aim (r_n = 1 mm * sqrt(n), golden angle)
  learned  shift the aim by the correction predicted from the contact signals (ridge model from E1)
  belief   grid belief over the hole offset (negative information + predicted correction, belief.py);
           aim where the posterior probability of entering is highest (also for the first try)
  slide    belief, but instead of retracting after a miss, slide to the new aim under a light axial
           preload and let the plug drop in; retract only if it does not
  oracle   shift the aim by the true lateral miss at contact (privileged upper bound for one-shot correction)
Perception error: lateral N(0, sigma) per axis, axial U(-2, 2) mm. New scenes (not in E1).
Results are written per scene (`<out>/scene_<seed>.json`); rerunning skips finished scenes."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from robogym import geometry as G

sys.path.insert(0, str(Path(__file__).parent))
import analyze_e1 as A  # noqa: E402
from belief import CaptureMap, HoleBelief  # noqa: E402
from common import JACKS, MAX_LEAD, Attempt, Probe, make_env, outcome, prepare  # noqa: E402
from e1_offset_sweep import pack  # noqa: E402

OUT = A.OUT
K = 1  # decide 100 ms after contact (steps c0, c0 + 1)
GOLDEN = np.pi * (3 - np.sqrt(5))
STRATS = ("blind", "spiral", "learned", "belief", "slide", "oracle")
SLIDE_LEAD = 0.0015  # axial preload while sliding (force-limited lead, stiff lateral tracking)
SLIDE_SPEED = 0.010  # m/s
DROP = 0.0015  # axial advance beyond the contact position that counts as "dropped in"


def train_models(src="e1"):
    """Linear in/out + 4-class models, ridge correction with cross-validated residual covariances per
    class, and the capture map P(entry | tip offset at contact), all from E1 attempts."""
    res = A.load_runs(src)
    X, y, corr, scene, _aux = A.build(res, K)
    sd = A.Std().fit(X)
    W_bin = A.fit_logreg(sd(X), (y != 0).astype(int), 2)
    W4 = A.fit_logreg(sd(X), y, 4)
    m = np.isin(y, [1, 2])
    B = A.fit_ridge(sd(X[m]), corr[m], lam=10.0)
    resid = np.zeros_like(corr)
    for te in A.folds(scene):
        mt = ~te & m
        Bf = A.fit_ridge(A.Std().fit(X[~te])(X[mt]), corr[mt], lam=10.0)
        resid[te] = A.predict_ridge(Bf, A.Std().fit(X[~te])(X[te])) - corr[te]
    S = {c: np.cov(resid[y == i].T) + np.eye(2) * 1e-8 for c, i in (("ramp", 1), ("face", 2))}
    keep = y != 3  # the capture map is about plug contacts, not blocked grippers
    cap = CaptureMap(-corr[keep], y[keep] == 0)
    # systematic pre-contact drift of the compliant controller (tip offset at contact minus the aim, error
    # removed): every strategy compensates it in its first aim, as a calibrated controller would
    drift = np.array([-c - a["err"][1:] for c, a in zip(corr, _aux)]).mean(0)
    return dict(mu=sd.mu, sd=sd.sd, W_bin=W_bin, W4=W4, B=B, S_ramp=S["ramp"], S_face=S["face"], cap_P=cap.P, cap_half=cap.half, cap_res=cap.res,
                drift_mean=drift)


def capture_map(models):
    cap = CaptureMap.__new__(CaptureMap)
    cap.P, cap.half, cap.res = models["cap_P"], models["cap_half"], models["cap_res"]
    cap.n = cap.P.shape[0]
    return cap


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class Guided(Attempt):
    def __init__(self, *a, strategy="belief", models=None, max_tries=6, budget_s=9.0, noise_seed=0, prior_mean=(0.0, 0.0), prior_sigma=None,
                 **kw):
        super().__init__(*a, **kw)
        self.strategy, self.models = strategy, models
        self.max_tries = 1 if strategy == "blind" else max_tries
        self.budget_s = budget_s
        self.rng = np.random.default_rng([noise_seed, 17])
        self.prior_mean = np.asarray(prior_mean, float)
        self.prior_sigma = prior_sigma
        self.Fn, self.Mn, self.taun = [], [], []

    # noisy sensor stream, kept in step with self.rows
    def log(self, phase, tp_cmd):
        super().log(phase, tp_cmd)
        r = self.rows[-1]
        self.Fn.append(r["F"] + self.rng.normal(0, A.SIG_F, 3))
        self.Mn.append(r["M"] + self.rng.normal(0, A.SIG_M, 3))
        self.taun.append(r["tau"] + self.rng.normal(0, A.SIG_TAU, 7))

    def stiff_to(self, goal, T=0.4, phase="move"):
        start = self.rows[-1]["cmd"] if self.rows else self.to_bel(self.tip0)
        n = max(2, int(round(T / self.env.dt)))
        for i in range(1, n + 1):
            u = G.min_jerk(i / n)
            self.step(self.tcp_target_for_tip(start + u * (np.asarray(goal) - start)), False, phase)

    def decide(self, c0):
        ts = pack(self.rows)
        F, M, tau = np.array(self.Fn), np.array(self.Mn), np.array(self.taun)
        x = A.features(F, M, tau, ts, c0, K)[None]
        m = self.models
        xs = (x - m["mu"]) / m["sd"]
        out = A.predict_logreg(m["W_bin"], xs)[0] == 1
        p4 = softmax(np.hstack([xs, np.ones((1, 1))]) @ m["W4"])[0]
        p_ramp = p4[1] / max(p4[1] + p4[2], 1e-9)
        corr = A.predict_ridge(m["B"], xs)[0]
        true_corr = -self.rows[c0]["tip_true"][1:]
        return out, corr, true_corr, p_ramp

    def push(self, yz, x_to, phase="push", decide=True):
        """Compliant force-limited push along the believed axis at lateral aim yz. Returns a dict with
        res in ('latched' | 'out' | 'in_no_click') and, for 'out', the correction estimate, the true
        correction, p(ramp) and the contact step. decide=False (pushing deeper after an entry) skips the
        in/out check: the plug is already in contact."""
        env, dt = self.env, self.env.dt
        x0 = self.rows[-1]["cmd"][0]
        n = max(1, int(round((x_to - x0) / self.speed / dt)))
        nh = int(round(self.hold_s / dt))
        c0, verdict = None, None
        start = len(self.rows)
        for i in range(1, n + nh + 1):
            u = G.min_jerk(min(1.0, i / n))
            tp = self.tcp_target_for_tip(np.array([x0 + u * (x_to - x0), *yz]))
            p_tcp, _ = env.tcp_pose(self.pb.arm)
            lead = (tp - p_tcp) @ self.R[:, 0]
            if lead > MAX_LEAD:
                tp = tp - (lead - MAX_LEAD) * self.R[:, 0]
            self.step(tp, True, phase)
            if env.task.latched[self.pb.c] >= 0:
                return dict(res="latched")
            j = len(self.rows) - 1
            if decide and c0 is None and np.linalg.norm(self.Fn[j]) > A.F_ON:
                c0 = j
                self.drift_pre = (self.rows[max(start, j - 1)]["tcp"] - self.rows[max(start, j - 1)]["cmd"])[1:]
            if c0 is not None and verdict is None and j >= c0 + K:
                out, corr, true_corr, p_ramp = self.decide(c0)
                verdict = (corr, true_corr, p_ramp)
                if out:
                    return dict(res="out", corr=corr, true_corr=true_corr, p_ramp=p_ramp, c0=c0)
            if env.time - self.t_start > self.budget_s:
                break
        if not decide:
            return dict(res="in_no_click")
        if verdict is None:  # never touched hard enough to decide
            if self.rows[-1]["tcp"][0] > 0.004:
                return dict(res="in_no_click")
            return dict(res="out", corr=np.zeros(2), true_corr=-self.rows[-1]["tip_true"][1:], p_ramp=0.5, c0=len(self.rows) - 1)
        return dict(res="in_no_click")

    def slide(self, a_from, a_to, x_contact):
        """Stiff lateral slide from a_from to a_to with a light axial preload; 'dropped' when the measured
        tip advances DROP beyond the contact position, 'latched', or 'no_drop'."""
        env, dt = self.env, self.env.dt
        n = max(3, int(round(np.linalg.norm(np.asarray(a_to) - a_from) / SLIDE_SPEED / dt)))
        for i in range(1, n + int(0.2 / dt) + 1):
            u = G.min_jerk(min(1.0, i / n))
            lat = np.asarray(a_from) + u * (np.asarray(a_to) - a_from)
            x_meas = self.rows[-1]["tcp"][0]
            self.step(self.tcp_target_for_tip(np.array([x_meas + SLIDE_LEAD, *lat])), False, "slide")
            if env.task.latched[self.pb.c] >= 0:
                return "latched"
            if self.rows[-1]["tcp"][0] - x_contact > DROP:
                return "dropped"
            if env.time - self.t_start > self.budget_s:
                break
        return "no_drop"

    def run(self):
        self.setup()
        self.t_start = self.env.time
        self.drift_pre = np.zeros(2)
        self.drops = 0
        m = self.models
        belief = None
        u_mean = self.prior_mean + m["drift_mean"]  # prior over u = perception error + controller drift
        if self.strategy in ("belief", "slide"):
            sig = self.prior_sigma if self.prior_sigma is not None else self.sigma_default
            belief = HoleBelief(capture_map(m), sig, m["S_ramp"], m["S_face"], mean=u_mean)
            aim, _ = belief.best_aim()
        else:
            aim = -u_mean
        aim0 = aim.copy()
        x1 = self.tc.seat_depth + self.overshoot
        self.stiff_to([self.pre_x, *aim], T=0.6, phase="pre")
        tries, deeper, n_spiral, slid = 0, 0, 0, 0
        self.result = dict(tries=0, deeper=0, slides=0, success=False)
        while tries < self.max_tries and self.env.time - self.t_start < self.budget_s:
            tries += 1
            r = self.push(aim, x1)
            if r["res"] == "out" and self.strategy == "slide":
                belief.update_out(aim, r["corr"], r["p_ramp"])
                new_aim, _ = belief.best_aim()
                slid += 1
                s = self.slide(aim, new_aim, self.rows[r["c0"]]["tcp"][0])
                aim = new_aim
                self.drops += s in ("dropped", "latched")
                if s == "latched":
                    r = dict(res="latched")
                elif s == "dropped":
                    r = self.push(aim, x1, phase="push", decide=False)
                else:
                    belief.update_out(aim, None)
                    aim, _ = belief.best_aim()
            if r["res"] == "in_no_click":  # partial insertion: push deeper along the axis (force-limited)
                deeper += 1
                if self.push(aim, x1 + 0.004, phase="deeper", decide=False)["res"] == "latched":
                    r = dict(res="latched")
            if r["res"] == "latched":
                self.result.update(success=True)
                break
            if self.strategy == "blind":
                break
            # retract stiffly, then re-aim
            self.stiff_to([self.pre_x, *self.rows[-1]["cmd"][1:]], T=0.35, phase="retract")
            if r["res"] == "out" and self.strategy != "slide":
                if self.strategy == "spiral":
                    n_spiral += 1
                    rr = 0.001 * np.sqrt(n_spiral)
                    aim = aim0 + rr * np.array([np.cos(n_spiral * GOLDEN), np.sin(n_spiral * GOLDEN)])
                elif self.strategy == "learned":
                    aim = aim + r["corr"]
                elif self.strategy == "oracle":
                    aim = aim + r["true_corr"]
                elif self.strategy == "belief":
                    belief.update_out(aim, r["corr"], r["p_ramp"])
                    aim, _ = belief.best_aim()
            aim = np.clip(aim, -0.008, 0.008)
            self.stiff_to([self.pre_x, *aim], T=0.35, phase="reaim")
        F = np.array([row["F"] for row in self.rows])
        # after a click the tip sits on the port axis (within the clearance), so minus the measured tip
        # position (believed frame) observes the lateral perception error; the drift is added back by the prior
        u_obs = -self.rows[-1]["tcp"][1:] if self.result["success"] else None
        self.result.update(tries=tries, deeper=deeper, slides=slid, drops=self.drops, time_s=self.env.time - self.t_start,
                           peak_F=float(np.linalg.norm(F, axis=1).max()), end=outcome(self.rows, self.tc), u_obs=u_obs)
        return self.rows

    sigma_default = 0.002


def run_scene(job):
    seed, jack, sigmas, n_err, strategies, models, out_dir = job
    shard = Path(out_dir) / f"scene_{seed}.json"
    if shard.exists():
        return json.loads(shard.read_text())
    env = make_env(JACKS[jack])
    got = prepare(env, seed)
    out = []
    if got is not None:
        _ex, snap = got
        port = int(env.scenario.task["cables"][0]["port"])
        pb = Probe(env)
        rng = np.random.default_rng([seed, 23])
        for sigma in sigmas:
            for e in range(n_err):
                err = np.array([rng.uniform(-0.002, 0.002), *rng.normal(0, sigma, 2)])
                for s in strategies:
                    g = Guided(env, pb, snap, port, err=err, strategy=s, models=models, noise_seed=seed * 1000 + e, prior_sigma=sigma)
                    g.run()
                    out.append(dict(scene=seed, jack=jack, sigma=sigma, err=err, strategy=s, unstable=bool(env.unstable),
                                    **{k: v for k, v in g.result.items() if k != "u_obs"}))
                    env.unstable = False
    tmp = shard.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    tmp.rename(shard)
    return out


def summarize(res, strategies=STRATS, log=print):
    blocked = {r["scene"] for r in res if r["strategy"] == strategies[0] and r["end"] == "blocked"}
    log(f"{len(res)} runs; scenes {len({r['scene'] for r in res})}; scenes with a blocked attempt (excluded) {sorted(blocked)}")
    rows = []
    for sigma in sorted({r["sigma"] for r in res}):
        log(f"\nsigma = {sigma * 1e3:.1f} mm lateral per axis")
        for s in strategies:
            rr = [r for r in res if r["strategy"] == s and r["sigma"] == sigma and not r["unstable"] and r["scene"] not in blocked]
            if not rr:
                continue
            k = sum(r["success"] for r in rr)
            lo, hi = wilson(k, len(rr))
            ok = [r for r in rr if r["success"]]
            row = dict(sigma=sigma, strategy=s, n=len(rr), success=k / len(rr), lo=lo, hi=hi, time_ok=float(np.mean([r["time_s"] for r in ok])) if ok else float("nan"),
                       tries=float(np.mean([r["tries"] for r in rr])), first_try=float(np.mean([r["success"] and r["tries"] == 1 for r in rr])),
                       peak_F=float(np.median([r["peak_F"] for r in rr])))
            rows.append(row)
            log(f"  {s:8s} success {row['success']:.2f} [{lo:.2f}, {hi:.2f}] (n={len(rr)})  first-try {row['first_try']:.2f}  mean tries {row['tries']:.2f}"
                f"  mean time-to-seat {row['time_ok']:.2f} s  median peak |F| {row['peak_F']:.0f} N")
    return rows


def paired(res, a, b, sigma, n_boot=4000):
    """Paired success difference a - b over identical (scene, error) cases, bootstrap 95 % interval."""
    blocked = {r["scene"] for r in res if r["end"] == "blocked"}
    by = {}
    for r in res:
        if r["scene"] in blocked or r["unstable"] or r["sigma"] != sigma:
            continue
        by.setdefault((r["scene"], tuple(np.round(r["err"], 7))), {})[r["strategy"]] = r["success"]
    d = np.array([v[a] - v[b] for v in by.values() if a in v and b in v], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n_err", type=int, default=15)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.001, 0.002])
    ap.add_argument("--strategies", nargs="+", default=list(STRATS))
    ap.add_argument("--jack", default="nominal", choices=sorted(JACKS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--name", default="e2")
    ap.add_argument("--out", default=None, help="shard directory (default runs/insertion_state/<name>)")
    ap.add_argument("--train", default="e1", help="E1 run name or shard directory")
    ap.add_argument("--seed0", type=int, default=3000)
    args = ap.parse_args()
    models = train_models(args.train)
    out_dir = Path(args.out) if args.out else OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, args.jack, args.sigmas, args.n_err, args.strategies, models, str(out_dir)) for s in range(args.seed0, args.seed0 + args.n)]
    res = []
    with Pool(args.workers) as p:
        for i, chunk in enumerate(p.imap_unordered(run_scene, jobs), 1):
            res += chunk
            print(f"  scene {i}/{len(jobs)} done", flush=True)
    summarize(res, args.strategies)
    for sigma in args.sigmas:
        for a, b in (("belief", "spiral"), ("slide", "belief"), ("belief", "learned")):
            if a in args.strategies and b in args.strategies:
                d, lo, hi = paired(res, a, b, sigma)
                print(f"  sigma {sigma * 1e3:.0f} mm  {a} - {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
