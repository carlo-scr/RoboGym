"""E3: remembering the port. Part of the perception error recurs: a bias b_k per panel / camera context k,
plus per-episode noise. A stream of episodes visits K contexts with sticky dwell times; a visual cue shows
the true context with probability rho. Every method runs the belief controller; they differ only in the
prior over the lateral offset at the start of an episode:
  none     N(0, sigma_b^2 + sigma_e^2)                      (memoryless)
  pooled   one memory over all episodes                      (ignores context)
  cue      one memory per cued context                       (trusts the cue)
  oracle   one memory per true context                       (upper bound)
A memory is a Gaussian over the recurring part of the lateral perception error: after a click the tip sits
on the port axis, so minus the measured tip position (believed frame) observes the error of that episode
(no privileged information). The controller adds its mean drift on top of the remembered error."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import JACKS, Probe, make_env, prepare  # noqa: E402
from e2_guided import OUT, Guided, train_models  # noqa: E402

METHODS = ("none", "pooled", "cue", "oracle")
SIG_Q = 0.0006  # where in the clearance the plug seats + drift variation between episodes (assumed)


class Memory:
    def __init__(self, sigma_b, sigma_e):
        self.sb2, self.se2 = sigma_b**2, sigma_e**2 + SIG_Q**2
        self.obs = {}

    def prior(self, key):
        """Prior over u (mean, per-axis sigma) for an episode whose memory key is `key`."""
        o = self.obs.get(key, [])
        n = len(o)
        if n == 0:
            return np.zeros(2), np.sqrt(self.sb2 + self.se2)
        post_var = 1.0 / (1.0 / self.sb2 + n / self.se2)
        post_mean = post_var * (np.sum(o, 0) / self.se2)
        return post_mean, np.sqrt(post_var + self.se2)

    def add(self, key, u_obs):
        self.obs.setdefault(key, []).append(np.asarray(u_obs))


def make_stream(seed, n, k_ctx, rho, dwell, sigma_b):
    rng = np.random.default_rng([seed, 31])
    biases = rng.normal(0, sigma_b, (k_ctx, 2))
    ctx, cue = [], []
    c = int(rng.integers(k_ctx))
    for _ in range(n):
        if rng.random() > 1 - 1 / dwell:
            c = int(rng.integers(k_ctx))
        ctx.append(c)
        cue.append(c if rng.random() < rho else int(rng.integers(k_ctx)))
    return biases, ctx, cue


def run_stream(job):
    seed, scenes, a = job["seed"], job["scenes"], job["args"]
    shard = Path(a["out_dir"]) / f"stream_{seed}.json"
    if shard.exists():
        return json.loads(shard.read_text())
    env = make_env(JACKS[a["jack"]])
    pb = Probe(env)
    models = job["models"]
    biases, ctx, cue = make_stream(seed, len(scenes), a["k_ctx"], a["rho"], a["dwell"], a["sigma_b"])
    rng = np.random.default_rng([seed, 37])
    mem = {m: Memory(a["sigma_b"], a["sigma_e"]) for m in METHODS}
    out = []
    for e, scene in enumerate(scenes):
        got = prepare(env, scene)
        if got is None:
            continue
        _ex, snap = got
        port = int(env.scenario.task["cables"][0]["port"])
        err = np.array([rng.uniform(-0.002, 0.002), *(biases[ctx[e]] + rng.normal(0, a["sigma_e"], 2))])
        for m in METHODS:
            key = {"none": None, "pooled": 0, "cue": cue[e], "oracle": ctx[e]}[m]
            mean, sig = mem[m].prior(key) if key is not None else (np.zeros(2), np.sqrt(a["sigma_b"] ** 2 + a["sigma_e"] ** 2))
            g = Guided(env, pb, snap, port, err=err, strategy="belief", models=models, prior_mean=mean, prior_sigma=sig, noise_seed=seed * 10000 + e)
            g.run()
            r = g.result
            if r["success"] and key is not None and r.get("u_obs") is not None:
                mem[m].add(key, r["u_obs"])
            first_visit = ctx[e] not in ctx[:e]
            out.append(dict(stream=seed, episode=e, scene=scene, method=m, ctx=ctx[e], cue=cue[e], first_visit=first_visit,
                            unstable=bool(env.unstable), **{k: v for k, v in r.items() if k != "u_obs"}))
            env.unstable = False
    tmp = shard.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    tmp.rename(shard)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--k_ctx", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--dwell", type=float, default=5.0)
    ap.add_argument("--sigma_b", type=float, default=0.002)
    ap.add_argument("--sigma_e", type=float, default=0.0005)
    ap.add_argument("--scene_pool", type=int, nargs=2, default=[5000, 5400])
    ap.add_argument("--train", default="e1", help="E1 run name or shard directory")
    ap.add_argument("--jack", default="nominal", choices=sorted(JACKS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--name", default="e3")
    ap.add_argument("--out", default=None, help="shard directory (default runs/insertion_state/<name>)")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    a = vars(args) | {"out_dir": str(out_dir)}
    models = train_models(args.train)
    pool = np.arange(*args.scene_pool)
    jobs = []
    for s in range(args.streams):
        rng = np.random.default_rng([s, 41])
        jobs.append(dict(seed=s, scenes=[int(x) for x in rng.choice(pool, args.episodes, replace=False)], args=a, models=models))
    res = []
    with Pool(args.workers) as p:
        for i, chunk in enumerate(p.imap_unordered(run_stream, jobs), 1):
            res += chunk
            print(f"  stream {i}/{len(jobs)} done", flush=True)
    summarize(res)


def summarize(res, log=print):
    """Episodes where any method was blocked by a neighbouring cable are dropped for all methods."""
    bad = {(r["stream"], r["episode"]) for r in res if r["unstable"] or r["end"] == "blocked"}
    res = [r for r in res if (r["stream"], r["episode"]) not in bad]
    log(f"{len(res)} episode-method runs ({len(bad)} blocked / unstable episodes excluded)")
    rows = []
    for m in METHODS:
        rr = [r for r in res if r["method"] == m]
        if not rr:
            continue
        ret = [r for r in rr if not r["first_visit"]]
        row = dict(method=m, n=len(rr), success=float(np.mean([r["success"] for r in rr])),
                   first_try=float(np.mean([r["success"] and r["tries"] == 1 for r in rr])),
                   first_try_returns=float(np.mean([r["success"] and r["tries"] == 1 for r in ret])) if ret else float("nan"),
                   tries=float(np.mean([r["tries"] for r in rr])), time_s=float(np.mean([r["time_s"] for r in rr])))
        rows.append(row)
        log(f"  {m:7s} success {row['success']:.2f}  first-try {row['first_try']:.2f}  first-try on returns {row['first_try_returns']:.2f}"
            f"  mean tries {row['tries']:.2f}  mean time {row['time_s']:.2f} s")
    return rows


if __name__ == "__main__":
    main()
