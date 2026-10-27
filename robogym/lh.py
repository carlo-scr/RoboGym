"""Long-horizon evaluation of policies.

For every episode the evaluation records, besides success / progress / duration: when each oracle subgoal
was first achieved, every *attempt* (an object grasped and released: which object, which subgoal the task
expected at that moment, whether the object ended up at its goal, at another goal, or was dropped), and
from those a first-failure attribution:

    none        success
    plan        at the first unachieved subgoal the policy manipulated the wrong object
    plan_goal   ... the right object, but put it at a different valid goal (other cube / spot / port / cell)
    exec_grasp  ... never got a grasp on the right object (missed grasp, stall, timeout)
    exec_place  ... grasped the right object but did not achieve the goal (drop, misplacement, jam)
    late        every subgoal was achieved but the episode still failed (toppled / still holding / timeout)
    unstable    the simulator diverged

This works for flat policies (nothing but the simulator's labels is used) and for hierarchical ones, where
`hier_log` adds the planner's plan validity and the per-subgoal executor outcomes. Summaries: success per
horizon (Wilson CI), conditional per-subgoal reliability r_k = P(k achieved | k-1 achieved), the
independent-subgoal prediction prod r_k against the observed success (compounding), and the attribution
mix per policy.
"""
from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import numpy as np

from . import bench
from .datagen import EpisodeSpec, build_scenario
from .policies.planning import adapter_for

ATTRIBUTIONS = ("none", "plan", "plan_goal", "exec_grasp", "exec_place", "late", "unstable")


def make_specs(task, horizons=None, n=20, split="test", variant=None, shifts=(), severity=0.5, regime=None, tag=None, offset=0):
    """Paired specs: the same `split` seeds at every horizon (and under every condition you call this with)."""
    horizons = list(horizons or bench.HORIZONS[task])
    out = []
    for h in horizons:
        for s in bench.seeds(split, n, offset=offset):
            out.append(EpisodeSpec(seed=s, category="ood" if shifts else ("regime_shift" if regime else "nominal"), shifts=list(shifts), shift_severity=severity,
                                   variant={"n": int(h), **(variant or {})}, regime_schedule=[[0.0, dict(regime)]] if regime else [],
                                   tag=tag or f"{split}:n{h}" + ("+" + "+".join(shifts) if shifts else "")))
    return out


def attribute(oracle, attempts, flags, success, unstable, ordered=True):
    """First-failure attribution (module docstring). For unordered tasks (cables) the failing subgoal is
    the first unachieved one that was attempted, and an object is "wrong" only if no unachieved subgoal
    needs it (`wrong_object` is decided at grasp time in run_episode)."""
    if unstable:
        return "unstable", None
    if success:
        return "none", None
    todo = [i for i, f in enumerate(flags) if not f]
    if not todo:
        return "late", len(flags)
    k = todo[0]
    if not ordered:
        k = next((i for i in todo if any(a["k_oracle"] == i for a in attempts)), todo[0])
    A = [a for a in attempts if a["k_oracle"] == k]
    if any(a["wrong_object"] for a in (attempts if not ordered else A)):
        return "plan", k
    if any(a["at_other_goal"] for a in A):
        return "plan_goal", k
    if A:
        return "exec_place", k
    return "exec_grasp", k


def run_episode(env, policy, spec: EpisodeSpec, max_s=None, settle_s=1.0, cams=(), max_s_scale=1.0):
    """One evaluation episode -> a flat dict (the row) with the labels described in the module docstring."""
    sc = build_scenario(env, spec)
    env.reset(sc)
    policy.reset(env, spec.seed)
    ad = adapter_for(env)
    oracle = ad.oracle(env)
    n_sub = len(oracle)
    lim = (max_s or env.episode_max_s) * max_s_scale
    frames = {c: [] for c in cams}
    attempts, open_att, streak = [], {}, {}
    sub_t = [None] * n_sub
    grasped_hist, n_done_hist, max_depth = [], [], {}
    t_ok = t_fin = None
    t_wall = time.time()
    while env.time < lim and not env.unstable:
        pr = env.privileged()
        flags = ad.oracle_flags(pr)
        kn = int(min(ad.oracle_next(pr), n_sub - 1)) if n_sub else 0
        for k, f in enumerate(flags):
            if f and sub_t[k] is None:
                sub_t[k] = round(env.time, 3)
        n_done_hist.append(float(sum(1 for f in flags if f)))
        g = pr["grasped"]
        grasped_hist.append(next((j for j, a in enumerate(g) if a == env.arms[0]), -1))
        for j, a in enumerate(g):
            s = streak.get(j, 0)
            streak[j] = (max(0, s) + 1) if a is not None else (min(0, s) - 1)
            if a is not None and streak[j] == 3 and j not in open_att:
                todo = [i for i, f in enumerate(flags) if not f]
                if ad.ordered:
                    k_att, wrong = kn, bool(n_sub and oracle[kn].obj != j)
                else:  # any unachieved subgoal that needs this object counts as the one attempted
                    mine = [i for i in todo if oracle[i].obj == j]
                    k_att, wrong = (mine[0] if mine else kn), not mine
                open_att[j] = dict(obj=int(j), k_oracle=int(k_att), wrong_object=wrong, t_grasp=round(env.time - 2 * env.dt, 3), t_release=None, at_other_goal=False, dropped=False)
            elif a is None and streak[j] == -3 and j in open_att:
                att = open_att.pop(j)
                att["t_release"] = round(env.time - 2 * env.dt, 3)
                att["dropped"] = not env.supported(ad.object_geoms(env, j))
                att["at_other_goal"] = bool(ad.at_other_goal(env, pr, j, att["k_oracle"]))
                attempts.append(att)
        if "plugs" in pr:
            for c, p in enumerate(pr["plugs"]):
                max_depth[c] = max(max_depth.get(c, -1.0), float(p["depth"]))
        for c in cams:
            frames[c].append(env.render(c))
        if policy.finished and t_fin is None:
            t_fin = env.time
        a = policy.act(env) if t_fin is None else policy.hold(env)
        env.step(a)
        if t_ok is None and env.info()["success"]:
            t_ok = env.time
        if t_ok is not None and env.time - t_ok > settle_s:
            break
        if t_fin is not None and env.time - t_fin > settle_s:
            break
    policy.finalize(env)
    pr = env.privileged()
    info = env.info(pr)
    flags = ad.oracle_flags(pr)
    for att in list(open_att.values()):
        att["at_other_goal"] = False
        attempts.append(att)
    for att in attempts:
        att["achieved"] = bool(flags[att["k_oracle"]]) if n_sub else False
    attribution, k_fail = attribute(oracle, attempts, flags, bool(info["success"]), env.unstable, ordered=ad.ordered)
    record = dict(grasped_k=grasped_hist, drops=int(sum(a["dropped"] for a in attempts)), n_seated=n_done_hist, max_depth=max_depth)
    try:
        failure_mode = env.task.outcome(env, record, SimpleNamespace(events=[]))
    except Exception:  # noqa: BLE001
        failure_mode = "other"
    pinfo = policy.info() or {}
    row = dict(
        policy=policy.name, task=env.task.name, seed=spec.seed, tag=spec.tag, horizon=n_sub, variant=json.dumps(spec.variant), shifts="+".join(spec.shifts),
        severity=spec.shift_severity if spec.shifts else 0.0, regime=json.dumps(spec.regime_schedule[0][1]) if spec.regime_schedule else "",
        success=bool(info["success"]), progress=float(info["progress"]), n_done=int(sum(1 for f in flags if f)), n_done_prefix=next((i for i, f in enumerate(flags) if not f), n_sub),
        duration=round(env.time, 3), t_success=t_ok, unstable=bool(env.unstable), attribution=attribution, first_fail_k=k_fail, failure_mode=failure_mode,
        n_attempts=len(attempts), n_wrong_object=int(sum(a["wrong_object"] for a in attempts)),
        n_dropped=int(sum(a["dropped"] for a in attempts)), subgoal_t=json.dumps(sub_t), attempts=json.dumps(attempts),
        plan_valid=pinfo.get("plan_valid"), plan_reason=pinfo.get("plan_reason"), n_plan=pinfo.get("n_plan"), policy_info=json.dumps(pinfo, default=str),
        wall_s=round(time.time() - t_wall, 2), privileged=bool(policy.privileged),
    )
    if cams:
        row["frames"] = frames
    return row


def evaluate(env, policy, specs, log=None, **kw):
    import pandas as pd

    rows = []
    for i, sp in enumerate(specs):
        rows.append(run_episode(env, policy, sp, **kw))
        if log and ((i + 1) % 5 == 0 or i + 1 == len(specs)):
            r = rows[-1]
            log(f"  [{policy.name}] {i + 1}/{len(specs)}  success so far {np.mean([x['success'] for x in rows]):.2f}  last: n{r['horizon']} {r['attribution']} ({r['duration']:.0f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- summaries
def summarize(df):
    """Per (policy, horizon): success rate with Wilson CI, mean progress / subgoals done / duration and the
    attribution mix."""
    import pandas as pd

    out = []
    for (pol, h), g in df.groupby(["policy", "horizon"]):
        r = bench.rate(g.success.values)
        row = dict(policy=pol, horizon=h, n=r["n"], success=r["rate"], lo=r["lo"], hi=r["hi"], progress=float(g.progress.mean()), n_done=float(g.n_done_prefix.mean()),
                   duration=float(g.duration.mean()), t_success=float(g.t_success.dropna().mean()) if g.t_success.notna().any() else float("nan"))
        for a in ATTRIBUTIONS:
            row[a] = float((g.attribution == a).mean())
        if g.plan_valid.notna().any():
            row["plan_valid"] = float(g.plan_valid.dropna().astype(bool).mean())
        out.append(row)
    return pd.DataFrame(out)


def reliability(df):
    """Conditional per-subgoal reliability per policy: r_k = P(subgoal k achieved | subgoals < k achieved),
    pooled over horizons > k. Returns {policy: {k: (r_k, n_k)}}."""
    out = {}
    for pol, g in df.groupby("policy"):
        rk = {}
        for k in range(int(g.horizon.max())):
            reached = g[(g.horizon > k) & (g.n_done_prefix >= k)]
            if len(reached):
                rk[k] = (float((reached.n_done_prefix > k).mean()), int(len(reached)))
        out[pol] = rk
    return out


def horizon_curve(df):
    """Per policy and horizon: observed success vs the independent-subgoal prediction prod_k r_k. A gap
    (observed < predicted) is compounding: later subgoals fail *because* of earlier ones."""
    import pandas as pd

    rel = reliability(df)
    rows = []
    for (pol, h), g in df.groupby(["policy", "horizon"]):
        r = bench.rate(g.success.values)
        pred = float(np.prod([rel[pol].get(k, (float("nan"), 0))[0] for k in range(int(h))])) if all(k in rel[pol] for k in range(int(h))) else float("nan")
        rows.append(dict(policy=pol, horizon=h, n=r["n"], success=r["rate"], lo=r["lo"], hi=r["hi"], predicted_independent=pred, compounding=r["rate"] - pred,
                         subgoals_done=float(g.n_done_prefix.mean()), done_frac=float((g.n_done_prefix / g.horizon).mean())))
    return pd.DataFrame(rows)


def attribution_table(df):
    """policy x attribution counts and the fraction of *failures* that are planning errors."""
    import pandas as pd

    t = pd.crosstab(df.policy, df.attribution)
    for a in ATTRIBUTIONS:
        if a not in t:
            t[a] = 0
    t = t[list(ATTRIBUTIONS)]
    fails = t.drop(columns=["none"]).sum(axis=1)
    t["plan_frac_of_failures"] = (t["plan"] + t["plan_goal"]) / fails.replace(0, np.nan)
    return t


def plan_exec_split(df):
    """Hierarchical policies only: plan validity rate and the executor's per-subgoal success given a valid
    plan (execution reliability isolated from planning), from hier_log."""
    import pandas as pd

    rows = []
    for pol, g in df.groupby("policy"):
        infos = [json.loads(x) for x in g.policy_info]
        logs = [(i, r) for i in infos if "hier_log" in i for r in i["hier_log"]]
        if not logs:
            continue
        valid = [bool(i["plan_valid"]) for i in infos if "plan_valid" in i]
        first = [(i, r) for i, r in logs if r["attempt"] == 0]
        ok_valid = [r["achieved"] for i, r in first if i["plan_valid"]]
        ok_all = [r["achieved"] for i, r in first]
        retr = [r["achieved"] for i, r in logs if r["attempt"] > 0]
        rows.append(dict(policy=pol, n_episodes=len(infos), plan_valid=float(np.mean(valid)) if valid else float("nan"), n_subgoals=len(first),
                         exec_success_valid_plan=float(np.mean(ok_valid)) if ok_valid else float("nan"), exec_success_all=float(np.mean(ok_all)) if ok_all else float("nan"),
                         retry_success=float(np.mean(retr)) if retr else float("nan"), timeouts=float(np.mean([r["timeout"] for i, r in first])) if first else float("nan"),
                         mean_subgoal_s=float(np.mean([r["t_end"] - r["t_start"] for i, r in first if r["t_end"] is not None])) if first else float("nan")))
    return pd.DataFrame(rows)


def save(df, out_dir, name="lh_eval", task=None):
    """episodes.parquet (frames dropped) + summary tables + a bench.Results JSON."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = df.drop(columns=[c for c in ("frames",) if c in df])
    d.to_parquet(out / "episodes.parquet")
    summ, curve, attr = summarize(d), horizon_curve(d), attribution_table(d)
    split = plan_exec_split(d)
    summ.to_csv(out / "summary.csv", index=False)
    curve.to_csv(out / "horizon_curve.csv", index=False)
    attr.to_csv(out / "attribution.csv")
    split.to_csv(out / "plan_exec_split.csv", index=False)
    res = bench.Results(name=name, task=task or str(d.task.iloc[0]), method="long_horizon_eval", config=dict(policies=sorted(d.policy.unique().tolist()), n=int(len(d))))
    for r in summ.to_dict("records"):
        res.add(metric="success", **{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    for r in curve.to_dict("records"):
        res.add(metric="horizon_curve", **{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    res.save(out / "results.json")
    return dict(summary=summ, curve=curve, attribution=attr, split=split)
