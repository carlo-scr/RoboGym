# %% [markdown]
# # Benchmark: train everything, then evaluate planning and execution
# One notebook that builds the whole stack for every task and **survives interruptions**: each stage
# checks its outputs on disk and skips what is done, datasets and evaluations run through the resumable
# job pool, training checkpoints every few hundred steps and resumes, and (on Colab with Drive mounted)
# `data/` and `runs/` live on Drive. After a disconnect, rerun top to bottom and it continues.
#
# Stages (`ROBOGYM_STAGES`, default all but `rq1`): **data** (DART demonstrations + a labelled evaluation
# set per task), **policies** (Diffusion Policy / ACT / MLP on state, DP on images), **wm** (frame
# autoencoder, latent world model, conformal monitor), **lh** (the long-horizon benchmark: scripted,
# learned and hierarchical planner + executor policies, with the planning / execution attribution),
# **rq1** (oracle ranking for steering, slow, optional), **summary**.
#
# Presets: `ROBOGYM_PRESET=smoke` (laptop, ~1 h for three tasks, tiny models) | `a100` (Colab A100 +
# Drive; the data stage runs on the VM's CPUs for a few hours, training a few hours, the benchmark ~1 h).
# `ROBOGYM_TASKS=cube_stack,lego,cable_plug`, `ROBOGYM_BUDGET_MIN=50` bounds each training call so a
# session can end cleanly (rerun to continue), `ROBOGYM_FORCE=1` redoes finished stages.

# %%
# @bootstrap

# %%
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

if "ROOT" not in globals():  # running the source as a plain script
    ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT))
from robogym import bench, lh
from robogym.config import make_cfg
from robogym.datagen import EpisodeSpec
from robogym.env import RoboGymEnv
from robogym.learn.data import list_episodes, load_many
from robogym.learn.encoders import load_ae, save_ae, train_ae
from robogym.learn.monitor import evaluate_monitor, headline, summarize as monitor_summary
from robogym.learn.policy import PolicyCfg, load_policy, save_policy, train_policy
from robogym.learn.wm import load_wm, save_wm, score_episode, train_wm, wm_features
from robogym.policies import ChunkPolicy
from robogym.tools import run_live

REPO = Path(ROOT)
PRESET = os.environ.get("ROBOGYM_PRESET", "a100" if torch.cuda.is_available() else "smoke")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = [t for t in os.environ.get("ROBOGYM_TASKS", "cube_stack,lego,cable_plug").split(",") if t]
STAGES = set(os.environ.get("ROBOGYM_STAGES", "data,policies,wm,lh,summary").split(","))
FORCE = os.environ.get("ROBOGYM_FORCE", "0") == "1"
BUDGET_S = float(os.environ["ROBOGYM_BUDGET_MIN"]) * 60 if os.environ.get("ROBOGYM_BUDGET_MIN") else None
CPU = os.cpu_count() or 4
P = dict(
    smoke=dict(n_train=40, n_eval=60, steps=dict(dp=400, act=400, mlp=400, dp_image=200), steps_ae=200, steps_wm=300, dims=(32, 64, 128), bs=128,
               workers=min(8, CPU), lh_n=4, n_img=16, n_branch=4, K=4, ckpt_every=100, hw=(72, 96), quick_n=2, wm_hidden=256),
    a100=dict(n_train=2000, n_eval=800, steps=dict(dp=60000, act=40000, mlp=20000, dp_image=40000), steps_ae=6000, steps_wm=8000, dims=(64, 128, 256), bs=256,
              workers=CPU, lh_n=50, n_img=500, n_branch=40, K=16, ckpt_every=2000, hw=(72, 96), quick_n=5, wm_hidden=512),
)[PRESET]
CAMS = ("right_wrist", "front")
CAM_CFG = json.dumps({"camera.width": P["hw"][1], "camera.height": P["hw"][0]})
OUT = REPO / "runs" / f"bench_{PRESET}"
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "status.json"
status = json.load(open(STATUS)) if STATUS.exists() else {}


def done(path):
    return Path(path).exists() and not FORCE


def mark(stage, task, **kw):
    status.setdefault(stage, {})[task] = dict(done=True, at=time.strftime("%Y-%m-%d %H:%M"), **kw)
    json.dump(status, open(STATUS, "w"), indent=1, default=str)


def run_script(name, *args):
    run_live([sys.executable, str(REPO / "scripts" / name), *map(str, args)], cwd=str(REPO))


def dataset_ready(d):
    return bool(list(Path(d).glob("index_*.parquet"))) and not FORCE


def data_dirs(task):
    return REPO / "data" / f"{task}_train_{PRESET}", REPO / "data" / f"{task}_eval_{PRESET}"


print(f"preset {PRESET} on {DEVICE}, tasks {TASKS}, stages {sorted(STAGES)}, out {OUT}")
print(P)

# %% [markdown]
# ## 1. Datasets
# Per task: DART demonstrations on the train seeds (random horizons) and a labelled evaluation set on the
# calibration seeds (nominal / injected failure / OOD / regime shift). `generate_dataset.py` resumes:
# finished episodes are skipped and the index is written only at the end, which is what the guard checks.

# %%
if "data" in STAGES:
    for task in TASKS:
        train_dir, eval_dir = data_dirs(task)
        if not dataset_ready(train_dir):
            run_script("generate_dataset.py", "--task", task, "--out", train_dir, "--n", P["n_train"], "--seed0", bench.SPLITS["train"][0], "--mix", "1,0,0,0",
                       "--dart", "0.002", "--cameras", *CAMS, "--no-states", "--workers", P["workers"], "--cfg", CAM_CFG)
        if not dataset_ready(eval_dir):
            run_script("generate_dataset.py", "--task", task, "--out", eval_dir, "--n", P["n_eval"], "--seed0", bench.SPLITS["cal"][0], "--mix", "0.5,0.25,0.2,0.05",
                       "--cameras", *CAMS, "--no-states", "--workers", P["workers"], "--cfg", CAM_CFG)
        it, ie = list_episodes(train_dir), list_episodes(eval_dir)
        print(f"[{task}] train {it.outcome.value_counts().to_dict()} | eval {ie.groupby(['category', 'outcome']).size().to_dict()}")
        mark("data", task, n_train=int(len(it)), n_eval=int(len(ie)))

# %% [markdown]
# ## 2. Policies: Diffusion Policy, ACT and MLP on state observations, Diffusion Policy on images
# Every run writes `ckpt.pt` every `ckpt_every` steps and resumes from it; `policy.pt` marks completion.
# A quick rollout on a few test seeds gives feedback; the real evaluation is stage 4.

# %%
_EPS = {}


def episodes_for(task, image=False):
    key = (task, image)
    if key not in _EPS:
        idx = list_episodes(data_dirs(task)[0])
        paths = idx[idx.outcome == "success"].path.tolist()
        _EPS[key] = load_many(paths[: P["n_img"]] if image else paths, cams=CAMS if image else (), frames=image)
    return _EPS[key]


def train_arch(task, arch, obs="state"):
    d = OUT / task / f"{arch}_{obs}"
    if done(d / "policy.pt"):
        return load_policy(d / "policy.pt", device=DEVICE)
    eps = episodes_for(task, image=obs == "image")
    cfg = PolicyCfg(arch=arch, obs_mode=obs, n_arms=eps[0]["n_arms"], task_dim=eps[0]["task"].shape[1], cams=CAMS, img_hw=P["hw"], down_dims=P["dims"])
    steps = P["steps"][f"{arch}_{obs}" if obs == "image" else arch]
    t0 = time.time()
    pol, hist = train_policy(eps, cfg, steps=steps, bs=P["bs"], device=DEVICE, log_every=max(50, steps // 10), ckpt_dir=d, ckpt_every=P["ckpt_every"], budget_s=BUDGET_S)
    if len(hist) < steps:
        print(f"[{task}/{arch}_{obs}] stopped at {len(hist)}/{steps} steps (time budget); rerun this cell to continue")
        return None
    save_policy(pol, d / "policy.pt")
    json.dump(dict(cfg=cfg.__dict__, steps=steps, loss=hist[:: max(1, len(hist) // 500)], train_s=time.time() - t0, n_episodes=len(eps)), open(d / "train.json", "w"), indent=1, default=str)
    env = RoboGymEnv(make_cfg(task, **json.loads(CAM_CFG)), render=obs == "image")
    rows = [lh.run_episode(env, ChunkPolicy(pol, device=DEVICE), sp) for sp in lh.make_specs(task, horizons=(bench.HORIZONS[task][0],), n=P["quick_n"])]
    env.close()
    print(f"[{task}/{arch}_{obs}] trained in {time.time() - t0:.0f}s; quick test success {np.mean([r['success'] for r in rows]):.2f} ({len(rows)} seeds)")
    mark("policies", f"{task}/{arch}_{obs}", train_s=round(time.time() - t0), quick_success=float(np.mean([r["success"] for r in rows])))
    return pol


if "policies" in STAGES:
    for task in TASKS:
        for arch in ("dp", "act", "mlp"):
            train_arch(task, arch, "state")
        train_arch(task, "dp", "image")
        _EPS.clear()  # free the frames

# %% [markdown]
# ## 3. Frame autoencoder, latent world model, conformal failure monitor (per task)
# Nominal episodes of the evaluation set train the autoencoder and the world model; the monitor is
# calibrated on held-out nominal episodes and scored on failures / OOD / regime shifts.

# %%
if "wm" in STAGES:
    for task in TASKS:
        d = OUT / task
        if done(d / "monitor.json"):
            continue
        idx_eval = list_episodes(data_dirs(task)[1])
        eps_eval = load_many(idx_eval.path.tolist(), cams=CAMS, frames=True)
        nominal = [e for e in eps_eval if e["ann"]["category"] == "nominal" and e["ann"]["outcome"] == "success"]
        if done(d / "ae.pt"):
            ae = load_ae(d / "ae.pt", device=DEVICE)
        else:
            frames_pool = np.concatenate([np.concatenate([e["frames"][c] for c in CAMS]) for e in nominal[: max(10, len(nominal) // 4)]])
            ae, _ = train_ae(frames_pool, latent=32, steps=P["steps_ae"], device=DEVICE, log_every=max(50, P["steps_ae"] // 5), ckpt_dir=d / "ae_ckpt", ckpt_every=P["ckpt_every"], budget_s=BUDGET_S)
            save_ae(ae, d / "ae.pt")

        def featurise(e):
            z = np.concatenate([ae.encode(e["frames"][c]) for c in CAMS], 1)
            return dict(x=wm_features(z, e["state"], e["n_arms"]), action=e["action"], ann=e["ann"], id=e["ann"]["spec"]["seed"])

        feats = [featurise(e) for e in eps_eval]
        nom_feats = [f for f in feats if f["ann"]["category"] == "nominal" and f["ann"]["outcome"] == "success"]
        n_tr, n_cal = int(0.5 * len(nom_feats)), int(0.75 * len(nom_feats))  # nominal: 50 % WM training, 25 % calibration, 25 % test (false alarms)
        if done(d / "wm.pt"):
            wm = load_wm(d / "wm.pt", device=DEVICE)
        else:
            wm, _ = train_wm(nom_feats[:n_tr], H=2, n_members=5, hidden=P["wm_hidden"], steps=P["steps_wm"], device=DEVICE, val=nom_feats[n_tr:n_cal], log_every=max(50, P["steps_wm"] // 5),
                             ckpt_dir=d / "wm_ckpt", ckpt_every=P["ckpt_every"], budget_s=BUDGET_S)
            save_wm(wm, d / "wm.pt")
        traces = {f["id"]: score_episode(wm, f["x"], f["action"], device=DEVICE) for f in feats}
        anns = {f["id"]: f["ann"] for f in feats}
        cal_ids = [f["id"] for f in nom_feats[n_tr:n_cal]]
        held = set(cal_ids) | {f["id"] for f in nom_feats[:n_tr]}
        splits = dict(nominal_cal=cal_ids, test=[f["id"] for f in feats if f["id"] not in held])  # nominal held-out + failures + OOD + regime shifts
        df_m, norm = evaluate_monitor(traces, anns, splits, key=("nll", "alea"), alpha=0.1)
        head = headline(df_m)
        json.dump(head, open(d / "monitor.json", "w"), indent=1, default=float)
        monitor_summary(df_m).to_csv(d / "monitor.csv")
        print(f"[{task}] monitor: {json.dumps(head, default=float)}")
        mark("wm", task, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in head.items()})
        del eps_eval, feats

# %% [markdown]
# ## 4. Long-horizon benchmark: planning and execution
# The same test seeds at every horizon for every policy: the scripted expert with execution knobs,
# the three learned architectures as flat policies and as goal-conditioned executors under an oracle
# planner, the hierarchical policy with rule-based, deliberately corrupted and (if an API key is set) LLM
# planners. `eval_policies.py` runs through the resumable job pool and writes `summary.csv`,
# `horizon_curve.csv`, `attribution.csv`, `plan_exec_split.csv`, `results.json`.

# %%
def lh_policies(task):
    d = OUT / task
    pols = ["expert", "expert:exec_noise=0.004", "hier:oracle+scripted", "hier:heuristic+scripted", "hier:corrupt(p_swap=0.5)+scripted",
            "hier:corrupt(p_goal=0.5)+scripted", "hier:corrupt(p_drop=0.5)+scripted", "hier:reverse+scripted", "hier:oracle+scripted(exec_noise=0.004)",
            "hier:oracle+scripted(failure=any,p_failure=0.3,severity=0.7):on_fail=retry"]
    for name in ("dp_state", "act_state", "mlp_state", "dp_image"):
        ck = d / name / "policy.pt"
        if ck.exists():
            pols.append(f"chunk:{ck}")
            if name != "dp_image":
                pols.append(f"hier:oracle+chunk({ck})")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401

            pols.append(f"hier:llm(cache_dir={OUT / 'llm_cache'})+scripted")
        except ImportError:
            print("ANTHROPIC_API_KEY is set but the anthropic package is missing: pip install anthropic")
    return pols


if "lh" in STAGES:
    for task in TASKS:
        out = OUT / task / "lh"
        pols = lh_policies(task)
        run_script("eval_policies.py", "--task", task, "--horizons", *bench.HORIZONS[task], "--n", P["lh_n"], "--workers", P["workers"], "--out", out,
                   "--cfg", CAM_CFG, "--device", "cpu", "--policies", *pols)
        summ = pd.read_csv(out / "summary.csv")
        mark("lh", task, n_policies=len(pols), episodes=int(summ.n.sum()))

# %% [markdown]
# ## 5. RQ1 oracle ranking (optional: `ROBOGYM_STAGES=...,rq1`)
# From N base-policy states, execute every candidate chunk from the same simulator state; reports how
# well random / progress / world-model uncertainty scores rank the candidates (steering headroom).

# %%
if "rq1" in STAGES:
    for task in TASKS:
        d = OUT / task
        if done(d / "rq1.json") or not (d / "dp_state" / "policy.pt").exists():
            continue
        run_script("rq1_oracle.py", "--policy", d / "dp_state" / "policy.pt", "--task", task, "--n", P["n_branch"], "--K", P["K"], "--wm", d / "wm.pt", "--ae", d / "ae.pt",
                   "--cams", *CAMS, "--out", d / "rq1.json", "--device", DEVICE)
        mark("rq1", task)

# %% [markdown]
# ## 6. Summary across tasks
# Success per policy and horizon, the planning / execution attribution, the hierarchical plan / execution
# split, monitor headlines; horizon curves as a figure. Everything is also in `runs/bench_<preset>/`.

# %%
if "summary" in STAGES:
    import matplotlib
    import matplotlib.pyplot as plt

    FAMILIES = [  # name, matcher on the policy name, colour (reference categorical palette, fixed order)
        ("expert", lambda n: n == "expert", "#2a78d6"),
        ("heuristic plan + scripted", lambda n: n.startswith("hier[heuristic+scripted]"), "#eb6834"),
        ("wrong-goal plans (p .5) + scripted", lambda n: "goal0.5" in n, "#1baf7a"),
        ("reversed plan + scripted", lambda n: n.startswith("hier[reverse"), "#eda100"),
        ("oracle plan + failing executor, retry", lambda n: "any0.3" in n and n.endswith(":retry"), "#e87ba4"),
        ("Diffusion Policy (state)", lambda n: n.startswith("dp_state"), "#008300"),
        ("ACT (state)", lambda n: n.startswith("act_state"), "#6250d6"),
        ("MLP (state)", lambda n: n.startswith("mlp_state"), "#e34948"),
    ]
    tables = {}
    for task in TASKS:
        out = OUT / task / "lh"
        if not (out / "summary.csv").exists():
            continue
        tables[task] = dict(summary=pd.read_csv(out / "summary.csv"), curve=pd.read_csv(out / "horizon_curve.csv"), attribution=pd.read_csv(out / "attribution.csv", index_col=0),
                            split=pd.read_csv(out / "plan_exec_split.csv"))
    if tables:
        pd.set_option("display.width", 220)
        allsum = pd.concat([t["summary"].assign(task=k) for k, t in tables.items()])
        allsum.to_csv(OUT / "benchmark_summary.csv", index=False)
        for task, t in tables.items():
            print(f"\n== {task}: success per policy / horizon ==")
            print(t["summary"].pivot(index="policy", columns="horizon", values="success").round(2).to_string())
            print(f"\n== {task}: first-failure attribution ==")
            print(t["attribution"].round(2).to_string())
            if len(t["split"]):
                print(f"\n== {task}: plan / execution split ==")
                print(t["split"].round(2).to_string(index=False))
            mon = OUT / task / "monitor.json"
            if mon.exists():
                print(f"\n== {task}: monitor == {open(mon).read().strip()}")
        ink, muted = "#1a1a19", "#8a8a85"
        matplotlib.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink, "axes.edgecolor": muted, "font.size": 10})
        fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.9), squeeze=False, facecolor="#fcfcfb")
        for ax, (task, t) in zip(axes[0], tables.items()):
            c = t["curve"]
            for (label, match, col), mk in zip(FAMILIES, "osD^vPX*"):  # marker shape = secondary encoding of identity
                g = c[c.policy.map(match)].sort_values("horizon")
                if len(g):
                    ax.plot(g.horizon, g.success, color=col, lw=2, marker=mk, ms=6, label=label)
            ax.set(title=task.replace("_", " "), xlabel="subgoals in the episode", ylim=(-0.03, 1.03), xticks=sorted(c.horizon.unique()), facecolor="#fcfcfb")
            ax.set_ylabel("success rate" if ax is axes[0][0] else "")
            ax.grid(True, color="#e6e6e3", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"Long-horizon success per policy family ({PRESET}, {P['lh_n']} paired test seeds per horizon)", color=ink)
        fig.tight_layout()
        fig.savefig(OUT / "horizon_curves.png", dpi=150, bbox_inches="tight")
        print(f"\nfigure: {OUT / 'horizon_curves.png'}")
        mark("summary", "all", tasks=list(tables))
    print(json.dumps(status, indent=1, default=str))
