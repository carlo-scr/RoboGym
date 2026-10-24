# %% [markdown]
# # Base policy, latent world model, monitor and steering on RoboGym
# Trains the pieces the research threads need and runs the RQ1 oracle ranking:
# 1. a Diffusion Policy (state-based, then image-based) on DART demonstrations,
# 2. a frame autoencoder + probabilistic latent world model on nominal episodes,
# 3. the conformal failure monitor on nominal / failure / OOD episodes,
# 4. the oracle ranking: execute K candidate chunks from the same state, see which scores rank them.
#
# `PRESET = "smoke"` runs in ~15 min on a laptop CPU (tiny, just to see every step work);
# `"a100"` is the real thing on a Colab A100 (dataset generation on the VM's CPUs, training on the GPU).

# %%
# @bootstrap

# %%
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

from robogym import bench
from robogym.config import make_cfg
from robogym.datagen import build_scenario
from robogym.env import RoboGymEnv
from robogym.learn.data import list_episodes, load_many
from robogym.learn.encoders import train_ae, save_ae
from robogym.learn.monitor import evaluate_monitor, headline, summarize
from robogym.learn.policy import PolicyCfg, PolicyRunner, rollout, save_policy, train_policy
from robogym.learn.steer import FeatureTracker, RandomScorer, WMScorer, capture_branches, rq1_metrics, rq1_ranking
from robogym.learn.wm import score_episode, train_wm, wm_features, save_wm
from robogym.tools import run_live

PRESET = os.environ.get("ROBOGYM_PRESET", "a100" if torch.cuda.is_available() else "smoke")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK = "cube_stack"
P = dict(smoke=dict(n_train=40, n_eval=160, steps_dp=1500, steps_ae=300, steps_wm=600, dims=(32, 64, 128), workers=8, K=4, n_branch=4, eval_n=8, hw=(72, 96)),
         a100=dict(n_train=3000, n_eval=1200, steps_dp=60000, steps_ae=6000, steps_wm=8000, dims=(64, 128, 256), workers=os.cpu_count(), K=16, n_branch=60, eval_n=100, hw=(72, 96)))[PRESET]
REPO = Path(ROOT)  # from the setup cell
OUT = REPO / "runs" / f"learn_{TASK}_{PRESET}"
OUT.mkdir(parents=True, exist_ok=True)
print(PRESET, DEVICE, P)

# %% [markdown]
# ## 1. Data: DART demonstrations (train split) and a labelled evaluation set (nominal / failures / OOD)

# %%
train_dir, eval_dir = REPO / "data" / f"{TASK}_train_{PRESET}", REPO / "data" / f"{TASK}_eval_{PRESET}"
GEN = str(REPO / "scripts" / "generate_dataset.py")
if not (train_dir / "dataset.json").exists():
    run_live([sys.executable, GEN, "--task", TASK, "--out", str(train_dir), "--n", str(P["n_train"]), "--seed0", str(bench.SPLITS["train"][0]),
              "--mix", "1,0,0,0", "--dart", "0.002", "--cameras", "right_wrist", "front", "--no-states", "--workers", str(P["workers"]),
              "--cfg", json.dumps({"camera.width": P["hw"][1], "camera.height": P["hw"][0]})], cwd=str(REPO))
if not (eval_dir / "dataset.json").exists():
    run_live([sys.executable, GEN, "--task", TASK, "--out", str(eval_dir), "--n", str(P["n_eval"]), "--seed0", str(bench.SPLITS["cal"][0]),
              "--mix", "0.5,0.25,0.2,0.05", "--cameras", "right_wrist", "front", "--no-states", "--workers", str(P["workers"]),
              "--cfg", json.dumps({"camera.width": P["hw"][1], "camera.height": P["hw"][0]})], cwd=str(REPO))
idx_train, idx_eval = list_episodes(train_dir), list_episodes(eval_dir)
print(idx_train.outcome.value_counts().to_dict(), "|", idx_eval.groupby(["category", "outcome"]).size().to_dict())

# %% [markdown]
# ## 2. Diffusion Policy (state observations: proprio + privileged task features)

# %%
cams = ("right_wrist", "front")
eps_train = load_many(idx_train[idx_train.outcome == "success"].path.tolist(), cams=(), frames=False)
cfg_dp = PolicyCfg(obs_mode="state", n_arms=eps_train[0]["n_arms"], task_dim=eps_train[0]["task"].shape[1], down_dims=P["dims"])
pol, hist = train_policy(eps_train, cfg_dp, steps=P["steps_dp"], bs=256 if PRESET == "a100" else 128, device=DEVICE, log_every=max(100, P["steps_dp"] // 10))
save_policy(pol, OUT / "dp_state.pt")
env_cfg = make_cfg(TASK, **{"camera.width": P["hw"][1], "camera.height": P["hw"][0]})
env = RoboGymEnv(env_cfg)
runner = PolicyRunner(pol, env, device=DEVICE)
rows = []
for sp in bench.split_specs(TASK, "test", n=P["eval_n"], horizon=2):
    r = rollout(env, runner, build_scenario(env, sp), seed=sp.seed)
    rows.append(r["success"])
print(f"state DP test success: {bench.rate(rows)}")

# %% [markdown]
# ## 3. Frame autoencoder + latent world model (nominal episodes), one-step scores on failures / OOD

# %%
eps_eval = load_many(idx_eval.path.tolist(), cams=cams, frames=True)
nominal = [e for e in eps_eval if e["ann"]["category"] == "nominal" and e["ann"]["outcome"] == "success"]
frames_pool = np.concatenate([np.concatenate([e["frames"][c] for c in cams]) for e in nominal[: max(10, len(nominal) // 4)]])
ae, _ = train_ae(frames_pool, latent=32, steps=P["steps_ae"], device=DEVICE, log_every=max(100, P["steps_ae"] // 5))
save_ae(ae, OUT / "ae.pt")
def featurise(e):
    z = np.concatenate([ae.encode(e["frames"][c]) for c in cams], 1)
    return dict(x=wm_features(z, e["state"], e["n_arms"]), action=e["action"], ann=e["ann"], id=e["ann"]["spec"]["seed"])
feats = [featurise(e) for e in eps_eval]
nom_feats = [f for f in feats if f["ann"]["category"] == "nominal" and f["ann"]["outcome"] == "success"]
n_tr = int(0.6 * len(nom_feats))
wm, _ = train_wm(nom_feats[:n_tr], H=2, n_members=5, hidden=256 if PRESET == "smoke" else 512, steps=P["steps_wm"], device=DEVICE, val=nom_feats[n_tr:], log_every=max(100, P["steps_wm"] // 5))
save_wm(wm, OUT / "wm.pt")
traces = {f["id"]: score_episode(wm, f["x"], f["action"], device=DEVICE) for f in feats}
anns = {f["id"]: f["ann"] for f in feats}
cal_ids = [f["id"] for f in nom_feats[n_tr:]]
splits = dict(nominal_cal=cal_ids, test=[f["id"] for f in feats if f["id"] not in set(cal_ids) and not (f["ann"]["category"] == "nominal" and f["ann"]["outcome"] == "success" and f["id"] in {g["id"] for g in nom_feats[:n_tr]})])
df, norm = evaluate_monitor(traces, anns, splits, key=("nll", "alea"), alpha=0.1)
print(json.dumps(headline(df), indent=1, default=float))
display(summarize(df))  # noqa: F821

# %% [markdown]
# ## 4. RQ1 oracle ranking: does the WM's uncertainty rank candidate chunks by eventual success?

# %%
tracker = FeatureTracker(ae, cams, n_arms=cfg_dp.n_arms, H=wm.H, hw=P["hw"])
scorers = [RandomScorer(0), WMScorer(wm, "alea", device=DEVICE), WMScorer(wm, "epi", device=DEVICE)]
branches = capture_branches(env, runner, tracker, bench.seeds("shift", P["n_branch"]), at_replans=(2, 4, 6))
succ, prog, scores = rq1_ranking(env, runner, tracker, scorers, branches, K=P["K"])
res, rows = rq1_metrics(succ, scores, prog)
print(json.dumps(res, indent=1))
for r in rows:
    print(r)
json.dump(dict(summary=res, scorers=rows), open(OUT / "rq1.json", "w"), indent=1)
