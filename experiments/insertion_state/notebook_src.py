# %% [markdown]
# # RJ45 insertion: contact state, guided search and port memory
#
# Can the robot tell from its own signals (wrist F/T, commanded vs measured TCP, joint torques) whether the
# plug went in, sits on the chamfer, or is next to the port on the panel face, and use that to get it in?
# And does remembering the recurring part of the perception error make the next insertion first-try?
#
# What the laptop runs showed (36 + 24 scenes, nominal jack; details in `notes/INSERTION_STATE.md`):
# * in vs out is identifiable 100 ms after contact (balanced acc 0.965, leave-scenes-out; 0.978 at 150 ms);
# * on the chamfer ramp the lateral force points to the hole (median 31 deg), on the face it does not (104 deg);
# * with retries, a contact-predicted correction beats a spiral search at 2 mm perception error
#   (0.84 vs 0.73 success, paired +0.11 [+0.04, +0.18]); a true one-shot correction reaches 1.00;
# * the privileged expert's own failures are the gripper hitting cables plugged in the row above, not alignment.
#
# This notebook scales that up and adds: a 100 Hz F/T stream and a 1-D CNN ensemble (GPU) vs the linear
# models; a tight, more realistic jack (0.6 mm clearance, 1 mm chamfer); belief-based search and sliding
# into the hole instead of retracting; and the port-memory stream (E3).
#
# **Run**: VS Code with a Colab A100 runtime (or Colab in the browser), Run All. `PRESET="a100"` takes
# about 3 h; the simulation runs on the VM's CPUs, only the CNN uses the GPU. Every stage writes shards and
# skips finished work, so after a disconnect just Run All again. Without a GPU the `smoke` preset runs
# (about 10 min on a laptop).

# %%
# @bootstrap

# %% [markdown]
# ## Experiment code
# The modules live in `RoboGym/experiments/insertion_state/` on the laptop and are not on the public repo
# yet, so they are inlined here. On Colab they are written to the clone; locally an existing file is never
# overwritten (set `INSERTION_FORCE_WRITE=1` to force).

# %%
EXP = pathlib.Path(ROOT) / "experiments" / "insertion_state"
EXP.mkdir(parents=True, exist_ok=True)
FORCE_WRITE = os.environ.get("INSERTION_FORCE_WRITE") == "1"


def write_module(name, src):
    p = EXP / name
    if p.exists() and not (IN_COLAB or FORCE_WRITE):
        if p.read_text() != src:
            print(f"{name}: the local file differs from the notebook copy, keeping the local file")
        return
    p.write_text(src)


if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

# %%
# @modules

# %%
import json
import time

import numpy as np
import torch
from IPython.display import Image, display

from robogym.tools import run_live

PRESET = os.environ.get("INSERTION_PRESET", "a100" if torch.cuda.is_available() else "smoke")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS = os.cpu_count()
P = dict(
    smoke=dict(jacks=["nominal"], e1_scenes=4, e1_radii=[1.0, 2.5, 4.0], cnn_steps=200, cnn_members=2,
               e2_scenes=3, e2_err=2, e2_sigmas=[0.002], e3_streams=2, e3_eps=6),
    a100=dict(jacks=["nominal", "tight"], e1_scenes=200, e1_radii=None, cnn_steps=4000, cnn_members=5,
              e2_scenes=48, e2_err=12, e2_sigmas=[0.001, 0.002, 0.003], e3_streams=24, e3_eps=40),
)[PRESET]
RUN = pathlib.Path(ROOT) / "runs" / "insertion_state_nb" / PRESET
RUN.mkdir(parents=True, exist_ok=True)
SUMMARY = RUN / "summary.json"
summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


def save_summary():
    SUMMARY.write_text(json.dumps(summary, indent=1, default=float))


def run(script, *args):
    run_live([sys.executable, str(EXP / script), *map(str, args)], cwd=str(ROOT))


print(PRESET, DEVICE, "workers", WORKERS, "->", RUN)
print(P)

# %% [markdown]
# ## 0. Check: the scene builds, the expert reaches the insertion, timing
# The VM runs the committed RoboGym; the laptop numbers came from a working tree with unrelated edits
# (cube stacking, hand-over), so this checks the cable task behaves the same.

# %%
from common import JACKS, Attempt, Probe, make_env, prepare  # noqa: E402

for jack in P["jacks"]:
    env = make_env(JACKS[jack])
    t0 = time.time()
    ex, snap = prepare(env, 1000)
    t_prep = time.time() - t0
    pb = Probe(env)
    port = int(env.scenario.task["cables"][0]["port"])
    t0 = time.time()
    rows = Attempt(env, pb, snap, port).run()
    print(f"{jack}: prepare {t_prep:.1f}s, one attempt {time.time() - t0:.1f}s, latched {rows[-1]['latched'] >= 0}, "
          f"high-rate wrench per step {rows[-1]['Fh'].shape}")

# %% [markdown]
# ## 1. E1: blind insertions with controlled port-estimate errors
# From the pre-insertion snapshot of each scene: lateral error on a polar grid (0-8 mm x 12 angles),
# axial error U(-2, 2) mm, one compliant force-limited push. Every step logs the 20 Hz signals, the 100 Hz
# wrench and the privileged labels. One shard per scene.

# %%
E1 = {}
for jack in P["jacks"]:
    E1[jack] = RUN / f"e1_{jack}"
    args = ["--n", P["e1_scenes"], "--seed0", 1000, "--jack", jack, "--workers", WORKERS, "--out", E1[jack]]
    if P["e1_radii"]:
        args += ["--radii", json.dumps(P["e1_radii"])]
    t0 = time.time()
    run("e1_offset_sweep.py", *args)
    print(f"{jack}: {time.time() - t0:.0f}s")

# %% [markdown]
# ## 2. Identifiability with linear models (leave-scenes-out)
# Contact is detected from the noisy tared wrench (|F| > 2.5 N); labels at that moment: `in` (enters the
# channel), `ramp` (on the chamfer), `face` (beyond it), `blocked` (gripper or plug on a neighbouring cable).
# In/out and 4-class logistic models, ridge regression of the lateral correction, 50-200 ms after contact.

# %%
import analyze_e1 as A  # noqa: E402

for jack in P["jacks"]:
    print(f"\n========== {jack} jack")
    res = A.load_runs(E1[jack])
    rows = A.analyze(res, fig_dir=RUN / f"figs_{jack}")
    summary.setdefault(jack, {})["linear"] = rows
    display(Image(str(RUN / f"figs_{jack}" / "e1_contact_state.png")))
    del res
save_summary()

# %% [markdown]
# ## 3. Temporal CNN ensemble on the 100 Hz window vs the linear models (GPU)
# Same split (one fifth of the scenes held out), same latency. The CNN also predicts a Gaussian over the
# correction; `cnn_cover_90` is the coverage of its 90 % ellipse on out-of-jack contacts (calibration matters
# for the belief search).

# %%
import seq_model as S  # noqa: E402

for jack in P["jacks"]:
    print(f"\n========== {jack} jack")
    res = A.load_runs(E1[jack])
    rows = S.evaluate(res, ks=(0, 1, 2), device=DEVICE, steps=P["cnn_steps"], n_members=P["cnn_members"])
    summary[jack]["cnn"] = rows
    del res
save_summary()
for jack in P["jacks"]:
    print(jack)
    for r in summary[jack]["cnn"]:
        print(f"  {r['ms']:3d} ms  in/out cnn {r['cnn_inout']:.3f} lin {r['lin_inout']:.3f} | 4-class cnn {r['cnn_4cls']:.3f} lin {r['lin_4cls']:.3f}"
              f" | dir ramp cnn {r['cnn_dir_ramp']:.0f} lin {r['lin_dir_ramp']:.0f} deg | dir face cnn {r['cnn_dir_face']:.0f} lin {r['lin_dir_face']:.0f} deg"
              f" | cover90 {r['cnn_cover_90']:.2f}")

# %% [markdown]
# ## 4. E2: closed-loop insertion with retries (new scenes)
# Shared in/out detector and push-deeper rule; all strategies start from the same drift-compensated aim.
# `blind` one push | `spiral` sunflower search | `learned` jump by the predicted correction | `belief` grid
# belief with negative information, aims where entry is most likely | `slide` belief, but slides to the new
# aim under a light preload instead of retracting | `oracle` jumps by the true miss. Budget 9 s, 6 tries.
# Scenes where the gripper hits a neighbouring cable are reported separately (lateral search cannot fix them).

# %%
import e2_guided as E2mod  # noqa: E402

for jack in P["jacks"]:
    print(f"\n========== {jack} jack")
    out = RUN / f"e2_{jack}"
    t0 = time.time()
    run("e2_guided.py", "--n", P["e2_scenes"], "--seed0", 3000, "--n_err", P["e2_err"], "--sigmas", *P["e2_sigmas"],
        "--jack", jack, "--workers", WORKERS, "--train", E1[jack], "--out", out)
    print(f"{jack}: {time.time() - t0:.0f}s")
    res = [r for f in sorted(out.glob("scene_*.json")) for r in json.loads(f.read_text())]
    summary[jack]["e2"] = E2mod.summarize(res, log=lambda *a: None)
    summary[jack]["e2_paired"] = {f"{a}-{b}@{s}": E2mod.paired(res, a, b, s) for s in P["e2_sigmas"]
                                  for a, b in (("belief", "spiral"), ("belief", "learned"), ("slide", "belief"), ("oracle", "belief"))}
save_summary()

# %% [markdown]
# ## 5. E3: remembering the port
# K=3 panel/camera contexts, each with a recurring lateral bias (sigma_b = 2 mm) plus per-episode noise
# (0.5 mm); sticky switching (mean dwell 5 episodes), cue correct with probability 0.8. All methods run the
# belief controller and differ only in the prior: none | pooled memory | memory per cued context | per true
# context. The memory sees only the robot's own measurement at the click.

# %%
import e3_recurring as E3mod  # noqa: E402

for jack in P["jacks"]:
    print(f"\n========== {jack} jack")
    out = RUN / f"e3_{jack}"
    t0 = time.time()
    run("e3_recurring.py", "--streams", P["e3_streams"], "--episodes", P["e3_eps"], "--jack", jack, "--workers", WORKERS,
        "--train", E1[jack], "--out", out)
    print(f"{jack}: {time.time() - t0:.0f}s")
    res = [r for f in sorted(out.glob("stream_*.json")) for r in json.loads(f.read_text())]
    summary[jack]["e3"] = E3mod.summarize(res)
save_summary()

# %% [markdown]
# ## 6. Summary

# %%
for jack in P["jacks"]:
    s = summary.get(jack, {})
    print(f"\n========== {jack} jack")
    for r in s.get("linear", []):
        if r["k"] == 1:
            print(f"linear @100 ms: in/out {r['inout_logreg']:.3f}, 4-class {r['cls4_logreg']:.3f}, dir ramp {r.get('dir_ramp_ridge', float('nan')):.0f} deg")
    for r in s.get("cnn", []):
        if r["k"] == 1:
            print(f"cnn    @100 ms: in/out {r['cnn_inout']:.3f}, 4-class {r['cnn_4cls']:.3f}, dir ramp {r['cnn_dir_ramp']:.0f} deg, cover90 {r['cnn_cover_90']:.2f}")
    print("E2 success (excl. blocked scenes):")
    for sig in P["e2_sigmas"]:
        cells = [r for r in s.get("e2", []) if r["sigma"] == sig]
        print(f"  sigma {sig * 1e3:.0f} mm: " + "  ".join(f"{r['strategy']} {r['success']:.2f}" for r in cells))
    for k, (d, lo, hi) in s.get("e2_paired", {}).items():
        print(f"  {k}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("E3:")
    for r in s.get("e3", []):
        print(f"  {r['method']:7s} success {r['success']:.2f}, first-try {r['first_try']:.2f}, first-try on returns {r['first_try_returns']:.2f}, tries {r['tries']:.2f}")
print(f"\nsummary -> {SUMMARY}")

# %% [markdown]
# The full summary as JSON, so the results travel back with the notebook outputs even when Drive is not
# mounted on the VM (VS Code keeps cell outputs in the local .ipynb).

# %%
print(json.dumps(summary, default=float))
