# %% [markdown]
# # Off-the-shelf policies: train, benchmark, GIFs and figures
# LeRobot's own implementations (ACT, Diffusion Policy, VQ-BeT, and the pretrained SmolVLA / pi0.5 as
# fine-tunes on a GPU) trained on RoboGym demonstrations, evaluated with the long-horizon benchmark next to
# the scripted expert, the hierarchical planner / executor policies and the in-house architectures from
# notebook 04 when they exist, then rollout GIFs and summary figures. Resumable like notebook 04: every
# stage checks its outputs on disk; LeRobot training resumes from its own checkpoints.
#
# Stages (`ROBOGYM_STAGES`, default all): **data** (demonstrations, as in 04), **export** (LeRobot dataset
# variants: `full` = state + 2 cameras + privileged features for ACT / Diffusion, `front` = one camera for
# VQ-BeT, `vla` = state + 2 cameras for the VLAs), **train** (`lerobot-train`, resumable), **eval** (the
# benchmark, `scripts/eval_policies.py`), **gifs** (captioned rollouts per policy), **figures**.
#
# Knobs: `ROBOGYM_PRESET` (smoke | fast | a100; notebook 06 is this notebook with `fast` preselected: an
# A100 run in about an hour), `ROBOGYM_TASKS`, `ROBOGYM_LEROBOT_POLICIES` (default `act,diffusion,vqbet`;
# add `smolvla,pi05` on a GPU: they download the pretrained weights from the hub), `ROBOGYM_PARALLEL_TRAIN`
# (trainings sharing the GPU at once), `ROBOGYM_OUT` (output directory), `LEROBOT_VIDEO=1` (video export
# instead of PNG frames; needs a working torchcodec at training time), `ROBOGYM_FORCE=1`. Needs
# `pip install lerobot` (done below on Colab).

# %%
# @bootstrap

# %%
import importlib, subprocess, sys
# lerobot 0.6 splits its dependencies into extras: the dataset module needs `datasets` and PyAV (`av`), the trainer
# needs `accelerate`; a bare `pip install lerobot` leaves them out (0.4 has them as core deps: the extras then only
# produce a warning). Verified with the exact imports the pipeline needs. lerobot caches a "package missing"
# verdict for the life of the kernel, so its modules are dropped first and the check re-imports them.


def _drop_modules(*tops):
    for _m in [m for m in sys.modules if m.split(".")[0] in tops]:
        del sys.modules[_m]
    importlib.invalidate_caches()


def _unpoison_datasets():
    """A kernel that imported `datasets`, dropped it and imports it again hits pyarrow's 'extension type already
    defined'; unregister those types so the fresh import can register them."""
    import pyarrow as pa

    for n in ("Array2D", "Array3D", "Array4D", "Array5D"):
        try:
            pa.unregister_extension_type(f"datasets.features.features.{n}ExtensionType")
        except Exception:  # noqa: BLE001
            pass
    _drop_modules("datasets")


def _lerobot_ready(retry=True):
    _drop_modules("lerobot")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401  (needs datasets + av)
        import accelerate  # noqa: F401  (the trainer)
        return True
    except ImportError as e:
        print("lerobot not ready:", e)
        return False
    except Exception as e:  # noqa: BLE001
        if retry and "already defined" in str(e):
            _unpoison_datasets()
            return _lerobot_ready(retry=False)
        raise RuntimeError(f"the kernel state is broken ({type(e).__name__}: {str(e)[:120]}): restart the kernel (VS Code: the Restart button "
                           "in the notebook toolbar; Colab: Runtime > Restart session), then run all") from None


if not _lerobot_ready():
    print("installing lerobot[dataset,training] ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lerobot[dataset,training]", "av", "datasets", "accelerate"], check=True)
    assert _lerobot_ready(), "still failing after the install: restart the kernel, then run all"


def _trainer_imports():
    """The trainer's import chain (cv2, torchvision, av, accelerate, ...) in a fresh process, the way the training subprocesses run it;
    the kernel-side dataset code imports less, so a broken module shows up only here."""
    r = subprocess.run([sys.executable, "-c", "import lerobot.scripts.lerobot_train"], capture_output=True, text=True)
    return r.returncode == 0, [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-10:]


_ok, _err = _trainer_imports()
if not _ok and any("cv2" in ln for ln in _err):
    # Colab ships opencv-python + opencv-contrib-python; lerobot adds opencv-python-headless and the mixed builds break `import cv2`
    print("fixing the opencv install (mixed builds):", *_err[-2:], sep="\n  ")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=True)
    _ok, _err = _trainer_imports()
assert _ok, "the lerobot trainer does not import in a fresh process:\n" + "\n".join(_err)
print("lerobot trainer imports OK")

# %%
import json, os, time
from pathlib import Path
import pandas as pd
import torch

if "ROOT" not in globals():  # running the source as a plain script
    ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT))
from robogym import bench, lh, viz
from robogym.env import RoboGymEnv
from robogym.learn.data import list_episodes
from robogym.learn.lerobot_io import POLICY_VARIANT, PRETRAINED, checkpoint_step, export_lerobot, train_lerobot_many
from robogym.policies import make_policy
from robogym.tools import run_live

REPO = Path(ROOT)
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
PRESET = os.environ.get("ROBOGYM_PRESET", "a100" if torch.cuda.is_available() else "smoke")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = [t for t in os.environ.get("ROBOGYM_TASKS", "cube_stack,lego,cable_plug").split(",") if t]
STAGES = set(os.environ.get("ROBOGYM_STAGES", "data,export,train,eval,gifs,figures").split(","))
FORCE = os.environ.get("ROBOGYM_FORCE", "0") == "1"
POLICIES = [p for p in os.environ.get("ROBOGYM_LEROBOT_POLICIES", "act,diffusion,vqbet").split(",") if p]
USE_VIDEO = os.environ.get("LEROBOT_VIDEO", "0") == "1"
CPU = os.cpu_count() or 4
P = dict(
    # smoke: a CPU pass through every stage in minutes (nothing learns anything).
    smoke=dict(n_train=40, steps=dict(act=60, diffusion=60, vqbet=60, smolvla=20, pi05=20), bs=dict(act=8, diffusion=8, vqbet=8, smolvla=4, pi05=2), workers=min(8, CPU),
               lh_n=4, gif_n=1, hw=(72, 96), extra=dict(vqbet=["--policy.n_vqvae_training_steps=20"]), parallel_train=1),
    # fast: first results on one A100 in about an hour (300 demos, 6k steps at batch 64 with the three policies training
    # concurrently, 20 paired seeds per horizon). Numbers are indicative; the a100 preset is the full run.
    fast=dict(n_train=300, steps=dict(act=6000, diffusion=6000, vqbet=6000, smolvla=3000, pi05=2000), bs=dict(act=64, diffusion=64, vqbet=64, smolvla=16, pi05=8),
              workers=CPU, lh_n=20, gif_n=2, hw=(72, 96), extra=dict(vqbet=["--policy.n_vqvae_training_steps=1500"]), parallel_train=3),
    a100=dict(n_train=2000, steps=dict(act=50000, diffusion=50000, vqbet=50000, smolvla=20000, pi05=10000), bs=dict(act=32, diffusion=32, vqbet=32, smolvla=16, pi05=8),
              workers=CPU, lh_n=50, gif_n=3, hw=(72, 96), extra=dict(vqbet=["--policy.n_vqvae_training_steps=10000"]), parallel_train=3),
)[PRESET]
PARALLEL_TRAIN = int(os.environ.get("ROBOGYM_PARALLEL_TRAIN", P["parallel_train"]))
CAMS = ("right_wrist", "front")
CAM_CFG = json.dumps({"camera.width": P["hw"][1], "camera.height": P["hw"][0]})
OUT = Path(os.environ.get("ROBOGYM_OUT") or REPO / "runs" / f"offtheshelf_{PRESET}")
OUT.mkdir(parents=True, exist_ok=True)
LEROBOT_ROOT = Path("/content/lerobot_data") if IN_COLAB else REPO / "data" / "lerobot"  # many small files: keep off Drive
WORK = Path("/content/lerobot_runs") if IN_COLAB else None  # trainings run here (local disk); the final checkpoint is copied to OUT (Drive has no symlinks)
STATUS = OUT / "status.json"
status = json.load(open(STATUS)) if STATUS.exists() else {}


def done(path):
    return Path(path).exists() and not FORCE


def mark(stage, key, **kw):
    status.setdefault(stage, {})[key] = dict(done=True, at=time.strftime("%Y-%m-%d %H:%M"), **kw)
    json.dump(status, open(STATUS, "w"), indent=1, default=str)


def run_script(name, *args):
    run_live([sys.executable, str(REPO / "scripts" / name), *map(str, args)], cwd=str(REPO))


def train_dir(task):
    return REPO / "data" / f"{task}_train_{PRESET}"


def ckpt_dir(task, ptype):
    return OUT / task / ptype


print(f"preset {PRESET} on {DEVICE}, tasks {TASKS}, policies {POLICIES}, stages {sorted(STAGES)}, {PARALLEL_TRAIN} concurrent trainings, out {OUT}, lerobot data {LEROBOT_ROOT}"
      + (f", training in {WORK}" if WORK else ""))

# %% [markdown]
# ## 1. Demonstrations (shared with notebook 04)

# %%
if "data" in STAGES:
    for task in TASKS:
        d = train_dir(task)
        if not list(d.glob("index_*.parquet")) or FORCE:
            run_script("generate_dataset.py", "--task", task, "--out", d, "--n", P["n_train"], "--seed0", bench.SPLITS["train"][0], "--mix", "1,0,0,0", "--dart", "0.002",
                       "--cameras", *CAMS, "--no-states", "--workers", P["workers"], "--cfg", CAM_CFG)
        idx = list_episodes(d)
        print(f"[{task}] {len(idx)} demonstrations, {idx.outcome.value_counts().to_dict()}")
        mark("data", task, n=int(len(idx)))

# %% [markdown]
# ## 2. Export to the LeRobot dataset format (one variant per policy family)

# %%
EXPORTS = {}
if "export" in STAGES or "train" in STAGES:
    for task in TASKS:
        for variant in sorted({POLICY_VARIANT[p] for p in POLICIES}):
            root = LEROBOT_ROOT / f"{task}_{PRESET}_{variant}"
            info = export_lerobot(train_dir(task), root, f"robogym/{task}_{PRESET}_{variant}", variant=variant, use_videos=USE_VIDEO, vcodec="h264")
            EXPORTS[(task, variant)] = info
            print(f"[{task}/{variant}] {info['n_episodes']} episodes, {info['n_frames']} frames, cams {info['cams']}, env_state {info['env_state']}")
            mark("export", f"{task}/{variant}", n_frames=info["n_frames"])

# %% [markdown]
# ## 3. Train LeRobot policies (resumable)
# ACT / Diffusion / VQ-BeT from scratch; SmolVLA / pi0.5 fine-tuned from the hub weights (GPU). Each run
# writes checkpoints under `runs/offtheshelf_<preset>/<task>/<policy>/checkpoints` and its log to
# `<policy>.train.log` beside it; a rerun resumes. `PARALLEL_TRAIN` runs share the GPU (these models are small, so three
# at once roughly triple the throughput of an A100).

# %%
if "train" in STAGES:
    jobs = []
    for task in TASKS:
        for ptype in POLICIES:
            if ptype in PRETRAINED and DEVICE != "cuda":
                print(f"[{task}/{ptype}] skipped: fine-tuning a VLA needs a GPU")
                continue
            steps = P["steps"][ptype]
            if (checkpoint_step(ckpt_dir(task, ptype)) or 0) >= steps and not FORCE:
                print(f"[{task}/{ptype}] trained ({steps} steps)")
                continue
            jobs.append(dict(name=f"{task}/{ptype}", policy_type=ptype, dataset_info=EXPORTS[(task, POLICY_VARIANT[ptype])], out_dir=ckpt_dir(task, ptype), steps=steps,
                             batch_size=P["bs"][ptype], device=DEVICE, num_workers=max(1, min(4, CPU // PARALLEL_TRAIN)), extra=P["extra"].get(ptype, []),
                             work_dir=WORK / task / ptype if WORK else None))
    if jobs:
        print(f"training {len(jobs)} policies, {PARALLEL_TRAIN} at a time: {[j['name'] for j in jobs]}")
        mins = train_lerobot_many(jobs, max_parallel=PARALLEL_TRAIN, poll_s=60)
        for j in jobs:
            mark("train", j["name"], steps=j["steps"], train_min=mins[j["name"]])

# %% [markdown]
# ## 4. Benchmark: off-the-shelf policies next to the expert, hierarchical and in-house policies
# Paired test seeds at every horizon; success, progress, first-failure attribution, horizon curves.

# %%
def policies_for(task):
    pols = ["expert", "hier:oracle+scripted", "hier:heuristic+scripted", "hier:corrupt(p_goal=0.5)+scripted", "hier:oracle+scripted(failure=any,p_failure=0.3,severity=0.7):on_fail=retry"]
    for ptype in POLICIES:
        if (checkpoint_step(ckpt_dir(task, ptype)) or 0) >= P["steps"][ptype]:
            pols.append(f"lerobot:{ckpt_dir(task, ptype)}")
    for arch in ("dp", "act", "mlp"):  # in-house architectures from notebook 04, if trained
        ck = REPO / "runs" / f"bench_{PRESET}" / task / f"{arch}_state" / "policy.pt"
        if ck.exists():
            pols.append(f"chunk:{ck}")
    return pols


if "eval" in STAGES:
    for task in TASKS:
        out = OUT / task / "lh"
        pols = policies_for(task)
        run_script("eval_policies.py", "--task", task, "--horizons", *bench.HORIZONS[task], "--n", P["lh_n"], "--workers", P["workers"], "--out", out, "--cfg", CAM_CFG,
                   "--device", DEVICE, "--policies", *pols)  # on CUDA every worker holds its policies on the GPU
        mark("eval", task, n_policies=len(pols))

# %% [markdown]
# ## 5. GIFs: captioned rollouts (front + wrist camera) per policy, successes and failures

# %%
if "gifs" in STAGES:
    items = []
    for task in TASKS:
        gdir = OUT / task / "gifs"
        gdir.mkdir(parents=True, exist_ok=True)
        env = RoboGymEnv(task, render=True)  # full-resolution cameras for the GIFs; policies render their own size
        n_top = bench.HORIZONS[task][-1]
        specs = lh.make_specs(task, horizons=(n_top,), n=P["gif_n"])
        for spec_s in ["expert", "hier:corrupt(p_goal=0.5)+scripted"] + [p for p in policies_for(task) if p.startswith(("lerobot:", "chunk:"))]:
            pol = make_policy(spec_s, device=DEVICE)
            for sp in specs:
                stem = f"{pol.name}_seed{sp.seed}_n{n_top}".replace("[", "_").replace("]", "").replace("(", "_").replace(")", "").replace("+", "_").replace(";", "_").replace(",", "_").replace(":", "_")
                gif = gdir / f"{stem}.gif"
                if done(gif):
                    items.append((gif, f"{task}: {pol.name}, seed {sp.seed}"))
                    continue
                frames, row = viz.rollout_frames(env, pol, sp)
                path, mb = viz.write_gif(frames, gif)
                viz.contact_sheet(frames, n=6, path=gdir / f"{stem}_sheet.png")
                items.append((gif, f"{task}: {pol.name}, seed {sp.seed}: {'success' if row['success'] else row['attribution']}"))
                print(f"[{task}] {pol.name} seed {sp.seed}: {'success' if row['success'] else row['attribution']} -> {path.name} ({mb} MB)")
        env.close()
        mark("gifs", task, n=len([i for i in items if str(i[0]).startswith(str(gdir))]))
    viz.gallery_html(items, OUT / "gallery.html", title=f"RoboGym off-the-shelf policies ({PRESET})")
    print(f"gallery: {OUT / 'gallery.html'}")

# %% [markdown]
# ## 6. Figures and tables

# %%
if "figures" in STAGES:
    FAMILIES = [
        ("expert", lambda n: n == "expert"),
        ("heuristic plan + scripted", lambda n: n.startswith("hier[heuristic+scripted]")),
        ("wrong-goal plans (p .5) + scripted", lambda n: "goal0.5" in n),
        ("LeRobot ACT", lambda n: n.startswith("lerobot_act")),
        ("LeRobot Diffusion", lambda n: n.startswith("lerobot_diffusion")),
        ("LeRobot VQ-BeT", lambda n: n.startswith("lerobot_vqbet")),
        ("LeRobot SmolVLA / pi0.5", lambda n: n.startswith(("lerobot_smolvla", "lerobot_pi0"))),
        ("in-house DP (state)", lambda n: n.startswith("dp_state")),
    ]
    curves, attrs, rels, summaries = {}, {}, {}, []
    for task in TASKS:
        out = OUT / task / "lh"
        if not (out / "episodes.parquet").exists():
            continue
        df = pd.read_parquet(out / "episodes.parquet")
        curves[task] = lh.horizon_curve(df)
        attrs[task] = lh.attribution_table(df)
        rels[task] = lh.reliability(df)
        summaries.append(lh.summarize(df).assign(task=task))
    if curves:
        summ = pd.concat(summaries)
        summ.to_csv(OUT / "summary.csv", index=False)
        pd.set_option("display.width", 220)
        for task in curves:
            print(f"\n== {task}: success per policy / horizon ==")
            print(summ[summ.task == task].pivot(index="policy", columns="horizon", values="success").round(2).to_string())
            print(f"\n== {task}: first-failure attribution ==")
            print(attrs[task].round(2).to_string())
        viz.horizon_figure(curves, FAMILIES, path=OUT / "horizon_curves.png", n_seeds=P["lh_n"])
        viz.attribution_figure(attrs, path=OUT / "attribution.png")
        viz.reliability_figure(rels, FAMILIES, path=OUT / "reliability.png")
        print(f"\nfigures: {OUT / 'horizon_curves.png'}, {OUT / 'attribution.png'}, {OUT / 'reliability.png'}")
        mark("figures", "all", tasks=list(curves))
    print(json.dumps(status, indent=1, default=str))
