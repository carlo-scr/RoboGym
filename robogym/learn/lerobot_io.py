"""Bridge to LeRobot (github.com/huggingface/lerobot, written against 0.4.4): export RoboGym episode
folders to the LeRobotDataset format, train LeRobot's own policies (ACT, Diffusion Policy, VQ-BeT, and
the pretrained SmolVLA / pi0.5 as fine-tunes) with `lerobot-train`, resumably, and load a checkpoint with
its pre/post-processors for closed-loop use behind robogym.policies.LeRobotPolicy.

Dataset variants (what a policy family sees):
  full   robot state (28 per arm) + both cameras + `observation.environment_state` = the privileged task
         features, so ACT / Diffusion are comparable to the in-house state-based policies
  front  one camera, no environment state (VQ-BeT accepts exactly one image feature)
  vla    both cameras, no environment state (the VLAs pad the state to 32 dims; 28 fits, 28 + features not)
Actions are the env's delta actions; with DART data the label is the expert's clean action.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

VARIANTS = {
    "full": dict(cams=("right_wrist", "front"), env_state=True),
    "front": dict(cams=("front",), env_state=False),
    "vla": dict(cams=("right_wrist", "front"), env_state=False),
}
POLICY_VARIANT = {"act": "full", "diffusion": "full", "vqbet": "front", "smolvla": "vla", "pi0": "vla", "pi05": "vla"}
PRETRAINED = {"smolvla": "lerobot/smolvla_base", "pi05": "lerobot/pi05_base", "pi0": "lerobot/pi0_base"}  # hub ids for the fine-tuned VLAs
# control runs at 20 Hz: a 32-step chunk is 1.6 s (ACT's 100 at 50 Hz is 2 s)
POLICY_ARGS = {
    "act": ["--policy.chunk_size=32", "--policy.n_action_steps=16"],
    "diffusion": ["--policy.horizon=16", "--policy.n_action_steps=8", "--policy.n_obs_steps=2", "--policy.noise_scheduler_type=DDIM", "--policy.num_inference_steps=16"],
    "vqbet": ["--policy.crop_shape=null", "--policy.n_obs_steps=3"],
    "smolvla": ["--policy.chunk_size=32", "--policy.n_action_steps=16"],
    "pi05": ["--policy.chunk_size=32", "--policy.n_action_steps=16"],
    "pi0": ["--policy.chunk_size=32", "--policy.n_action_steps=16"],
}
MARKER = "robogym_export.json"
SIDECAR = "robogym_lerobot.json"


def export_lerobot(data_root, out_root, repo_id, variant="full", fps=20.0, only_success=True, max_episodes=None, use_videos=False, vcodec="libsvtav1",
                   image_writer_threads=4, log=print):
    """RoboGym dataset folder -> LeRobotDataset at out_root (skipped if the export marker exists; a partial
    export is removed and redone). Returns the export info dict."""
    import datasets as hf_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .data import list_episodes, load_episode_arrays

    hf_datasets.disable_progress_bars()

    out_root = Path(out_root).resolve()
    if (out_root / MARKER).exists():
        return json.load(open(out_root / MARKER))
    if out_root.exists():
        shutil.rmtree(out_root)
    v = VARIANTS[variant]
    cams = tuple(v["cams"])
    idx = list_episodes(data_root)
    if only_success:
        idx = idx[idx.outcome == "success"]
    paths = idx.path.tolist()[:max_episodes]
    probe = load_episode_arrays(paths[0], cams=cams, frames=True)
    fr = json.load(open(Path(paths[0]) / "feature_report.json"))
    S, A, E = probe["state"].shape[1], probe["action"].shape[1], probe["task"].shape[1]
    H, W = probe["frames"][cams[0]].shape[1:3]
    features = {"observation.state": {"dtype": "float32", "shape": (S,), "names": list(fr["observation_state_names"])},
                "action": {"dtype": "float32", "shape": (A,), "names": list(fr["action_names"])}}
    if v["env_state"]:
        features["observation.environment_state"] = {"dtype": "float32", "shape": (E,), "names": list(fr["task_feature_names"])}
    for c in cams:
        features[f"observation.images.{c}"] = {"dtype": "video" if use_videos else "image", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id, fps=int(round(fps)), features=features, root=out_root, robot_type="xarm7", use_videos=use_videos,
                               image_writer_threads=image_writer_threads, vcodec=vcodec)
    t0, n_frames = time.time(), 0
    for i, p in enumerate(paths):
        ep = probe if i == 0 else load_episode_arrays(p, cams=cams, frames=True)
        acts = ep["action_expert"] if ep.get("action_expert") is not None else ep["action"]
        task = str(ep["ann"]["goal"])
        T = min(len(ep["state"]), *[len(ep["frames"][c]) for c in cams])
        for t in range(T):
            frame = {"observation.state": ep["state"][t].astype(np.float32), "action": acts[t].astype(np.float32), "task": task}
            if v["env_state"]:
                frame["observation.environment_state"] = ep["task"][t].astype(np.float32)
            for c in cams:
                frame[f"observation.images.{c}"] = np.ascontiguousarray(ep["frames"][c][t])
            ds.add_frame(frame)
        ds.save_episode()
        n_frames += T
        if log and ((i + 1) % 20 == 0 or i + 1 == len(paths)):
            log(f"[lerobot export] {i + 1}/{len(paths)} episodes, {n_frames} frames ({time.time() - t0:.0f}s)")
    ds.finalize()
    info = dict(repo_id=repo_id, root=str(out_root), variant=variant, cams=list(cams), env_state=bool(v["env_state"]), n_arms=int(probe["n_arms"]), state_dim=S, action_dim=A,
                task_dim=E, hw=[int(H), int(W)], fps=fps, n_episodes=len(paths), n_frames=n_frames, source=str(data_root), use_videos=use_videos)
    json.dump(info, open(out_root / MARKER, "w"), indent=1)
    return info


def checkpoint_step(out_dir) -> int | None:
    p = Path(out_dir) / "checkpoints" / "last" / "training_state" / "training_step.json"
    return int(json.load(open(p))["step"]) if p.exists() else None


def pretrained_dir(out_dir) -> Path:
    return Path(out_dir) / "checkpoints" / "last" / "pretrained_model"


def prune_checkpoints(out_dir, keep=1) -> int:
    """Delete all but the newest `keep` numbered checkpoint directories of a lerobot run (lerobot keeps every
    one, and a Diffusion checkpoint with its optimizer state is ~1.5 GB). `last` and the directory it points
    to are always kept. Returns the number of directories removed."""
    cdir = Path(out_dir) / "checkpoints"
    if not cdir.exists():
        return 0
    last = (cdir / "last").resolve() if (cdir / "last").exists() else None
    nums = sorted((p for p in cdir.iterdir() if p.is_dir() and not p.is_symlink() and p.name.isdigit()), key=lambda p: int(p.name))
    n = 0
    for p in nums[: len(nums) - max(0, keep)]:
        if last is not None and p.resolve() == last:
            continue
        shutil.rmtree(p, ignore_errors=True)
        n += 1
    return n


def train_lerobot(policy_type, dataset_info: dict, out_dir, steps, batch_size=8, device="cpu", save_freq=None, num_workers=2, log_freq=None, extra=(), pretrained=None,
                  python=None, log=print, log_path=None):
    """Run `lerobot-train` for `steps` (resumes from checkpoints/last if the run was interrupted; returns
    at once if the final checkpoint exists). `pretrained`: a hub id / path for fine-tuning (VLAs); else
    `--policy.type=<policy_type>` from scratch. `log_path`: send the trainer's output to that file instead
    of the console (used when several trainings run at once). Returns the pretrained_model directory."""
    from ..tools import run_live

    out_dir = Path(out_dir).resolve()
    root = Path(dataset_info["root"]).resolve()
    done = checkpoint_step(out_dir)
    if done is not None and done >= steps:
        return pretrained_dir(out_dir)
    base = [python or sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    if done is not None:
        cmd = base + [f"--config_path={pretrained_dir(out_dir) / 'train_config.json'}", "--resume=true", f"--steps={steps}"]
        if log:
            log(f"[lerobot train] resuming {policy_type} from step {done}/{steps}")
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)  # a run that died before its first checkpoint
        pol = [f"--policy.path={pretrained or PRETRAINED[policy_type]}"] if (pretrained or policy_type in PRETRAINED) else [f"--policy.type={policy_type}"]
        cmd = base + [f"--dataset.repo_id={dataset_info['repo_id']}", f"--dataset.root={root}", *pol, f"--policy.device={device}", "--policy.push_to_hub=false",
                      f"--output_dir={out_dir}", f"--steps={steps}", f"--batch_size={batch_size}", f"--save_freq={save_freq or max(50, steps // 10)}", "--eval_freq=0",
                      f"--log_freq={log_freq or max(10, steps // 50)}", f"--num_workers={num_workers}", "--wandb.enable=false", *POLICY_ARGS.get(policy_type, []), *extra]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    run_live(cmd, cwd=str(out_dir.parent), skip=("Warning", "warn"), sink=log_path)
    done = checkpoint_step(out_dir)
    if done is None or done < steps:
        raise RuntimeError(f"lerobot training of {policy_type} ended at step {done} < {steps}")
    prune_checkpoints(out_dir, keep=1)
    json.dump(dict(policy_type=policy_type, steps=steps, dataset=dataset_info, cmd=[str(c) for c in cmd]), open(out_dir / SIDECAR, "w"), indent=1)
    return pretrained_dir(out_dir)


def _progress(log_path) -> str:
    """The last `step:... loss:...` fragment of a lerobot training log ("" if none yet)."""
    try:
        lines = [ln for ln in Path(log_path).read_text(errors="replace").splitlines() if "step:" in ln]
    except OSError:
        return ""
    if not lines:
        return ""
    toks = lines[-1].split()
    keep = [t for t in toks if t.startswith(("step:", "epch:", "loss:", "updt_s:", "data_s:"))]
    return " ".join(keep)


def train_log(out_dir) -> Path:
    out_dir = Path(out_dir).resolve()
    return out_dir.parent / f"{out_dir.name}.train.log"


def train_lerobot_many(jobs, max_parallel=1, poll_s=60, log=print):
    """Run several `train_lerobot` calls, up to `max_parallel` at a time (each is its own subprocess, so
    small policies can share one GPU). `jobs`: dicts of train_lerobot keyword arguments plus a `name`;
    each trainer's output goes to <out_dir>.train.log (beside the run directory, which lerobot insists on
    creating itself) and one progress line per running job is printed every `poll_s` seconds. Already
    finished runs return at once; older numbered checkpoints are pruned as the runs go (the newest two,
    then only the last). Returns {name: minutes}; after all jobs ended, raises RuntimeError listing the
    failed ones (the log tail is in each .train.log)."""
    from concurrent.futures import ThreadPoolExecutor, wait

    def one(kw):
        kw = dict(kw)
        name = kw.pop("name")
        t0 = time.time()
        train_lerobot(**kw, log=None, log_path=train_log(kw["out_dir"]))
        return name, (time.time() - t0) / 60

    mins, errors = {}, {}
    pending = {}
    with ThreadPoolExecutor(max(1, int(max_parallel))) as ex:
        for kw in jobs:
            pending[ex.submit(one, kw)] = kw
        t_log = time.time()
        while pending:
            done, _ = wait(list(pending), timeout=poll_s)
            for f in done:
                kw = pending.pop(f)
                try:
                    name, m = f.result()
                    mins[name] = round(m, 1)
                    if log:
                        log(f"[{name}] trained ({kw['steps']} steps, {m:.1f} min)")
                except Exception as e:  # noqa: BLE001
                    errors[kw["name"]] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
                    if log:
                        log(f"[{kw['name']}] FAILED: {errors[kw['name']]} (see {train_log(kw['out_dir'])})")
            if pending and time.time() - t_log >= poll_s - 1:
                running = [kw for f, kw in pending.items() if f.running()]
                for kw in running[:max_parallel]:
                    prune_checkpoints(kw["out_dir"], keep=2)  # keep the disk (or Drive) from filling with 1.5 GB checkpoints
                    if log:
                        log(f"  [{kw['name']}] {_progress(train_log(kw['out_dir'])) or 'starting'} / {kw['steps']} steps")
                t_log = time.time()
    if errors:
        raise RuntimeError("lerobot training failed for " + ", ".join(f"{k} ({v})" for k, v in errors.items()))
    return mins


def load_lerobot(ckpt, device="cpu"):
    """A checkpoint (`pretrained_model` dir, or a training output dir) -> (policy, preprocessor,
    postprocessor, info). `info` is the export info (cameras, env state, image size) when the sidecar exists."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    ckpt = Path(ckpt)
    if (ckpt / "checkpoints").exists():
        ckpt = pretrained_dir(ckpt)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.pretrained_path = ckpt
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}})
    info = {}
    for d in (ckpt, *ckpt.parents[:3]):
        if (d / SIDECAR).exists():
            info = json.load(open(d / SIDECAR))["dataset"]
            break
    if not info:  # fall back to the checkpoint's own feature list
        info = dict(cams=[k.split(".")[-1] for k in cfg.input_features if k.startswith("observation.images.")], env_state="observation.environment_state" in cfg.input_features,
                    hw=[cfg.input_features[k].shape[1:] for k in cfg.input_features if k.startswith("observation.images.")][0] if any(k.startswith("observation.images.") for k in cfg.input_features) else None)
    return policy, pre, post, info
