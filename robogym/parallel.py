"""Resumable, time-budgeted process pool for episode generation / evaluation jobs.

Each worker builds one RoboGymEnv per EnvCfg (cached by cfg hash) and runs `fn(env, job)`; results are
written in shards as they arrive so a killed run can resume. Workers use one BLAS thread each.
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import pickle
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ENVS: dict = {}


@contextlib.contextmanager
def _init_lock(render: bool):
    """Serialise render-context creation across worker processes. On macOS two processes creating their
    GL contexts at the same moment can deadlock inside Apple's shader cache (a file lock in CoreFSCache
    that is never released), which stalls the whole pool; creating them one at a time avoids the race."""
    if not render or os.name != "posix":
        yield
        return
    import fcntl

    with open(Path(tempfile.gettempdir()) / "robogym_env_init.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _init_worker():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if os.name == "posix" and not os.environ.get("DISPLAY") and os.uname().sysname == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")


def _env_for(cfg_dict: dict, render: bool):
    from .config import EnvCfg
    from .env import RoboGymEnv

    key = (json.dumps(cfg_dict, sort_keys=True, default=str), render)
    if key not in _ENVS:
        _ENVS.clear()  # one env per worker is plenty; avoid leaking GL contexts
        with _init_lock(render):
            _ENVS[key] = RoboGymEnv(EnvCfg.from_dict(cfg_dict), render=render)
    return _ENVS[key]


def _run(fn, cfg_dict, render, job):
    try:
        env = _env_for(cfg_dict, render)
        t0 = time.time()
        res = fn(env, job)
        return dict(ok=True, job=job, result=res, wall_s=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, job=job, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())


def run_jobs(fn, jobs, cfg, out_dir=None, workers=None, render=True, budget_s=None, flush_every=20, log=print, job_key=lambda j: str(j), start_method=None):
    """Run fn(env, job) for every job across `workers` processes. Results are returned as a list of dicts
    (ok, job, result | error) and, if out_dir is given, appended to shard_*.pkl files; already finished
    jobs (by `job_key`) are skipped on resume. `budget_s` stops submitting new jobs after the budget.
    `start_method` ("spawn" | "fork" | "forkserver"; default: ROBOGYM_MP_START or the platform default):
    use "spawn" when the workers run policies on CUDA, which cannot be used from a forked process."""
    cfg_dict = cfg.to_dict()
    done_keys = set()
    out = []
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(out_dir.glob("shard_*.pkl")):
            for r in pickle.load(open(p, "rb")):
                done_keys.add(job_key(r["job"]))
                out.append(r)
        if done_keys:
            log(f"resuming: {len(done_keys)} jobs already done")
    todo = [j for j in jobs if job_key(j) not in done_keys]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t_start = time.time()
    buf, n_ok, n_err = [], 0, 0
    shard_i = len(list(out_dir.glob("shard_*.pkl"))) if out_dir else 0

    def flush():
        nonlocal buf, shard_i
        if out_dir and buf:
            pickle.dump(buf, open(out_dir / f"shard_{shard_i:04d}.pkl", "wb"))
            shard_i += 1
            buf = []

    start_method = start_method or os.environ.get("ROBOGYM_MP_START") or None
    ctx = multiprocessing.get_context(start_method) if start_method else None
    with ProcessPoolExecutor(workers, initializer=_init_worker, mp_context=ctx) as ex:
        futs = {}
        it = iter(todo)
        # keep ~2 jobs per worker in flight so a time budget can stop submission
        def submit_next():
            try:
                j = next(it)
            except StopIteration:
                return False
            futs[ex.submit(_run, fn, cfg_dict, render, j)] = j
            return True

        for _ in range(2 * workers):
            if not submit_next():
                break
        t_log = time.time()
        while futs:
            for f in as_completed(list(futs)):
                futs.pop(f)
                r = f.result()
                out.append(r)
                buf.append(r)
                n_ok += r["ok"]
                n_err += not r["ok"]
                if not r["ok"]:
                    log(f"job failed: {r['error']}")
                if len(buf) >= flush_every:
                    flush()
                if budget_s is None or time.time() - t_start < budget_s:
                    submit_next()
                if time.time() - t_log > 30:
                    done = n_ok + n_err
                    rate = done / max(1e-9, time.time() - t_start)
                    log(f"  {done}/{len(todo)} done, {n_err} errors, {rate * 60:.1f}/min, {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min left")
                    t_log = time.time()
                break
    flush()
    log(f"finished {n_ok + n_err}/{len(todo)} jobs ({n_err} errors) in {(time.time() - t_start) / 60:.1f} min")
    return out


# ---------------------------------------------------------------------- the standard dataset job
def episode_job(env, job):
    """job = dict(spec=EpisodeSpec dict, out=path, cameras=[...], save_states=bool). Writes the episode, returns the summary row."""
    from .datagen import EpisodeSpec, run_episode, summary_row, write_episode

    spec = EpisodeSpec.from_dict(job["spec"])
    ep = run_episode(env, spec, cameras=job.get("cameras"), save_states=job.get("save_states", True))
    write_episode(ep, job["out"], fps=env.cfg.sim.control_hz, cfg=env.cfg)
    return summary_row(ep["annotations"], job["out"])
