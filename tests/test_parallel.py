"""The resumable job pool: results, resume, and the spawn start method used for CUDA workers."""
from robogym.config import make_cfg
from robogym.parallel import run_jobs


def _job(env, j):
    import multiprocessing

    env.reset(seed=j["seed"])
    return j["seed"] * 2, type(multiprocessing.current_process()).__name__


def test_run_jobs_resume_and_spawn(tmp_path):
    cfg = make_cfg("cube_stack")
    jobs = [dict(seed=s) for s in range(4)]
    logs = []
    res = run_jobs(_job, jobs[:2], cfg, out_dir=tmp_path / "j", workers=2, render=False, log=logs.append, job_key=lambda j: str(j["seed"]), start_method="spawn")
    assert sorted(r["result"][0] for r in res) == [0, 2] and all(r["ok"] and r["result"][1] == "SpawnProcess" for r in res)
    res = run_jobs(_job, jobs, cfg, out_dir=tmp_path / "j", workers=2, render=False, log=logs.append, job_key=lambda j: str(j["seed"]))
    assert sorted(r["result"][0] for r in res) == [0, 2, 4, 6] and any("resuming: 2 jobs" in m for m in logs)
