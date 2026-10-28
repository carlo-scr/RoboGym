"""LeRobot bridge: export, a few training steps, load and closed-loop run (skipped without lerobot)."""
import numpy as np
import pytest

from robogym.datagen import EpisodeSpec, run_episode, write_episode
from robogym.env import RoboGymEnv

pytest.importorskip("lerobot")


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    import pandas as pd

    from robogym.datagen import summary_row

    root = tmp_path_factory.mktemp("ds")
    env = RoboGymEnv("cube_stack", render=True, **{"camera.width": 96, "camera.height": 72})
    rows = []
    for seed in range(2):
        ep = run_episode(env, EpisodeSpec(seed=seed, variant={"n": 2}, dart=0.002), cameras=["right_wrist", "front"], save_states=False, max_s=3.0)
        ep["annotations"]["outcome"] = "success"  # short episodes: keep them regardless
        out = write_episode(ep, root / f"ep_{seed:06d}", fps=20.0, cfg=env.cfg)
        rows.append(summary_row(ep["annotations"], str(out)))
    pd.DataFrame(rows).to_parquet(root / "index_000000.parquet")
    env.close()
    return root


@pytest.mark.parametrize("ptype,variant", [("act", "full"), ("diffusion", "full"), ("vqbet", "front")])
def test_export_train_run(tiny_root, tmp_path, ptype, variant):
    from robogym import lh
    from robogym.learn.lerobot_io import checkpoint_step, export_lerobot, train_lerobot
    from robogym.policies import LeRobotPolicy, make_policy

    info = export_lerobot(tiny_root, tmp_path / f"lr_{variant}", f"test/robogym_{variant}", variant=variant, log=None)
    assert info["n_episodes"] == 2 and info["env_state"] == (variant == "full") and (tmp_path / f"lr_{variant}" / "meta" / "info.json").exists()
    assert export_lerobot(tiny_root, tmp_path / f"lr_{variant}", f"test/robogym_{variant}", variant=variant, log=None) == info  # idempotent
    extra = {"vqbet": ["--policy.n_vqvae_training_steps=2"], "diffusion": ["--policy.down_dims=[32,64]", "--policy.diffusion_step_embed_dim=32"],
             "act": ["--policy.dim_model=64", "--policy.n_encoder_layers=1", "--policy.n_decoder_layers=1", "--policy.pretrained_backbone_weights=null"]}[ptype]
    out = tmp_path / ptype
    d = train_lerobot(ptype, info, out, steps=4, batch_size=2, device="cpu", save_freq=2, num_workers=0, log_freq=2, extra=extra, log=None)
    assert checkpoint_step(out) == 4 and (d / "config.json").exists()
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["000004", "last"]  # step-2 checkpoint pruned
    assert train_lerobot(ptype, info, out, steps=4, batch_size=2, device="cpu", log=None) == d  # already done
    train_lerobot(ptype, info, out, steps=6, batch_size=2, device="cpu", save_freq=2, num_workers=0, log_freq=2, extra=extra, log=None)  # resume 4 -> 6
    assert checkpoint_step(out) == 6 and sorted(p.name for p in (out / "checkpoints").iterdir()) == ["000006", "last"]
    pol = make_policy(f"lerobot:{out}")
    assert isinstance(pol, LeRobotPolicy) and pol.name == f"lerobot_{ptype}" and pol.needs_render and pol.cams == tuple(info["cams"])
    env = RoboGymEnv("cube_stack", render=True)
    row = lh.run_episode(env, pol, EpisodeSpec(seed=5, variant={"n": 2}), max_s=0.5)
    assert 0.0 <= row["progress"] <= 1.0 and row["policy"] == pol.name and np.isfinite(row["duration"])


def test_train_many(tiny_root, tmp_path):
    from robogym.learn.lerobot_io import checkpoint_step, export_lerobot, train_lerobot, train_lerobot_many, train_log
    from robogym.policies import LeRobotPolicy, make_policy

    info = export_lerobot(tiny_root, tmp_path / "lr_full", "test/robogym_full", variant="full", log=None)
    tiny = ["--policy.dim_model=64", "--policy.n_encoder_layers=1", "--policy.n_decoder_layers=1", "--policy.pretrained_backbone_weights=null"]
    jobs = [dict(name=f"act{i}", policy_type="act", dataset_info=info, out_dir=tmp_path / f"act{i}", steps=4, batch_size=2, device="cpu", save_freq=2, num_workers=0,
                 log_freq=2, extra=tiny) for i in range(2)]
    logs = []
    mins = train_lerobot_many(jobs, max_parallel=2, poll_s=5, log=logs.append)
    assert set(mins) == {"act0", "act1"} and all(checkpoint_step(tmp_path / f"act{i}") == 4 for i in range(2))
    assert all("step:" in train_log(tmp_path / f"act{i}").read_text() for i in range(2)) and any("trained (4 steps" in m for m in logs)
    assert train_lerobot_many(jobs, max_parallel=2, log=None) == {"act0": 0.0, "act1": 0.0}  # already done: returns at once
    bad = [dict(jobs[0], name="bad", policy_type="no_such_policy", out_dir=tmp_path / "bad")]
    with pytest.raises(RuntimeError, match="bad") as ei:
        train_lerobot_many(bad, max_parallel=1, log=logs.append)
    assert "end of the bad log" in str(ei.value) and any("[bad] FAILED" in m and "\n    | " in m for m in logs)  # the log tail is shown
    # a run on local disk whose final checkpoint is published to out_dir (Colab: out_dir on Drive, which has no symlinks)
    out, work = tmp_path / "pub", tmp_path / "work" / "pub"
    job = dict(jobs[0], name="pub", out_dir=out, work_dir=work)
    assert set(train_lerobot_many([job], max_parallel=1, log=None)) == {"pub"}
    last = out / "checkpoints" / "last"
    assert checkpoint_step(out) == 4 and checkpoint_step(work) == 4 and not last.is_symlink() and (last / "pretrained_model" / "config.json").exists()
    assert [q.name for q in (last / "training_state").iterdir()] == ["training_step.json"]  # no optimizer state in the published copy
    assert train_log(out).read_text().startswith("$ ") and "step:" in train_log(out).read_text()  # the log is copied beside out_dir
    kw = {k: v for k, v in job.items() if k != "name"}
    assert train_lerobot(**kw, log=None) == last / "pretrained_model"  # published: returns at once
    train_lerobot(**dict(kw, steps=6), log=None)  # resumes the local run 4 -> 6 and republishes
    assert checkpoint_step(out) == 6 and checkpoint_step(work) == 6 and isinstance(make_policy(f"lerobot:{out}"), LeRobotPolicy)


def test_fast_window_query(tiny_root, tmp_path):
    """The window query patch returns exactly what lerobot's reader returns (skipped on lerobot <= 0.4, no reader)."""
    import torch

    from robogym.learn.lerobot_fast import patch_reader
    from robogym.learn.lerobot_io import export_lerobot

    pytest.importorskip("lerobot.datasets.dataset_reader")
    from lerobot.datasets.dataset_reader import DatasetReader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = export_lerobot(tiny_root, tmp_path / "lr_full", "test/robogym_full", variant="full", log=None)
    dts = {"action": [i / 20 for i in range(32)], "observation.images.front": [-1 / 20, 0], "observation.state": [-1 / 20, 0]}
    ds = LeRobotDataset(info["repo_id"], root=info["root"], delta_timestamps=dts)
    orig_q, orig_t = DatasetReader._query_hf_dataset, DatasetReader._get_query_timestamps
    idx = [0, 3, len(ds) - 1]
    before = [ds[i] for i in idx]
    try:
        assert patch_reader() and patch_reader()  # idempotent
        after = [ds[i] for i in idx]
    finally:
        DatasetReader._query_hf_dataset, DatasetReader._get_query_timestamps, DatasetReader._robogym_fast = orig_q, orig_t, False
    for a, b in zip(before, after):
        assert a.keys() == b.keys()
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k]), k
            else:
                assert a[k] == b[k], k
