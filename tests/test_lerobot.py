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


@pytest.mark.parametrize("ptype,variant", [("act", "full"), ("vqbet", "front")])
def test_export_train_run(tiny_root, tmp_path, ptype, variant):
    from robogym import lh
    from robogym.learn.lerobot_io import checkpoint_step, export_lerobot, train_lerobot
    from robogym.policies import LeRobotPolicy, make_policy

    info = export_lerobot(tiny_root, tmp_path / f"lr_{variant}", f"test/robogym_{variant}", variant=variant, log=None)
    assert info["n_episodes"] == 2 and info["env_state"] == (variant == "full") and (tmp_path / f"lr_{variant}" / "meta" / "info.json").exists()
    assert export_lerobot(tiny_root, tmp_path / f"lr_{variant}", f"test/robogym_{variant}", variant=variant, log=None) == info  # idempotent
    extra = ["--policy.n_vqvae_training_steps=2"] if ptype == "vqbet" else ["--policy.dim_model=64", "--policy.n_encoder_layers=1", "--policy.n_decoder_layers=1", "--policy.pretrained_backbone_weights=null"]
    out = tmp_path / ptype
    d = train_lerobot(ptype, info, out, steps=4, batch_size=2, device="cpu", save_freq=2, num_workers=0, log_freq=2, extra=extra, log=None)
    assert checkpoint_step(out) == 4 and (d / "config.json").exists()
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["000004", "last"]  # step-2 checkpoint pruned
    assert train_lerobot(ptype, info, out, steps=4, batch_size=2, device="cpu", log=None) == d  # already done
    pol = make_policy(f"lerobot:{out}")
    assert isinstance(pol, LeRobotPolicy) and pol.name == f"lerobot_{ptype}" and pol.needs_render and pol.cams == tuple(info["cams"])
    env = RoboGymEnv("cube_stack", render=True)
    row = lh.run_episode(env, pol, EpisodeSpec(seed=5, variant={"n": 2}), max_s=0.5)
    assert 0.0 <= row["progress"] <= 1.0 and row["policy"] == pol.name and np.isfinite(row["duration"])


def test_train_many(tiny_root, tmp_path):
    from robogym.learn.lerobot_io import checkpoint_step, export_lerobot, train_lerobot_many, train_log

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
    with pytest.raises(RuntimeError, match="bad"):
        train_lerobot_many(bad, max_parallel=1, log=None)
