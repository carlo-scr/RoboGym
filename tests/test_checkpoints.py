"""Interrupted training resumes from the checkpoint and ends with the same number of steps."""
import numpy as np
import pytest

from robogym.datagen import EpisodeSpec, run_episode, write_episode
from robogym.env import RoboGymEnv

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from robogym.learn.data import load_episode_arrays

    root = tmp_path_factory.mktemp("ds")
    env = RoboGymEnv("cube_stack", render=False)
    eps = []
    for seed in range(2):
        ep = run_episode(env, EpisodeSpec(seed=seed, variant={"n": 2}, dart=0.002), cameras=[], save_states=False, max_s=4.0)
        eps.append(load_episode_arrays(write_episode(ep, root / f"ep_{seed:06d}", fps=20.0, cfg=env.cfg), cams=(), frames=False))
    return eps


@pytest.mark.parametrize("arch", ["mlp", "dp"])
def test_policy_resume(tiny, tmp_path, arch):
    from robogym.learn.policy import PolicyCfg, train_policy

    cfg = PolicyCfg(arch=arch, horizon=8, n_act=4, down_dims=(16, 32), cond_dim=32, diffusion_steps=20, ddim_steps=3, mlp_hidden=32)
    d = tmp_path / arch
    # budget 0: stops right after the first checkpoint (step 4 of 12)
    pol1, h1 = train_policy(tiny, cfg, steps=12, bs=16, log=None, ckpt_dir=d, ckpt_every=4, budget_s=0)
    assert len(h1) == 4 and (d / "ckpt.pt").exists()
    pol2, h2 = train_policy(tiny, cfg, steps=12, bs=16, log=None, ckpt_dir=d, ckpt_every=4)
    assert len(h2) == 12 and np.allclose(h2[:4], h1)  # the first 4 losses are the checkpointed ones
    # a different `steps` is a different run: starts from scratch
    _, h3 = train_policy(tiny, cfg, steps=6, bs=16, log=None, ckpt_dir=tmp_path / "other", ckpt_every=3)
    assert len(h3) == 6


def test_ae_and_wm_resume(tiny, tmp_path):
    from robogym.learn.encoders import train_ae
    from robogym.learn.wm import train_wm, wm_features

    frames = np.random.default_rng(0).integers(0, 255, (16, 24, 32, 3), dtype=np.uint8)
    ae1, h1 = train_ae(frames, latent=8, steps=6, bs=4, log=None, ckpt_dir=tmp_path / "ae", ckpt_every=2, budget_s=0)
    ae2, h2 = train_ae(frames, latent=8, steps=6, bs=4, log=None, ckpt_dir=tmp_path / "ae", ckpt_every=2)
    assert len(h1) == 2 and len(h2) == 6 and h2[:2] == h1
    eps = [dict(x=wm_features(np.zeros((len(e["state"]), 4), np.float32), e["state"], 1), action=e["action"]) for e in tiny]
    wm1, g1 = train_wm(eps, H=2, n_members=2, hidden=16, steps=6, bs=32, log=None, val=eps[:1], log_every=2, ckpt_dir=tmp_path / "wm", ckpt_every=3, budget_s=0)
    wm2, g2 = train_wm(eps, H=2, n_members=2, hidden=16, steps=6, bs=32, log=None, val=eps[:1], log_every=2, ckpt_dir=tmp_path / "wm", ckpt_every=3)
    assert len(g1) == 3 and len(g2) == 6 and g2[:3] == g1 and len(wm2.val_hist) >= 2
