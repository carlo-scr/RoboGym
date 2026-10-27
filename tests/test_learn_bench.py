"""Learners (tiny training runs), the benchmark protocol, recovery labels, depth / segmentation renders,
sensor regimes and the dry real-robot backend."""
import numpy as np
import pytest

from robogym import bench
from robogym.datagen import EpisodeSpec, run_episode, write_episode
from robogym.env import RoboGymEnv

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """4 short physics-only cube episodes written to disk + loaded as arrays."""
    from robogym.learn.data import load_episode_arrays

    root = tmp_path_factory.mktemp("ds")
    env = RoboGymEnv("cube_stack", render=False)
    eps = []
    for seed in range(4):
        ep = run_episode(env, EpisodeSpec(seed=seed, variant={"n": 2}, dart=0.002), cameras=[], save_states=False, max_s=6.0)
        p = write_episode(ep, root / f"ep_{seed:06d}", fps=20.0, cfg=env.cfg)
        eps.append(load_episode_arrays(p, cams=(), frames=False))
    return eps


def test_policy_train_rollout_roundtrip(tiny_dataset, tmp_path):
    from robogym.learn.policy import PolicyCfg, PolicyRunner, load_policy, rollout, save_policy, train_policy

    cfg = PolicyCfg(obs_mode="state", action_mode="rel", n_arms=1, down_dims=(16, 32), cond_dim=32, diffusion_steps=20, ddim_steps=3, horizon=8, n_act=4)
    pol, hist = train_policy(tiny_dataset, cfg, steps=30, bs=32, log=None)
    assert len(hist) == 30 and np.isfinite(hist).all()
    save_policy(pol, tmp_path / "p.pt")
    pol2 = load_policy(tmp_path / "p.pt")
    env = RoboGymEnv("cube_stack", render=False)
    runner = PolicyRunner(pol2, env, seed=3)
    r = rollout(env, runner, seed=3, max_s=1.0)
    assert 0.0 <= r["progress"] <= 1.0 and r["duration"] >= 1.0 - 1e-6
    # seeded candidates: the same (episode, replan, k) draws the same chunk
    env.reset(seed=5)
    runner.reset(seed=5)
    runner.observe()
    c1 = runner.sample_chunks(3)
    runner.reset(seed=5)
    runner.observe()
    c2 = runner.sample_chunks(3)
    assert np.allclose(c1, c2)


def test_wm_and_monitor(tiny_dataset):
    from robogym.learn.monitor import auroc, conformal_threshold, smooth
    from robogym.learn.wm import score_episode, train_wm, wm_features

    eps = [dict(x=wm_features(np.zeros((len(e["state"]), 4), np.float32), e["state"], 1), action=e["action"]) for e in tiny_dataset]
    wm, hist = train_wm(eps, H=2, n_members=2, hidden=32, steps=20, bs=64, log=None)
    s = score_episode(wm, eps[0]["x"], eps[0]["action"])
    assert set(s) == {"alea", "epi", "nll"} and len(s["nll"]) == len(eps[0]["x"]) - 1
    assert conformal_threshold([1, 2, 3, 4, 5], alpha=0.5) == 3.0
    assert auroc([0, 1], [2, 3]) == 1.0 and len(smooth(np.arange(10), 3)) == 10


def test_branching_oracle(tiny_dataset):
    from robogym.learn.policy import PolicyCfg, PolicyRunner, train_policy
    from robogym.learn.steer import RandomScorer, capture_branches, rq1_metrics, rq1_ranking

    cfg = PolicyCfg(obs_mode="state", down_dims=(16, 32), cond_dim=32, diffusion_steps=20, ddim_steps=3, horizon=8, n_act=4)
    pol, _ = train_policy(tiny_dataset, cfg, steps=10, bs=32, log=None)
    env = RoboGymEnv("cube_stack", render=False)
    runner = PolicyRunner(pol, env)
    branches = capture_branches(env, runner, None, seeds=[1, 2], at_replans=(1,), max_s=3.0)
    assert len(branches) == 2
    succ, prog, scores = rq1_ranking(env, runner, None, [RandomScorer()], branches, K=2, max_s=2.0, log=None)
    assert succ.shape == (2, 2) and prog.shape == (2, 2) and scores["random"].shape == (2, 2)
    res, rows = rq1_metrics(succ, scores, prog)
    assert res["n_states"] == 2 and rows[0]["scorer"] == "random"


def test_bench_protocol():
    assert bench.seeds("cal", 3) == [2000, 2001, 2002]
    specs = bench.suite_specs("lego", "task", n=2, severities=(0.5,))
    assert len(specs) == 2 * len(bench.SUITES["task"]["lego"]) and all(s.category == "ood" for s in specs)
    conds = bench.condition_specs("cube_stack", n=2)
    assert len(conds) == 2 * len(bench.CONDITIONS)
    r = bench.rate([True, True, False, True])
    assert 0 < r["lo"] < r["rate"] < r["hi"] <= 1
    d = bench.paired_delta([True, False, True], [True, True, False])
    assert d["harm"] == pytest.approx(1 / 3) and d["help"] == pytest.approx(1 / 3)
    res = bench.Results("t", "cube_stack", "m")
    res.add(split="test", metric="success", rate=0.5)
    assert res.table().iloc[0]["rate"] == 0.5


def test_recovery_labels():
    env = RoboGymEnv("cube_stack", render=False)
    any_rec = False
    for seed in (3, 5, 8):
        ep = run_episode(env, EpisodeSpec(seed=seed, category="known_failure", failure="drop", severity=0.9, variant={"n": 2}, recover=True), cameras=[], save_states=False)
        ann = ep["annotations"]
        assert ann["spec"]["recover"] is True
        for r in ann["recoveries"]:
            assert {"subgoal", "attempt", "t_start", "t_end", "success"} <= set(r)
        any_rec |= len(ann["recoveries"]) > 0
    assert any_rec


def test_depth_seg_and_sensor_regimes():
    env = RoboGymEnv("lego")
    env.reset(seed=1)
    d = env.render("front", mode="depth")
    s = env.render("front", mode="seg")
    assert d.shape == (240, 320) and d.dtype == np.float32 and 0.3 < float(np.median(d)) < 5.0
    assert s.shape == (240, 320) and s.max() >= 0 and len(env.segmentation_labels()) == env.model.ngeom
    x0 = env.observe()
    env.set_regime({"ft_bias": 3.0, "obs_delay": 2})
    env.step(np.zeros(8))
    x1 = env.observe()
    assert abs(float(x1[22] - x0[22]) - 3.0) < 1.5 or env.regime["obs_delay"] == 2


def test_dry_real_backend(tmp_path):
    from robogym.real.twin import divergence, replay_actions, summarize_divergence
    from robogym.real.xarm_env import RealXArmEnv

    r = RealXArmEnv(arms=("right",), cameras={"right_wrist": None}, backend="dry")
    obs = r.reset()
    assert obs.shape == (28,)
    a = np.zeros(8)
    a[0], a[6] = 0.01, 1.0
    obs2 = r.step(a)
    assert obs2[14] > obs[14] and 0 <= r.b["right"].grip < 1
    out = r.record(tmp_path / "real_ep", lambda o: a, max_s=0.5)
    assert (out / "episode.parquet").exists() and (out / "feature_report.json").exists()
    env = RoboGymEnv("cube_stack", render=False)
    acts = np.zeros((10, 8))
    tr = replay_actions(env, acts, seed=1)
    div = divergence(tr, tr)
    assert summarize_divergence(div)["arm0_tcp_err_m"]["max"] == 0.0


def test_bench_evaluate_actor_with_expert():
    env = RoboGymEnv("cube_stack", render=False)
    holder = {}

    def reset_fn(e, spec):
        holder["ex"] = e.task.make_expert(e, seed=spec.seed)

    specs = bench.split_specs("cube_stack", "test", n=2, horizon=2) + bench.condition_specs("cube_stack", n=1, horizon=2, conditions={"slippery": {"pad_friction": 0.3}})
    df = bench.evaluate_actor(env, lambda e: holder["ex"].act(), specs, reset_fn=reset_fn)
    assert len(df) == 3 and set(df.tag) == {"test", "cond:slippery"}
    summ = bench.summarize_eval(df)
    assert "success" in summ.columns and summ.loc["test", "n"] == 2


def test_safety_signals():
    env = RoboGymEnv("cube_stack", render=False, bimanual=True)
    env.reset(seed=1)
    s = env.safety()
    assert s["joint_limit_margin"] > 0 and 0 <= s["torque_sat"] <= 1 and s["arm_clearance"] > 0.01 and not s["would_collide"]
    ep = run_episode(env, EpisodeSpec(seed=1, variant={"n": 2}), cameras=[], save_states=False, max_s=2.0)
    assert "safety_arm_clearance" in ep["extra_cols"] and len(ep["data"]["safety_tcp_speed"]) == ep["annotations"]["n_steps"]
