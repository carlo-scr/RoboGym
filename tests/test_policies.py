"""Policies behind one interface, the planner / executor split, the long-horizon evaluation and its
attribution, the ACT / MLP architectures and the policy spec parser."""
import json

import numpy as np
import pytest

from robogym import lh
from robogym.datagen import EpisodeSpec, run_episode, write_episode
from robogym.env import RoboGymEnv
from robogym.policies import (
    CorruptedPlanner,
    ExpertPolicy,
    HierarchicalPolicy,
    HoldPolicy,
    LLMPlanner,
    OraclePlanner,
    PermutedPlanner,
    ScriptedExecutor,
    Subgoal,
    adapter_for,
    make_policy,
    parse_spec,
)
from robogym.policies.planning import plan_diff


def test_spec_parser():
    p = parse_spec("hier:corrupt(p_swap=0.5,base=heuristic)+scripted(exec_noise=0.003):on_fail=retry,max_retries=2")
    assert p["planner"] == ("corrupt", [], {"p_swap": 0.5, "base": "heuristic"}) and p["executor"][2] == {"exec_noise": 0.003}
    assert p["opts"] == {"on_fail": "retry", "max_retries": 2}
    assert parse_spec("expert:exec_noise=0.004,recover=true")["opts"] == {"exec_noise": 0.004, "recover": True}
    assert parse_spec("chunk:runs/x/policy.pt:K=4")["pos"] == ["runs/x/policy.pt"]
    pol = make_policy("hier:reverse+scripted")
    assert isinstance(pol, HierarchicalPolicy) and pol.privileged
    with pytest.raises(ValueError):
        make_policy("nope")


def test_expert_policy_eval_and_attribution(env_for):
    env = env_for("cube_stack")
    row = lh.run_episode(env, ExpertPolicy(), EpisodeSpec(seed=1, variant={"n": 3}))
    assert row["success"] and row["attribution"] == "none" and row["n_done_prefix"] == row["horizon"] == 2
    att = json.loads(row["attempts"])
    assert len(att) == 2 and [a["obj"] for a in att] == [m[0] for m in env.scenario.task["moves"]] and all(a["achieved"] for a in att)
    assert row["subgoal_t"] != "[null, null]" and row["failure_mode"] == "none"
    # the no-op policy: no attempts, the first subgoal is an execution (grasp) failure
    row = lh.run_episode(env, HoldPolicy(), EpisodeSpec(seed=1, variant={"n": 2}), max_s=1.0)
    assert not row["success"] and row["attribution"] == "exec_grasp" and row["n_attempts"] == 0


def test_hierarchical_oracle_and_bad_plans(env_for):
    env = env_for("cube_stack")
    ad = adapter_for(env)
    row = lh.run_episode(env, HierarchicalPolicy(OraclePlanner(), ScriptedExecutor()), EpisodeSpec(seed=1, variant={"n": 3}))
    info = json.loads(row["policy_info"])
    assert row["success"] and row["plan_valid"] and info["n_plan"] == 2 and all(r["achieved"] for r in info["hier_log"])
    # reversed order: the first manipulated cube is the wrong one -> a planning error
    row = lh.run_episode(env, HierarchicalPolicy(PermutedPlanner("reverse"), ScriptedExecutor()), EpisodeSpec(seed=1, variant={"n": 3}))
    assert row["plan_valid"] is False and row["n_wrong_object"] >= 1 and (row["success"] or row["attribution"] == "plan")
    # wrong goal with probability 1: right cube, other support -> plan_goal (or plan if the swap changed the object)
    pol = HierarchicalPolicy(CorruptedPlanner(p_goal=1.0), ScriptedExecutor())
    row = lh.run_episode(env, pol, EpisodeSpec(seed=1, variant={"n": 3}))
    assert row["plan_valid"] is False and row["attribution"] in ("plan", "plan_goal", "exec_place")
    # plan diff bookkeeping
    env.reset(seed=1, variant={"n": 3})
    oracle = ad.oracle(env)
    d = plan_diff(oracle[::-1], oracle)
    assert not d["exact"] and d["same_set"] and d["prefix"] == 0 and d["missing"] == 0
    assert ad.valid(env, oracle)[0] and not ad.valid(env, oracle[:1])[0]


@pytest.mark.parametrize("task,seed,n", [("lego", 4, 2), ("cable_plug", 2, 1)])
def test_hierarchical_other_tasks(env_for, task, seed, n):
    env = env_for(task)
    row = lh.run_episode(env, make_policy("hier:heuristic+scripted"), EpisodeSpec(seed=seed, variant={"n": n}))
    assert row["success"] and row["plan_valid"] and row["attribution"] == "none"


def test_lego_plan_validity(env_for):
    env = env_for("lego")
    env.reset(seed=4, variant={"n": 3})
    ad = adapter_for(env)
    plan = ad.oracle(env)
    assert ad.valid(env, plan)[0]
    lv = [env.scenario.task["goals"][k][2] for k in range(3)]
    if max(lv) > 0:  # a stacked brick before its support is invalid
        top = int(np.argmax(lv))
        bad = [Subgoal(top)] + [s for s in plan if s.obj != top]
        assert ad.valid(env, bad) == (False, "unsupported")
    assert ad.valid(env, plan[:2])[0] is False


def test_llm_planner_offline(env_for):
    env = env_for("cube_stack")
    env.reset(seed=2, variant={"n": 3})
    ad = adapter_for(env)
    cols = env.scenario.task["colors"]
    oracle = ad.oracle(env)
    good = json.dumps([{"cube": cols[s.obj], "on": cols[s.goal[0]]} for s in oracle])
    pl = LLMPlanner(ask=lambda system, user: f"Here is the plan:\n```json\n{good}\n```")
    assert [s.key() for s in pl.plan(env, 2)] == [s.key() for s in oracle] and pl.info()["llm_error"] is None
    assert "Task:" in ad.scene_text(env) and "JSON" in ad.llm_format(env)
    bad = LLMPlanner(ask=lambda system, user: "I cannot do that")
    assert bad.plan(env, 2) == [] and bad.info()["llm_error"]
    for task in ("lego", "cable_plug"):
        e = env_for(task)
        e.reset(seed=3, variant={"n": 2})
        a = adapter_for(e)
        text = a.scene_text(e)
        assert len(text) > 50 and a.heuristic(e) and a.candidates(e)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from robogym.learn.data import load_episode_arrays

    root = tmp_path_factory.mktemp("ds")
    env = RoboGymEnv("cube_stack", render=False)
    eps = []
    for seed in range(3):
        ep = run_episode(env, EpisodeSpec(seed=seed, variant={"n": 2}, dart=0.002), cameras=[], save_states=False, max_s=5.0)
        eps.append(load_episode_arrays(write_episode(ep, root / f"ep_{seed:06d}", fps=20.0, cfg=env.cfg), cams=(), frames=False))
    return eps


@pytest.mark.parametrize("arch", ["mlp", "act"])
def test_architectures_train_and_execute(tiny_dataset, tmp_path, arch):
    pytest.importorskip("torch")
    from robogym.learn.policy import PolicyCfg, load_policy, save_policy, train_policy
    from robogym.policies import ChunkPolicy, LearnedExecutor

    cfg = PolicyCfg(arch=arch, obs_mode="state", horizon=8, n_act=4, d_model=32, n_heads=4, n_enc=1, n_dec=1, mlp_hidden=64)
    pol, hist = train_policy(tiny_dataset, cfg, steps=15, bs=16, log=None)
    assert np.isfinite(hist).all()
    save_policy(pol, tmp_path / "p.pt")
    pol2 = load_policy(tmp_path / "p.pt")
    assert pol2.cfg.arch == arch and type(pol2).__name__ == type(pol).__name__
    env = RoboGymEnv("cube_stack", render=False)
    row = lh.run_episode(env, ChunkPolicy(pol2), EpisodeSpec(seed=5, variant={"n": 2}), max_s=0.6)
    assert row["policy"].startswith(arch) and 0.0 <= row["progress"] <= 1.0
    hp = HierarchicalPolicy(OraclePlanner(), LearnedExecutor(pol2), budget_s=0.3)
    row = lh.run_episode(env, hp, EpisodeSpec(seed=5, variant={"n": 3}), max_s=0.8)
    info = json.loads(row["policy_info"])
    assert info["n_plan"] == 2 and len(info["hier_log"]) >= 2 and info["hier_log"][0]["timeout"]
    # the executor's features follow the planned subgoal, not the env's current one
    ad = adapter_for(env)
    pr = env.privileged()
    f_plan = ad.features(env, pr, 1, ad.oracle(env)[1], ad.oracle(env))
    assert f_plan.shape == env.task_features(pr).shape and f_plan[1] == 1


def test_summaries_from_rows():
    import pandas as pd

    rows = []
    for pol, succ, attr, nd in (("a", True, "none", 2), ("a", False, "exec_place", 1), ("b", False, "plan", 0), ("b", True, "none", 2)):
        rows.append(dict(policy=pol, horizon=2, success=succ, progress=nd / 2, n_done_prefix=nd, duration=10.0, t_success=5.0 if succ else None, attribution=attr,
                         plan_valid=None, policy_info="{}"))
    df = pd.DataFrame(rows)
    s = lh.summarize(df)
    assert set(s.policy) == {"a", "b"} and s.loc[s.policy == "a", "success"].iloc[0] == 0.5
    c = lh.horizon_curve(df)
    assert "predicted_independent" in c and np.isfinite(c.predicted_independent).all()
    t = lh.attribution_table(df)
    assert t.loc["b", "plan"] == 1 and t.loc["b", "plan_frac_of_failures"] == 1.0
    assert len(lh.plan_exec_split(df)) == 0
    specs = lh.make_specs("cube_stack", horizons=(2, 3), n=2)
    assert len(specs) == 4 and specs[0].seed == specs[2].seed and specs[2].variant["n"] == 3
