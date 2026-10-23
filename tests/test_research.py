"""Research hooks: deterministic branching, candidate evaluation, monitor labels, gym wrapper."""
import numpy as np
import pytest

from robogym.datagen import EpisodeSpec, build_episode
from robogym.research import ObsDict, branch_outcomes, capture_states, monitor_labels, rollout_policy, seed_for
from robogym.tools import expert_at, run_until


def test_branching_is_deterministic(env_for):
    env = env_for("cube_stack")
    sc, _, ex = build_episode(env, EpisodeSpec(seed=9, variant={"n": 2}))
    run_until(env, ex, "carry", subgoal=0)
    state = env.get_state()
    t_limit = env.time + 1.0
    rng = np.random.default_rng(0)
    hold = np.zeros(8)
    hold[6] = 1.0
    chunks = [np.repeat(hold[None], 8, 0), np.repeat(hold[None], 8, 0) + np.concatenate([rng.normal(0, 0.004, 3), np.zeros(5)])]
    r1 = branch_outcomes(env, state, chunks, lambda e: hold, max_s=t_limit)
    r2 = branch_outcomes(env, state, chunks, lambda e: hold, max_s=t_limit)
    assert [(r["progress"], r["duration_s"]) for r in r1] == [(r["progress"], r["duration_s"]) for r in r2]
    assert len(r1) == 2 and all(0.0 <= r["progress"] <= 1.0 for r in r1)
    # restoring the snapshot afterwards reproduces the exact state
    env.set_state(state)
    assert np.abs(env.get_state()["physics"] - state["physics"]).max() == 0.0


def test_capture_and_expert_at(env_for):
    env = env_for("cube_stack")
    spec = EpisodeSpec(seed=12, variant={"n": 2})
    sc, _, ex = build_episode(env, spec)
    states = capture_states(env, ex.act, at_steps=(10, 30))
    assert set(states) == {10, 30}
    ex2 = expert_at(env, spec, 30)
    assert env.step_count == 30 and ex2.phase == ex2.phase
    assert np.abs(env.get_state()["physics"] - states[30]["physics"]).max() == 0.0


def test_rollout_policy_and_labels(env_for):
    env = env_for("cube_stack")
    env.reset(seed=2)
    ex = env.task.make_expert(env, seed=2)
    rec = rollout_policy(env, lambda obs: ex.act(), max_s=2.0, snapshot_every=10)
    assert len(rec["t"]) == 40 and len(rec["snapshots"]) == 4
    assert seed_for(1, 2, 3) == seed_for(1, 2, 3) != seed_for(1, 2, 4)
    o = ObsDict(env)
    assert o.state.shape[0] == 28 and o.features.ndim == 1


def test_monitor_labels(env_for, tmp_path):
    from robogym.datagen import load_episode, run_episode, write_episode

    env = env_for("cube_stack")
    for seed in (3, 5, 8, 11):  # the first drop that ends in a failure (a late drop can still land seated)
        ep = run_episode(env, EpisodeSpec(seed=seed, category="known_failure", failure="drop", severity=0.8, variant={"n": 2}), cameras=[], save_states=False)
        if ep["annotations"]["outcome"] == "failure":
            break
    write_episode(ep, tmp_path / "ep", fps=20.0, cfg=env.cfg)
    E = load_episode(tmp_path / "ep")
    lab = monitor_labels(E["df"], E["annotations"], lead_s=1.0)
    assert lab["onset_s"] is not None and lab["pre_failure"].any() and np.isfinite(lab["time_to_failure"]).all()
    assert len(lab["stage"]) == len(E["df"])


def test_gym_wrapper():
    pytest.importorskip("gymnasium")
    from robogym.gym_wrapper import RoboGymEnvGym

    g = RoboGymEnvGym("cube_stack")
    obs, info = g.reset(seed=1)
    assert g.observation_space.contains({k: np.asarray(v, dtype=g.observation_space[k].dtype) for k, v in obs.items()})
    obs, r, term, trunc, info = g.step(g.action_space.sample())
    assert "progress" in info and isinstance(r, float)
    g.close()
