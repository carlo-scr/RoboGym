"""Fast end-to-end checks per task: spaces, nominal expert success, exact restore + replay, an injected
failure that fails and is labelled, export / load round trip."""
import json

import numpy as np
import pytest

from robogym.config import EnvCfg
from robogym.datagen import EpisodeSpec, load_episode, restore, run_episode, write_episode
from robogym.env import STATE_PER_ARM, action_names, state_names

TASKS = ("cube_stack", "lego", "cable_plug")
SUCCESS_SEEDS = {"cube_stack": (1, 2), "lego": (4, 5), "cable_plug": (2, 3)}


@pytest.mark.parametrize("task", TASKS)
def test_spaces(env_for, task):
    env = env_for(task)
    cfg = env.cfg
    assert len(env.observe()) == len(state_names(cfg)) == STATE_PER_ARM * cfg.robot.n_arms
    assert len(action_names(cfg)) == 8 * cfg.robot.n_arms
    pr = env.privileged()
    assert len(env.task.task_features(env, pr)) == len(env.task.feature_names())
    info = env.info(pr)
    assert 0.0 <= info["progress"] <= 1.0 and info["stage_name"] in env.task.stages()


@pytest.mark.parametrize("task", TASKS)
def test_nominal_success(env_for, task):
    env = env_for(task)
    for seed in SUCCESS_SEEDS[task]:
        ep = run_episode(env, EpisodeSpec(seed=seed, variant={"n": 2}), cameras=[], save_states=False)
        assert ep["annotations"]["outcome"] == "success", (task, seed, ep["annotations"]["failure_mode"])
        assert ep["annotations"]["final_progress"] == 1.0


@pytest.mark.parametrize("task", TASKS)
def test_restore_replay_exact(env_for, task):
    env = env_for(task)
    ep = run_episode(env, EpisodeSpec(seed=7, variant={"n": 2}), cameras=[])
    st, n = ep["data"]["state"], len(ep["data"]["state"])
    k = n // 2
    env.set_state(dict(physics=ep["physics"][k], aux=ep["aux"][k]))
    for j in range(k, n - 1):
        env.step(ep["actions64"][j])
        assert np.abs(env.observe() - st[j + 1]).max() == 0.0


def test_drop_is_injected_and_labelled(env_for):
    """The injected drop opens the jaw mid-carry: the object is lost (drop event + onset time) and the
    physical outcome is whatever happens next (usually a failure; a late drop can land the cube seated)."""
    env = env_for("cube_stack")
    outcomes = []
    for seed in (3, 5):
        ep = run_episode(env, EpisodeSpec(seed=seed, category="known_failure", failure="drop", severity=0.8), cameras=[], save_states=False)
        ann = ep["annotations"]
        assert ann["drops"] >= 1 and ann["drop_times_s"] and ann["injection_time_s"] is not None
        assert abs(ann["drop_times_s"][0] - ann["injection_time_s"]) < 1.5
        assert ann["failure_mode"] in ("none", "dropped", "toppled", "misplaced", "incomplete")
        outcomes.append(ann["outcome"])
    assert "failure" in outcomes


def test_export_round_trip(env_for, tmp_path):
    env = env_for("cube_stack", render=True)
    ep = run_episode(env, EpisodeSpec(seed=6, variant={"n": 2}), cameras=["right_wrist"])
    write_episode(ep, tmp_path / "ep", fps=env.cfg.sim.control_hz, cfg=env.cfg)
    E = load_episode(tmp_path / "ep", load_frames=True)
    ts = E["df"]["timestamp"].values.astype(np.float64) * 1e-9
    assert np.allclose(np.diff(ts), 1 / env.cfg.sim.control_hz)
    assert E["frames"]["right_wrist"].shape[1:] == (env.cfg.camera.height, env.cfg.camera.width, 3)
    assert len(E["frames"]["right_wrist"]) == len(E["df"])
    ann = json.load(open(tmp_path / "ep" / "annotations.json"))
    assert ann["outcome"] == "success" and ann["task"] == "cube_stack"
    assert len(E["df"]["task.features"][0]) == len(E["features"]["task_feature_names"])
    # counterfactual restart from the recording
    restore(env, E, len(E["df"]) // 3)
    assert env.step_count == len(E["df"]) // 3


def test_cfg_round_trip():
    cfg = EnvCfg(task="lego")
    cfg2 = EnvCfg.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    cfg3 = cfg.with_overrides({"task.bricks": ["2x2", "2x2"], "robot.n_arms": 2})
    assert cfg3.task.bricks == ["2x2", "2x2"] and cfg3.robot.n_arms == 2 and cfg3.bench.y_half >= 0.75
