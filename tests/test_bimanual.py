"""Two xArm 7: the dual-arm expert assigns objects by side, takes turns at the shared structure, and the
data path records both arms."""
import numpy as np
import pytest

from robogym.config import make_cfg
from robogym.datagen import EpisodeSpec, run_episode
from robogym.env import STATE_PER_ARM, RoboGymEnv


@pytest.fixture(scope="module")
def envs():
    return {t: RoboGymEnv(t, render=False, bimanual=True) for t in ("cube_stack", "lego", "cable_plug")}


def test_two_arms_spaces(envs):
    env = envs["cube_stack"]
    assert env.arms == ("right", "left") and len(env.observe()) == 2 * STATE_PER_ARM
    assert "left_wrist" in env.cfg.camera.record and "right_wrist" in env.cfg.camera.record
    assert make_cfg("lego", bimanual=True).robot.n_arms == 2


@pytest.mark.parametrize("task,variant,seeds", [("cube_stack", {"n": 4}, (0, 1)), ("lego", {"n": 3}, (1, 3))])
def test_bimanual_success_uses_both_arms(envs, task, variant, seeds):
    """Every episode succeeds; across the seeds both arms do grasp (an episode whose objects all lie on
    one side is legitimately handled by one arm)."""
    env = envs[task]
    grasping = set()
    for seed in seeds:
        ep = run_episode(env, EpisodeSpec(seed=seed, variant=variant), cameras=[], save_states=False)
        ann = ep["annotations"]
        assert ann["outcome"] == "success", (task, seed, ann["failure_mode"])
        assert ann["active_arm"] == "both"
        grasping |= {e[0].split(":")[0] for e in ann["phase_events"] if e[0].endswith(":grasp")}
        assert "grasped_k_left" in ep["extra_cols"]
    assert grasping == {"right", "left"}, grasping


def test_bimanual_cable_runs_and_is_labelled(envs):
    env = envs["cable_plug"]
    ep = run_episode(env, EpisodeSpec(seed=0, variant={"n": 2}), cameras=[], save_states=False)
    ann = ep["annotations"]
    assert ann["failure_mode"] in env.task.outcome_modes
    assert all(e[0].split(":")[0] in ("right", "left") for e in ann["phase_events"])


def test_single_arm_expert_on_two_arm_scene(envs):
    env = envs["cube_stack"]
    env.reset(seed=2)
    ex = env.task.make_expert(env, seed=2, bimanual=False)
    assert ex.arm in env.arms
    a = ex.act()
    assert a.shape == (16,) and np.isfinite(a).all()
