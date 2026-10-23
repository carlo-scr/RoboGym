"""Every OOD shift, every injected failure, regimes and streams run without errors and produce the
expected labels."""
import numpy as np
import pytest

from robogym import shifts as S
from robogym.datagen import EpisodeSpec, make_plan, run_episode
from robogym.failures import all_failure_types
from robogym.scenario import sample_scenario
from robogym.streams import make_stream, regime_table

TASKS = ("cube_stack", "lego", "cable_plug")


@pytest.mark.parametrize("task", TASKS)
def test_all_shifts_sample_and_reset(env_for, task):
    env = env_for(task, render=True)
    for shift in S.ALL_SHIFTS + tuple(env.task.shift_types):
        sc = sample_scenario(env.cfg, 11, shifts=[shift], severity=0.8, task=env.task)
        assert sc.shifts == [shift]
        env.reset(sc)
        img = env.render(env.cfg.camera.record[0])
        assert img.shape == (env.cfg.camera.height, env.cfg.camera.width, 3)
        if S.group_of(shift) == "physics":
            assert sc.regime != sample_scenario(env.cfg, 11, task=env.task).regime


@pytest.mark.parametrize("task", TASKS)
def test_all_failures_run(env_for, task):
    env = env_for(task)
    for kind in all_failure_types(env.task):
        ep = run_episode(env, EpisodeSpec(seed=21, category="known_failure", failure=kind, severity=0.9, variant={"n": 2}), cameras=[], save_states=False, max_s=12.0)
        ann = ep["annotations"]
        assert ann["injected_failure"]["kind"] == kind
        assert ann["failure_mode"] in env.task.outcome_modes


def test_regime_schedule_and_snapshot(env_for):
    env = env_for("cube_stack")
    spec = EpisodeSpec(seed=5, category="regime_shift", regime_schedule=[[1.0, {"pad_friction": 0.3, "action_delay": 2}]])
    ep = run_episode(env, spec, cameras=[], max_s=3.0)
    rid = ep["data"]["regime_id"]
    assert rid[0] == 0 and rid[-1] == 1
    assert env.regime["pad_friction"] == 0.3 and env.delay == 2
    # restore mid-episode after the switch reproduces exactly
    st = ep["data"]["state"]
    k = len(st) - 10
    env.set_state(dict(physics=ep["physics"][k], aux=ep["aux"][k], regime=env.regime))
    for j in range(k, len(st) - 1):
        env.step(ep["actions64"][j])
        assert np.abs(env.observe() - st[j + 1]).max() == 0.0


def test_streams_and_plan(env_for):
    env = env_for("cube_stack")
    for name in ("sudden", "gradual", "recurring", "mixed", "in_episode"):
        specs = make_stream(name, 8, seed0=100)
        assert len(specs) == 8 and len(regime_table(specs)) == 8
    plan = make_plan(env.task, 20, seed0=0)
    assert {p.category for p in plan} <= {"nominal", "known_failure", "ood", "regime_shift"}
    assert all(p.failure for p in plan if p.category == "known_failure")


def test_online_regime_change(env_for):
    env = env_for("cube_stack")
    env.reset(seed=3)
    f0 = env.model.geom_friction[next(iter(env.arm["right"]["all_pads"])), 0]
    env.set_regime({"pad_friction": 0.5})
    assert np.isclose(env.model.geom_friction[next(iter(env.arm["right"]["all_pads"])), 0], 0.5 * f0 / env.scenario.regime["pad_friction"])
    env.set_regime({"pad_friction": env.scenario.regime["pad_friction"]})
    assert np.isclose(env.model.geom_friction[next(iter(env.arm["right"]["all_pads"])), 0], f0)
