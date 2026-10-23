import pytest

from robogym.env import RoboGymEnv

_ENVS = {}


@pytest.fixture(scope="session")
def env_for():
    """Session-cached envs per task (building a model takes ~0.1 s; the first reset settles physics)."""

    def get(task, render=False):
        key = (task, render)
        if key not in _ENVS:
            _ENVS[key] = RoboGymEnv(task, render=render)
        return _ENVS[key]

    return get
