# %% [markdown]
# # RoboGym tour
# One or two UFACTORY xArm 7 on a lab bench, three tasks (cube stacking, oversized LEGO, RJ45 plugging),
# cameras, the scripted experts, and what an episode looks like. Everything in this notebook runs on a
# laptop CPU in a few minutes; the setup cell also works on a Colab runtime (clones the repo).

# %%
# @bootstrap

# %%
import numpy as np
import matplotlib.pyplot as plt
from robogym.env import RoboGymEnv
from robogym.tools import image_grid, mosaic, video_html, run_until
from robogym.datagen import EpisodeSpec, build_episode

# %% [markdown]
# ## The three tasks
# Each env compiles its own MuJoCo model (~0.1 s). The free camera shows the bench, the robot and the task
# objects; translucent "ghosts" mark the goal (LEGO) where the goal is not readable from the objects alone.

# %%
envs = {t: RoboGymEnv(t) for t in ("cube_stack", "lego", "cable_plug")}
imgs, titles = [], []
for name, env in envs.items():
    env.reset(seed=3)
    imgs.append(env.render_free(width=640, height=400))
    titles.append(f"{name}: {env.task.goal_description(env.scenario.task)}")
image_grid(imgs, titles, cols=3, scale=5)
plt.show()

# %% [markdown]
# ## Cameras
# Robot-side cameras: a wrist camera per arm (RealSense-on-the-camera-stand assumption) and three fixed
# cameras (front / side / top). The dataset records `cfg.camera.record` (wrist + front by default).

# %%
env = envs["cube_stack"]
image_grid([env.render(c) for c in env.cam_ids], list(env.cam_ids), cols=4, scale=3.5)
plt.show()

# %% [markdown]
# ## State, action, privileged state
# * `observe()` -> 28-D per arm: q(7), qd(7), TCP pos(3), TCP quat(4), grip width(1), wrist wrench(6)
# * action -> 8-D per arm: TCP delta (m), rotation-vector delta (rad), grip in [0, 1], compliant flag
# * `privileged()` / `info()` -> ground truth + stage / progress / success labels

# %%
obs = env.observe()
pr = env.privileged()
print("state", obs.shape, "| features", env.task_features(pr).shape, "| info", env.info(pr))

# %% [markdown]
# ## A scripted expert episode
# The experts are privileged, closed-loop "teleoperators" written as generators over Cartesian primitives.

# %%
for name, env in envs.items():
    sc, failure, ex = build_episode(env, EpisodeSpec(seed=5))
    frames = []
    while not ex.done and env.time < env.episode_max_s and not env.unstable:
        env.step(ex.act())
        if env.step_count % 4 == 0:
            frames.append(mosaic(env, ghost=True))
    info = env.info()
    print(f"{name}: success={info['success']} progress={info['progress']:.2f} t={env.time:.1f}s phases={[e[0] for e in ex.events][:12]}...")
    display(video_html(frames, fps=5))  # noqa: F821
