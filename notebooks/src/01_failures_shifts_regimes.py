# %% [markdown]
# # Failures, OOD shifts and test-time regimes
# Injected failures corrupt the expert (the *injected* label) and the physics decides the *outcome*;
# OOD shifts push one factor outside the nominal randomisation; physics regimes can be switched on
# mid-episode (`env.set_regime`) or scheduled (`EpisodeSpec.regime_schedule`).

# %%
# @bootstrap

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from robogym.env import RoboGymEnv
from robogym import shifts as S
from robogym.failures import all_failure_types
from robogym.datagen import EpisodeSpec, run_episode
from robogym.scenario import sample_scenario
from robogym.tools import image_grid, run_until

TASK = "cube_stack"
env = RoboGymEnv(TASK)

# %% [markdown]
# ## Injected failures -> physical outcomes
# Every failure kind at severity 0.9 on a 2-object task. Note that the outcome label is what *happened*
# (a mild misplacement can still succeed).

# %%
rows = []
for kind in all_failure_types(env.task):
    ep = run_episode(env, EpisodeSpec(seed=21, category="known_failure", failure=kind, severity=0.9, variant={"n": 2}), cameras=[], save_states=False)
    a = ep["annotations"]
    rows.append(dict(injected=kind, outcome=a["outcome"], mode=a["failure_mode"], onset_s=a["injection_time_s"], drops=a["drops"], progress=a["final_progress"]))
pd.DataFrame(rows)

# %% [markdown]
# ## OOD shifts (wrist + front camera at severity 0.8)

# %%
imgs, titles = [], []
for sh in S.ALL_SHIFTS + tuple(env.task.shift_types):
    env.reset(sample_scenario(env.cfg, 5, shifts=[sh], severity=0.8, task=env.task))
    ex = env.task.make_expert(env, seed=5)
    run_until(env, ex, "carry", subgoal=0)
    imgs.append(np.concatenate([env.render(env.cfg.camera.record[0]), env.render("front")], 1))
    titles.append(f"{sh} ({S.group_of(sh)})")
image_grid(imgs, titles, cols=3, scale=4.5)
plt.show()

# %% [markdown]
# ## A regime switch inside an episode
# Pad friction drops to 30 % and two control steps of actuation delay appear at t = 2 s. The per-step
# `regime_id` label and the wrist force trace show where it happens.

# %%
spec = EpisodeSpec(seed=4, category="regime_shift", regime_schedule=[[2.0, {"pad_friction": 0.3, "action_delay": 2}]])
ep = run_episode(env, spec, cameras=[], save_states=False)
d = ep["data"]
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(d["t"], d["progress"], label="progress")
ax[0].plot(d["t"], d["regime_id"], label="regime id")
ax[0].legend()
ax[1].plot(d["t"], d["state"][:, 22:25])
ax[1].set_ylabel("wrist force [N]")
ax[1].set_xlabel("t [s]")
print("outcome:", ep["annotations"]["outcome"], ep["annotations"]["failure_mode"])
plt.show()
