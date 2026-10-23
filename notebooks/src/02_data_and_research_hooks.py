# %% [markdown]
# # Datasets, counterfactual branching, TTA streams
# * `run_episode` / `write_episode` write the lab's real-data layout (parquet + mp4 + annotations)
# * `get_state` / `set_state` are bit-exact: branch K action chunks from one state (latent-steering RQ1)
# * `streams.make_stream` builds continual-TTA protocols with ground-truth regime labels

# %%
# @bootstrap

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from robogym.env import RoboGymEnv
from robogym.datagen import EpisodeSpec, run_episode, write_episode, load_episode, restore, build_episode
from robogym.research import branch_outcomes, monitor_labels
from robogym.streams import make_stream, regime_table
from robogym.tools import run_until

env = RoboGymEnv("lego")
OUT = Path("data/demo_lego")

# %% [markdown]
# ## Write, load, restore

# %%
ep = run_episode(env, EpisodeSpec(seed=2, dart=0.002), cameras=["right_wrist", "front"])
path = write_episode(ep, OUT / "ep_000002", fps=env.cfg.sim.control_hz, cfg=env.cfg)
E = load_episode(path)
print(E["df"].columns.tolist())
print("outcome:", E["annotations"]["outcome"], "| horizon:", E["annotations"]["horizon"], "| stage times:", E["annotations"]["stage_times_s"])
k = len(E["df"]) // 2
restore(env, E, k)
st = np.stack(E["df"]["observation.state"].values)
dev = 0.0
for j in range(k, len(st) - 1):
    env.step(E["actions64"][j])
    dev = max(dev, float(np.abs(env.observe() - st[j + 1]).max()))
print("restore + replay max state deviation:", dev)

# %% [markdown]
# ## Monitor labels from a recorded failure

# %%
ep_f = run_episode(env, EpisodeSpec(seed=3, category="known_failure", failure="drop", severity=0.8), cameras=[], save_states=False)
write_episode(ep_f, OUT / "ep_fail", fps=env.cfg.sim.control_hz, cfg=env.cfg)
Ef = load_episode(OUT / "ep_fail")
lab = monitor_labels(Ef["df"], Ef["annotations"], lead_s=1.0)
t = Ef["df"]["timestamp"].values * 1e-9
plt.figure(figsize=(9, 3))
plt.plot(t, lab["progress"], label="progress")
plt.plot(t, lab["pre_failure"], label="pre-failure window")
plt.axvline(lab["onset_s"], color="r", ls="--", label="onset")
plt.legend(); plt.xlabel("t [s]"); plt.show()

# %% [markdown]
# ## Counterfactual branching (the RQ1 oracle mechanism)
# Snapshot while carrying the first brick, then execute 6 candidate chunks (hold + random lateral drift)
# and let the expert-free "hold" continuation finish. Differences between branches come from the chunks alone.

# %%
sc, _, ex = build_episode(env, EpisodeSpec(seed=7))
run_until(env, ex, "carry", subgoal=0)
state = env.get_state()
rng = np.random.default_rng(0)
hold = np.zeros(8); hold[6] = 1.0
chunks = [np.repeat(hold[None], 8, 0) + np.concatenate([rng.normal(0, s, 3), np.zeros(5)]) for s in (0, 0, 0.003, 0.006, 0.01, 0.015)]
res = branch_outcomes(env, state, chunks, lambda e: hold, max_s=env.time + 2.0)
pd.DataFrame(res)

# %% [markdown]
# ## TTA stream protocols

# %%
for name in ("sudden", "gradual", "recurring", "mixed", "in_episode"):
    specs = make_stream(name, 12, seed0=0)
    print(f"{name:11s}", [r[1] for r in regime_table(specs)])
