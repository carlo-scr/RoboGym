"""Policies for the benchmark: scripted, learned, off-the-shelf and hierarchical, behind one interface
(robogym.policies.base.Policy). `make_policy(spec)` builds one from a short string, so evaluations and
worker processes can name policies on the command line:

    expert                               the task's scripted expert (privileged upper bound)
    expert:exec_noise=0.004              ... with actuation noise (m/step) on every executed action
    expert:failure=drop,severity=0.8     ... with an injected failure kind (p_failure=1 per episode)
    expert:recover=1                     ... noticing failed subgoals and retrying them
    chunk:runs/dp_cubes/policy.pt        a robogym.learn checkpoint (Diffusion Policy / ACT / MLP)
    hier:oracle+scripted                 planner + executor (robogym.policies.hierarchical)
    hier:heuristic+scripted(exec_noise=0.003)
    hier:corrupt(p_swap=0.5)+scripted:on_fail=retry
    hier:reverse+chunk(runs/dp_cubes/policy.pt)
    hier:llm+scripted                    Claude plans from a text scene description (needs the SDK + key)
    hier:llm(model=claude-sonnet-5-5,cache_dir=runs/llm_cache)+scripted
    lerobot:runs/offtheshelf/cube_stack/act    a LeRobot checkpoint (robogym.learn.lerobot_io), needs `pip install lerobot`
    random | hold | replay:data/x/ep_000003
"""
from __future__ import annotations

import re

from .base import FunctionPolicy, HoldPolicy, Policy, RandomPolicy, ReplayPolicy
from .hierarchical import Executor, HierarchicalPolicy, LearnedExecutor, ScriptedExecutor
from .learned import ChunkPolicy, LeRobotPolicy
from .planning import CorruptedPlanner, HeuristicPlanner, LLMPlanner, OraclePlanner, PermutedPlanner, Planner, Subgoal, adapter_for
from .scripted import ExpertPolicy

__all__ = ["Policy", "FunctionPolicy", "HoldPolicy", "RandomPolicy", "ReplayPolicy", "ExpertPolicy", "ChunkPolicy", "LeRobotPolicy",
           "HierarchicalPolicy", "Executor", "ScriptedExecutor", "LearnedExecutor", "Planner", "OraclePlanner", "HeuristicPlanner",
           "PermutedPlanner", "CorruptedPlanner", "LLMPlanner", "Subgoal", "adapter_for", "make_policy", "parse_spec"]


def _val(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _kw(args: str) -> tuple[list, dict]:
    pos, kw = [], {}
    for part in filter(None, (p.strip() for p in args.split(","))):
        if "=" in part:
            k, v = part.split("=", 1)
            kw[k.strip()] = _val(v)
        else:
            pos.append(_val(part))
    return pos, kw


def _call(s: str) -> tuple[str, list, dict]:
    """'name(a, k=v)' -> (name, [a], {k: v}); 'name' -> (name, [], {})."""
    m = re.fullmatch(r"\s*([\w.\-]+)\s*(?:\((.*)\))?\s*", s, re.S)
    if not m:
        raise ValueError(f"bad spec part {s!r}")
    pos, kw = _kw(m.group(2) or "")
    return m.group(1), pos, kw


def _split_top(s: str, sep: str) -> list:
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def parse_spec(spec: str) -> dict:
    kind, _, rest = spec.strip().partition(":")
    kind = kind.strip()
    if kind == "hier":
        combo, *opts = _split_top(rest, ":")
        parts = _split_top(combo, "+")
        if len(parts) != 2:
            raise ValueError(f"hier spec needs '<planner>+<executor>': {spec!r}")
        _p, kw = _kw(":".join(opts))
        return dict(kind="hier", planner=_call(parts[0]), executor=_call(parts[1]), opts=kw)
    pos, kw = _kw(rest) if kind in ("expert", "random", "hold") else ([rest] if rest else [], {})
    if kind in ("chunk", "lerobot") and rest:
        path, *opts = _split_top(rest, ":")
        _p, kw = _kw(":".join(opts))
        pos = [path]
    return dict(kind=kind, pos=pos, opts=kw)


def make_planner(name, pos, kw, seed=0):
    if name == "oracle":
        return OraclePlanner()
    if name == "heuristic":
        return HeuristicPlanner()
    if name in ("reverse", "shuffle"):
        return PermutedPlanner(name, seed=kw.get("seed", seed))
    if name == "corrupt":
        base = kw.pop("base", "oracle")
        return CorruptedPlanner(make_planner(base, [], {}, seed), seed=kw.pop("seed", seed), **kw)
    if name == "llm":
        return LLMPlanner(**kw)
    raise ValueError(f"unknown planner {name!r}")


def make_executor(name, pos, kw, device="cpu"):
    if name == "scripted":
        return ScriptedExecutor(**kw)
    if name in ("chunk", "learned"):
        path = pos[0] if pos else kw.pop("path")
        return LearnedExecutor(path, device=kw.pop("device", device), **kw)
    raise ValueError(f"unknown executor {name!r}")


def make_policy(spec: str, device="cpu", seed=0) -> Policy:
    p = parse_spec(spec)
    k = p["kind"]
    if k == "hier":
        pl = make_planner(*p["planner"], seed=seed)
        ex = make_executor(*p["executor"], device=device)
        return HierarchicalPolicy(pl, ex, **p["opts"])
    if k == "expert":
        return ExpertPolicy(**p["opts"])
    if k == "chunk":
        return ChunkPolicy(p["pos"][0], device=p["opts"].pop("device", device), **p["opts"])
    if k == "lerobot":
        return LeRobotPolicy(p["pos"][0], device=p["opts"].pop("device", device), **p["opts"])
    if k == "random":
        return RandomPolicy(**p["opts"])
    if k == "hold":
        return HoldPolicy()
    if k == "replay":
        return ReplayPolicy(p["pos"][0])
    raise ValueError(f"unknown policy spec {spec!r}")
