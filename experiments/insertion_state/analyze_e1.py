"""E1 analysis: is the contact state identifiable from non-privileged signals shortly after first contact,
and does the contact tell the robot which way to correct? Leave-scenes-out evaluation.

Contact is detected without privileged information (|F| > F_ON after taring, with sensor noise); the
labels come from the simulator at the same moment:
  in       the tip passes the face plane within 3 steps (entered the channel, possibly via the chamfer)
  ramp     the plug edge overlaps the jack wall by <= the chamfer width (on the lead-in, should be guidable)
  face     overlap beyond the chamfer (wedge front or face strip: no lateral guidance from the geometry)
  blocked  the gripper / plug touches a neighbouring cable first
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "runs" / "insertion_state"
F_ON = 2.5  # N, contact detection threshold on the tared wrench
SIG_F, SIG_M, SIG_TAU = 0.3, 0.01, 0.2  # assumed sensor noise (N, Nm, Nm)
HALF_CLEAR = 0.0008  # (channel - plug) / 2 per side
CHAMFER = 0.0025
CLASSES = ("in", "ramp", "face", "blocked")
K_MAX = 3


def noisy(a, i):
    rng = np.random.default_rng([i, 5])
    ts = a["ts"]
    F = ts["F"] + rng.normal(0, SIG_F, ts["F"].shape)
    M = ts["M"] + rng.normal(0, SIG_M, ts["M"].shape)
    tau = ts["tau"] + rng.normal(0, SIG_TAU, ts["tau"].shape)
    return F, M, tau


def first_contact(F, phase):
    idx = np.where((phase >= 1) & (np.linalg.norm(F, axis=1) > F_ON))[0]
    return int(idx[0]) if len(idx) else None


def label(ts, c0):
    """Privileged contact-state label at the (non-privileged) detection step c0."""
    w = slice(max(0, c0 - 1), c0 + 2)
    blocked = (ts["f_grip_occ"][w] + ts["f_grip_panel"][w] + ts["f_plug_occ"][w]).max()
    plug = (ts["f_plug_face"][w] + ts["f_plug_wedge"][w] + ts["f_plug_back"][w]).max()
    if blocked > 1.0 and blocked >= plug:
        return "blocked"
    depth = ts["tip_true"][c0 : c0 + 4, 0]
    if (depth > 0.0003).any() or (ts["latched"][c0 : c0 + 4] >= 0).any():
        return "in"
    y, z = np.abs(ts["tip_true"][c0, 1:])
    overlap = max(y, z) - HALF_CLEAR  # crude: the larger-axis overlap of the plug edge into the jack wall
    return "ramp" if overlap <= CHAMFER else "face"


def features(F, M, tau, ts, c0, k):
    """Non-privileged features at step c0 + k relative to the last pre-contact step c0 - 1."""
    b = max(0, c0 - 1)
    j = min(c0 + k, len(F) - 1)
    lag = ts["tcp"] - ts["cmd"]
    rows = []
    for s in range(c0, j + 1):
        rows.append(np.concatenate([
            F[s], M[s],
            [lag[s, 0] - lag[b, 0], ts["tcp"][s, 0] - ts["tcp"][b, 0]],  # axial lag growth, axial travel since contact
            lag[s, 1:] - lag[b, 1:],  # lateral deflection since contact
            tau[s] - tau[b],
        ]))
    while len(rows) < k + 1:
        rows.append(rows[-1])
    return np.concatenate(rows).astype(np.float64)


FEAT_NAMES_STEP = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "lag_x", "travel_x", "defl_y", "defl_z"] + [f"tau{i}" for i in range(7)]


def build(res, k):
    X, y, corr, scene, aux = [], [], [], [], []
    for i, a in enumerate(res):
        ts = a["ts"]
        F, M, tau = noisy(a, i)
        c0 = first_contact(F, ts["phase"])
        if c0 is None:
            continue
        X.append(features(F, M, tau, ts, c0, k))
        y.append(CLASSES.index(label(ts, c0)))
        corr.append(-ts["tip_true"][c0, 1:])  # lateral move that would centre the tip on the port
        scene.append(a["scene"])
        aux.append(dict(i=i, c0=c0, seated=bool(ts["latched"][-1] >= 0), err=a["err"]))
    return np.array(X), np.array(y), np.array(corr), np.array(scene), aux


# ---------------------------------------------------------------- small numpy models
class Std:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def fit_logreg(X, y, n_cls, lam=1e-2, iters=600, lr=0.5):
    n, d = X.shape
    W = np.zeros((d + 1, n_cls))
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(n_cls)[y]
    w_cls = 1.0 / np.maximum(1, np.bincount(y, minlength=n_cls))  # class-balanced
    sw = w_cls[y] / w_cls[y].mean()
    for _ in range(iters):
        Z = Xb @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = Xb.T @ ((P - Y) * sw[:, None]) / n + lam * np.vstack([W[:-1], np.zeros((1, n_cls))])
        W -= lr * G
    return W


def predict_logreg(W, X):
    return np.argmax(np.hstack([X, np.ones((len(X), 1))]) @ W, 1)


def fit_ridge(X, Y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)


def predict_ridge(B, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ B


def balanced_acc(y, p, n_cls):
    accs = [np.mean(p[y == c] == c) for c in range(n_cls) if (y == c).any()]
    return float(np.mean(accs))


def ang_err(pred, true):
    a = np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(true[:, 1], true[:, 0])
    return np.degrees(np.abs((a + np.pi) % (2 * np.pi) - np.pi))


def folds(scene, n_folds=6, seed=0):
    us = np.unique(scene)
    rng = np.random.default_rng(seed)
    rng.shuffle(us)
    return [np.isin(scene, us[f::n_folds]) for f in range(n_folds)]


def cv_eval(X, y, corr, scene):
    n_cls = len(CLASSES)
    p_multi = np.zeros(len(y), int)
    p_bin = np.zeros(len(y), int)
    p_rule = np.zeros(len(y), int)
    p_corr = np.zeros_like(corr)
    p_corr_rule = np.zeros_like(corr)
    i_travel = FEAT_NAMES_STEP.index("travel_x")
    n_step = len(FEAT_NAMES_STEP)
    for te in folds(scene):
        tr = ~te
        sd = Std().fit(X[tr])
        W = fit_logreg(sd(X[tr]), y[tr], n_cls)
        p_multi[te] = predict_logreg(W, sd(X[te]))
        yb = (y != 0).astype(int)
        Wb = fit_logreg(sd(X[tr]), yb[tr], 2)
        p_bin[te] = predict_logreg(Wb, sd(X[te]))
        # one-feature rule: axial travel since contact at the last step, best threshold on train
        trav = X[:, (X.shape[1] // n_step - 1) * n_step + i_travel]
        ths = np.quantile(trav[tr], np.linspace(0.01, 0.99, 99))
        best = max(ths, key=lambda th: balanced_acc(yb[tr], (trav[tr] < th).astype(int), 2))
        p_rule[te] = (trav[te] < best).astype(int)
        # direction: ridge on all features, trained on out-of-jack plug contacts (ramp + face)
        m_tr = tr & np.isin(y, [1, 2])
        B = fit_ridge(sd(X[m_tr]), corr[m_tr], lam=10.0)
        p_corr[te] = predict_ridge(B, sd(X[te]))
        # rule: correction along +/- the lateral force at contact, sign fit on train per true class
        # (an upper bound for the rule: it is told whether the contact is ramp or face)
        Flat = X[:, 1:3]
        for cls in (1, 2):
            m_c = tr & (y == cls)
            s = np.sign(np.sum(np.einsum("ij,ij->i", Flat[m_c], corr[m_c]))) if m_c.any() else 1.0
            p_corr_rule[te & (y == cls)] = s * Flat[te & (y == cls)]
    return p_multi, p_bin, p_rule, p_corr, p_corr_rule


def report(k, X, y, corr, scene, aux, preds, log=print):
    p_multi, p_bin, p_rule, p_corr, p_corr_rule = preds
    yb = (y != 0).astype(int)
    out = dict(k=k, ms=50 * (k + 1), n=len(y), inout_logreg=balanced_acc(yb, p_bin, 2), inout_travel_rule=balanced_acc(yb, p_rule, 2),
               cls4_logreg=balanced_acc(y, p_multi, 4))
    log(f"\n--- {50 * (k + 1)} ms after contact (k={k}); n={len(y)} attempts, {len(np.unique(scene))} scenes; class counts "
        + ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))
    log(f"  in vs out   : logistic balanced acc {out['inout_logreg']:.3f}   travel-threshold rule {out['inout_travel_rule']:.3f}")
    log(f"  4-class     : logistic balanced acc {out['cls4_logreg']:.3f}")
    cm = np.zeros((4, 4), int)
    for t, p in zip(y, p_multi):
        cm[t, p] += 1
    log("  confusion (rows true, cols pred): " + "  ".join(f"{c[:4]}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        log(f"     {c:8s} {cm[i]}")
    for name, cls in (("ramp", 1), ("face", 2)):
        m = y == cls
        if m.sum() < 5:
            continue
        e_l = ang_err(p_corr[m], corr[m])
        e_r = ang_err(p_corr_rule[m], corr[m])
        mag = np.linalg.norm(corr[m], axis=1) * 1e3
        mag_err = np.abs(np.linalg.norm(p_corr[m], axis=1) * 1e3 - mag)
        out[f"dir_{name}_ridge"], out[f"dir_{name}_force_rule"] = float(np.median(e_l)), float(np.median(e_r))
        log(f"  direction on {name:4s} (n={m.sum():4d}, true miss {np.median(mag):.1f} mm): learned median {np.median(e_l):5.1f} deg "
            f"(<45 deg: {np.mean(e_l < 45):.2f}), |mag| err {np.median(mag_err):.2f} mm;  lateral-force rule median {np.median(e_r):5.1f} deg")
    return out


def figures(res, X, y, corr, aux, preds_k, fig_dir=None):
    import matplotlib

    if fig_dir is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir) if fig_dir else OUT / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols = {"in": "#2a9d8f", "ramp": "#e9c46a", "face": "#e76f51", "blocked": "#6c757d"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) capture map in true-miss coordinates at contact
    for i, c in enumerate(CLASSES):
        m = y == i
        ax[0].scatter(-corr[m, 0] * 1e3, -corr[m, 1] * 1e3, s=5, c=cols[c], label=c, alpha=0.6)
    for r in (HALF_CLEAR, HALF_CLEAR + CHAMFER):
        ax[0].add_patch(plt.Rectangle((-r * 1e3, -r * 1e3), 2 * r * 1e3, 2 * r * 1e3, fill=False, ls="--", lw=0.8, color="k"))
    ax[0].set(xlabel="tip y - port y at contact [mm]", ylabel="tip z - port z [mm]", title="contact state vs true miss", xlim=(-9, 9), ylim=(-9, 9), aspect="equal")
    ax[0].legend(fontsize=8, markerscale=3)
    # (b) axial force / travel after contact
    for i, c in enumerate(CLASSES[:3]):
        idx = np.where(y == i)[0][:150]
        for j in idx:
            a = res[aux[j]["i"]]
            ts, c0 = a["ts"], aux[j]["c0"]
            s = slice(max(0, c0 - 2), min(len(ts["t"]), c0 + 8))
            tt = (np.arange(s.start, s.stop) - c0) * 50
            ax[1].plot(tt, (ts["tcp"][s, 0] - ts["tcp"][max(0, c0 - 1), 0]) * 1e3, c=cols[c], lw=0.5, alpha=0.3)
            ax[2].plot(tt, ts["F"][s, 0], c=cols[c], lw=0.5, alpha=0.3)
    ax[1].set(xlabel="ms since detected contact", ylabel="axial TCP travel since contact [mm]", title="measured TCP (encoders)")
    ax[2].set(xlabel="ms since detected contact", ylabel="axial wrist force [N]", title="wrist F/T")
    for c in CLASSES[:3]:
        ax[1].plot([], [], c=cols[c], label=c)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "e1_contact_state.png", dpi=130)
    # (c) direction predictions on ramp / face
    p_corr = preds_k[3]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    for a_, cls, nm in ((ax[0], 1, "ramp"), (ax[1], 2, "face")):
        m = y == cls
        e = ang_err(p_corr[m], corr[m])
        a_.hist(e, bins=36, range=(0, 180), color=cols[nm])
        a_.set(title=f"{nm}: correction-direction error (learned)", xlabel="deg", ylabel="attempts")
    fig.tight_layout()
    fig.savefig(fig_dir / "e1_direction.png", dpi=130)
    print(f"\nfigures -> {fig_dir}")


def load_runs(name_or_path):
    """E1 attempts from a shard directory, a pickle, or a run name under runs/insertion_state/."""
    p = Path(name_or_path)
    if not p.exists():
        p = OUT / name_or_path if (OUT / name_or_path).is_dir() else OUT / f"{name_or_path}.pkl"
    if p.is_dir():
        res = []
        for f in sorted(p.glob("scene_*.pkl")):
            res += pickle.load(open(f, "rb"))
    else:
        res = pickle.load(open(p, "rb"))
    return [a for a in res if not a["unstable"]]


def analyze(res, ks=range(K_MAX + 1), fig_dir=None, log=print):
    log(f"{len(res)} attempts; end outcomes: " + str({o: sum(a['outcome'] == o for a in res) for o in sorted({a['outcome'] for a in res})}))
    rows, keep = [], None
    for k in ks:
        X, y, corr, scene, aux = build(res, k)
        preds = cv_eval(X, y, corr, scene)
        rows.append(report(k, X, y, corr, scene, aux, preds, log=log))
        if k == 1:
            keep = (X, y, corr, aux, preds)
    if keep is not None:
        figures(res, *keep, fig_dir=fig_dir)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e1")
    args = ap.parse_args()
    analyze(load_runs(args.name))


if __name__ == "__main__":
    main()
