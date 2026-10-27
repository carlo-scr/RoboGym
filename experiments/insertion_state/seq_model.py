"""Temporal contact-state model: a small 1-D CNN ensemble over the high-rate wrench window around the
detected contact (plus the 20 Hz kinematic features), predicting the contact class and a Gaussian over the
lateral correction. Compared against the linear models of analyze_e1 at the same latency, leave-scenes-out.

Window: control steps c0-2 .. c0+k, each contributing its high-rate wrench samples (ts["Fh"], 100 Hz) and
the per-step features (lag xyz, axial travel, joint torques) repeated across its samples."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import analyze_e1 as A

PRE = 2


def window(a, i, k, rng=None):
    """(channels, length) window for attempt a, plus the detection step; None if no contact is detected."""
    ts = a["ts"]
    rng = rng or np.random.default_rng([i, 5])
    F, M, tau = A.noisy(a, i)
    c0 = A.first_contact(F, ts["phase"])
    if c0 is None:
        return None, None
    hr = ts.get("Fh")
    n = len(ts["t"])
    steps = [min(max(s, 0), n - 1) for s in range(c0 - PRE, c0 + k + 1)]
    b = max(0, c0 - 1)
    lag = ts["tcp"] - ts["cmd"]
    cols = []
    for s in steps:
        low = np.concatenate([lag[s] - lag[b], [ts["tcp"][s, 0] - ts["tcp"][b, 0]], tau[s] - tau[b]]) * np.r_[[1e3] * 4, [1.0] * 7]
        if hr is not None:
            w = hr[s] + rng.normal(0, 1, hr[s].shape) * np.r_[[A.SIG_F] * 3, [A.SIG_M] * 3]
        else:
            w = np.concatenate([F[s], M[s]])[None]
        for r in w:
            cols.append(np.concatenate([r[:3], r[3:] * 10.0, low]))
    return np.array(cols, np.float32).T, c0


def build(res, k):
    X, y, corr, scene = [], [], [], []
    for i, a in enumerate(res):
        x, c0 = window(a, i, k)
        if x is None:
            continue
        X.append(x)
        y.append(A.CLASSES.index(A.label(a["ts"], c0)))
        corr.append(-a["ts"]["tip_true"][c0, 1:] * 1e3)  # mm
        scene.append(a["scene"])
    return np.stack(X), np.array(y), np.array(corr, np.float32), np.array(scene)


class Net(nn.Module):
    def __init__(self, c_in, width=64, n_cls=4):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(c_in, width, 3, padding=1), nn.GELU(), nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
                               nn.Conv1d(width, width, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * width + c_in, width), nn.GELU(), nn.Linear(width, n_cls + 4))
        self.n_cls = n_cls

    def forward(self, x):
        h = self.f(x)
        z = torch.cat([h.mean(-1), h.amax(-1), x[:, :, -1]], -1)
        o = self.head(z)
        return o[:, : self.n_cls], o[:, self.n_cls : self.n_cls + 2], o[:, self.n_cls + 2 :].clamp(-6, 6)


def train_ensemble(X, y, corr, n_members=5, steps=3000, bs=512, lr=2e-3, device="cpu", seed=0, log=print):
    mu = X.mean((0, 2), keepdims=True)
    sd = X.std((0, 2), keepdims=True) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=device)
    yt = torch.tensor(y, device=device)
    ct = torch.tensor(corr, device=device)
    out_mask = torch.tensor(np.isin(y, [1, 2]), device=device)
    w_cls = torch.tensor(1.0 / np.maximum(1, np.bincount(y, minlength=4)), dtype=torch.float32, device=device)
    w_cls = w_cls / w_cls.mean()
    members = []
    for mi in range(n_members):
        torch.manual_seed(seed * 100 + mi)
        net = Net(X.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            idx = torch.randint(0, len(Xt), (bs,), device=device)
            logits, m, lv = net(Xt[idx])
            loss = nn.functional.cross_entropy(logits, yt[idx], weight=w_cls)
            om = out_mask[idx]
            if om.any():
                r = ct[idx][om] - m[om]
                loss = loss + 0.5 * (r * r * torch.exp(-lv[om]) + lv[om]).sum(-1).mean() * 0.2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        members.append(net.eval())
        log(f"  member {mi}: final loss {loss.item():.3f}")
    return dict(members=members, mu=mu, sd=sd, device=device)


@torch.no_grad()
def predict(model, X):
    """Class probabilities and a Gaussian over the correction [mm] (ensemble mixture moments)."""
    Xt = torch.tensor((X - model["mu"]) / model["sd"], device=model["device"])
    P, Ms, Vs = [], [], []
    for net in model["members"]:
        logits, m, lv = net(Xt)
        P.append(torch.softmax(logits, -1))
        Ms.append(m)
        Vs.append(torch.exp(lv))
    P = torch.stack(P).mean(0)
    Ms, Vs = torch.stack(Ms), torch.stack(Vs)
    mean = Ms.mean(0)
    var = Vs.mean(0) + Ms.var(0, unbiased=False)
    return P.cpu().numpy(), mean.cpu().numpy(), var.cpu().numpy()


def evaluate(res, ks=(0, 1, 2), device="cpu", steps=3000, n_members=5, log=print):
    """Leave-scenes-out comparison of the CNN ensemble and the linear models at each latency k."""
    rows = []
    for k in ks:
        X, y, corr, scene = build(res, k)
        Xl, yl, corrl, scenel, _ = A.build(res, k)
        assert len(y) == len(yl)
        te = A.folds(scene, n_folds=5)[0]
        model = train_ensemble(X[~te], y[~te], corr[~te], n_members=n_members, device=device, steps=steps, log=log)
        P, mean, var = predict(model, X[te])
        pred = P.argmax(1)
        out = np.isin(y[te], [1, 2])
        z2 = ((corr[te] - mean) ** 2 / var).sum(1)
        # linear baseline on the same split
        sd = A.Std().fit(Xl[~te])
        Wb = A.fit_logreg(sd(Xl[~te]), (yl[~te] != 0).astype(int), 2)
        W4 = A.fit_logreg(sd(Xl[~te]), yl[~te], 4)
        m_tr = ~te & np.isin(yl, [1, 2])
        B = A.fit_ridge(sd(Xl[m_tr]), corrl[m_tr], lam=10.0)
        pl = A.predict_ridge(B, sd(Xl[te])) * 1e3
        row = dict(
            k=k, ms=50 * (k + 1), n_test=int(te.sum()),
            cnn_inout=A.balanced_acc((y[te] != 0).astype(int), (pred != 0).astype(int), 2),
            lin_inout=A.balanced_acc((yl[te] != 0).astype(int), A.predict_logreg(Wb, sd(Xl[te])), 2),
            cnn_4cls=A.balanced_acc(y[te], pred, 4), lin_4cls=A.balanced_acc(yl[te], A.predict_logreg(W4, sd(Xl[te])), 4),
            cnn_dir_ramp=float(np.median(A.ang_err(mean[out & (y[te] == 1)], corr[te][out & (y[te] == 1)]))),
            lin_dir_ramp=float(np.median(A.ang_err(pl[out & (y[te] == 1)], corr[te][out & (y[te] == 1)]))),
            cnn_dir_face=float(np.median(A.ang_err(mean[out & (y[te] == 2)], corr[te][out & (y[te] == 2)]))),
            lin_dir_face=float(np.median(A.ang_err(pl[out & (y[te] == 2)], corr[te][out & (y[te] == 2)]))),
            cnn_cover_90=float(np.mean(z2[out] < 4.605)),  # chi2(2) 90 % ellipse coverage on out contacts
        )
        log(row)
        rows.append(row)
    return rows
