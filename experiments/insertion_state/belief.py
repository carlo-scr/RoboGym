"""Grid belief over the unknown lateral offset of the hole, updated from failed contacts.

Coordinates are lateral (y, z) in the believed port frame. Aiming the tip at `a` puts it at
m = a + u relative to the true port at first contact, where u (perception error + controller drift) is
unknown but constant within an episode. A failed contact contributes two pieces of evidence:
  - negative information: the plug did not enter at m, likelihood 1 - p_in(m) from the E1 capture map;
  - the contact signals: the regressor's correction c_hat ~ N(-m, S_class), mixed over ramp / face by the
    classifier's probability.
The next aim maximises the posterior probability of entering, sum_u post(u) p_in(a + u)."""
from __future__ import annotations

import numpy as np


class CaptureMap:
    """P(entry | tip offset at contact), a smoothed 2-D histogram on a regular grid."""

    def __init__(self, m, entered, half=0.012, res=0.0005, smooth=1.0):
        n = int(round(2 * half / res)) + 1
        self.half, self.res, self.n = half, res, n
        idx = np.clip(np.round((np.asarray(m) + half) / res).astype(int), 0, n - 1)
        hit = np.zeros((n, n))
        tot = np.zeros((n, n))
        np.add.at(tot, (idx[:, 0], idx[:, 1]), 1.0)
        np.add.at(hit, (idx[:, 0], idx[:, 1]), np.asarray(entered, float))
        if smooth > 0:
            from scipy.ndimage import gaussian_filter

            hit, tot = gaussian_filter(hit, smooth), gaussian_filter(tot, smooth)
        # pseudo-count towards 0 far from data (outside the sampled disk the plug never enters)
        self.P = hit / (tot + 0.05)

    def __call__(self, m):
        idx = np.round((np.asarray(m) + self.half) / self.res).astype(int)
        ok = (idx >= 0).all(-1) & (idx < self.n).all(-1)
        out = np.zeros(idx.shape[:-1])
        out[ok] = self.P[idx[ok][:, 0], idx[ok][:, 1]]
        return out


class HoleBelief:
    def __init__(self, cap: CaptureMap, sigma, S_ramp, S_face, mean=(0.0, 0.0), half=0.010, res=0.00025, drift_sigma=0.0003):
        g = np.arange(-half, half + 1e-12, res)
        U = np.stack(np.meshgrid(g, g, indexing="ij"), -1)  # (n, n, 2)
        self.U = U.reshape(-1, 2)
        self.n = len(g)
        self.cap = cap
        d = self.U - np.asarray(mean)
        sig2 = np.broadcast_to(np.asarray(sigma, float) ** 2, (2,))
        self.logp = -0.5 * np.sum(d * d / sig2, -1)
        self.S = {"ramp": np.asarray(S_ramp), "face": np.asarray(S_face)}
        self.Si = {k: np.linalg.inv(v) for k, v in self.S.items()}
        self.logdet = {k: np.linalg.slogdet(v)[1] for k, v in self.S.items()}
        self.drift_sigma, self.res = drift_sigma, res
        self._normalise()

    def _normalise(self):
        self.logp -= self.logp.max()
        p = np.exp(self.logp)
        self.p = p / p.sum()

    def _diffuse(self):
        """Process noise between tries (the plug can shift in the grasp after a failed push)."""
        if self.drift_sigma <= 0:
            return
        from scipy.ndimage import gaussian_filter

        P = gaussian_filter(self.p.reshape(self.n, self.n), self.drift_sigma / self.res)
        self.logp = np.log(P.reshape(-1) + 1e-300)
        self._normalise()

    def update_out(self, aim, c_hat=None, p_ramp=0.5):
        m = np.asarray(aim)[None] + self.U
        ll = np.log(np.clip(1.0 - self.cap(m), 1e-3, 1.0))
        if c_hat is not None:
            r = np.asarray(c_hat)[None] + m  # c_hat - (-m)
            comp = []
            for k, w in (("ramp", p_ramp), ("face", 1.0 - p_ramp)):
                q = np.einsum("ni,ij,nj->n", r, self.Si[k], r)
                comp.append(np.log(max(w, 1e-6)) - 0.5 * (q + self.logdet[k]))
            ll = ll + np.logaddexp(*comp)
        self.logp = np.log(self.p + 1e-300) + ll
        self._normalise()
        self._diffuse()

    def mean(self):
        return self.p @ self.U

    def best_aim(self, half=0.008, res=0.0005, mass=0.999):
        """argmax_a sum_u post(u) p_in(a + u), with u restricted to the support holding `mass` of the belief."""
        order = np.argsort(self.p)[::-1]
        k = int(np.searchsorted(np.cumsum(self.p[order]), mass)) + 1
        sup, w = self.U[order[:k]], self.p[order[:k]]
        g = np.arange(-half, half + 1e-12, res)
        A = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        best, best_s = np.zeros(2), -1.0
        for chunk in np.array_split(A, max(1, len(A) * len(sup) // 2_000_000 + 1)):
            s = (self.cap(chunk[:, None] + sup[None]) * w[None]).sum(1)
            i = int(np.argmax(s))
            if s[i] > best_s:
                best, best_s = chunk[i], float(s[i])
        return best, best_s
