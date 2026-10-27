"""Frame encoders for the latent world model: a small conv autoencoder (runs anywhere) and the lab's
frozen NVIDIA Cosmos tokenizer (GPU nodes). Both expose `encode(frames uint8 [N,H,W,3]) -> [N, D]`."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    def __init__(self, hw=(72, 96), latent=32):
        super().__init__()
        H, W = hw
        assert H % 8 == 0 and W % 8 == 0, "frame size must be divisible by 8"
        self.hw, self.latent = (H, W), latent
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(), nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU(),
        )
        self.h8, self.w8 = H // 8, W // 8
        self.fc_e = nn.Linear(64 * self.h8 * self.w8, latent)
        self.fc_d = nn.Linear(latent, 64 * self.h8 * self.w8)
        self.dec = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.SiLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def _x(self, frames):
        return torch.as_tensor(frames).permute(0, 3, 1, 2).float() / 255.0

    def forward_latent(self, x):
        return self.fc_e(self.enc(x * 2 - 1).flatten(1))

    def decode(self, z):
        return torch.sigmoid(self.dec(self.fc_d(z).view(-1, 64, self.h8, self.w8)))

    @torch.no_grad()
    def encode(self, frames, bs=512):
        dev = next(self.parameters()).device
        out = [self.forward_latent(self._x(frames[i : i + bs]).to(dev)).cpu() for i in range(0, len(frames), bs)]
        return torch.cat(out).numpy().astype(np.float32)


def edge_weights(x, strength):
    g = x.mean(1, keepdim=True)
    m = F.pad((g[..., :, 1:] - g[..., :, :-1]).abs(), (0, 1)) + F.pad((g[..., 1:, :] - g[..., :-1, :]).abs(), (0, 0, 0, 1))
    m = m / (m.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return 1 + strength * (m / 10).clamp(max=1)


def train_ae(frames: np.ndarray, latent=32, steps=3000, bs=128, lr=1e-3, device="cpu", log=print, log_every=500, edge=0.0, ckpt_dir=None, ckpt_every=1000,
             budget_s=None):
    """ckpt_dir / ckpt_every / budget_s: checkpoint and resume as in robogym.learn.policy.train_policy."""
    from pathlib import Path

    from .ckpt import load_ckpt, rng_state, save_ckpt, set_rng_state

    ae = ConvAE(frames.shape[1:3], latent).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, hist = time.time(), []
    start = 0
    ck = Path(ckpt_dir) / "ckpt.pt" if ckpt_dir else None
    st = load_ckpt(ck, device) if ck else None
    if st is not None and st["steps"] == steps:
        ae.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        hist, start = list(st["hist"]), int(st["step"])
        set_rng_state(st.get("rng"))
        if log:
            log(f"[ae] resumed from {ck} at step {start}/{steps}")
    for it in range(start, steps):
        j = np.random.randint(0, len(frames), bs)
        x = ae._x(frames[j]).to(device)
        err = (ae.decode(ae.forward_latent(x)) - x) ** 2
        if edge > 0:
            w = edge_weights(x, edge)
            loss = (w * err).mean() / w.mean()
        else:
            loss = err.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(err.mean().item())
        if log and (it % log_every == 0 or it == steps - 1):
            log(f"[ae] step {it:5d}  mse {np.mean(hist[-log_every:]):.5f}  ({time.time() - t0:.0f}s)")
        if ck is not None and ((it + 1) % ckpt_every == 0 or it == steps - 1):
            save_ckpt(ck, dict(step=it + 1, steps=steps, model=ae.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, rng=rng_state()))
            if budget_s is not None and time.time() - t0 > budget_s and it < steps - 1:
                if log:
                    log(f"[ae] time budget reached at step {it + 1}/{steps}; checkpoint written")
                break
    return ae.eval(), hist


class CosmosEncoder:
    """Frozen Cosmos continuous image tokenizer -> spatially pooled latent vector (GPU nodes)."""

    def __init__(self, ckpt_dir, name="Cosmos-0.1-Tokenizer-CI16x16", device="cuda", pool=(3, 4)):
        from cosmos_tokenizer.image_lib import ImageTokenizer  # noqa: import only on GPU nodes

        self.tok = ImageTokenizer(checkpoint_enc=f"{ckpt_dir}/{name}/encoder.jit", device=device)
        self.device, self.pool = device, pool

    @torch.no_grad()
    def encode(self, frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            x = torch.as_tensor(frames[i : i + bs]).permute(0, 3, 1, 2).to(self.device).to(torch.bfloat16) / 127.5 - 1
            (z,) = self.tok.encode(x)
            out.append(F.adaptive_avg_pool2d(z.float(), self.pool).flatten(1).cpu())
        return torch.cat(out).numpy().astype(np.float32)


def save_ae(ae: ConvAE, path):
    torch.save(dict(hw=ae.hw, latent=ae.latent, state=ae.state_dict()), path)


def load_ae(path, device="cpu") -> ConvAE:
    ck = torch.load(path, map_location=device, weights_only=False)
    ae = ConvAE(tuple(ck["hw"]), ck["latent"])
    ae.load_state_dict(ck["state"])
    return ae.to(device).eval()
