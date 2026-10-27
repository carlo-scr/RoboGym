"""Alternative chunk-policy architectures with the DiffusionPolicy interface (robogym.learn.policy):
`cfg`, `obs_norm`, `act_norm`, `loss(low, act, imgs)`, `sample(low, imgs, K, ...) -> [B, K, horizon, act_dim]`,
so PolicyRunner, rollout, steering and the RQ1 oracle work unchanged (`PolicyCfg.arch` selects them).

* MLPChunkPolicy: chunked behaviour cloning, an MLP regressing the whole action chunk (MSE). The simplest
  baseline; deterministic, so the K "candidates" of a replan are identical.
* ACTPolicy: Action Chunking with Transformers (Zhao et al. 2023). A CVAE: at training a transformer
  encoder infers a style latent z from the chunk and the observation; a transformer encoder-decoder with
  H learned queries predicts the chunk from (z, observation tokens). L1 + beta KL. At test z = 0
  (deterministic) unless cfg.z_sample, in which case the per-candidate seeded noise drives z.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import STATE_PER_ARM
from .data import Normalizer
from .policy import ImageEncoder, PolicyCfg


def _low_dim(cfg: PolicyCfg) -> int:
    return cfg.state_dim + (cfg.task_dim if cfg.obs_mode == "state" else 0) + (cfg.tgt_dim if cfg.action_mode in ("abs", "rel") else 0)


class MLPChunkPolicy(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.low_dim, self.act_dim = _low_dim(cfg), cfg.act_dim
        img = 0
        if cfg.obs_mode == "image":
            self.img_enc = ImageEncoder(len(cfg.cams), cfg.img_feat)
            img = len(cfg.cams) * cfg.img_feat
        h = cfg.mlp_hidden
        self.net = nn.Sequential(nn.Linear(cfg.n_obs * (self.low_dim + img), h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, h), nn.Mish(),
                                 nn.Linear(h, cfg.horizon * self.act_dim))
        self.obs_norm, self.act_norm = Normalizer(), Normalizer()

    def cond(self, low, imgs=None):
        B = low.shape[0]
        parts = [low]
        if self.cfg.obs_mode == "image":
            parts.append(self.img_enc(imgs.flatten(0, 1)).view(B, self.cfg.n_obs, -1))
        return torch.cat(parts, -1).flatten(1)

    def forward(self, low, imgs=None):
        return self.net(self.cond(low, imgs)).view(low.shape[0], self.cfg.horizon, self.act_dim)

    def loss(self, low, act, imgs=None):
        return F.mse_loss(self(low, imgs), act)

    @torch.no_grad()
    def sample(self, low, imgs=None, K=1, steps=None, eta=None, generator=None, noise=None):
        return self(low, imgs)[:, None].repeat(1, K, 1, 1)


class ACTPolicy(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d_model, cfg.horizon
        self.low_dim, self.act_dim = _low_dim(cfg), cfg.act_dim
        self.low_proj = nn.Linear(self.low_dim, d)
        if cfg.obs_mode == "image":
            self.img_enc = ImageEncoder(len(cfg.cams), cfg.img_feat)
            self.img_proj = nn.Linear(len(cfg.cams) * cfg.img_feat, d)
        n_obs_tok = cfg.n_obs * (2 if cfg.obs_mode == "image" else 1)
        self.obs_pos = nn.Parameter(torch.randn(n_obs_tok, d) * 0.02)
        # CVAE encoder (training only): [cls, action tokens, obs tokens] -> z
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.act_proj = nn.Linear(self.act_dim, d)
        self.enc_pos = nn.Parameter(torch.randn(1 + H + n_obs_tok, d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)  # noqa: E731
        self.cvae_enc = nn.TransformerEncoder(layer(), cfg.n_enc, enable_nested_tensor=False)
        self.z_head = nn.Linear(d, 2 * cfg.latent)
        # decoder: memory = encoder over [z token, obs tokens]; queries -> chunk
        self.z_proj = nn.Linear(cfg.latent, d)
        self.mem_pos = nn.Parameter(torch.randn(1 + n_obs_tok, d) * 0.02)
        self.enc = nn.TransformerEncoder(layer(), cfg.n_enc, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(H, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, cfg.n_heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True), cfg.n_dec)
        self.out = nn.Linear(d, self.act_dim)
        self.obs_norm, self.act_norm = Normalizer(), Normalizer()

    def obs_tokens(self, low, imgs=None):
        toks = [self.low_proj(low)]
        if self.cfg.obs_mode == "image":
            B = low.shape[0]
            toks.append(self.img_proj(self.img_enc(imgs.flatten(0, 1)).view(B, self.cfg.n_obs, -1)))
        return torch.cat(toks, 1) + self.obs_pos

    def encode(self, obs_tok, act):
        B = act.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.act_proj(act), obs_tok], 1) + self.enc_pos
        mu, logvar = self.z_head(self.cvae_enc(x)[:, 0]).chunk(2, -1)
        return mu, logvar

    def decode(self, obs_tok, z):
        mem = torch.cat([self.z_proj(z)[:, None], obs_tok], 1) + self.mem_pos
        mem = self.enc(mem)
        q = self.queries[None].expand(mem.shape[0], -1, -1)
        return self.out(self.dec(q, mem))

    def loss(self, low, act, imgs=None):
        obs_tok = self.obs_tokens(low, imgs)
        mu, logvar = self.encode(obs_tok, act)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        pred = self.decode(obs_tok, z)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean() / self.cfg.latent
        return F.l1_loss(pred, act) + self.cfg.kl_weight * kl

    @torch.no_grad()
    def sample(self, low, imgs=None, K=1, steps=None, eta=None, generator=None, noise=None):
        B = low.shape[0]
        obs_tok = self.obs_tokens(low, imgs).repeat_interleave(K, 0)
        if self.cfg.z_sample and noise is not None:
            z = noise.flatten(1)[:, : self.cfg.latent]
        elif self.cfg.z_sample:
            z = torch.randn((B * K, self.cfg.latent), device=low.device, generator=generator)
        else:
            z = torch.zeros((B * K, self.cfg.latent), device=low.device)
        return self.decode(obs_tok, z).view(B, K, self.cfg.horizon, self.act_dim)


__all__ = ["ACTPolicy", "MLPChunkPolicy", "STATE_PER_ARM"]
