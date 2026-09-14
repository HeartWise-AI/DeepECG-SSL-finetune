#!/usr/bin/env python3
"""Class-conditional 1-D diffusion model over single ECG beats.

Classes: 0 = normal (no acute coronary syndrome), 1 = LEFT (LAD or Left Main) complete
occlusion, 2 = RCA, 3 = LCX. Class 4 is the null token used for classifier-free guidance.

Unlike the morph, which pushes a normal beat until a classifier is satisfied, this model learns
P(beat | territory) from real occluded beats. An ACO-positive example is then a SAMPLE, so its
morphology comes from the data rather than from the classifier's decision boundary.

Schedule note: the cosine schedule sends alpha_bar to ~1e-15 at the terminal step, and
predict_x0 divides by sqrt(alpha_bar), amplifying any epsilon error by ~1e7 -- the bug recorded
in DeepECG-PNG/scripts/explainability/HANDOVER.md section 4, which clamped every early sample to
the amplitude bound. alpha_bar is clipped to ALPHA_MIN here and sampling starts at the last
timestep above it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BEAT_LEN, PAD_LEN = 250, 256
N_CLASSES = 4                      # normal, LEFT, RCA, LCX
NULL_CLASS = 4                     # classifier-free guidance token
T_STEPS = 1000
ALPHA_MIN = 1e-4


# ------------------------------------------------------------------ schedule
def cosine_alpha_bar(T=T_STEPS, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(min=ALPHA_MIN, max=1.0)
    return ab[1:].float()          # alpha_bar[t], t = 0..T-1


class Schedule:
    def __init__(self, T=T_STEPS, device="cuda"):
        self.T = T
        self.ab = cosine_alpha_bar(T).to(device)
        self.t_max = int((self.ab > ALPHA_MIN).nonzero().max().item())
        self.sqrt_ab = self.ab.sqrt()
        self.sqrt_1mab = (1 - self.ab).sqrt()

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t][:, None, None] * x0 + self.sqrt_1mab[t][:, None, None] * noise

    def predict_x0(self, xt, t, eps):
        return ((xt - self.sqrt_1mab[t][:, None, None] * eps)
                / self.sqrt_ab[t][:, None, None])


# ------------------------------------------------------------------ model
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cemb, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, cin)
        self.c1 = nn.Conv1d(cin, cout, 5, padding=2)
        self.emb = nn.Linear(cemb, cout * 2)
        self.n2 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv1d(cout, cout, 5, padding=2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None].chunk(2, 1)
        h = F.silu(self.n2(h) * (1 + scale) + shift)
        return self.skip(x) + self.c2(h)


class BeatUNet(nn.Module):
    def __init__(self, in_ch=12, base=96, mults=(1, 2, 3), cemb=384, n_res=2):
        super().__init__()
        self.cemb = cemb
        self.t_mlp = nn.Sequential(nn.Linear(base, cemb), nn.SiLU(), nn.Linear(cemb, cemb))
        self.base = base
        self.cls_emb = nn.Embedding(N_CLASSES + 1, cemb)
        chs = [base * m for m in mults]

        self.stem = nn.Conv1d(in_ch, chs[0], 5, padding=2)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        skip_ch = [chs[0]]
        c = chs[0]
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(n_res):
                blocks.append(ResBlock(c, ch, cemb))
                c = ch
                skip_ch.append(c)
            self.down.append(blocks)
            if i < len(chs) - 1:
                self.downsample.append(nn.Conv1d(c, c, 4, 2, 1))
                skip_ch.append(c)
            else:
                self.downsample.append(nn.Identity())

        self.mid1 = ResBlock(c, c, cemb)
        self.mid2 = ResBlock(c, c, cemb)

        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            blocks = nn.ModuleList()
            for _ in range(n_res + 1):
                blocks.append(ResBlock(c + skip_ch.pop(), ch, cemb))
                c = ch
            self.up.append(blocks)
            if i > 0:
                self.upsample.append(nn.ConvTranspose1d(c, c, 4, 2, 1))
            else:
                self.upsample.append(nn.Identity())

        self.out_n = nn.GroupNorm(8, c)
        self.out_c = nn.Conv1d(c, in_ch, 5, padding=2)
        nn.init.zeros_(self.out_c.weight)
        nn.init.zeros_(self.out_c.bias)

    def forward(self, x, t, y):
        emb = self.t_mlp(timestep_embedding(t, self.base)) + self.cls_emb(y)
        h = self.stem(x)
        hs = [h]
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, emb)
                hs.append(h)
            if not isinstance(ds, nn.Identity):
                h = ds(h)
                hs.append(h)
        h = self.mid2(self.mid1(h, emb), emb)
        for blocks, us in zip(self.up, self.upsample):
            for b in blocks:
                h = b(torch.cat([h, hs.pop()], 1), emb)
            if not isinstance(us, nn.Identity):
                h = us(h)
        return self.out_c(F.silu(self.out_n(h)))


# ------------------------------------------------------------------ sampling
@torch.no_grad()
def ddim_sample(model, sched, n, y, device="cuda", steps=100, guidance=3.0, in_ch=12):
    """Classifier-free-guided DDIM. Starts at t_max, never at the clipped terminal step."""
    ts = torch.linspace(sched.t_max, 0, steps + 1).long().to(device)
    x = torch.randn(n, in_ch, PAD_LEN, device=device) * sched.sqrt_1mab[ts[0]]
    y = y.to(device)
    y_null = torch.full_like(y, NULL_CLASS)
    for i in range(steps):
        t = ts[i].repeat(n)
        eps_c = model(x, t, y)
        if guidance != 0:
            eps_u = model(x, t, y_null)
            eps = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = eps_c
        x0 = sched.predict_x0(x, t, eps).clamp(-6, 6)
        t_next = ts[i + 1].repeat(n)
        ab_next = sched.ab[t_next][:, None, None]
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x[:, :, :BEAT_LEN]
