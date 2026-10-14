"""Small class-conditional DDPM (epsilon-prediction) with classifier-free guidance and DDIM sampling.

Sized for a 4 GB GPU: ~10-15 M parameters at 64x64 with the default config. Everything is written from
scratch in plain PyTorch so you can explain every line in an interview.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads if ch % heads == 0 else 1
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _Seq(nn.ModuleList):
    """Applies ResBlock(+Attention) layers that need the embedding."""


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 64,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base), nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)  # last index = "null" label for CFG
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)

        self.down = nn.ModuleList()
        chans = [base]
        ch, res = base, image_size
        for level, m in enumerate(mults):
            for _ in range(num_res_blocks):
                blocks = nn.ModuleList([ResBlock(ch, base * m, emb_dim, dropout)])
                ch = base * m
                if res in attn_resolutions:
                    blocks.append(SelfAttention(ch))
                self.down.append(blocks)
                chans.append(ch)
            if level != len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                chans.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), SelfAttention(ch), ResBlock(ch, ch, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, m in reversed(list(enumerate(mults))):
            for i in range(num_res_blocks + 1):
                blocks = nn.ModuleList([ResBlock(ch + chans.pop(), base * m, emb_dim, dropout)])
                ch = base * m
                if res in attn_resolutions:
                    blocks.append(SelfAttention(ch))
                if level != 0 and i == num_res_blocks:
                    blocks.append(Upsample(ch))
                    res *= 2
                self.up.append(blocks)
        self.out = nn.Sequential(nn.GroupNorm(_groups(ch), ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t) + self.class_emb(y)
        h = self.in_conv(x)
        skips = [h]
        for blocks in self.down:
            for layer in blocks:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for blocks in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in blocks:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.out(h)


# --------------------------------------------------------------------------------------
# Diffusion process
# --------------------------------------------------------------------------------------
def make_betas(timesteps: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    s = 0.008
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    abar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    abar = abar / abar[0]
    betas = 1 - abar[1:] / abar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        self.timesteps = timesteps
        betas = make_betas(timesteps, schedule)
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def to(self, device) -> "GaussianDiffusion":
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        abar = self.alphas_cumprod[t][:, None, None, None]
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    def loss(self, model: UNet, x0: torch.Tensor, y: torch.Tensor, p_uncond: float = 0.1) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        drop = torch.rand(b, device=x0.device) < p_uncond
        y = torch.where(drop, torch.full_like(y, model.num_classes), y)
        pred = model(self.q_sample(x0, t, noise), t, y)
        return F.mse_loss(pred.float(), noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: UNet,
        labels: torch.Tensor,
        image_size: int,
        steps: int = 50,
        guidance_scale: float = 3.0,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Class-conditional DDIM with classifier-free guidance. Returns images in [-1, 1]."""
        device = labels.device
        b = labels.shape[0]
        x = torch.randn(b, 3, image_size, image_size, device=device, generator=generator)
        seq = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        null = torch.full_like(labels, model.num_classes)
        for i, t in enumerate(seq):
            tt = torch.full((b,), int(t), device=device, dtype=torch.long)
            if guidance_scale == 1.0:
                eps = model(x, tt, labels)
            else:
                eps_c, eps_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([labels, null])).chunk(2)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            abar_t = self.alphas_cumprod[t]
            abar_prev = self.alphas_cumprod[seq[i + 1]] if i + 1 < len(seq) else torch.tensor(1.0, device=device)
            x0 = ((x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).sqrt()
            dir_xt = (1 - abar_prev - sigma**2).clamp(min=0).sqrt() * eps
            noise = torch.randn(x.shape, device=device, generator=generator) if eta > 0 else 0.0
            x = abar_prev.sqrt() * x0 + dir_xt + sigma * noise
        return x.clamp(-1, 1)


class EMA:
    """Exponential moving average of weights (essential for decent diffusion samples)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def build_unet(cfg, num_classes: int) -> UNet:
    d = cfg.diffusion
    return UNet(
        num_classes=num_classes,
        image_size=d.image_size,
        base=d.base_channels,
        mults=tuple(d.channel_mults),
        num_res_blocks=d.num_res_blocks,
        attn_resolutions=tuple(d.attn_resolutions),
    )
