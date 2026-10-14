"""Convolutional VAE trained on HEALTHY leaves only; reconstruction error = anomaly score."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class ConvVAE(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 128, base: int = 32):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        n_down = max(2, int(math.log2(image_size // 4)))
        self.n_down = n_down
        if image_size % 2**n_down:
            raise ValueError(f"image_size must be divisible by {2**n_down} (use 32/48/64/96/128)")
        chans = [min(base * 2**i, 256) for i in range(n_down)]

        enc, cin = [], 3
        for c in chans:
            enc += [nn.Conv2d(cin, c, 4, stride=2, padding=1), _gn(c), nn.LeakyReLU(0.2, inplace=True)]
            cin = c
        self.encoder = nn.Sequential(*enc)
        self.final_res = image_size // 2**n_down
        flat = cin * self.final_res**2
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        self._dec_ch = cin

        dec = []
        rev = list(reversed(chans))
        for i, c in enumerate(rev):
            cout = rev[i + 1] if i + 1 < len(rev) else base
            dec += [nn.ConvTranspose2d(c, cout, 4, stride=2, padding=1), _gn(cout), nn.LeakyReLU(0.2, inplace=True)]
        dec += [nn.Conv2d(base, 3, 3, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self._dec_ch, self.final_res, self.final_res)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(x, recon, mu, logvar, sigma: float = 0.1, beta: float = 1.0):
        """Gaussian-decoder ELBO per image: SSE / (2 sigma^2) + beta * KL."""
        sse = F.mse_loss(recon, x, reduction="none").flatten(1).sum(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        rec = sse / (2 * sigma**2)
        return (rec + beta * kl).mean(), sse.mean() / x[0].numel(), kl.mean()

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, mode: str = "topk", topk_frac: float = 0.10) -> torch.Tensor:
        """Per-image anomaly score. 'topk' = mean of the worst `topk_frac` pixel errors, which reacts to
        small local lesions that a plain mean-squared error would average away."""
        recon, _, _ = self.forward(x)  # eval mode -> deterministic mean of the posterior
        err = (x - recon).pow(2).mean(1).flatten(1)
        if mode == "mean":
            return err.mean(1)
        k = max(1, int(err.shape[1] * topk_frac))
        return err.topk(k, dim=1).values.mean(1)
