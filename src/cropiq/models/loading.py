"""Checkpoint (de)serialisation helpers shared by training, evaluation, API and dashboard."""
from __future__ import annotations

from pathlib import Path

import torch

from cropiq.models.classifier import LeafClassifier
from cropiq.models.ddpm import GaussianDiffusion, UNet
from cropiq.models.snn import IrrigationSNN, SpikeEncoder
from cropiq.models.vae import ConvVAE
from cropiq.utils import load_checkpoint


def load_classifier(path: str | Path, device: str | torch.device = "cpu"):
    ck = load_checkpoint(path, device)
    model = LeafClassifier(len(ck["class_names"]), 3, ck["backbone"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck


def load_vae(path: str | Path, device: str | torch.device = "cpu"):
    ck = load_checkpoint(path, device)
    model = ConvVAE(ck["image_size"], ck["latent_dim"], ck["base"])
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck


def load_snn(path: str | Path, device: str | torch.device = "cpu"):
    ck = load_checkpoint(path, device)
    h = ck["hyper"]
    model = IrrigationSNN(
        h["in_channels"], h["hidden"], h["n_out"], h["beta"], h["beta_out"], h["threshold"], h["slope"], h["backend"],
        readout_frac=h.get("readout_frac", 1.0),
    )
    # LIF constants (beta, threshold, ...) are hyper-parameters stored in `hyper`; backend-specific buffers
    # (snnTorch registers them in the state dict, the native LIF does not) are ignored so a checkpoint
    # trained with snnTorch also loads where snnTorch is not installed.
    state = {k: v for k, v in ck["model_state"].items() if not k.startswith("lif")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in [*missing, *unexpected] if not k.startswith("lif")]
    if bad:
        raise RuntimeError(f"SNN checkpoint does not match the model definition: {bad}")
    enc = SpikeEncoder(ck["encoder"]["pop"], ck["encoder"]["sub_steps"], ck["encoder"]["delta_gain"])
    return model.to(device).eval(), enc, ck


def load_ddpm(path: str | Path, device: str | torch.device = "cpu", use_ema: bool = True):
    ck = load_checkpoint(path, device)
    d = ck["unet"]
    model = UNet(
        len(ck["class_names"]),
        d["image_size"],
        d["base"],
        tuple(d["mults"]),
        d["num_res_blocks"],
        tuple(d["attn_resolutions"]),
    )
    model.load_state_dict(ck["ema_state"] if use_ema else ck["model_state"])
    diffusion = GaussianDiffusion(ck["timesteps"], ck["schedule"]).to(device)
    return model.to(device).eval(), diffusion, ck
