from pathlib import Path

import pytest
import torch

from cropiq.config import load_config
from cropiq.data.toy import make_toy_dataset
from cropiq.models.classifier import LeafClassifier
from cropiq.models.snn import build_snn
from cropiq.models.vae import ConvVAE
from cropiq.utils import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def toy_cfg(tmp_path_factory):
    """Smoke config pointing at a tiny procedural dataset in a temp dir."""
    tmp = tmp_path_factory.mktemp("cropiq")
    make_toy_dataset(tmp / "toy", size=64, seed=1)
    overrides = [
        f"paths.data_root={tmp / 'toy'}",
        f"paths.manifest={tmp / 'manifest.csv'}",
        f"paths.runs_dir={tmp / 'runs'}",
        f"paths.synthetic_dir={tmp / 'synthetic'}",
        f"models.classifier_ckpt={tmp / 'runs/clf/best.pt'}",
        f"models.vae_ckpt={tmp / 'runs/vae/best.pt'}",
        f"models.snn_ckpt={tmp / 'runs/snn/best.pt'}",
    ]
    return load_config(ROOT / "configs/smoke.yaml", overrides)


@pytest.fixture(scope="session")
def toy_manifest(toy_cfg):
    from cropiq.data.manifest import build_manifest

    df = build_manifest(toy_cfg)
    df.to_csv(toy_cfg.paths.manifest, index=False)
    return df


@pytest.fixture(scope="session")
def random_artifacts(toy_cfg, toy_manifest):
    """Random-weight checkpoints in the exact on-disk format of the trainers (fast API/inference tests)."""
    from cropiq.data.datasets import load_frames

    cfg = toy_cfg
    names = load_frames(cfg).class_names
    size = cfg.data.image_size
    clf = LeafClassifier(len(names), 3, "small_cnn", pretrained=False)
    save_checkpoint(cfg.models.classifier_ckpt, model_state=clf.state_dict(), class_names=names, backbone="small_cnn",
                    image_size=size, temperature=1.0, severity_names=["mild", "moderate", "severe"])
    vae = ConvVAE(size, cfg.vae.latent_dim, cfg.vae.base_channels)
    save_checkpoint(cfg.models.vae_ckpt, model_state=vae.state_dict(), image_size=size, latent_dim=cfg.vae.latent_dim,
                    base=cfg.vae.base_channels, threshold=0.05, score_mode="topk", topk_frac=0.1)
    snn, enc = build_snn(cfg)
    s = cfg.snn
    save_checkpoint(cfg.models.snn_ckpt, model_state=snn.state_dict(),
                    hyper=dict(in_channels=enc.n_channels, hidden=s.hidden, n_out=3, beta=s.beta, beta_out=s.beta_out,
                               threshold=s.threshold, slope=s.surrogate_slope, backend=s.backend, readout_frac=s.readout_frac),
                    encoder=dict(pop=s.pop_neurons, sub_steps=s.sub_steps, delta_gain=s.delta_gain),
                    ranges={k: list(v) for k, v in cfg.sensors.ranges.items()}, actions=["wait", "irrigate", "alert"],
                    window_hours=cfg.sensors.window_hours)
    return cfg
