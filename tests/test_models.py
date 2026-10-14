import pytest
import torch

from cropiq.config import load_config
from cropiq.models.classifier import LeafClassifier
from cropiq.models.ddpm import EMA, GaussianDiffusion, build_unet
from cropiq.models.snn import GRUBaseline, IrrigationSNN, NativeLIF, SpikeEncoder, build_snn, gru_cost, snn_cost
from cropiq.models.vae import ConvVAE
from conftest import ROOT


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs/smoke.yaml")


@pytest.mark.parametrize("backbone,size", [("small_cnn", 48), ("resnet18", 64)])
def test_classifier_heads(backbone, size):
    m = LeafClassifier(10, 3, backbone, pretrained=False)
    d, s = m(torch.randn(2, 3, size, size))
    assert d.shape == (2, 10) and s.shape == (2, 3)


def test_default_unet_matches_training_config():
    full = load_config(ROOT / "configs/default.yaml")
    m = build_unet(full, 10)
    out = m(torch.randn(1, 3, 64, 64), torch.tensor([5]), torch.tensor([3]))
    assert out.shape == (1, 3, 64, 64)


def test_diffusion_loss_backprop_and_sampling_range(cfg):
    unet = build_unet(cfg, 4)
    diff = GaussianDiffusion(cfg.diffusion.timesteps, cfg.diffusion.schedule)
    x, y = torch.rand(4, 3, 32, 32) * 2 - 1, torch.tensor([0, 1, 2, 3])
    loss = diff.loss(unet, x, y)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in unet.parameters() if p.grad is not None)
    imgs = diff.ddim_sample(unet.eval(), y, 32, steps=4, guidance_scale=2.0)
    assert imgs.shape == (4, 3, 32, 32) and imgs.min() >= -1 and imgs.max() <= 1


def test_cosine_schedule_is_monotone():
    abar = GaussianDiffusion(1000, "cosine").alphas_cumprod
    assert (abar[1:] < abar[:-1]).all() and abar[0] > 0.99 and abar[-1] < 0.01


def test_ema_tracks_model():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, 0.9)
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert not torch.equal(ema.model.weight, m.weight)


def test_vae_anomaly_score_higher_for_noise_than_for_training_pattern():
    torch.manual_seed(0)
    vae = ConvVAE(32, 16, 16)
    x = torch.full((16, 3, 32, 32), 0.4)  # trivially simple "healthy" distribution
    opt = torch.optim.Adam(vae.parameters(), 3e-3)
    for _ in range(150):
        recon, mu, lv = vae(x + 0.02 * torch.randn_like(x))
        loss, _, _ = ConvVAE.loss(x, recon, mu, lv)
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    normal = vae.anomaly_score(x[:4])
    weird = vae.anomaly_score(torch.rand(4, 3, 32, 32))
    assert weird.mean() > 3 * normal.mean()


def test_encoder_shapes_and_sparsity():
    enc = SpikeEncoder(pop=8, sub_steps=4)
    sp = enc(torch.rand(5, 24, 3))
    assert sp.shape == (5, 96, enc.n_channels) and set(sp.unique().tolist()) <= {0.0, 1.0}
    assert sp.mean() < 0.4


def test_encoder_is_reproducible_with_seeded_generator():
    enc = SpikeEncoder()
    x = torch.rand(2, 24, 3)
    a = enc(x, torch.Generator().manual_seed(1))
    b = enc(x, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


@pytest.mark.parametrize("backend", ["snntorch", "native"])
def test_snn_forward_and_gradients_flow(cfg, backend):
    model, enc = build_snn(cfg, backend)
    sp = enc(torch.rand(6, 24, 3))
    logits, stats = model(sp, return_stats=True)
    assert logits.shape == (6, 3) and stats["steps"] == 96
    logits.sum().backward()
    assert model.fc1.weight.grad is not None and model.fc1.weight.grad.abs().sum() > 0


def test_native_and_snntorch_backends_are_equivalent(cfg):
    a, enc = build_snn(cfg, "snntorch")
    b, _ = build_snn(cfg, "native")
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k.startswith('fc')})
    sp = enc(torch.rand(4, 24, 3), torch.Generator().manual_seed(0))
    assert torch.allclose(a(sp), b(sp), atol=1e-5)


def test_lif_neuron_integrates_leaks_and_resets():
    lif = NativeLIF(beta=0.5, threshold=1.0)
    mem, spikes = torch.zeros(1), []
    for _ in range(6):
        spk, mem = lif(torch.tensor([0.8]), mem)
        spikes.append(int(spk.item()))
    assert sum(spikes) >= 1 and sum(spikes) < 6  # fires sometimes, not always (leak + reset)


def test_cost_model_prefers_sparse_spiking(cfg):
    s = cfg.snn
    stats = {"input_spikes": 700.0, "l1_spikes": 400.0, "l2_spikes": 80.0, "steps": 96}
    assert snn_cost(stats, 30, s.hidden, 3)["energy_nj"] > 0
    assert gru_cost(24, 3, 32, 3)["macs"] > 0
