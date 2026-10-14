import numpy as np
import pytest
import torch

from cropiq.config import load_config
from cropiq.edge.export_snn import EdgeSNN, export
from cropiq.models.snn import build_snn
from cropiq.utils import save_checkpoint
from conftest import ROOT


@pytest.mark.parametrize("backend", ["snntorch", "native"])
def test_numpy_runtime_matches_pytorch_logits(tmp_path, backend):
    cfg = load_config(ROOT / "configs/smoke.yaml", [f"snn.backend={backend}", "snn.weight_gain=2.0"])
    torch.manual_seed(0)
    model, enc = build_snn(cfg)
    s = cfg.snn
    save_checkpoint(tmp_path / "snn.pt", model_state=model.state_dict(),
                    hyper=dict(in_channels=enc.n_channels, hidden=s.hidden, n_out=3, beta=s.beta, beta_out=s.beta_out,
                               threshold=s.threshold, slope=s.surrogate_slope, backend=backend, readout_frac=s.readout_frac),
                    encoder=dict(pop=s.pop_neurons, sub_steps=s.sub_steps, delta_gain=s.delta_gain),
                    ranges={k: list(v) for k, v in cfg.sensors.ranges.items()}, actions=["wait", "irrigate", "alert"], window_hours=24)
    out = export(tmp_path / "snn.pt", tmp_path / "edge")
    edge = EdgeSNN.load(out)

    window = np.random.default_rng(0).uniform([10, 20, 40], [50, 35, 95], size=(24, 3)).astype(np.float32)
    spikes = edge.encode(window, np.random.default_rng(1))
    with torch.no_grad():
        ref = model(torch.from_numpy(spikes).unsqueeze(0)).numpy()[0]
    got = edge.forward_spikes(spikes)
    np.testing.assert_allclose(got, ref, atol=1e-4)
    assert edge.predict(window)["action"] in ("wait", "irrigate", "alert")


def test_numpy_encoder_matches_torch_probabilities(tmp_path):
    cfg = load_config(ROOT / "configs/smoke.yaml")
    model, enc = build_snn(cfg)
    s = cfg.snn
    save_checkpoint(tmp_path / "snn.pt", model_state=model.state_dict(),
                    hyper=dict(in_channels=enc.n_channels, hidden=s.hidden, n_out=3, beta=s.beta, beta_out=s.beta_out,
                               threshold=s.threshold, slope=s.surrogate_slope, backend="native", readout_frac=s.readout_frac),
                    encoder=dict(pop=s.pop_neurons, sub_steps=s.sub_steps, delta_gain=s.delta_gain),
                    ranges={k: list(v) for k, v in cfg.sensors.ranges.items()}, actions=["wait", "irrigate", "alert"], window_hours=24)
    edge = EdgeSNN.load(export(tmp_path / "snn.pt", tmp_path / "edge"))
    xn = np.random.default_rng(3).uniform(0, 1, (24, 3)).astype(np.float32)
    np.testing.assert_allclose(edge.probabilities(xn), enc.probabilities(torch.from_numpy(xn)[None])[0].numpy(), atol=1e-5)
