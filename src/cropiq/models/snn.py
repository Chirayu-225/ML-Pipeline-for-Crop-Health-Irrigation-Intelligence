"""Spiking neural network for irrigation decisions, plus spike encoders and an ANN baseline.

Pipeline:  hourly readings (24 x 3) -> normalise -> spike encoder (population + delta coding, `sub_steps`
spike-train steps per hour) -> Linear-LIF-Linear-LIF-Linear-(leaky integrator readout) -> action logits.

LIF dynamics (identical in snnTorch `Leaky`, in the native fallback and in the NumPy edge runtime):
    mem_t = beta * mem_{t-1} + I_t - threshold * H(mem_{t-1} - threshold)      (reset by subtraction)
    spk_t = H(mem_t - threshold)
Training uses a fast-sigmoid surrogate gradient for H.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Spike encoder
# --------------------------------------------------------------------------------------


class SpikeEncoder:
    """Population (Gaussian tuning curves) + delta (temporal change) coding with Bernoulli spikes.

    Per feature: `pop` neurons tuned to different levels (resolves the absolute value), plus an UP and a DOWN
    neuron driven by the hour-to-hour change (resolves rain/irrigation onset, flat-lined or jumping probes).
    """

    def __init__(self, pop: int = 8, sub_steps: int = 4, delta_gain: float = 10.0, n_features: int = 3):
        self.pop, self.sub_steps, self.delta_gain, self.n_features = pop, sub_steps, delta_gain, n_features
        self.sigma = 0.7 / max(pop - 1, 1)

    @property
    def n_channels(self) -> int:
        return self.n_features * self.pop + 2 * self.n_features

    def probabilities(self, xn: torch.Tensor) -> torch.Tensor:
        """xn: [B, T, F] in [0,1] -> spike probabilities [B, T, C]."""
        centers = torch.linspace(0, 1, self.pop, device=xn.device)
        tuning = torch.exp(-((xn.unsqueeze(-1) - centers) ** 2) / (2 * self.sigma**2)).flatten(2)
        d = torch.diff(xn, dim=1, prepend=xn[:, :1])
        up = (d * self.delta_gain).clamp(0, 1)
        down = (-d * self.delta_gain).clamp(0, 1)
        return torch.cat([tuning, up, down], dim=-1)

    def __call__(self, xn: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """-> binary spike trains [B, T * sub_steps, C]."""
        p = self.probabilities(xn).repeat_interleave(self.sub_steps, dim=1)
        u = torch.rand(p.shape, device=p.device, generator=generator)
        return (u < p).float()


# --------------------------------------------------------------------------------------
# LIF neurons
# --------------------------------------------------------------------------------------
class _FastSigmoidSpike(torch.autograd.Function):
    slope = 25.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (_FastSigmoidSpike.slope * x.abs() + 1.0) ** 2


class NativeLIF(nn.Module):
    """Dependency-free LIF layer with the same dynamics as snnTorch `Leaky(reset_mechanism='subtract')`."""

    def __init__(self, beta: float, threshold: float, slope: float = 25.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold
        _FastSigmoidSpike.slope = slope

    def forward(self, cur: torch.Tensor, mem: torch.Tensor):
        reset = (mem > self.threshold).float().detach()
        mem = self.beta * mem + cur - reset * self.threshold
        return _FastSigmoidSpike.apply(mem - self.threshold), mem


def make_lif(backend: str, beta: float, threshold: float, slope: float) -> nn.Module:
    if backend == "snntorch":
        try:
            import snntorch as snn
            from snntorch import surrogate

            return snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=surrogate.fast_sigmoid(slope=slope),
                init_hidden=False,
                reset_mechanism="subtract",
            )
        except ImportError:  # pragma: no cover
            pass
    return NativeLIF(beta, threshold, slope)


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------
class IrrigationSNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        n_out: int = 3,
        beta: float = 0.9,
        beta_out: float = 0.95,
        threshold: float = 1.0,
        slope: float = 25.0,
        backend: str = "snntorch",
        weight_gain: float = 1.0,
        readout_frac: float = 0.25,
    ):
        super().__init__()
        self.in_channels, self.hidden, self.n_out = in_channels, hidden, n_out
        self.readout_frac = readout_frac
        self.beta, self.beta_out, self.threshold = beta, beta_out, threshold
        self.fc1 = nn.Linear(in_channels, hidden)
        self.lif1 = make_lif(backend, beta, threshold, slope)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = make_lif(backend, beta, threshold, slope)
        self.fc3 = nn.Linear(hidden, n_out)
        with torch.no_grad():  # sparse binary inputs give tiny currents; scale init so neurons can reach threshold
            for fc in (self.fc1, self.fc2):
                fc.weight.mul_(weight_gain)

    def forward(self, spikes: torch.Tensor, return_stats: bool = False):
        """spikes: [B, S, C] binary -> logits [B, n_out].

        The readout is the membrane of a non-spiking leaky integrator, averaged over the LAST `readout_frac`
        of the spike train: the decision is about the present state, so early steps should not dilute it.
        """
        b, steps, _ = spikes.shape
        dev = spikes.device
        mem1 = torch.zeros(b, self.hidden, device=dev)
        mem2 = torch.zeros(b, self.hidden, device=dev)
        out = torch.zeros(b, self.n_out, device=dev)
        acc = torch.zeros(b, self.n_out, device=dev)
        n1 = torch.zeros(b, device=dev)
        n2 = torch.zeros(b, device=dev)
        read_from = steps - max(1, int(round(steps * self.readout_frac)))
        for t in range(steps):
            spk1, mem1 = self.lif1(self.fc1(spikes[:, t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            out = self.beta_out * out + self.fc3(spk2)
            if t >= read_from:
                acc = acc + out
            if return_stats:
                n1 = n1 + spk1.detach().sum(1)
                n2 = n2 + spk2.detach().sum(1)
        logits = acc / (steps - read_from)
        if return_stats:
            stats = {"input_spikes": spikes.detach().sum((1, 2)), "l1_spikes": n1, "l2_spikes": n2, "steps": steps}
            return logits, stats
        return logits


class GRUBaseline(nn.Module):
    """Small GRU on the raw normalised windows: the conventional ANN to compare the SNN against."""

    def __init__(self, n_features: int = 3, hidden: int = 32, n_out: int = 3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_out)
        self.hidden, self.n_features = hidden, n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


# --------------------------------------------------------------------------------------
# Cost model (a *proxy*, not a hardware measurement)
# --------------------------------------------------------------------------------------
E_MAC_PJ = 4.6  # 32-bit float multiply-accumulate, 45 nm (Horowitz, ISSCC 2014)
E_AC_PJ = 0.9  # 32-bit float accumulate only (spike-driven synaptic event)


def snn_cost(mean_stats: dict, in_channels: int, hidden: int, n_out: int) -> dict:
    """Synaptic accumulates (spike-driven) + neuron-state MACs per decision, and an energy proxy in nJ."""
    steps = mean_stats["steps"]
    sops = (
        mean_stats["input_spikes"] * hidden
        + mean_stats["l1_spikes"] * hidden
        + mean_stats["l2_spikes"] * n_out
    )
    neuron_macs = steps * (2 * hidden + n_out)
    return {
        "synaptic_accumulates": float(sops),
        "neuron_update_macs": float(neuron_macs),
        "energy_nj": float((sops * E_AC_PJ + neuron_macs * E_MAC_PJ) / 1000.0),
    }


def gru_cost(window: int, n_features: int, hidden: int, n_out: int) -> dict:
    macs = window * 3 * (n_features * hidden + hidden * hidden) + hidden * n_out
    return {"macs": float(macs), "energy_nj": float(macs * E_MAC_PJ / 1000.0)}


def build_snn(cfg, backend: str | None = None) -> tuple[IrrigationSNN, SpikeEncoder]:
    s = cfg.snn
    enc = SpikeEncoder(s.pop_neurons, s.sub_steps, s.delta_gain)
    model = IrrigationSNN(
        enc.n_channels,
        s.hidden,
        len(cfg.advisory.actions),
        s.beta,
        s.beta_out,
        s.threshold,
        s.surrogate_slope,
        backend or s.backend,
        s.get("weight_gain", 1.0),
        s.get("readout_frac", 0.25),
    )
    return model, enc
