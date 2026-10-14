"""Export the trained SNN to a dependency-light NumPy runtime for low-cost edge boards.

    python -m cropiq.edge.export_snn --ckpt artifacts/runs/snn/best.pt --out artifacts/edge

The runtime needs only NumPy (no PyTorch/snnTorch), so it runs on a Raspberry Pi-class device. Parity with the
PyTorch model is unit-tested (tests/test_snn_edge.py): identical input spikes give identical logits.
Next steps for real microcontrollers (see docs/ROADMAP.md): int8 weight quantisation and a C port of `forward_spikes`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cropiq.utils import ensure_dir, load_checkpoint


def export(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    ck = load_checkpoint(ckpt_path, "cpu")
    sd = {k: v.cpu().numpy().astype(np.float32) for k, v in ck["model_state"].items()}
    out = ensure_dir(out_dir)
    np.savez(
        out / "snn_edge.npz",
        w1=sd["fc1.weight"], b1=sd["fc1.bias"], w2=sd["fc2.weight"], b2=sd["fc2.bias"], w3=sd["fc3.weight"], b3=sd["fc3.bias"],
    )
    meta = {
        "hyper": ck["hyper"],
        "encoder": ck["encoder"],
        "ranges": ck["ranges"],
        "actions": ck["actions"],
        "window_hours": ck["window_hours"],
        "n_parameters": int(sum(v.size for v in sd.values())),
        "size_kb_float32": round(sum(v.size for v in sd.values()) * 4 / 1024, 1),
    }
    (out / "snn_edge.json").write_text(json.dumps(meta, indent=2))
    return out


class EdgeSNN:
    """Pure-NumPy inference with the same LIF dynamics as training."""

    def __init__(self, weights: dict, meta: dict):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        h, e = meta["hyper"], meta["encoder"]
        self.beta, self.beta_out, self.thr = h["beta"], h["beta_out"], h["threshold"]
        self.readout_frac = h.get("readout_frac", 1.0)
        self.pop, self.sub_steps, self.gain = e["pop"], e["sub_steps"], e["delta_gain"]
        self.sigma = 0.7 / max(self.pop - 1, 1)
        self.ranges, self.actions = meta["ranges"], meta["actions"]
        self.window = meta["window_hours"]

    @classmethod
    def load(cls, directory: str | Path) -> "EdgeSNN":
        d = Path(directory)
        meta = json.loads((d / "snn_edge.json").read_text())
        return cls(dict(np.load(d / "snn_edge.npz")), meta)

    # ---- encoding (physical units -> spikes) ------------------------------------------------
    def normalize(self, window: np.ndarray) -> np.ndarray:
        feats = ["soil_moisture", "temperature", "humidity"]
        lo = np.array([self.ranges[f][0] for f in feats], dtype=np.float32)
        hi = np.array([self.ranges[f][1] for f in feats], dtype=np.float32)
        return np.clip((window - lo) / (hi - lo), 0, 1).astype(np.float32)

    def probabilities(self, xn: np.ndarray) -> np.ndarray:
        centers = np.linspace(0, 1, self.pop, dtype=np.float32)
        tuning = np.exp(-((xn[..., None] - centers) ** 2) / (2 * self.sigma**2)).reshape(xn.shape[0], -1)
        d = np.diff(xn, axis=0, prepend=xn[:1])
        return np.concatenate([tuning, np.clip(d * self.gain, 0, 1), np.clip(-d * self.gain, 0, 1)], axis=1)

    def encode(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = np.repeat(self.probabilities(self.normalize(window)), self.sub_steps, axis=0)
        return (rng.random(p.shape) < p).astype(np.float32)

    # ---- dynamics --------------------------------------------------------------------------
    def forward_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """spikes [S, C] -> logits [n_out] (identical maths to IrrigationSNN.forward)."""
        w = self.w
        mem1 = np.zeros(w["w1"].shape[0], np.float32)
        mem2 = np.zeros(w["w2"].shape[0], np.float32)
        out = np.zeros(w["w3"].shape[0], np.float32)
        acc = np.zeros_like(out)
        read_from = len(spikes) - max(1, int(round(len(spikes) * self.readout_frac)))
        for t, s in enumerate(spikes):
            reset1 = (mem1 > self.thr).astype(np.float32)
            mem1 = self.beta * mem1 + (w["w1"] @ s + w["b1"]) - reset1 * self.thr
            spk1 = (mem1 > self.thr).astype(np.float32)
            reset2 = (mem2 > self.thr).astype(np.float32)
            mem2 = self.beta * mem2 + (w["w2"] @ spk1 + w["b2"]) - reset2 * self.thr
            spk2 = (mem2 > self.thr).astype(np.float32)
            out = self.beta_out * out + (w["w3"] @ spk2 + w["b3"])
            if t >= read_from:
                acc += out
        return acc / (len(spikes) - read_from)

    def predict(self, window: np.ndarray, n_draws: int = 5, seed: int = 0) -> dict:
        window = np.asarray(window, dtype=np.float32)[-self.window :]
        logits = np.mean([self.forward_spikes(self.encode(window, np.random.default_rng(seed + k))) for k in range(n_draws)], axis=0)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return {"action": self.actions[int(probs.argmax())], "probabilities": dict(zip(self.actions, map(float, probs)))}


def main() -> None:
    p = argparse.ArgumentParser(description="Export the SNN for NumPy/edge inference")
    p.add_argument("--ckpt", default="artifacts/runs/snn/best.pt")
    p.add_argument("--out", default="artifacts/edge")
    args = p.parse_args()
    out = export(args.ckpt, args.out)
    meta = json.loads((out / "snn_edge.json").read_text())
    print(f"Exported to {out} | {meta['n_parameters']} parameters (~{meta['size_kb_float32']} KB float32)")


if __name__ == "__main__":
    main()
