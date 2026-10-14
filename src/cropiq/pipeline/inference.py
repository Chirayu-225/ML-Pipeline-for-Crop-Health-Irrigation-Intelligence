"""CropAdvisor: loads the trained models and turns (leaf image, sensor window) into an advisory.

    python -m cropiq.pipeline.inference --image leaf.jpg --sensors readings.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cropiq.config import Config, base_parser, load_config
from cropiq.data.datasets import build_transform, pil_to_tensor_01
from cropiq.data.sensors import FEATURES, dataframe_to_window, load_sensor_csv, normalize
from cropiq.models.loading import load_classifier, load_snn, load_vae
from cropiq.pipeline.advisory import build_advisory
from cropiq.utils import get_device, get_logger

log = get_logger("inference")
CLASSIFIER_FALLBACKS = ("clf_synthetic", "clf_balanced", "clf_real")


class ModelUnavailable(RuntimeError):
    pass


class CropAdvisor:
    def __init__(self, cfg: Config, device: str | None = None):
        self.cfg = cfg
        self.device = get_device(device or cfg.get("device", "auto"))
        self.classifier = self.classifier_ck = None
        self.vae = self.vae_ck = None
        self.snn = self.snn_enc = self.snn_ck = None
        self.errors: dict[str, str] = {}
        self._load_all()

    # ------------------------------------------------------------------ loading
    def _resolve_classifier(self) -> Path:
        primary = Path(self.cfg.models.classifier_ckpt)
        if primary.exists():
            return primary
        for name in CLASSIFIER_FALLBACKS:
            cand = Path(self.cfg.paths.runs_dir) / name / "best.pt"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No classifier checkpoint found (looked for {primary} and fallbacks).")

    def _load_all(self) -> None:
        loaders = {
            "classifier": lambda: self._set("classifier", load_classifier(self._resolve_classifier(), self.device)),
            "vae": lambda: self._set("vae", load_vae(self.cfg.models.vae_ckpt, self.device)),
            "snn": lambda: self._set("snn", load_snn(self.cfg.models.snn_ckpt, self.device)),
        }
        for name, fn in loaders.items():
            try:
                fn()
            except Exception as exc:  # a missing model should degrade the advisory, not crash the service
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                log.warning("Could not load %s: %s", name, self.errors[name])

    def _set(self, name: str, loaded: tuple) -> None:
        if name == "classifier":
            self.classifier, self.classifier_ck = loaded
        elif name == "vae":
            self.vae, self.vae_ck = loaded
        else:
            self.snn, self.snn_enc, self.snn_ck = loaded

    def available(self) -> dict:
        return {
            "classifier": self.classifier is not None,
            "vae": self.vae is not None,
            "snn": self.snn is not None,
            "errors": self.errors,
        }

    # ------------------------------------------------------------------ image branch
    @torch.no_grad()
    def predict_image(self, image: Image.Image, return_recon: bool = False) -> dict:
        if self.classifier is None:
            raise ModelUnavailable(self.errors.get("classifier", "classifier not loaded"))
        ck = self.classifier_ck
        rgb = np.asarray(image.convert("RGB"))
        x = build_transform(ck["image_size"], False)(image=rgb)["image"].unsqueeze(0).to(self.device)
        d_logits, s_logits = self.classifier(x)
        probs = torch.softmax(d_logits / ck["temperature"], 1)[0].cpu().numpy()
        names = ck["class_names"]
        top = int(probs.argmax())
        label = names[top]
        sev_idx = int(s_logits.argmax(1))
        severity = "none" if label.endswith("healthy") else ck["severity_names"][sev_idx]
        order = np.argsort(-probs)[:5]
        out = {
            "disease": label,
            "confidence": float(probs[top]),
            "severity": severity,
            "top_k": [{"disease": names[i], "probability": float(probs[i])} for i in order],
            "all_probs": {n: float(p) for n, p in zip(names, probs)},
            "anomaly_score": None,
            "anomaly_threshold": None,
            "anomaly_flag": False,
        }
        if self.vae is not None:
            v = self.vae_ck
            xv = pil_to_tensor_01(image, v["image_size"]).unsqueeze(0).to(self.device)
            score = float(self.vae.anomaly_score(xv, v["score_mode"], v["topk_frac"]).item())
            out.update(anomaly_score=score, anomaly_threshold=float(v["threshold"]), anomaly_flag=score > v["threshold"])
            if return_recon:
                recon = self.vae(xv)[0][0].permute(1, 2, 0).cpu().numpy()
                out["reconstruction"] = (recon * 255).clip(0, 255).astype(np.uint8)
        return out

    # ------------------------------------------------------------------ irrigation branch
    @torch.no_grad()
    def predict_irrigation(self, readings, n_draws: int = 5) -> dict:
        if self.snn is None:
            raise ModelUnavailable(self.errors.get("snn", "SNN not loaded"))
        window = self.snn_ck["window_hours"]
        arr = dataframe_to_window(readings, window)
        xn = torch.from_numpy(normalize(arr, self.snn_ck["ranges"])).unsqueeze(0).to(self.device)
        logits = 0
        stats = None
        for k in range(n_draws):  # average over spike-train draws for a stable, reproducible decision
            gen = torch.Generator(device=self.device).manual_seed(int(self.cfg.snn.seed_inference) + k)
            lg, st = self.snn(self.snn_enc(xn, gen), return_stats=True)
            logits = logits + lg / n_draws
            stats = st
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        actions = self.snn_ck["actions"]
        return {
            "action": actions[int(probs.argmax())],
            "probabilities": {a: float(p) for a, p in zip(actions, probs)},
            "spikes_per_decision": {k: float(v.mean()) for k, v in stats.items() if k != "steps"},
            "sensor_summary": {
                "soil_moisture": float(arr[-1, 0]),
                "temperature": float(arr[-1, 1]),
                "humidity": float(arr[-1, 2]),
                "humidity_mean_24h": float(arr[:, 2].mean()),
            },
        }

    # ------------------------------------------------------------------ fusion
    def advise(self, image: Image.Image | None = None, readings=None) -> dict:
        if image is None and readings is None:
            raise ValueError("Provide a leaf image, sensor readings, or both.")
        img = self.predict_image(image) if image is not None else None
        irr = self.predict_irrigation(readings) if readings is not None else None
        adv = build_advisory(img, irr, (irr or {}).get("sensor_summary"), self.cfg.advisory)
        img_public = None if img is None else {k: v for k, v in img.items() if k != "all_probs"}
        return {"advisory": adv, "image_analysis": img_public, "irrigation": irr}


def load_advisor(config_path: str | Path = "configs/default.yaml", overrides: list[str] | None = None) -> CropAdvisor:
    return CropAdvisor(load_config(config_path, overrides))


def main() -> None:
    parser = base_parser("Run the combined advisory from the command line")
    parser.add_argument("--image", default=None)
    parser.add_argument("--sensors", default=None, help="CSV with hourly soil_moisture,temperature,humidity (>=24 rows)")
    args = parser.parse_args()
    advisor = load_advisor(args.config, args.override)
    image = Image.open(args.image) if args.image else None
    readings = load_sensor_csv(args.sensors)[FEATURES] if args.sensors else None
    print(json.dumps(advisor.advise(image, readings), indent=2))


if __name__ == "__main__":
    main()
