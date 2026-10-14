"""Small shared helpers: seeding, devices, class-name normalisation, checkpoints, tracking."""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Class-name normalisation (PlantVillage ships with several folder-naming conventions)
# --------------------------------------------------------------------------------------
_ALIASES = {"yellowleaf": "yellow_leaf"}


def canon_class(name: str, crop: str = "tomato") -> str:
    """Normalise a class/folder name.

    'Tomato___Early_blight' / 'Tomato_Early_blight' / 'tomato early blight' -> 'early_blight'
    With crop='all' the crop prefix is kept ('apple_apple_scab').
    """
    n = name.strip().lower().replace(" ", "_").replace("-", "_")
    n = re.sub(r"[(),]", "", n)
    n = re.sub(r"_+", "_", n).strip("_")
    for old, new in _ALIASES.items():
        n = n.replace(old, new)
    n = re.sub(r"_+", "_", n)
    if crop and crop != "all":
        n = re.sub(rf"^{re.escape(crop.lower())}_", "", n)
    return n


def match_label(name: str, labels: list[str], crop: str = "tomato") -> str:
    """Map a user-typed class name (any spelling) onto one of the dataset's canonical labels."""
    for cand in (name, canon_class(name, "all"), canon_class(name, crop)):
        if cand in labels:
            return cand
    raise ValueError(f"Class {name!r} not found. Available classes: {sorted(labels)}")


def is_healthy(label: str) -> bool:
    return label.endswith("healthy")


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(path: str | Path, **payload: Any) -> None:
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    # weights_only=False: our checkpoints hold plain python metadata (class names, config).
    # Only load checkpoints you trained yourself.
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------------------
# Experiment tracking: none | mlflow | wandb (all optional, lazily imported)
# --------------------------------------------------------------------------------------
class Tracker:
    def __init__(self, cfg, run_name: str, params: dict | None = None):
        self.backend = (cfg.get("tracking", {}) or {}).get("backend", "none")
        self.experiment = (cfg.get("tracking", {}) or {}).get("experiment", "cropiq")
        self.run_name = run_name
        self._mlflow = None
        self._wandb = None
        log = get_logger("tracker")
        try:
            if self.backend == "mlflow":
                import mlflow

                mlflow.set_experiment(self.experiment)
                mlflow.start_run(run_name=run_name)
                self._mlflow = mlflow
                if params:
                    mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            elif self.backend == "wandb":
                import wandb

                self._wandb = wandb.init(project=self.experiment, name=run_name, config=params)
        except Exception as exc:  # tracking must never break training
            log.warning("Tracking backend %s unavailable (%s); continuing without.", self.backend, exc)
            self._mlflow = self._wandb = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
        if self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step)
        if self._wandb is not None:
            self._wandb.log(clean, step=step)

    def finish(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
        if self._wandb is not None:
            self._wandb.finish()
