"""Evaluation helpers: calibration (ECE, temperature scaling), per-class metrics, plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Post-hoc temperature scaling (Guo et al., 2017) fitted on VALIDATION logits only."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def classification_summary(logits: np.ndarray, labels: np.ndarray, class_names: list[str], rare: list[str], temperature: float = 1.0) -> dict:
    probs = softmax_np(logits, temperature)
    pred = probs.argmax(1)
    per_class = f1_score(labels, pred, labels=list(range(len(class_names))), average=None, zero_division=0)
    f1_by_name = {c: float(f) for c, f in zip(class_names, per_class)}
    rare_f1 = [f1_by_name[c] for c in rare if c in f1_by_name]
    return {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": float(np.mean(per_class)),
        "rare_macro_f1": float(np.mean(rare_f1)) if rare_f1 else None,
        "per_class_f1": f1_by_name,
        "ece": expected_calibration_error(probs, labels),
        "temperature": temperature,
    }


def plot_confusion(labels: np.ndarray, pred: np.ndarray, class_names: list[str], path: str | Path) -> None:
    cm = confusion_matrix(labels, pred, labels=list(range(len(class_names))))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.7 * len(class_names) + 3, 0.7 * len(class_names) + 2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
