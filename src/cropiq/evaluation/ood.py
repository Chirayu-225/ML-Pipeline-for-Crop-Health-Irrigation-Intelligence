"""Feature-space novelty ("unknown disease") detectors built on the trained classifier, used as baselines for the VAE.

All scores follow one convention: HIGHER = MORE LIKELY UNKNOWN.

* msp          1 - max softmax probability (Hendrycks & Gimpel, 2017)
* energy       -T * logsumexp(logits / T)   (Liu et al., 2020)
* mahalanobis  min over known classes of the Mahalanobis distance of the penultimate features, class means +
               a shared Ledoit-Wolf covariance fitted on TRAIN features (Lee et al., 2018)
* knn          1 - cosine similarity to the k-th nearest TRAIN feature (Sun et al., 2022)

Thresholds are set on known-class VALIDATION images (never on the unknown class), at a chosen false-alarm rate.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def classifier_outputs(model, dataset, device, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """-> (penultimate features [N, D], disease logits [N, C]); model must be in eval mode (dropout off)."""
    model.eval()
    feats, logits = [], []
    for i in range(0, len(dataset), batch_size):
        x = torch.stack([dataset[j][0] for j in range(i, min(i + batch_size, len(dataset)))]).to(device)
        f = model.features(x).float()
        feats.append(f.cpu())
        logits.append(model.disease_head(f).float().cpu())
    if not feats:
        return np.zeros((0, model.feature_dim), np.float32), np.zeros((0, model.disease_head.out_features), np.float32)
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


class NoveltyScorer:
    def __init__(self, k: int = 10):
        self.k = k
        self.means = self.precision = self.bank = None

    def fit(self, train_feats: np.ndarray, train_labels: np.ndarray) -> "NoveltyScorer":
        train_labels = np.asarray(train_labels)
        classes = np.unique(train_labels[train_labels >= 0])
        self.means = np.stack([train_feats[train_labels == c].mean(0) for c in classes]).astype(np.float64)
        centred = np.concatenate([train_feats[train_labels == c] - m for c, m in zip(classes, self.means)])
        self.precision = LedoitWolf().fit(centred).precision_
        self.bank = F.normalize(torch.as_tensor(train_feats, dtype=torch.float32), dim=1)
        return self

    def mahalanobis(self, feats: np.ndarray) -> np.ndarray:
        f = feats.astype(np.float64)
        fp = f @ self.precision
        d = (fp * f).sum(1, keepdims=True) - 2 * fp @ self.means.T + ((self.means @ self.precision) * self.means).sum(1)[None]
        return d.min(1)

    def knn(self, feats: np.ndarray, chunk: int = 1024) -> np.ndarray:
        q = F.normalize(torch.as_tensor(feats, dtype=torch.float32), dim=1)
        k = min(self.k, len(self.bank))
        out = [1.0 - (q[i : i + chunk] @ self.bank.T).topk(k, dim=1).values[:, -1] for i in range(0, len(q), chunk)]
        return torch.cat(out).numpy() if out else np.zeros(0)

    def score(self, feats: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
        z = torch.as_tensor(logits, dtype=torch.float32) / temperature
        return {
            "msp": (1.0 - torch.softmax(z, 1).max(1).values).numpy(),
            "energy": (-temperature * torch.logsumexp(z, 1)).numpy(),
            "mahalanobis": self.mahalanobis(feats),
            "knn": self.knn(feats),
        }


def threshold_at_false_alarm(known_val_scores: np.ndarray, false_alarm: float = 0.05) -> float:
    return float(np.percentile(known_val_scores, 100 * (1 - false_alarm))) if len(known_val_scores) else float("inf")


def open_set_report(scores: np.ndarray, threshold: float, unknown: np.ndarray, healthy: np.ndarray) -> dict:
    """Detection of unknown diseases vs. false alarms on KNOWN classes, split into healthy / known-diseased."""
    flag = scores > threshold
    known = ~unknown
    known_dis = known & ~healthy
    known_hea = known & healthy
    out = {
        "threshold": float(threshold),
        "detection_rate_unknown": float(flag[unknown].mean()) if unknown.any() else None,
        "false_alarm_known_all": float(flag[known].mean()) if known.any() else None,
        "false_alarm_known_healthy": float(flag[known_hea].mean()) if known_hea.any() else None,
        "false_alarm_known_diseased": float(flag[known_dis].mean()) if known_dis.any() else None,
    }
    if unknown.any() and known.any():
        out["auroc_unknown_vs_known"] = float(roc_auc_score(unknown, scores))
    return out
