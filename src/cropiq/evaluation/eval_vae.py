"""Evaluate the VAE as an unknown-disease screen.

Reports (on the held-out REAL test split):
  * AUROC of healthy-vs-diseased, overall and per disease class;
  * detection rate per disease at the calibrated threshold, and the false-alarm rate on healthy leaves;
  * if `data.holdout_classes` is set (open-set experiment) and a classifier exists: how many truly UNSEEN
    diseases are caught by (VAE alarm OR low classifier confidence) vs. the false-alarm rate on known classes.

    python -m cropiq.evaluation.eval_vae --run-name vae
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cropiq.config import base_parser, load_config
from cropiq.data.datasets import LeafDataset, RawImageDataset, build_transform, load_frames
from cropiq.models.loading import load_classifier, load_vae
from cropiq.utils import get_device, get_logger, save_json

log = get_logger("eval_vae")


def evaluate(cfg, run_name: str = "vae", classifier_run: str | None = None) -> dict:
    device = get_device(cfg.device)
    run_dir = Path(cfg.paths.runs_dir) / run_name
    model, ck = load_vae(run_dir / "best.pt", device)
    frames = load_frames(cfg, with_synthetic=False)
    test = frames.test
    ds = RawImageDataset(test, frames.class_to_idx, ck["image_size"], augment=False)
    scores = []
    for i in range(0, len(ds), 64):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(ds)))]).to(device)
        scores.append(model.anomaly_score(x, ck["score_mode"], ck["topk_frac"]).cpu())
    scores = torch.cat(scores).numpy()
    thr = ck["threshold"]
    healthy = test.is_healthy.to_numpy()
    flagged = scores > thr

    res: dict = {"threshold": thr, "false_alarm_rate_on_healthy": float(flagged[healthy].mean()) if healthy.any() else None}
    if healthy.any() and (~healthy).any():
        res["auroc_healthy_vs_diseased"] = float(roc_auc_score(~healthy, scores))
    per_class = {}
    for label in sorted(test.label.unique()):
        m = (test.label == label).to_numpy()
        if label.endswith("healthy"):
            continue
        entry = {"n": int(m.sum()), "detection_rate": float(flagged[m].mean())}
        if healthy.any():
            both = m | healthy
            entry["auroc_vs_healthy"] = float(roc_auc_score(m[both], scores[both]))
        per_class[label] = entry
    res["per_disease"] = per_class

    # ---------------- open-set: unseen diseases ------------------------------------------------
    if frames.holdout:
        unknown = test.is_unknown.to_numpy()
        res["holdout_classes"] = frames.holdout
        res["vae_detection_rate_unseen"] = float(flagged[unknown].mean())
        clf_path = Path(cfg.paths.runs_dir) / (classifier_run or "clf_real") / "best.pt"
        if clf_path.exists():
            clf, cck = load_classifier(clf_path, device)
            tds = LeafDataset(test, frames.class_to_idx, build_transform(cck["image_size"], False))
            maxp = []
            with torch.no_grad():
                for i in range(0, len(tds), 64):
                    x = torch.stack([tds[j][0] for j in range(i, min(i + 64, len(tds)))]).to(device)
                    maxp.append(torch.softmax(clf(x)[0] / cck["temperature"], 1).max(1).values.cpu())
            lowconf = torch.cat(maxp).numpy() < cfg.advisory.min_confidence
            combined = flagged | lowconf
            res["classifier_lowconf_rate_unseen"] = float(lowconf[unknown].mean())
            res["combined_detection_rate_unseen"] = float(combined[unknown].mean())
            res["combined_false_alarm_rate_known"] = float(combined[~unknown].mean())
    save_json(res, run_dir / "metrics.json")
    log.info("VAE: AUROC %.3f | false-alarm(healthy) %.3f | mean disease detection %.3f",
             res.get("auroc_healthy_vs_diseased", float("nan")), res["false_alarm_rate_on_healthy"] or float("nan"),
             float(np.mean([v["detection_rate"] for v in per_class.values()])) if per_class else float("nan"))
    return res


def main() -> None:
    parser = base_parser("Evaluate the VAE anomaly screen")
    parser.add_argument("--run-name", default="vae")
    parser.add_argument("--classifier-run", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    evaluate(cfg, args.run_name, args.classifier_run)


if __name__ == "__main__":
    main()
