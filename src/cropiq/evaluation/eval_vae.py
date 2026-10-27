"""Evaluate the VAE (a) as an abnormality screen and (b) as an unknown-disease detector, against baselines.

(a) Abnormality screen (always): the VAE is trained on HEALTHY leaves only, so it fires on any disease, known or
    not. Reported: healthy-vs-diseased AUROC, per-disease detection rate at the deployed threshold (calibrated on
    healthy validation leaves) and the false-alarm rate on healthy leaves.

(b) Open-set experiment (when `data.holdout_classes` is set and a classifier trained WITH that holdout exists):
    * every novelty detector (VAE, msp, energy, mahalanobis, knn) gets a threshold at `--false-alarm` (default 5%)
      on KNOWN-class validation images, then is scored on test: unknown detection rate, AUROC unknown-vs-known,
      and false alarms on known HEALTHY and known DISEASED leaves reported separately;
    * the DEPLOYED rule is evaluated exactly as the app runs it: `pipeline.advisory.classify_disease_status`
      with the classifier's temperature-scaled confidence and the VAE flag at its deployed threshold.

    python -m cropiq.evaluation.eval_vae --run-name vae_openset --classifier-run clf_openset \
        --override data.holdout_classes=[target_spot]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cropiq.config import base_parser, load_config
from cropiq.data.datasets import LeafDataset, RawImageDataset, build_transform, load_frames
from cropiq.evaluation.ood import NoveltyScorer, classifier_outputs, open_set_report, threshold_at_false_alarm
from cropiq.evaluation.metrics import softmax_np
from cropiq.models.loading import load_classifier, load_vae
from cropiq.pipeline.advisory import classify_disease_status
from cropiq.utils import get_device, get_logger, save_json

log = get_logger("eval_vae")


@torch.no_grad()
def vae_scores(model, ck, frame, class_to_idx, device, batch_size: int = 64) -> np.ndarray:
    ds = RawImageDataset(frame, class_to_idx, ck["image_size"], augment=False)
    out = []
    for i in range(0, len(ds), batch_size):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).to(device)
        out.append(model.anomaly_score(x, ck["score_mode"], ck["topk_frac"]).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def abnormality_screen(scores: np.ndarray, thr: float, test) -> dict:
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
    return res


def open_set(cfg, frames, vae, vck, test_vae_scores, classifier_run, false_alarm, device) -> dict:
    test = frames.test
    unknown = test.is_unknown.to_numpy()
    healthy = test.is_healthy.to_numpy()
    out: dict = {"holdout_classes": frames.holdout, "false_alarm_target": false_alarm, "classifier_run": classifier_run}

    clf_path = Path(cfg.paths.runs_dir) / classifier_run / "best.pt"
    if not clf_path.exists():
        out["skipped"] = f"{clf_path} not found"
        return out
    clf, cck = load_classifier(clf_path, device)
    if list(cck["class_names"]) != list(frames.class_names):
        out["skipped"] = ("classifier was not trained with the same holdout classes "
                          f"(it knows {len(cck['class_names'])} classes, the config expects {len(frames.class_names)}); "
                          "train one with the same data.holdout_classes override")
        return out
    T = float(cck["temperature"])
    tf = build_transform(cck["image_size"], False)
    c2i = frames.class_to_idx
    tr_f, tr_l = classifier_outputs(clf, LeafDataset(frames.train, c2i, tf), device)
    va_f, va_l = classifier_outputs(clf, LeafDataset(frames.val, c2i, tf), device)
    te_f, te_l = classifier_outputs(clf, LeafDataset(test, c2i, tf), device)
    train_y = np.array([c2i[l] for l in frames.train.label])

    scorer = NoveltyScorer(k=10).fit(tr_f, train_y)
    val_scores = scorer.score(va_f, va_l, T)
    test_scores = scorer.score(te_f, te_l, T)
    val_scores["vae"] = vae_scores(vae, vck, frames.val, c2i, device)
    test_scores["vae"] = test_vae_scores

    methods = {}
    for name in ["vae", "msp", "energy", "mahalanobis", "knn"]:
        thr = threshold_at_false_alarm(val_scores[name], false_alarm)
        methods[name] = open_set_report(test_scores[name], thr, unknown, healthy)
    out["detectors_at_matched_false_alarm"] = methods

    # ---- the rule the app actually runs (pipeline/advisory.py) --------------------------------------------
    probs = softmax_np(te_l, T)
    conf, pred = probs.max(1), probs.argmax(1)
    vae_flag = test_vae_scores > float(vck["threshold"])
    min_conf = cfg.advisory.get("min_confidence", 0.6)
    status = np.array([
        classify_disease_status({"disease": frames.class_names[p], "confidence": float(c), "anomaly_flag": bool(f)}, min_conf)
        for p, c, f in zip(pred, conf, vae_flag)
    ])

    def rates(flag: np.ndarray) -> dict:
        known = ~unknown
        return {
            "detection_rate_unknown": float(flag[unknown].mean()) if unknown.any() else None,
            "false_alarm_known_healthy": float(flag[known & healthy].mean()) if (known & healthy).any() else None,
            "false_alarm_known_diseased": float(flag[known & ~healthy].mean()) if (known & ~healthy).any() else None,
        }

    out["deployed_rule"] = {
        "min_confidence": min_conf,
        "vae_threshold": float(vck["threshold"]),
        "possible_unknown_disease (VAE flag AND low confidence)": rates(status == "possible_unknown_disease"),
        "needs_review (possible_unknown OR uncertain)": rates(np.isin(status, ["possible_unknown_disease", "uncertain"])),
        "unknown_leaves_by_status": {s: int((status[unknown] == s).sum()) for s in np.unique(status[unknown])},
    }
    return out


def evaluate(cfg, run_name: str = "vae", classifier_run: str | None = None, false_alarm: float = 0.05) -> dict:
    device = get_device(cfg.device)
    run_dir = Path(cfg.paths.runs_dir) / run_name
    vae, vck = load_vae(run_dir / "best.pt", device)
    frames = load_frames(cfg, with_synthetic=False)
    scores = vae_scores(vae, vck, frames.test, frames.class_to_idx, device)

    res = {"abnormality_screen": abnormality_screen(scores, float(vck["threshold"]), frames.test)}
    if frames.holdout:
        res["open_set"] = open_set(cfg, frames, vae, vck, scores, classifier_run or "clf_real", false_alarm, device)
    save_json(res, run_dir / "metrics.json")

    a = res["abnormality_screen"]
    log.info("Abnormality screen: AUROC healthy-vs-diseased %.3f | false alarms on healthy %.3f",
             a.get("auroc_healthy_vs_diseased", float("nan")), a["false_alarm_rate_on_healthy"] or float("nan"))
    o = res.get("open_set", {})
    if "skipped" in o:
        log.warning("Open-set evaluation skipped: %s", o["skipped"])
    for name, m in o.get("detectors_at_matched_false_alarm", {}).items():
        log.info("open-set %-12s AUROC %.3f | unknown detected %.3f | false alarm healthy %.3f / known-diseased %.3f",
                 name, m.get("auroc_unknown_vs_known", float("nan")), m["detection_rate_unknown"] or 0.0,
                 m["false_alarm_known_healthy"] or 0.0, m["false_alarm_known_diseased"] or 0.0)
    if "deployed_rule" in o:
        log.info("deployed rule: %s", o["deployed_rule"]["possible_unknown_disease (VAE flag AND low confidence)"])
    return res


def main() -> None:
    parser = base_parser("Evaluate the VAE abnormality screen and the unknown-disease detectors")
    parser.add_argument("--run-name", default="vae")
    parser.add_argument("--classifier-run", default=None, help="classifier trained WITH the same holdout classes")
    parser.add_argument("--false-alarm", type=float, default=0.05, help="target false-alarm rate on known-class val")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    evaluate(cfg, args.run_name, args.classifier_run, args.false_alarm)


if __name__ == "__main__":
    main()
