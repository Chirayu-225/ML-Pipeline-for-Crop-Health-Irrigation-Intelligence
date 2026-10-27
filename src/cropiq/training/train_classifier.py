"""Train the disease + severity classifier.

Three runs make up the augmentation ablation (all evaluated on the SAME real, untouched test split):

    python -m cropiq.training.train_classifier --run-name clf_real
    python -m cropiq.training.train_classifier --run-name clf_balanced  --balanced-sampler
    python -m cropiq.training.train_classifier --run-name clf_synthetic --use-synthetic --balanced-sampler
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from cropiq.config import base_parser, load_config
from cropiq.data.datasets import IGNORE, LeafDataset, build_transform, class_counts, load_frames, make_sampler, rare_signature
from cropiq.evaluation.metrics import classification_summary, fit_temperature, plot_confusion, softmax_np
from cropiq.models.classifier import LeafClassifier
from cropiq.utils import (
    Tracker,
    count_params,
    ensure_dir,
    get_device,
    get_logger,
    save_checkpoint,
    save_json,
    seed_everything,
)

log = get_logger("train_classifier")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, sev_logits, labels, sev = [], [], [], []
    for x, y, s in loader:
        d, sv = model(x.to(device))
        logits.append(d.float().cpu())
        sev_logits.append(sv.float().cpu())
        labels.append(y)
        sev.append(s)
    return (
        torch.cat(logits).numpy(),
        torch.cat(sev_logits).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(sev).numpy(),
    )


def main() -> None:
    parser = base_parser("Train the leaf disease/severity classifier")
    parser.add_argument("--run-name", default="clf_real")
    parser.add_argument("--use-synthetic", action="store_true", help="append diffusion images to TRAIN")
    parser.add_argument("--balanced-sampler", action="store_true", help="inverse-frequency class sampling")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    cc = cfg.classifier
    run_dir = ensure_dir(Path(cfg.paths.runs_dir) / args.run_name)

    frames = load_frames(cfg, with_synthetic=args.use_synthetic)
    names, c2i = frames.class_names, frames.class_to_idx
    signature = rare_signature(frames)
    if args.use_synthetic:
        report_path = Path(cfg.paths.synthetic_dir) / "generation_report.json"
        syn_sig = json.loads(report_path.read_text()).get("_meta", {}).get("rare_signature") if report_path.exists() else None
        if syn_sig not in (None, signature):
            raise SystemExit(f"Synthetic images in {cfg.paths.synthetic_dir} were made from a different rare subset "
                             f"({syn_sig} vs {signature}); regenerate them for this keep-fraction.")
    counts = dict(zip(names, class_counts(frames.train, names).tolist()))
    log.info("Train class counts: %s", counts)

    size = cfg.data.image_size
    train_ds = LeafDataset(frames.train, c2i, build_transform(size, True, cc.downscale_aug_p))
    val_ds = LeafDataset(frames.val, c2i, build_transform(size, False))
    known_test = frames.test[~frames.test.is_unknown]
    test_ds = LeafDataset(known_test, c2i, build_transform(size, False))
    kw = dict(num_workers=cfg.data.num_workers, pin_memory=device.type == "cuda")
    bs = cfg.data.batch_size
    sampler = make_sampler(frames.train, names) if args.balanced_sampler else None
    train_loader = DataLoader(train_ds, bs, sampler=sampler, shuffle=sampler is None, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, bs * 2, **kw)
    test_loader = DataLoader(test_ds, bs * 2, **kw)

    model = LeafClassifier(len(names), 3, cc.backbone, cc.pretrained).to(device)
    log.info("Model params: %.2fM | device=%s", count_params(model) / 1e6, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cc.lr, weight_decay=cc.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cc.lr, epochs=cc.epochs, steps_per_epoch=max(1, len(train_loader)))
    use_amp = bool(cc.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    tracker = Tracker(cfg, args.run_name, {"synthetic": args.use_synthetic, "balanced": args.balanced_sampler, **counts})

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, cc.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y, s in train_loader:
            x, y, s = x.to(device, non_blocking=True), y.to(device), s.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                d_logits, s_logits = model(x)
                loss = F.cross_entropy(d_logits, y, label_smoothing=cc.label_smoothing)
                if (s != IGNORE).any():
                    loss = loss + cc.severity_loss_weight * F.cross_entropy(s_logits, s, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        logits, _, labels, _ = predict(model, val_loader, device)
        val_f1 = f1_score(labels, logits.argmax(1), average="macro", labels=list(range(len(names))), zero_division=0)
        log.info("epoch %02d/%d loss %.4f val macro-F1 %.4f (%.0fs)", epoch, cc.epochs, total / max(1, len(train_loader)), val_f1, time.time() - t0)
        tracker.log({"train_loss": total / max(1, len(train_loader)), "val_macro_f1": val_f1}, step=epoch)
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cc.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    val_logits, _, val_labels, _ = predict(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    t_logits, t_sev_logits, t_labels, t_sev = predict(model, test_loader, device)
    summary = classification_summary(t_logits, t_labels, names, frames.rare, temperature)
    raw_ece = classification_summary(t_logits, t_labels, names, frames.rare, 1.0)["ece"]
    mask = t_sev != IGNORE
    sev_pred = t_sev_logits.argmax(1)
    summary.update(
        {
            "ece_uncalibrated": raw_ece,
            "severity_accuracy": float((sev_pred[mask] == t_sev[mask]).mean()) if mask.any() else None,
            "severity_within_one": float((np.abs(sev_pred[mask] - t_sev[mask]) <= 1).mean()) if mask.any() else None,
        }
    )
    metrics = {
        "run": args.run_name,
        "seed": int(cfg.seed),
        "rare_keep_fraction": float(cfg.data.rare_keep_fraction),
        "rare_signature": signature,
        "synthetic_dir": str(cfg.paths.synthetic_dir) if args.use_synthetic else None,
        "use_synthetic": args.use_synthetic,
        "balanced_sampler": args.balanced_sampler,
        "train_class_counts": counts,
        "rare_classes": frames.rare,
        "best_val_macro_f1": best_f1,
        "test": summary,
    }
    save_json(metrics, run_dir / "metrics.json")
    plot_confusion(t_labels, softmax_np(t_logits, temperature).argmax(1), names, run_dir / "confusion_matrix.png")
    save_checkpoint(
        run_dir / "best.pt",
        model_state=model.state_dict(),
        class_names=names,
        backbone=cc.backbone,
        image_size=size,
        temperature=temperature,
        severity_names=["mild", "moderate", "severe"],
        rare_signature=signature,
    )
    tracker.log({"test_macro_f1": summary["macro_f1"], "test_ece": summary["ece"]})
    tracker.finish()
    log.info("TEST accuracy %.4f | macro-F1 %.4f | rare macro-F1 %s | ECE %.4f -> %.4f (T=%.2f)",
             summary["accuracy"], summary["macro_f1"], summary["rare_macro_f1"], raw_ece, summary["ece"], temperature)


if __name__ == "__main__":
    main()
