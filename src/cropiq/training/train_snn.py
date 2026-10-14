"""Train the irrigation SNN (and a GRU baseline for an honest comparison).

    python -m cropiq.training.train_snn

Reports macro-F1 / per-class recall on held-out FIELDS, robustness to extra sensor noise, spike sparsity and
an energy *proxy* versus the GRU. The labels come from an agronomic rule policy (see data/sensors.py), so
read the numbers as "policy imitation under noisy, faulty sensors", not as field-validated agronomy.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from cropiq.config import base_parser, load_config
from cropiq.data.sensors import ACTIONS, FEATURES, build_sensor_splits, normalize
from cropiq.evaluation.metrics import plot_confusion
from cropiq.models.snn import GRUBaseline, build_snn, gru_cost, snn_cost
from cropiq.utils import Tracker, count_params, ensure_dir, get_device, get_logger, save_checkpoint, save_json, seed_everything

log = get_logger("train_snn")


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def eval_snn(model, enc, xn: torch.Tensor, y: np.ndarray, device, seed: int, with_stats: bool = False):
    model.eval()
    gen = make_generator(device, seed)
    preds, stats_acc = [], {"input_spikes": [], "l1_spikes": [], "l2_spikes": []}
    steps = 0
    for i in range(0, len(xn), 256):
        spikes = enc(xn[i : i + 256].to(device), gen)
        if with_stats:
            logits, st = model(spikes, return_stats=True)
            steps = st["steps"]
            for k in stats_acc:
                stats_acc[k].append(st[k].cpu())
        else:
            logits = model(spikes)
        preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    out = {"pred": pred, "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))}
    if with_stats:
        out["mean_stats"] = {k: float(torch.cat(v).mean()) for k, v in stats_acc.items()} | {"steps": steps}
    return out


@torch.no_grad()
def eval_gru(model, xn, y, device):
    model.eval()
    pred = torch.cat([model(xn[i : i + 512].to(device)).argmax(1).cpu() for i in range(0, len(xn), 512)]).numpy()
    return {"pred": pred, "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))}


def class_weights(y: np.ndarray, n_classes: int = 3, power: float = 0.5) -> torch.Tensor:
    """(Softened) inverse-frequency weights; power=1 is full inverse frequency (favours recall on rare 'alert')."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main() -> None:
    parser = base_parser("Train the irrigation SNN + GRU baseline")
    parser.add_argument("--run-name", default="snn")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    s, sc = cfg.snn, cfg.sensors
    run_dir = ensure_dir(Path(cfg.paths.runs_dir) / args.run_name)

    splits = build_sensor_splits(cfg)
    log.info("Windows - train %d | val %d | test %d | class balance (train): %s", len(splits["train"][1]), len(splits["val"][1]), len(splits["test"][1]),
             dict(zip(ACTIONS, np.bincount(splits["train"][1], minlength=3).tolist())))
    ranges = {k: list(v) for k, v in sc.ranges.items()}
    xn = {k: torch.from_numpy(normalize(v[0], ranges)) for k, v in splits.items()}
    y = {k: v[1] for k, v in splits.items()}

    model, enc = build_snn(cfg)
    model.to(device)
    log.info("SNN backend=%s | params %.1fk | channels %d | steps/decision %d", s.backend, count_params(model) / 1e3, enc.n_channels, sc.window_hours * s.sub_steps)
    weights = class_weights(y["train"], power=s.get("class_weight_power", 0.5)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=s.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s.epochs)
    tracker = Tracker(cfg, args.run_name, {"backend": s.backend})
    gen = make_generator(device, cfg.seed)

    best_f1, best_state = -1.0, None
    n = len(xn["train"])
    ytr = torch.from_numpy(y["train"])
    for epoch in range(1, s.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        t0, total = time.time(), 0.0
        for i in range(0, n - s.batch_size + 1, s.batch_size):
            idx = perm[i : i + s.batch_size]
            spikes = enc(xn["train"][idx].to(device), gen)
            logits = model(spikes)
            loss = F.cross_entropy(logits, ytr[idx].to(device), weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item()
        sched.step()
        val = eval_snn(model, enc, xn["val"], y["val"], device, seed=1)
        log.info("epoch %02d/%d loss %.4f | val macro-F1 %.4f (%.0fs)", epoch, s.epochs, total / max(1, n // s.batch_size), val["macro_f1"], time.time() - t0)
        tracker.log({"loss": total / max(1, n // s.batch_size), "val_macro_f1": val["macro_f1"]}, step=epoch)
        if val["macro_f1"] > best_f1:
            best_f1, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    # ---------------------------- test: clean and noisier sensors -------------------------------
    test = eval_snn(model, enc, xn["test"], y["test"], device, seed=2, with_stats=True)
    report = classification_report(y["test"], test["pred"], labels=[0, 1, 2], target_names=ACTIONS, output_dict=True, zero_division=0)
    plot_confusion(y["test"], test["pred"], ACTIONS, run_dir / "confusion_matrix.png")
    rng = np.random.default_rng(0)
    robustness = {}
    for mult in (1.0, 2.0, 3.0):
        sigma = np.array([sc.noise_std[f] for f in FEATURES], dtype=np.float32) * mult
        noisy = splits["test"][0] + rng.normal(0, 1, splits["test"][0].shape).astype(np.float32) * sigma
        r = eval_snn(model, enc, torch.from_numpy(normalize(noisy, ranges)), y["test"], device, seed=3)
        robustness[f"extra_noise_x{mult:g}"] = r["macro_f1"]

    # ---------------------------- GRU baseline on identical data ------------------------------
    mu, sd = xn["train"].mean((0, 1)), xn["train"].std((0, 1)).clamp(min=1e-3)  # z-score: a fair, well-conditioned baseline
    zn = {k: (v - mu) / sd for k, v in xn.items()}
    gru = GRUBaseline(len(FEATURES), 32, 3).to(device)
    gopt = torch.optim.Adam(gru.parameters(), lr=3e-3)
    g_best, g_state = -1.0, None
    for epoch in range(1, s.epochs + 1):
        gru.train()
        perm = torch.randperm(n)
        for i in range(0, n - s.batch_size + 1, s.batch_size):
            idx = perm[i : i + s.batch_size]
            loss = F.cross_entropy(gru(zn["train"][idx].to(device)), ytr[idx].to(device), weight=weights)
            gopt.zero_grad(set_to_none=True)
            loss.backward()
            gopt.step()
        v = eval_gru(gru, zn["val"], y["val"], device)["macro_f1"]
        if v > g_best:
            g_best, g_state = v, copy.deepcopy(gru.state_dict())
    gru.load_state_dict(g_state)
    gtest = eval_gru(gru, zn["test"], y["test"], device)
    g_report = classification_report(y["test"], gtest["pred"], labels=[0, 1, 2], target_names=ACTIONS, output_dict=True, zero_division=0)

    metrics = {
        "snn": {
            "test_macro_f1": test["macro_f1"],
            "per_class": {a: {k: report[a][k] for k in ("precision", "recall", "f1-score", "support")} for a in ACTIONS},
            "robustness_macro_f1": robustness,
            "mean_spikes_per_decision": test["mean_stats"],
            "cost_proxy": snn_cost(test["mean_stats"], enc.n_channels, s.hidden, 3),
            "params": count_params(model),
        },
        "gru_baseline": {
            "test_macro_f1": gtest["macro_f1"],
            "per_class": {a: {k: g_report[a][k] for k in ("precision", "recall", "f1-score", "support")} for a in ACTIONS},
            "cost_proxy": gru_cost(sc.window_hours, len(FEATURES), 32, 3),
            "params": count_params(gru),
        },
        "note": "Labels come from a rule policy on hidden simulator state; cost numbers are a 45nm op-count proxy, not hardware measurements.",
    }
    save_json(metrics, run_dir / "metrics.json")
    save_checkpoint(
        run_dir / "best.pt",
        model_state=model.state_dict(),
        hyper=dict(in_channels=enc.n_channels, hidden=s.hidden, n_out=3, beta=s.beta, beta_out=s.beta_out,
                   threshold=s.threshold, slope=s.surrogate_slope, backend=s.backend,
                   readout_frac=s.get("readout_frac", 0.25)),
        encoder=dict(pop=s.pop_neurons, sub_steps=s.sub_steps, delta_gain=s.delta_gain),
        ranges=ranges,
        actions=ACTIONS,
        window_hours=sc.window_hours,
    )
    tracker.finish()
    log.info("TEST macro-F1 SNN %.4f | GRU %.4f | alert recall SNN %.3f | noise robustness %s", test["macro_f1"], gtest["macro_f1"], report["alert"]["recall"], robustness)
    log.info("Energy proxy per decision: SNN %.1f nJ vs GRU %.1f nJ", metrics["snn"]["cost_proxy"]["energy_nj"], metrics["gru_baseline"]["cost_proxy"]["energy_nj"])


if __name__ == "__main__":
    main()
