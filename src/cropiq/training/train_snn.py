"""Train the irrigation SNN (and a GRU baseline for an honest comparison).

    python -m cropiq.training.train_snn

Reports, for the SNN, a GRU, the observed-values threshold rule and a small decision tree, all on the SAME
held-out-field windows: macro-F1, per-class recall (alert recall is the safety-critical headline), robustness
to extra sensor noise, and measured CPU latency per decision. The op-count energy proxy is kept, but it only
describes neuromorphic hardware: on a CPU / Raspberry Pi the measured latency is the relevant cost.
The labels come from an agronomic rule policy (see data/sensors.py), so read the numbers as
"policy imitation under noisy, faulty sensors", not as field-validated agronomy.
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
from cropiq.data.sensors import ACTIONS, FEATURES, build_sensor_splits, normalize, observed_rule_label
from cropiq.edge.export_snn import EdgeSNN
from cropiq.evaluation.sensor_baselines import fit_tree, handcrafted_features, latency_ms, rule_predict
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

    # ---------------------------- non-learned / classical baselines on identical windows --------------
    X_train_phys, X_test_phys = splits["train"][0], splits["test"][0]
    rule_pred = rule_predict(X_test_phys, cfg)
    tree = fit_tree(X_train_phys, y["train"], seed=cfg.seed)
    tree_pred = tree.predict(handcrafted_features(X_test_phys))
    # Same tree, trained on the clean windows plus noisier copies (x1, x2, x3 the simulator's sensor noise).
    # Labels are unchanged: they come from the hidden true state, so extra observation noise does not alter them.
    aug_rng = np.random.default_rng(cfg.seed + 1)
    base_sigma = np.array([sc.noise_std[f] for f in FEATURES], dtype=np.float32)
    aug_X = [X_train_phys] + [X_train_phys + aug_rng.normal(0, 1, X_train_phys.shape).astype(np.float32) * base_sigma * m
                              for m in (1.0, 2.0, 3.0)]
    tree_aug = fit_tree(np.concatenate(aug_X), np.tile(y["train"], len(aug_X)), seed=cfg.seed)
    tree_aug_pred = tree_aug.predict(handcrafted_features(X_test_phys))

    def summarise(pred: np.ndarray) -> dict:
        rep_ = classification_report(y["test"], pred, labels=[0, 1, 2], target_names=ACTIONS, output_dict=True, zero_division=0)
        return {
            "test_macro_f1": float(f1_score(y["test"], pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "per_class": {a: {k: rep_[a][k] for k in ("precision", "recall", "f1-score", "support")} for a in ACTIONS},
        }

    # ---------------------------- robustness: the SAME noisy test windows for every model -------------
    rng = np.random.default_rng(0)
    robustness = {m: {} for m in ("snn", "gru", "rule", "tree", "tree_noiseaug")}
    for mult in (1.0, 2.0, 3.0):
        sigma = np.array([sc.noise_std[f] for f in FEATURES], dtype=np.float32) * mult
        noisy = X_test_phys + rng.normal(0, 1, X_test_phys.shape).astype(np.float32) * sigma
        noisy_n = torch.from_numpy(normalize(noisy, ranges))
        key = f"extra_noise_x{mult:g}"
        f1 = lambda p: float(f1_score(y["test"], p, average="macro", labels=[0, 1, 2], zero_division=0))  # noqa: E731
        robustness["snn"][key] = eval_snn(model, enc, noisy_n, y["test"], device, seed=3)["macro_f1"]
        robustness["gru"][key] = eval_gru(gru, (noisy_n - mu) / sd, y["test"], device)["macro_f1"]
        robustness["rule"][key] = f1(rule_predict(noisy, cfg))
        robustness["tree"][key] = f1(tree.predict(handcrafted_features(noisy)))
        robustness["tree_noiseaug"][key] = f1(tree_aug.predict(handcrafted_features(noisy)))

    # ---------------------------- measured CPU latency per decision (batch size 1, one thread) ---------
    n_lat = min(200, len(X_test_phys))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    snn_cpu, gru_cpu = copy.deepcopy(model).cpu().eval(), copy.deepcopy(gru).cpu().eval()
    sd_np = {k: v.detach().cpu().numpy() for k, v in snn_cpu.state_dict().items()}
    edge = EdgeSNN(
        {"w1": sd_np["fc1.weight"], "b1": sd_np["fc1.bias"], "w2": sd_np["fc2.weight"], "b2": sd_np["fc2.bias"],
         "w3": sd_np["fc3.weight"], "b3": sd_np["fc3.bias"]},
        {"hyper": {"beta": s.beta, "beta_out": s.beta_out, "threshold": s.threshold, "readout_frac": s.get("readout_frac", 0.25)},
         "encoder": {"pop": s.pop_neurons, "sub_steps": s.sub_steps, "delta_gain": s.delta_gain},
         "ranges": ranges, "actions": ACTIONS, "window_hours": sc.window_hours},
    )
    lat_gen = torch.Generator().manual_seed(0)
    xn_cpu, zn_cpu = xn["test"][:n_lat], zn["test"][:n_lat].cpu()
    with torch.no_grad():
        latency = {
            "snn_torch_1draw": latency_ms(lambda i: snn_cpu(enc(xn_cpu[i : i + 1], lat_gen)), list(range(n_lat))),
            "snn_numpy_edge_5draws": latency_ms(lambda w: edge.predict(w, n_draws=5), list(X_test_phys[:n_lat])),
            "gru_torch": latency_ms(lambda i: gru_cpu(zn_cpu[i : i + 1]), list(range(n_lat))),
            "rule": latency_ms(lambda w: observed_rule_label(w, cfg), list(X_test_phys[:n_lat])),
            "tree": latency_ms(lambda w: tree.predict(handcrafted_features(w[None])), list(X_test_phys[:n_lat])),
            "tree_noiseaug": latency_ms(lambda w: tree_aug.predict(handcrafted_features(w[None])), list(X_test_phys[:n_lat])),
        }
    torch.set_num_threads(threads)

    snn_proxy = snn_cost(test["mean_stats"], enc.n_channels, s.hidden, 3)
    snn_proxy["encoder_random_draws"] = int(test["mean_stats"]["steps"] * enc.n_channels)
    rule_s, tree_s, tree_aug_s = summarise(rule_pred), summarise(tree_pred), summarise(tree_aug_pred)
    summary = {
        name: {
            "macro_f1": round(m1, 4),
            "alert_recall": round(rec, 4),
            "macro_f1_noise_x3": round(robustness[key]["extra_noise_x3"], 4),
            "latency_median_ms": round(latency[lat]["median_ms"], 4),
        }
        for name, key, m1, rec, lat in [
            ("snn", "snn", test["macro_f1"], report["alert"]["recall"], "snn_torch_1draw"),
            ("gru", "gru", gtest["macro_f1"], g_report["alert"]["recall"], "gru_torch"),
            ("rule", "rule", rule_s["test_macro_f1"], rule_s["per_class"]["alert"]["recall"], "rule"),
            ("tree", "tree", tree_s["test_macro_f1"], tree_s["per_class"]["alert"]["recall"], "tree"),
            ("tree_noiseaug", "tree_noiseaug", tree_aug_s["test_macro_f1"], tree_aug_s["per_class"]["alert"]["recall"], "tree_noiseaug"),
        ]
    }
    metrics = {
        "snn": {
            "test_macro_f1": test["macro_f1"],
            "per_class": {a: {k: report[a][k] for k in ("precision", "recall", "f1-score", "support")} for a in ACTIONS},
            "robustness_macro_f1": robustness["snn"],
            "mean_spikes_per_decision": test["mean_stats"],
            "cost_proxy": snn_proxy,
            "params": count_params(model),
        },
        "gru_baseline": {
            "test_macro_f1": gtest["macro_f1"],
            "per_class": {a: {k: g_report[a][k] for k in ("precision", "recall", "f1-score", "support")} for a in ACTIONS},
            "robustness_macro_f1": robustness["gru"],
            "cost_proxy": gru_cost(sc.window_hours, len(FEATURES), 32, 3),
            "params": count_params(gru),
        },
        "rule_baseline": {**summarise(rule_pred), "robustness_macro_f1": robustness["rule"]},
        "tree_baseline": {**summarise(tree_pred), "robustness_macro_f1": robustness["tree"],
                          "depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves())},
        "tree_noiseaug_baseline": {**tree_aug_s, "robustness_macro_f1": robustness["tree_noiseaug"],
                                   "depth": int(tree_aug.get_depth()), "leaves": int(tree_aug.get_n_leaves())},
        "latency_cpu_1thread": latency,
        "summary": summary,
        "note": ("Labels come from a rule policy on hidden simulator state. cost_proxy is a 45 nm op-count model that only "
                 "applies to neuromorphic hardware (spike = accumulate) and ignores the encoder's random draws; on a CPU "
                 "use latency_cpu_1thread, measured with batch size 1 on one thread."),
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
    log.info("%-13s %9s %13s %15s %12s", "model", "macro-F1", "ALERT recall", "F1 @ noise x3", "CPU ms/dec")
    for name, m in summary.items():
        log.info("%-13s %9.4f %13.3f %15.4f %12.3f", name, m["macro_f1"], m["alert_recall"], m["macro_f1_noise_x3"], m["latency_median_ms"])
    log.info("NumPy edge SNN (5 draws, as served): %.3f ms/decision", latency["snn_numpy_edge_5draws"]["median_ms"])
    log.info("Energy proxy (neuromorphic-hardware assumption only): SNN %.1f nJ vs GRU %.1f nJ",
             snn_proxy["energy_nj"], metrics["gru_baseline"]["cost_proxy"]["energy_nj"])


if __name__ == "__main__":
    main()
