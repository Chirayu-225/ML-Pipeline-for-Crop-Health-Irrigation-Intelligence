"""Tests for the ablation-seeding, open-set and sensor-baseline fixes."""
import json

import numpy as np
import pandas as pd

from cropiq.data.datasets import load_frames, rare_signature, rare_subset
from cropiq.evaluation.ablation import build_seed_table
from cropiq.evaluation.ood import NoveltyScorer, open_set_report, threshold_at_false_alarm
from cropiq.evaluation.sensor_baselines import FEATURE_NAMES, fit_tree, handcrafted_features, rule_predict


# ---------------------------------------------------------------- rare subsets (ablation fairness)
def _grp(n=200):
    return pd.DataFrame({"path": [f"leaf_mold/img_{i:04d}.jpg" for i in range(n)], "label": "leaf_mold"})


def test_rare_subset_is_nested_across_keep_fractions():
    g = _grp()
    small, mid, big = (set(rare_subset(g, "leaf_mold", f, 0).path) for f in (0.02, 0.05, 0.15))
    assert small < mid < big
    assert len(mid) == 10


def test_rare_subset_ignores_row_order_and_depends_only_on_rare_seed():
    g = _grp()
    a = set(rare_subset(g, "leaf_mold", 0.05, 0).path)
    b = set(rare_subset(g.sample(frac=1, random_state=3), "leaf_mold", 0.05, 0).path)
    c = set(rare_subset(g, "leaf_mold", 0.05, 1).path)
    assert a == b and a != c


def test_training_seed_does_not_change_rare_subset(toy_cfg, toy_manifest):
    other = type(toy_cfg)({**toy_cfg, "seed": toy_cfg.seed + 7})
    f1, f2 = load_frames(toy_cfg), load_frames(other)
    assert rare_signature(f1) == rare_signature(f2)
    assert set(f1.train.path) == set(f2.train.path)


def test_seed_table_reports_mean_std_and_paired_delta(tmp_path):
    for arm, base in (("clf_balanced", 0.50), ("clf_synthetic", 0.55)):
        for s in (0, 1, 2):
            d = tmp_path / f"{arm}_s{s}"
            d.mkdir()
            f1 = base + 0.01 * s
            (d / "metrics.json").write_text(json.dumps({
                "rare_classes": ["leaf_mold"], "rare_keep_fraction": 0.05, "rare_signature": "abc",
                "test": {"accuracy": 0.9, "macro_f1": f1, "rare_macro_f1": f1, "ece": 0.02, "per_class_f1": {"leaf_mold": f1}},
            }))
    table = build_seed_table(tmp_path, ["clf_balanced", "clf_synthetic"], [0, 1, 2])
    assert "0.5100 ± 0.0100" in table           # mean ± std of clf_balanced
    assert "0.0500 ± 0.0000" in table           # paired delta synthetic - balanced
    assert "WARNING" not in table


# ---------------------------------------------------------------- open-set detectors
def test_feature_detectors_flag_far_away_points():
    rng = np.random.default_rng(0)
    # Like post-ReLU CNN features: each class points in its own direction, away from the origin
    # (cosine kNN only sees direction, so a class centred on the origin would be meaningless for it).
    centers = np.array([[8, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0]], float)
    y = np.repeat([0, 1, 2], 100)
    train = centers[y] + rng.normal(0, 1, (300, 4))
    known = centers[np.repeat([0, 1, 2], 30)] + rng.normal(0, 1, (90, 4))
    unknown = np.array([0, 0, 0, 8]) + rng.normal(0, 1, (30, 4))
    feats = np.concatenate([known, unknown]).astype(np.float32)
    logits = -np.linalg.norm(feats[:, None, :] - centers[None], axis=2)
    scorer = NoveltyScorer(k=5).fit(train.astype(np.float32), y)
    scores = scorer.score(feats, logits)
    is_unknown = np.r_[np.zeros(90, bool), np.ones(30, bool)]
    for name in ("mahalanobis", "knn", "energy"):
        rep = open_set_report(scores[name], threshold_at_false_alarm(scores[name][:90], 0.05), is_unknown, np.zeros(120, bool))
        assert rep["auroc_unknown_vs_known"] > 0.9, name


def test_open_set_report_separates_false_alarm_types():
    scores = np.array([0.9, 0.1, 0.9, 0.9, 0.1])
    unknown = np.array([True, False, False, False, False])
    healthy = np.array([False, True, True, False, False])
    rep = open_set_report(scores, 0.5, unknown, healthy)
    assert rep["detection_rate_unknown"] == 1.0
    assert rep["false_alarm_known_healthy"] == 0.5
    assert rep["false_alarm_known_diseased"] == 0.5
    assert rep["false_alarm_known_all"] == 0.5


# ---------------------------------------------------------------- sensor baselines
def test_sensor_baselines_shapes_and_fit(toy_cfg):
    from cropiq.data.sensors import make_windows, simulate_field

    X, y = make_windows(simulate_field(5, toy_cfg, days=20), toy_cfg.sensors.window_hours, 3)
    feats = handcrafted_features(X)
    assert feats.shape == (len(X), len(FEATURE_NAMES)) and np.isfinite(feats).all()
    pred = rule_predict(X, toy_cfg)
    assert pred.shape == y.shape and set(np.unique(pred)) <= {0, 1, 2}
    tree = fit_tree(X, y, seed=0, max_depth=4)
    assert tree.get_depth() <= 4 and (tree.predict(feats) == y).mean() > 0.5


def test_snn_seed_aggregation_table():
    from cropiq.evaluation.snn_seeds import aggregate

    row = lambda f1: {"macro_f1": f1, "alert_recall": 0.7, "macro_f1_noise_x3": 0.6, "latency_median_ms": 1.0}  # noqa: E731
    table = aggregate({0: {"gru": row(0.84), "tree": row(0.90)}, 1: {"gru": row(0.86), "tree": row(0.90)}})
    assert "| gru | 0.8500 ± 0.0141" in table
    assert "| tree | 0.9000 ± 0.0000" in table
