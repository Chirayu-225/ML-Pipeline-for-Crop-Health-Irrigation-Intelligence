import numpy as np
import pandas as pd

from cropiq.data.datasets import load_frames, class_counts, make_sampler
from cropiq.data.severity import lesion_fraction, severity_from_fraction


def test_manifest_splits_are_disjoint_and_stratified(toy_manifest):
    df = toy_manifest
    assert df.path.is_unique
    assert set(df.split) == {"train", "val", "test"}
    for label, g in df.groupby("label"):
        assert {"train", "val", "test"} <= set(g.split), f"{label} missing a split"
    assert df[df.is_healthy].severity.eq(0).all()
    assert df[~df.is_healthy].severity.between(1, 3).all()


def test_severity_proxy_tracks_ground_truth(toy_cfg, toy_manifest):
    gt = pd.read_csv(f"{toy_cfg.paths.data_root}/ground_truth.csv")
    j = toy_manifest.merge(gt, on="path")
    healthy, diseased = j[j.is_healthy], j[~j.is_healthy]
    assert healthy.lesion_frac.mean() < 0.02  # healthy leaves have ~no lesion area
    assert np.corrcoef(diseased.lesion_frac, diseased.true_lesion_frac)[0, 1] > 0.6


def test_severity_thresholds():
    assert severity_from_fraction(0.02) == 1
    assert severity_from_fraction(0.15) == 2
    assert severity_from_fraction(0.60) == 3
    assert lesion_fraction(np.full((64, 64, 3), 128, np.uint8)) == 0.0  # blank grey image: no leaf found


def test_rare_classes_reduced_in_train_only(toy_cfg, toy_manifest):
    full = toy_manifest[toy_manifest.split == "train"].label.value_counts()
    frames = load_frames(toy_cfg)
    reduced = frames.train.label.value_counts()
    for r in frames.rare:
        assert reduced[r] < full[r]
    non_rare = [c for c in frames.class_names if c not in frames.rare]
    assert all(reduced[c] == full[c] for c in non_rare)
    # val/test untouched
    assert len(frames.val) == (toy_manifest.split == "val").sum()
    assert len(frames.test) == (toy_manifest.split == "test").sum()


def test_no_train_val_test_leakage(toy_cfg, toy_manifest):
    f = load_frames(toy_cfg)
    tr, va, te = set(f.train.path), set(f.val.path), set(f.test.path)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_holdout_classes_become_unknown_at_test_time(toy_cfg, toy_manifest):
    cfg = type(toy_cfg)({**toy_cfg, "data": type(toy_cfg.data)({**toy_cfg.data, "holdout_classes": ["Tomato___Target_Spot"]})})
    f = load_frames(cfg)
    assert "target_spot" not in f.class_names
    assert "target_spot" not in set(f.train.label) | set(f.val.label)
    assert f.test[f.test.is_unknown].label.eq("target_spot").all() and f.test.is_unknown.any()


def test_balanced_sampler_equalises_classes(toy_cfg, toy_manifest):
    f = load_frames(toy_cfg)
    sampler = make_sampler(f.train, f.class_names)
    idx = np.concatenate([np.fromiter(iter(sampler), dtype=int) for _ in range(30)])  # many epochs -> low variance
    drawn = f.train.label.iloc[idx].value_counts()
    counts = class_counts(f.train, f.class_names)
    assert counts.max() / counts.min() > 3  # original data really is imbalanced
    assert drawn.max() / drawn.min() < 1.6  # sampler flattens it
