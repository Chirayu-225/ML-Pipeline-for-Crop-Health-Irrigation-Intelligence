"""Non-learned and classical baselines for the irrigation task, plus a CPU latency helper.

The simulator's labels come from a threshold policy, so the first question an examiner will ask is
"does a hand-written rule or a tiny tree already solve this?". These baselines answer it on identical windows.

* rule  `data.sensors.observed_rule_label`: the agronomic thresholds applied to OBSERVED values only.
* tree  a depth-limited decision tree on a dozen hand-crafted window features (last values, 3 h deltas,
        recent range / jumps / zeros for fault detection, extremes). Interpretable, microsecond inference.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cropiq.data.sensors import observed_rule_label

FEATURE_NAMES = [
    "sm_last", "temp_last", "rh_last",
    "sm_delta3h", "temp_delta3h", "rh_delta3h",
    "sm_range6h", "sm_max_jump", "sm_min", "sm_max",
    "temp_max", "rh_mean", "sm_zero_recent",
]


def handcrafted_features(X: np.ndarray) -> np.ndarray:
    """X: [N, window, 3] in PHYSICAL units (soil moisture, temperature, humidity) -> [N, 13]."""
    X = np.asarray(X, dtype=np.float32)
    sm, t, h = X[..., 0], X[..., 1], X[..., 2]
    feats = [
        sm[:, -1], t[:, -1], h[:, -1],
        sm[:, -1] - sm[:, -4], t[:, -1] - t[:, -4], h[:, -1] - h[:, -4],
        np.ptp(sm[:, -6:], axis=1), np.abs(np.diff(sm, axis=1)).max(1), sm.min(1), sm.max(1),
        t.max(1), h.mean(1), (sm[:, -6:] <= 0.5).any(1).astype(np.float32),
    ]
    return np.stack(feats, axis=1).astype(np.float32)


def rule_predict(X: np.ndarray, cfg) -> np.ndarray:
    return np.array([observed_rule_label(w, cfg) for w in np.asarray(X)], dtype=np.int64)


def fit_tree(X: np.ndarray, y: np.ndarray, seed: int = 0, max_depth: int = 6) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=seed)
    return tree.fit(handcrafted_features(X), y)


def latency_ms(fn: Callable, inputs: Sequence, warmup: int = 5) -> dict:
    """Wall-clock per single-window decision (batch size 1), median and p95 in milliseconds."""
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "n": int(len(t))}
