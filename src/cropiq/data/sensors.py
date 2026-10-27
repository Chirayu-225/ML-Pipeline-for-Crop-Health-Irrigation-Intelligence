"""Sensor time-series: a physically-motivated field simulator, a CSV loader and window builders.

Three hourly features: soil moisture (% volumetric water content), air temperature (deg C) and relative
humidity (%). Each window of `window_hours` readings is mapped to an action:
    0 = wait, 1 = irrigate, 2 = alert (sensor fault, critical dryness or heat stress).

IMPORTANT (be honest in write-ups): labels in the *simulator* come from an agronomic rule policy evaluated on
the hidden, noise-free state. The model sees only noisy/faulty observations, so it is not a trivial copy of
the rule, but the benchmark measures policy imitation + fault robustness - not real-world agronomic skill.
Plug in real data (AgriSense or your own logger) through `load_sensor_csv` to go beyond this.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["soil_moisture", "temperature", "humidity"]
ACTIONS = ["wait", "irrigate", "alert"]
WAIT, IRRIGATE, ALERT = 0, 1, 2

_COLUMN_ALIASES = {
    "soil_moisture": ["soil_moisture", "soilmoisture", "moisture", "soil moisture", "sm", "vwc"],
    "temperature": ["temperature", "temp", "air_temperature", "air_temp", "t"],
    "humidity": ["humidity", "rh", "relative_humidity", "air_humidity"],
}


# --------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------
def simulate_field(seed: int, cfg, days: int | None = None) -> pd.DataFrame:
    """Simulate one field for `days` days of hourly data (loosely tuned to a Karnataka-like climate)."""
    sc = cfg.sensors
    rng = np.random.default_rng(seed)
    days = days or sc.days_per_field
    n = days * 24
    t = np.arange(n)
    day, hod = t // 24, t % 24
    doy = (int(rng.integers(0, 365)) + day) % 365
    doy_day = doy[::24][:days]

    # ---- weather ------------------------------------------------------------------
    mean_t = 24.5 + 4.0 * np.sin(2 * np.pi * (doy - 100) / 365)
    anomaly = rng.normal(0, 1.5, days)
    for _ in range(rng.poisson(days / 60)):  # heat waves
        s = int(rng.integers(0, max(1, days - 4)))
        anomaly[s : s + int(rng.integers(2, 5))] += rng.uniform(5, 8)
    diurnal = 5.0 * np.sin(2 * np.pi * (hod - 9) / 24)

    monsoon = (doy_day >= 152) & (doy_day <= 273)
    rain_day = rng.random(days) < np.where(monsoon, 0.45, 0.07)
    rain_mm = np.zeros(n)
    for d in np.flatnonzero(rain_day):
        start = d * 24 + int(rng.integers(12, 21))
        rain_mm[start : start + int(rng.integers(1, 5))] += rng.uniform(2, 10)
    rain_mm = rain_mm[:n]
    rain_smooth = np.convolve((rain_mm > 0).astype(float), np.ones(4) / 4, mode="same")

    temp_true = mean_t + anomaly[day] + diurnal - 3.0 * rain_smooth
    hum_true = np.clip(78 - 2.0 * (temp_true - 24.5) + 18 * rain_smooth + rng.normal(0, 3, n), 20, 100)

    # ---- soil water balance with a noisy "farmer" irrigating --------------------------------
    sm = np.zeros(n)
    sm[0] = rng.uniform(28, 42)
    irrigated = np.zeros(n, dtype=bool)
    farmer_thr, farmer_p = rng.uniform(20, 31), rng.uniform(0.3, 0.8)
    remaining = 0
    for i in range(n - 1):
        daylight = np.sin(np.pi * (hod[i] - 6) / 12) if 6 <= hod[i] <= 18 else 0.0
        et = 0.30 * max(temp_true[i] - 8, 0) / 20 * (1.2 - hum_true[i] / 100) * max(daylight, 0)
        if hod[i] == 6 and sm[i] < farmer_thr and rng.random() < farmer_p:
            remaining = int(rng.integers(3, 6))
        irr = 0.0
        if remaining > 0:
            irr, remaining, irrigated[i] = 2.0, remaining - 1, True
        drain = 0.15 * max(sm[i] - 42, 0)
        sm[i + 1] = np.clip(sm[i] - et + 0.55 * rain_mm[i] + irr - drain, 5, 60)

    # ---- observations: noise + sensor faults on the soil probe ---------------------------
    sm_obs = sm + rng.normal(0, sc.noise_std["soil_moisture"], n)
    temp_obs = temp_true + rng.normal(0, sc.noise_std["temperature"], n)
    hum_obs = np.clip(hum_true + rng.normal(0, sc.noise_std["humidity"], n), 0, 100)
    fault = np.zeros(n, dtype=bool)
    for _ in range(rng.poisson(days / 7)):
        kind = rng.choice(["flatline", "dropout", "spike"])
        s = int(rng.integers(0, max(1, n - 24)))
        if kind == "flatline":
            e = s + int(rng.integers(6, 20))
            sm_obs[s:e] = sm_obs[s]
        elif kind == "dropout":
            e = s + int(rng.integers(1, 4))
            sm_obs[s:e] = 0.0
        else:
            e = s + int(rng.integers(1, 3))
            sm_obs[s:e] += rng.choice([-1, 1]) * rng.uniform(25, 40)
        fault[s:e] = True

    # ---- rule-based label from the hidden clean state ----------------------------------------
    labels = np.zeros(n, dtype=np.int64)
    thr, wilt = sc.irrigate_threshold, sc.wilting_point
    for i in range(n):
        lo = max(0, i - 5)
        if fault[lo : i + 1].any() or sm[i] < wilt or (temp_true[i] >= 36 and sm[i] < thr + 5):
            labels[i] = ALERT
        elif rain_mm[max(0, i - 2) : i + 1].sum() > 0 or irrigated[max(0, i - 3) : i + 1].any():
            labels[i] = WAIT
        elif sm[i] < thr:
            labels[i] = IRRIGATE
    return pd.DataFrame(
        {
            "field_id": seed,
            "hour": t,
            "soil_moisture": sm_obs,
            "temperature": temp_obs,
            "humidity": hum_obs,
            "label": labels,
            "true_soil_moisture": sm,
            "fault": fault,
        }
    )


# --------------------------------------------------------------------------------------
# Real data (AgriSense-style CSV)
# --------------------------------------------------------------------------------------
def observed_rule_label(window: np.ndarray, cfg) -> int:
    """Fallback labeller that only uses *observed* values (for real CSVs without labels)."""
    sm = window[:, 0]
    sc = cfg.sensors
    lo, hi = sc.ranges["soil_moisture"]
    if (
        np.ptp(sm[-6:]) < 0.05
        or np.abs(np.diff(sm)).max() > 15
        or sm.min() < lo - 1e-6 and sm[-1] <= lo
        or sm[-1] > hi
        or sm[-1] < sc.wilting_point
    ):
        return ALERT
    if sm[-1] - sm[-4] > 2.0:  # rising: rain or irrigation under way
        return WAIT
    return IRRIGATE if sm[-1] < sc.irrigate_threshold else WAIT


def _find_column(df: pd.DataFrame, feature: str) -> str:
    lowered = {c.lower().strip(): c for c in df.columns}
    for alias in _COLUMN_ALIASES[feature]:
        if alias in lowered:
            return lowered[alias]
    raise KeyError(f"CSV needs a column for {feature!r} (accepted names: {_COLUMN_ALIASES[feature]})")


def load_sensor_csv(path: str | Path) -> pd.DataFrame:
    """Load an hourly sensor CSV into the canonical schema.

    Required columns (case-insensitive aliases accepted): soil_moisture, temperature, humidity.
    Optional: timestamp (resampled to hourly), field_id, label (wait|irrigate|alert or 0|1|2).
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame({f: raw[_find_column(raw, f)].astype(float) for f in FEATURES})
    lowered = {c.lower().strip(): c for c in raw.columns}
    out["field_id"] = raw[lowered["field_id"]] if "field_id" in lowered else 0
    if "timestamp" in lowered:
        out["timestamp"] = pd.to_datetime(raw[lowered["timestamp"]])
        parts = []
        for fid, g in out.groupby("field_id"):
            g = g.set_index("timestamp").sort_index()
            g = g[FEATURES].resample("1h").mean().interpolate(limit=3).dropna()
            g["field_id"] = fid
            parts.append(g.reset_index(drop=True))
        out = pd.concat(parts, ignore_index=True)
    if "label" in lowered:
        lab = raw[lowered["label"]]
        if lab.dtype == object:
            lab = lab.str.lower().map({a: i for i, a in enumerate(ACTIONS)})
        out["label"] = lab.astype("Int64")
    return out


# --------------------------------------------------------------------------------------
# Windows and splits
# --------------------------------------------------------------------------------------
def make_windows(df: pd.DataFrame, window: int, stride: int, cfg=None):
    """Slice one field into (X [N,window,3], y [N]). Label = action at the last hour of the window."""
    values = df[FEATURES].to_numpy(dtype=np.float32)
    X, y = [], []
    has_label = "label" in df.columns and df["label"].notna().all()
    for end in range(window - 1, len(df), stride):
        w = values[end - window + 1 : end + 1]
        X.append(w)
        y.append(int(df["label"].iloc[end]) if has_label else observed_rule_label(w, cfg))
    return np.stack(X), np.asarray(y, dtype=np.int64)


def normalize(x: np.ndarray, ranges: dict) -> np.ndarray:
    """Scale [..., 3] physical values to [0, 1] using fixed sensor ranges (not data-dependent)."""
    lo = np.array([ranges[f][0] for f in FEATURES], dtype=np.float32)
    hi = np.array([ranges[f][1] for f in FEATURES], dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_sensor_splits(cfg, seed: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split by FIELD (simulator) or by contiguous time blocks (single real logger) - never by random window."""
    sc = cfg.sensors
    seed = cfg.seed if seed is None else seed
    w, s = sc.window_hours, sc.stride_hours
    csv = cfg.paths.get("sensor_csv")
    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    if csv:
        df = load_sensor_csv(csv)
        ids = sorted(df["field_id"].unique())
        if len(ids) >= 3:  # split by field
            rng = np.random.default_rng(seed)
            rng.shuffle(ids)
            n_tr, n_va = max(1, int(0.7 * len(ids))), max(1, int(0.15 * len(ids)))
            assign = {i: "train" if k < n_tr else "val" if k < n_tr + n_va else "test" for k, i in enumerate(ids)}
            for fid, g in df.groupby("field_id"):
                splits[assign[fid]].append(make_windows(g.reset_index(drop=True), w, s, cfg))
        else:  # few loggers: contiguous time blocks, never random windows
            for _, g in df.groupby("field_id"):
                g = g.reset_index(drop=True)
                n = len(g)
                blocks = {
                    "train": g.iloc[: int(0.7 * n)],
                    "val": g.iloc[int(0.7 * n) : int(0.85 * n)],
                    "test": g.iloc[int(0.85 * n) :],
                }
                for k, b in blocks.items():
                    if len(b) > w:
                        splits[k].append(make_windows(b.reset_index(drop=True), w, s, cfg))
    else:
        offsets = {"train": 0, "val": 10_000, "test": 20_000}
        counts = {"train": sc.fields_train, "val": sc.fields_val, "test": sc.fields_test}
        for name in splits:
            for k in range(counts[name]):
                splits[name].append(make_windows(simulate_field(seed + offsets[name] + k, cfg), w, s))

    out = {}
    for name, parts in splits.items():
        if not parts:
            raise RuntimeError(f"No sensor windows for split {name!r}; need more data.")
        out[name] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    return out


def dataframe_to_window(readings: list[dict] | pd.DataFrame, window: int) -> np.ndarray:
    """Take the last `window` hourly readings -> array [window, 3] (physical units)."""
    df = pd.DataFrame(readings)
    arr = df[[_find_column(df, f) for f in FEATURES]].to_numpy(dtype=np.float32)
    if len(arr) < window:
        raise ValueError(f"Need at least {window} hourly readings, got {len(arr)}.")
    return arr[-window:]
