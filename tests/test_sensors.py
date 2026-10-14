import numpy as np
import pandas as pd
import pytest

from cropiq.config import load_config
from cropiq.data.sensors import (ACTIONS, FEATURES, build_sensor_splits, dataframe_to_window, load_sensor_csv,
                                 make_windows, normalize, simulate_field)
from conftest import ROOT


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs/smoke.yaml")


def test_simulator_shapes_ranges_and_determinism(cfg):
    a, b = simulate_field(5, cfg), simulate_field(5, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == cfg.sensors.days_per_field * 24
    assert a.true_soil_moisture.between(5, 60).all()
    assert a.humidity.between(0, 100).all()


def test_all_three_actions_occur(cfg):
    labels = np.concatenate([simulate_field(s, cfg, days=90).label.to_numpy() for s in range(6)])
    for i, name in enumerate(ACTIONS):
        assert (labels == i).mean() > 0.01, f"action {name} too rare in simulation"


def test_windows_use_last_hour_label(cfg):
    df = simulate_field(1, cfg)
    X, y = make_windows(df, 24, 3)
    assert X.shape[1:] == (24, 3)
    assert y[0] == df.label.iloc[23] and y[1] == df.label.iloc[26]


def test_splits_are_by_field_not_window(cfg):
    sp = build_sensor_splits(cfg)
    assert set(sp) == {"train", "val", "test"}
    tr, te = sp["train"][0], sp["test"][0]
    # no identical window appears in both train and test (they come from different simulated fields)
    train_set = {w.tobytes() for w in tr}
    assert not any(w.tobytes() in train_set for w in te)


def test_normalize_is_bounded(cfg):
    x = np.array([[100.0, -10.0, 50.0]], dtype=np.float32)
    n = normalize(x, cfg.sensors.ranges)
    assert n.min() >= 0 and n.max() <= 1


def test_csv_loader_accepts_aliases_and_resamples(tmp_path):
    ts = pd.date_range("2026-06-01", periods=60, freq="30min")
    raw = pd.DataFrame({"Timestamp": ts, "Moisture": np.linspace(40, 30, 60), "Temp": 28.0, "RH": 65.0})
    path = tmp_path / "real.csv"
    raw.to_csv(path, index=False)
    df = load_sensor_csv(path)
    assert set(FEATURES) <= set(df.columns) and len(df) == 30  # 30-min readings -> hourly


def test_dataframe_to_window_requires_enough_rows():
    rows = [{"soil_moisture": 30, "temperature": 28, "humidity": 60}] * 10
    with pytest.raises(ValueError):
        dataframe_to_window(rows, 24)
    assert dataframe_to_window(rows * 3, 24).shape == (24, 3)
