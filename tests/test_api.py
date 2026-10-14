import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cropiq.api.main import create_app
from cropiq.data.sensors import FEATURES, simulate_field


@pytest.fixture(scope="module")
def client(random_artifacts):
    cfg = random_artifacts
    overrides = [f"{k}={v}" for k, v in [
        ("paths.runs_dir", cfg.paths.runs_dir), ("models.classifier_ckpt", cfg.models.classifier_ckpt),
        ("models.vae_ckpt", cfg.models.vae_ckpt), ("models.snn_ckpt", cfg.models.snn_ckpt)]]
    from conftest import ROOT
    return TestClient(create_app(str(ROOT / "configs/smoke.yaml"), overrides))


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), (60, 140, 50)).save(buf, "JPEG")
    return buf.getvalue()


def readings(n=24, seed=2, cfg=None):
    from conftest import ROOT
    from cropiq.config import load_config
    df = simulate_field(seed, load_config(ROOT / "configs/smoke.yaml"))
    return df.iloc[100 : 100 + n][FEATURES].to_dict("records")


def test_health_reports_all_models(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["models"]["classifier"] and r["models"]["vae"] and r["models"]["snn"]


def test_predict_image(client):
    r = client.post("/predict/image", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["confidence"] <= 1 and len(body["top_k"]) == 5 and "anomaly_flag" in body


def test_predict_image_rejects_non_images(client):
    r = client.post("/predict/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_predict_irrigation_and_validation(client):
    r = client.post("/predict/irrigation", json={"readings": readings()})
    assert r.status_code == 200 and r.json()["action"] in ("wait", "irrigate", "alert")
    assert abs(sum(r.json()["probabilities"].values()) - 1) < 1e-4
    assert client.post("/predict/irrigation", json={"readings": readings(10)}).status_code == 422  # too short
    bad = readings(); bad[0]["humidity"] = 900
    assert client.post("/predict/irrigation", json={"readings": bad}).status_code == 422  # out of range


def test_irrigation_decision_is_deterministic(client):
    a = client.post("/predict/irrigation", json={"readings": readings()}).json()
    b = client.post("/predict/irrigation", json={"readings": readings()}).json()
    assert a["probabilities"] == b["probabilities"]


def test_fused_advisory(client):
    r = client.post("/advisory", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")}, data={"readings": json.dumps({"readings": readings()})})
    assert r.status_code == 200
    body = r.json()
    assert body["advisory"]["risk_level"] in ("low", "moderate", "high", "critical")
    assert body["image_analysis"] and body["irrigation"] and body["advisory"]["recommendations"]


def test_advisory_needs_some_input(client):
    assert client.post("/advisory").status_code == 422


def test_missing_model_gives_503_not_500(random_artifacts):
    from conftest import ROOT
    app = create_app(str(ROOT / "configs/smoke.yaml"), ["models.classifier_ckpt=/nonexistent.pt", "paths.runs_dir=/nonexistent", "models.vae_ckpt=/nonexistent.pt", "models.snn_ckpt=/nonexistent.pt"])
    c = TestClient(app)
    assert c.get("/health").json()["models"]["classifier"] is False
    assert c.post("/predict/image", files={"file": ("l.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 503
