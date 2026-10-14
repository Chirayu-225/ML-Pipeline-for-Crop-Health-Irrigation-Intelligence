from cropiq.config import Config
from cropiq.pipeline.advisory import build_advisory

ADV = Config(min_confidence=0.6, humid_rh=85)


def img(disease="healthy", conf=0.95, sev="none", flag=False):
    return {"disease": disease, "confidence": conf, "severity": sev, "anomaly_flag": flag, "anomaly_score": 0.1, "anomaly_threshold": 0.05}


def irr(action="wait"):
    return {"action": action}


def text(adv):
    return " ".join(adv["recommendations"]).lower()


def test_healthy_leaf_and_no_irrigation_is_low_risk():
    a = build_advisory(img(), irr("wait"), {"humidity_mean_24h": 60}, ADV)
    assert a["risk_level"] == "low" and a["disease_status"] == "healthy"


def test_foliar_disease_changes_irrigation_advice():
    a = build_advisory(img("early_blight", 0.9, "moderate"), irr("irrigate"), {"humidity_mean_24h": 60}, ADV)
    assert "base" in text(a) and "overhead" in text(a)  # drip / avoid wetting foliage
    plain = build_advisory(img("healthy"), irr("irrigate"), {}, ADV)
    assert "overhead" not in text(plain)


def test_high_humidity_adds_infection_pressure_warning_only_for_wet_diseases():
    wet = build_advisory(img("late_blight", 0.9, "mild"), irr("wait"), {"humidity_mean_24h": 92}, ADV)
    assert "humidity" in text(wet)
    viral = build_advisory(img("tomato_mosaic_virus", 0.9, "mild"), irr("wait"), {"humidity_mean_24h": 92}, ADV)
    assert "infection pressure" not in text(viral)


def test_low_confidence_plus_anomaly_means_possible_unknown_disease():
    a = build_advisory(img("septoria_leaf_spot", 0.35, "mild", flag=True), irr("wait"), {}, ADV)
    assert a["disease_status"] == "possible_unknown_disease" and a["risk_level"] in ("high", "critical", "moderate")
    assert "agronomist" in text(a)


def test_low_confidence_without_anomaly_is_uncertain_not_alarming():
    a = build_advisory(img("septoria_leaf_spot", 0.35, "mild"), None, None, ADV)
    assert a["disease_status"] == "uncertain" and "retake" in text(a)


def test_severe_late_blight_is_critical():
    a = build_advisory(img("late_blight", 0.97, "severe"), irr("wait"), {}, ADV)
    assert a["risk_level"] == "critical"


def test_viral_disease_advice_mentions_vector_control_and_no_cure():
    a = build_advisory(img("tomato_yellow_leaf_curl_virus", 0.9, "severe"), None, None, ADV)
    assert "whitefly" in text(a) and "no cure" in text(a)


def test_sensor_alert_surfaces_inspection_advice():
    a = build_advisory(None, irr("alert"), {}, ADV)
    assert "inspect" in text(a) and a["risk_level"] in ("moderate", "high", "critical")


def test_mites_plus_dry_soil_prioritises_watering():
    a = build_advisory(img("spider_mites_two_spotted_spider_mite", 0.9, "mild"), irr("irrigate"), {}, ADV)
    assert "mite" in text(a) and "overhead" not in text(a)


def test_output_always_has_disclaimer_and_summary():
    a = build_advisory(img(), irr(), {}, ADV)
    assert a["disclaimer"] and a["summary"].endswith(".")
