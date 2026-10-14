"""Fusion layer: disease evidence + irrigation decision + sensor context -> one advisory.

Pure Python (no torch) so it is trivially unit-testable. The design principle is *interaction awareness*:
irrigation advice changes when a foliar disease is present (how and when to water), and a disease
prediction is down-weighted to 'uncertain' / 'possible unknown disease' when the classifier is unsure.
"""
from __future__ import annotations

from cropiq.pipeline.knowledge import info_for

SEVERITY_WEIGHT = {"none": 0.0, "mild": 0.4, "moderate": 0.7, "severe": 1.0}
IRRIGATION_URGENCY = {"wait": 0.0, "irrigate": 0.7, "alert": 1.0}
DISCLAIMER = (
    "Decision support only, not a diagnosis. Confirm in the field and check treatment options with your "
    "local agricultural extension service (e.g. Krishi Vigyan Kendra) before spraying anything."
)


def classify_disease_status(img: dict, min_conf: float) -> str:
    healthy = img["disease"].endswith("healthy")
    flagged = bool(img.get("anomaly_flag"))
    conf = img["confidence"]
    if flagged and conf < min_conf:
        return "possible_unknown_disease"
    if conf < min_conf:
        return "uncertain"
    if healthy:
        return "healthy_flagged" if flagged else "healthy"
    return "known_disease"


def disease_risk(status: str, img: dict) -> float:
    if status == "known_disease":
        risk = SEVERITY_WEIGHT.get(img.get("severity", "mild"), 0.4) * img["confidence"]
        return min(1.0, risk + (0.15 if info_for(img["disease"]).get("urgent") else 0.0))
    return {"possible_unknown_disease": 0.65, "uncertain": 0.4, "healthy_flagged": 0.25, "healthy": 0.0}[status]


def risk_level(score: float) -> str:
    return "low" if score < 0.2 else "moderate" if score < 0.45 else "high" if score < 0.7 else "critical"


def build_advisory(img: dict | None, irr: dict | None, sensors: dict | None, adv_cfg) -> dict:
    """img: output of CropAdvisor.predict_image; irr: output of predict_irrigation; sensors: last-reading summary."""
    min_conf = adv_cfg.get("min_confidence", 0.6)
    humid_rh = adv_cfg.get("humid_rh", 85)
    recs: list[dict] = []

    # ---------------------------------------------------------------- disease branch
    status, d_risk, info = "not_assessed", 0.0, {}
    if img is not None:
        status = classify_disease_status(img, min_conf)
        d_risk = disease_risk(status, img)
        info = info_for(img["disease"])
        name = info["display"]
        if status == "known_disease":
            sev = img.get("severity", "mild")
            recs.append({"priority": 1, "text": f"{name} detected ({sev} severity, {img['confidence']:.0%} confidence). {info.get('note', '')}".strip()})
            if info.get("urgent"):
                recs.append({"priority": 1, "text": "Act today: remove badly affected foliage and contact your agri-extension officer immediately; this disease can spread across a field in days."})
            if info["kind"] == "viral":
                recs.append({"priority": 2, "text": "No cure for viral disease: isolate or remove infected plants" + (f" and control the {info['vector']} vector." if info.get("vector") else " and disinfect tools and hands between plants.")})
            elif info["kind"] in ("fungal", "bacterial", "oomycete"):
                recs.append({"priority": 2, "text": "Remove severely affected leaves, avoid working in the crop while it is wet, and ask your extension service for an approved treatment."})
            elif info["kind"] == "pest":
                recs.append({"priority": 2, "text": "Inspect leaf undersides for mites and consult your extension service for suitable control."})
            if img.get("anomaly_flag"):
                recs.append({"priority": 3, "text": "The leaf also looks unusual to the anomaly screen, so a second problem (co-infection or nutrient stress) is possible."})
        elif status == "possible_unknown_disease":
            recs.append({"priority": 1, "text": "This leaf does not match the known diseases and looks abnormal. Possible unknown disease: isolate the plant, photograph several leaves and escalate to an agronomist."})
        elif status == "uncertain":
            recs.append({"priority": 2, "text": f"The disease prediction is uncertain ({img['confidence']:.0%}, top guess: {info['display']}). Retake the photo (single leaf, good light, plain background)."})
        elif status == "healthy_flagged":
            recs.append({"priority": 3, "text": "Leaf classified healthy, but the anomaly screen raised a mild flag. Re-scan a few more leaves to rule out early symptoms."})
        else:
            recs.append({"priority": 4, "text": "Leaf looks healthy."})

    # ---------------------------------------------------------------- irrigation branch
    action, i_urg = None, 0.0
    if irr is not None:
        action = irr["action"]
        i_urg = IRRIGATION_URGENCY[action]
        wet_risk = bool(info.get("wet_foliage_risk")) or status in ("possible_unknown_disease", "uncertain")
        humidity = (sensors or {}).get("humidity_mean_24h")
        if action == "irrigate":
            if wet_risk and img is not None:
                recs.append({"priority": 1, "text": "Irrigate, but at the base (drip or furrow) in the early morning. Avoid overhead watering and evening irrigation: wet foliage helps this problem spread."})
            elif info.get("dry_stress_risk"):
                recs.append({"priority": 1, "text": "Irrigate now. Water-stressed plants suffer worse mite outbreaks, so keep soil moisture steady."})
            else:
                recs.append({"priority": 2, "text": "Irrigate now (soil moisture is below the target). Early morning is best."})
        elif action == "wait":
            recs.append({"priority": 4, "text": "No irrigation needed right now."})
            if info.get("dry_stress_risk"):
                recs.append({"priority": 3, "text": "Hot, dry spells favour mites: keep monitoring soil moisture closely."})
        else:  # alert
            recs.append({"priority": 1, "text": "Sensor/field alert: possible probe fault, critically dry soil or heat stress. Inspect the field and the soil probe before relying on automatic decisions."})
        if wet_risk and humidity is not None and humidity >= humid_rh:
            recs.append({"priority": 2, "text": f"Humidity has averaged {humidity:.0f}% over the last 24 h, which raises infection pressure: improve airflow, avoid dense canopy and late-day wetting."})

    # Combined priority: a serious disease dominates on its own; irrigation need adds on top of it.
    if img is not None and irr is not None:
        score = max(d_risk, 0.65 * d_risk + 0.35 * i_urg)
    elif irr is not None:
        score = 0.6 * i_urg
    else:
        score = d_risk
    level = risk_level(score)
    recs.sort(key=lambda r: r["priority"])

    parts = []
    if img is not None:
        if status in ("known_disease", "healthy", "healthy_flagged"):
            label = info["display"] + (f" ({img['severity']})" if status == "known_disease" else "")
        else:
            label = status.replace("_", " ")
        parts.append(f"Leaf: {label}")
    if irr is not None:
        parts.append(f"Irrigation: {action}")
    summary = f"{level.upper()} priority. " + "; ".join(parts) + "."

    return {
        "risk_level": level,
        "risk_score": round(float(score), 3),
        "disease_status": status,
        "irrigation_action": action,
        "summary": summary,
        "recommendations": [r["text"] for r in recs],
        "disclaimer": DISCLAIMER,
    }
