"""Small, hand-written disease knowledge base used by the advisory layer (tomato).

Deliberately generic: it states WHAT kind of problem it is and which water-management interactions matter,
and defers product/dose decisions to local agri-extension (e.g. a Krishi Vigyan Kendra). Do not turn this
into a spray-dose recommender without domain-expert review.
"""
from __future__ import annotations

DISEASE_INFO: dict[str, dict] = {
    "bacterial_spot": {
        "display": "Bacterial spot",
        "kind": "bacterial",
        "wet_foliage_risk": True,
        "note": "Spreads through splashing water and handling of wet plants.",
    },
    "early_blight": {
        "display": "Early blight",
        "kind": "fungal",
        "wet_foliage_risk": True,
        "note": "Fungal; favoured by warm, humid spells. Lower, older leaves are hit first.",
    },
    "late_blight": {
        "display": "Late blight",
        "kind": "oomycete",
        "wet_foliage_risk": True,
        "urgent": True,
        "note": "Spreads very fast in cool, wet, humid weather and can destroy a crop within days.",
    },
    "leaf_mold": {
        "display": "Leaf mold",
        "kind": "fungal",
        "wet_foliage_risk": True,
        "note": "Fungal; thrives at high humidity with poor air circulation (common in polyhouses).",
    },
    "septoria_leaf_spot": {
        "display": "Septoria leaf spot",
        "kind": "fungal",
        "wet_foliage_risk": True,
        "note": "Fungal; spores spread by water splash from infected lower leaves.",
    },
    "spider_mites_two_spotted_spider_mite": {
        "display": "Spider mites",
        "kind": "pest",
        "dry_stress_risk": True,
        "note": "Pest rather than pathogen; populations explode in hot, dry conditions and on water-stressed plants.",
    },
    "target_spot": {
        "display": "Target spot",
        "kind": "fungal",
        "wet_foliage_risk": True,
        "note": "Fungal; favoured by prolonged leaf wetness and humid canopies.",
    },
    "tomato_yellow_leaf_curl_virus": {
        "display": "Tomato yellow leaf curl virus",
        "kind": "viral",
        "vector": "whitefly",
        "note": "Viral and incurable; spread by whiteflies. Control is vector management and removing infected plants.",
    },
    "tomato_mosaic_virus": {
        "display": "Tomato mosaic virus",
        "kind": "viral",
        "note": "Viral and incurable; spreads mechanically via hands, tools and seed. Sanitation is the control.",
    },
    "healthy": {"display": "Healthy", "kind": "none"},
}


def info_for(label: str) -> dict:
    if label in DISEASE_INFO:
        return DISEASE_INFO[label]
    pretty = label.replace("_", " ").strip().capitalize()
    return {"display": pretty, "kind": "unknown", "note": ""}
