from pathlib import Path

import pytest

from cropiq.config import load_config
from cropiq.utils import canon_class, is_healthy, match_label

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Tomato___Early_blight", "early_blight"),
        ("Tomato_Early_blight", "early_blight"),
        ("Tomato___Tomato_mosaic_virus", "tomato_mosaic_virus"),
        ("Tomato___Spider_mites Two-spotted_spider_mite", "spider_mites_two_spotted_spider_mite"),
        ("Tomato_Spider_mites_Two_spotted_spider_mite", "spider_mites_two_spotted_spider_mite"),
        ("Tomato__Tomato_YellowLeaf__Curl_Virus", "tomato_yellow_leaf_curl_virus"),
        ("Tomato___healthy", "healthy"),
        ("Tomato_healthy", "healthy"),
    ],
)
def test_canon_class_handles_all_known_spellings(raw, expected):
    assert canon_class(raw, "tomato") == expected


def test_canon_class_keeps_crop_when_all():
    assert canon_class("Apple___Apple_scab", "all") == "apple_apple_scab"
    assert is_healthy(canon_class("Apple___healthy", "all"))


def test_match_label_accepts_user_spellings():
    labels = ["early_blight", "tomato_mosaic_virus", "healthy"]
    assert match_label("Tomato___Early_blight", labels) == "early_blight"
    assert match_label("tomato_mosaic_virus", labels) == "tomato_mosaic_virus"
    with pytest.raises(ValueError):
        match_label("not_a_class", labels)


def test_config_inheritance_and_overrides():
    cfg = load_config(ROOT / "configs/smoke.yaml", ["classifier.epochs=7", "data.image_size=40"])
    assert cfg.classifier.epochs == 7 and cfg.data.image_size == 40
    assert cfg.classifier.lr == 0.0005  # inherited from default.yaml
    assert cfg.diffusion.generation.ddim_steps == 5  # deep-merged override
    assert cfg.diffusion.generation.guidance_scale == 3.0  # untouched sibling key survives the merge
