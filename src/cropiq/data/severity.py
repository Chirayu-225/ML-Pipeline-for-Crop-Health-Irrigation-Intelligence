"""Lesion-area severity proxy.

PlantVillage has **no severity annotations**. We derive weak labels from the fraction of the leaf area
that is not healthy green (necrotic, chlorotic or discoloured). The result is a *proxy*: it is good for
lesion-type diseases (blights, spots, mould) and weak for systemic ones (viruses, mites). It must be
described as such in any write-up. Replace with expert labels if you can get them.
"""
from __future__ import annotations

import cv2
import numpy as np

SEVERITY_NAMES = {0: "none", 1: "mild", 2: "moderate", 3: "severe"}


def lesion_fraction(rgb: np.ndarray, work_size: int = 128) -> float:
    """Fraction (0..1) of leaf pixels that are not healthy green. `rgb` is a uint8 HxWx3 array."""
    img = cv2.resize(rgb, (work_size, work_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 1) leaf region: saturated, not-too-dark pixels (PlantVillage backgrounds are low-saturation grey)
    candidate = ((s > 40) & (v > 35)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if n <= 1:
        return 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)

    # 2) fill holes so necrotic (dark / unsaturated) centres still count as leaf area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    leaf_area = int(filled.sum())
    if leaf_area < 0.05 * mask.size:
        return 0.0

    # 3) healthy green (OpenCV hue range is 0-179; green ~ 33-85)
    green = (h >= 33) & (h <= 85) & (s > 40) & (v > 50)
    lesion = filled.astype(bool) & ~green
    return float(lesion.sum() / leaf_area)


def severity_from_fraction(frac: float, mild_max: float = 0.08, moderate_max: float = 0.25) -> int:
    """1 = mild, 2 = moderate, 3 = severe (0 = none is assigned by the caller for healthy leaves)."""
    if frac <= mild_max:
        return 1
    if frac <= moderate_max:
        return 2
    return 3
