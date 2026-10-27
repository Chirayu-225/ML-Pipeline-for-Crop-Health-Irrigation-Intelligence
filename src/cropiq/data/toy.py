"""Procedural PlantVillage look-alike for CPU smoke tests and unit tests (NOT for real results).

Creates `<root>/color/Tomato___<Class>/img_XXXX.jpg` with class-specific lesion styles and a known
ground-truth lesion fraction (written to `ground_truth.csv`), so the severity heuristic can be tested.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# folder name -> (images, lesion colour RGB, n_lesions range, (min,max) area fraction, style)
TOY_CLASSES = {
    "Tomato___Bacterial_spot": (24, (90, 55, 30), (10, 18), (0.03, 0.18), "spot"),
    "Tomato___Early_blight": (14, (120, 80, 35), (3, 6), (0.05, 0.30), "ring"),
    "Tomato___Late_blight": (22, (50, 45, 30), (2, 4), (0.10, 0.45), "patch"),
    "Tomato___Leaf_Mold": (12, (205, 190, 70), (3, 6), (0.05, 0.30), "patch"),
    "Tomato___Septoria_leaf_spot": (20, (190, 190, 175), (14, 24), (0.02, 0.15), "spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": (20, (200, 195, 90), (20, 35), (0.03, 0.20), "spot"),
    "Tomato___Target_Spot": (18, (165, 125, 70), (5, 9), (0.04, 0.25), "ring"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (40, (215, 205, 60), (1, 2), (0.30, 0.60), "patch"),
    "Tomato___Tomato_mosaic_virus": (8, (170, 200, 90), (6, 10), (0.05, 0.20), "patch"),
    "Tomato___healthy": (24, (0, 0, 0), (0, 0), (0.0, 0.0), "none"),
}


def _leaf(rng: np.random.Generator, size: int):
    bg = int(rng.integers(115, 160))
    img = np.full((size, size, 3), (bg, bg + 3, bg + 6), np.uint8)
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8)
    center = (int(size * rng.uniform(0.45, 0.55)), int(size * rng.uniform(0.45, 0.55)))
    axes = (int(size * rng.uniform(0.34, 0.42)), int(size * rng.uniform(0.22, 0.30)))
    angle = float(rng.uniform(0, 180))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    green = np.array([rng.integers(40, 70), rng.integers(120, 160), rng.integers(30, 60)])
    leaf_pixels = np.where(mask[..., None] > 0, green, img).astype(np.uint8)
    # midrib
    dx, dy = np.cos(np.deg2rad(angle)) * axes[0], np.sin(np.deg2rad(angle)) * axes[0]
    p1 = (int(center[0] - dx), int(center[1] - dy))
    p2 = (int(center[0] + dx), int(center[1] + dy))
    rib = leaf_pixels.copy()
    cv2.line(rib, p1, p2, tuple(int(c * 0.85) for c in green), 1)
    leaf_pixels = np.where(mask[..., None] > 0, rib, leaf_pixels)
    return leaf_pixels, mask


def make_image(rng: np.random.Generator, size: int, colour, n_range, area_range, style: str):
    img, leaf = _leaf(rng, size)
    lesion = np.zeros_like(leaf)
    if style != "none":
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        target = rng.uniform(*area_range) * leaf.sum()
        ys, xs = np.nonzero(leaf)
        radius = max(1.5, float(np.sqrt(target / max(n, 1) / np.pi)))
        for _ in range(n):
            k = int(rng.integers(0, len(xs)))
            c = (int(xs[k]), int(ys[k]))
            r = int(max(1, radius * rng.uniform(0.8, 1.2)))
            cv2.circle(lesion, c, r, 1, -1)
        lesion &= leaf
        jitter = rng.integers(-12, 13, 3)
        col = np.clip(np.array(colour) + jitter, 0, 255).astype(np.uint8)
        img = np.where(lesion[..., None] > 0, col, img).astype(np.uint8)
        if style == "ring":  # darker core to look like concentric rings
            core = np.zeros_like(lesion)
            ys2, xs2 = np.nonzero(lesion)
            for k in rng.choice(len(xs2), size=max(1, len(xs2) // 40), replace=False):
                cv2.circle(core, (int(xs2[k]), int(ys2[k])), 1, 1, -1)
            img = np.where(core[..., None] > 0, (col * 0.6).astype(np.uint8), img)
    frac = float(lesion.sum() / max(leaf.sum(), 1))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, frac


def make_toy_dataset(root: str | Path, size: int = 96, seed: int = 0, scale: float = 1.0) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    for folder, (n_img, colour, n_range, area_range, style) in TOY_CLASSES.items():
        out = root / "color" / folder
        out.mkdir(parents=True, exist_ok=True)
        for i in range(max(4, int(round(n_img * scale)))):
            img, frac = make_image(rng, size, colour, n_range, area_range, style)
            rel = f"color/{folder}/img_{i:04d}.jpg"
            cv2.imwrite(str(root / rel), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"path": rel, "class": folder, "true_lesion_frac": frac})
    pd.DataFrame(rows).to_csv(root / "ground_truth.csv", index=False)
    return root


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a tiny synthetic PlantVillage-like dataset")
    p.add_argument("--out", default="artifacts_smoke/toy_plantvillage")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--scale", type=float, default=1.0, help="multiply images per class")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    root = make_toy_dataset(args.out, args.size, args.seed, args.scale)
    print(f"Toy dataset written to {root}")


if __name__ == "__main__":
    main()
