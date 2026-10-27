"""Scan a PlantVillage folder, compute proxy severity labels and write a stratified split manifest.

    python -m cropiq.data.manifest --config configs/default.yaml

The manifest is the single source of truth for splits: every training script reads it, so there is no
way for validation/test images to leak into training (or into the diffusion model).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cropiq.config import base_parser, load_config
from cropiq.data.severity import lesion_fraction, severity_from_fraction
from cropiq.utils import canon_class, ensure_dir, get_logger, is_healthy, seed_everything

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VARIANTS = {"color", "grayscale", "segmented"}
log = get_logger("manifest")


def discover_images(root: Path, crop: str, variant: str) -> dict[str, list[Path]]:
    """Return {canonical_label: [image paths]} from any folder nesting of PlantVillage."""
    found: dict[str, list[Path]] = defaultdict(list)
    other_variants = VARIANTS - {variant}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        parts = {part.lower() for part in d.relative_to(root).parts}
        if parts & other_variants:
            continue
        imgs = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
        if not imgs:
            continue
        label = canon_class(d.name, crop)
        if crop != "all":
            raw = canon_class(d.name, "all")
            if not raw.startswith(crop.lower() + "_"):
                continue
        found[label].extend(sorted(imgs))
    return found


def stratified_split(n: int, fractions: dict, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(n)
    n_train = int(round(n * fractions["train"]))
    n_val = int(round(n * fractions["val"]))
    if n >= 3:  # make sure every split has at least one image
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    split = np.empty(n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train : n_train + n_val]] = "val"
    split[idx[n_train + n_val :]] = "test"
    return split


def build_manifest(cfg) -> pd.DataFrame:
    root = Path(cfg.paths.data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download PlantVillage first (see README: 'Getting the data')."
        )
    rng = np.random.default_rng(cfg.seed)
    found = discover_images(root, cfg.data.crop, cfg.data.variant)
    if not found:
        raise RuntimeError(f"No images found under {root} for crop={cfg.data.crop!r}.")
    log.info("Found %d classes, %d images", len(found), sum(len(v) for v in found.values()))

    rows = []
    for label, paths in sorted(found.items()):
        splits = stratified_split(len(paths), cfg.data.split, rng)
        for path, split in zip(tqdm(paths, desc=label, leave=False), splits):
            with Image.open(path) as im:
                frac = lesion_fraction(np.asarray(im.convert("RGB")))
            healthy = is_healthy(label)
            sev = 0 if healthy else severity_from_fraction(
                frac, cfg.severity.mild_max, cfg.severity.moderate_max
            )
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "is_healthy": healthy,
                    "split": split,
                    "lesion_frac": round(frac, 4),
                    "severity": sev,
                    "source": "real",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = base_parser("Build the dataset manifest (splits + proxy severity labels)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    df = build_manifest(cfg)
    out = Path(cfg.paths.manifest)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    summary = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print("\nSeverity distribution (diseased only):")
    print(df[~df.is_healthy].severity.value_counts().sort_index().to_string())
    log.info("Wrote %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
