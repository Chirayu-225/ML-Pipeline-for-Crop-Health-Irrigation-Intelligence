"""Frames (train/val/test tables), PyTorch datasets and augmentation pipelines for leaf images."""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from cropiq.utils import match_label

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE = -100  # ignore_index for the severity loss


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
@dataclass
class Frames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]
    holdout: list[str] = field(default_factory=list)
    rare: list[str] = field(default_factory=list)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.class_names)}

    @property
    def healthy_idx(self) -> list[int]:
        return [i for i, c in enumerate(self.class_names) if c.endswith("healthy")]


def _with_abs_path(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    df = df.copy()
    df["abs_path"] = [str(root / p) for p in df["path"]]
    return df


def load_frames(cfg, with_synthetic: bool = False) -> Frames:
    """Build leak-free train/val/test tables.

    * rare classes are down-sampled in TRAIN only (controlled low-data experiment);
    * holdout classes are removed from train/val (they act as unknown diseases at test time);
    * synthetic images are appended to TRAIN only, and only if `with_synthetic`.
    """
    crop = cfg.data.crop
    df = pd.read_csv(cfg.paths.manifest)
    df = _with_abs_path(df, Path(cfg.paths.data_root))

    all_labels = sorted(df["label"].unique())
    holdout = [match_label(c, all_labels, crop) for c in cfg.data.get("holdout_classes", []) or []]
    rare = [match_label(c, all_labels, crop) for c in cfg.data.get("rare_classes", []) or []]
    class_names = [c for c in all_labels if c not in holdout]

    train = df[(df.split == "train") & df.label.isin(class_names)]
    rare_seed = int(cfg.data.get("rare_seed", 0))
    parts = []
    for label, grp in train.groupby("label"):
        if label in rare and cfg.data.rare_keep_fraction < 1.0:
            grp = rare_subset(grp, label, cfg.data.rare_keep_fraction, rare_seed)
        parts.append(grp)
    train = pd.concat(parts).reset_index(drop=True)

    if with_synthetic:
        syn_path = Path(cfg.paths.synthetic_dir) / "manifest.csv"
        if not syn_path.exists():
            raise FileNotFoundError(f"{syn_path} not found - run the diffusion generation step first.")
        syn = _with_abs_path(pd.read_csv(syn_path), Path(cfg.paths.synthetic_dir))
        syn = syn[syn.label.isin(class_names)]
        cap = cfg.classifier.get("synthetic_max_per_class", None)
        if cap:
            syn = pd.concat(
                [g.sample(n=min(cap, len(g)), random_state=cfg.seed) for _, g in syn.groupby("label")],
                ignore_index=True,
            )
        train = pd.concat([train, syn], ignore_index=True)

    val = df[(df.split == "val") & df.label.isin(class_names)].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    test["is_unknown"] = test.label.isin(holdout)
    return Frames(train, val, test, class_names, holdout, rare)


def rare_subset(grp: pd.DataFrame, label: str, keep_fraction: float, rare_seed: int = 0) -> pd.DataFrame:
    """Deterministic, NESTED subset of one rare class's TRAIN images.

    * It depends only on `rare_seed` and the class name, never on the training seed, so every training seed of
      every ablation arm (and the diffusion model) sees exactly the same rare images. Otherwise the DDPM could
      have been trained on rare images the classifier never saw, which would unfairly favour the synthetic arm.
    * The images are ranked once by a fixed permutation and the first `keep` are taken, so a 2 % subset is
      always contained in the 5 % subset, which is contained in the 15 % subset (clean keep-fraction sweeps).
    """
    grp = grp.sort_values("path")
    keep = min(len(grp), max(3, int(round(len(grp) * keep_fraction))))
    seed = (rare_seed * 1_000_003 + zlib.crc32(label.encode("utf-8"))) % (2**32)
    order = np.random.default_rng(seed).permutation(len(grp))
    return grp.iloc[np.sort(order[:keep])]


def rare_signature(frames: "Frames") -> str:
    """Short hash of the rare-class TRAIN image list; stored in checkpoints to detect subset mismatches."""
    paths = sorted(frames.train.loc[frames.train.label.isin(frames.rare) & (frames.train.get("source", "real") == "real"), "path"])
    return hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()[:12]


def class_counts(frame: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    vc = frame["label"].value_counts()
    return np.array([vc.get(c, 0) for c in class_names])


def make_sampler(frame: pd.DataFrame, class_names: list[str]) -> WeightedRandomSampler:
    """Inverse-frequency sampler (the strong classical baseline for the diffusion ablation)."""
    counts = class_counts(frame, class_names).astype(float)
    per_class = 1.0 / np.maximum(counts, 1.0)
    idx = frame["label"].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    weights = per_class[idx]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(frame), replacement=True)


# --------------------------------------------------------------------------------------
# Transforms for the classifier
# --------------------------------------------------------------------------------------
def build_transform(size: int, train: bool, downscale_p: float = 0.0) -> A.Compose:
    norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    if not train:
        return A.Compose([A.Resize(size, size), *norm])
    aug = [
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
    ]
    if downscale_p > 0:
        aug.append(A.Downscale(scale_range=(0.5, 0.9), p=downscale_p))
    return A.Compose([*aug, *norm])


class LeafDataset(Dataset):
    """Returns (image tensor, disease idx or -1 if unknown class, severity idx 0..2 or IGNORE)."""

    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform: A.Compose):
        self.paths = frame["abs_path"].tolist()
        self.labels = [class_to_idx.get(l, -1) for l in frame["label"]]
        sev = frame["severity"].to_numpy()
        self.severity = [int(s) - 1 if s > 0 else IGNORE for s in sev]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = np.asarray(Image.open(self.paths[i]).convert("RGB"))
        x = self.transform(image=img)["image"]
        return x, self.labels[i], self.severity[i]


# --------------------------------------------------------------------------------------
# Raw [0,1] / [-1,1] images for the generative models (VAE, DDPM)
# --------------------------------------------------------------------------------------
class RawImageDataset(Dataset):
    """Pre-loads and resizes all images (they are small) and serves float tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        size: int,
        augment: bool = False,
        value_range: str = "01",
    ):
        self.size = size
        self.augment = augment
        self.value_range = value_range
        self.labels = [class_to_idx.get(l, -1) for l in frame["label"]]
        self.images = np.stack(
            [
                np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC))
                for p in frame["abs_path"]
            ]
        ) if len(frame) else np.zeros((0, size, size, 3), np.uint8)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        if self.augment:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
        if self.value_range == "-11":
            x = x * 2.0 - 1.0
        return x, self.labels[i]


def pil_to_tensor_01(img: Image.Image, size: int) -> torch.Tensor:
    arr = np.array(img.convert("RGB").resize((size, size), Image.BICUBIC))
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
