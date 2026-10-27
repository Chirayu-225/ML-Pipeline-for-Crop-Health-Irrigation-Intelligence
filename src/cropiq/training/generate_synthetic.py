"""Generate synthetic disease images for under-represented classes, with quality gates.

For each target class:
  1. sample with DDIM + classifier-free guidance (64x64), up-sample to the classifier resolution;
  2. TEACHER GATE: keep only images that a classifier trained on REAL data only assigns to the target class
     with probability >= teacher_min_prob (removes off-class / incoherent samples);
  3. NEAR-DUPLICATE GATE: drop images whose feature cosine similarity to any real training image exceeds
     near_duplicate_sim (guards against the model memorising and replaying training images);
  4. compute the same lesion-area severity proxy as for real images.

    python -m cropiq.training.generate_synthetic --teacher artifacts/runs/clf_real/best.pt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from cropiq.config import base_parser, load_config
from cropiq.data.datasets import LeafDataset, build_transform, load_frames, rare_signature
from cropiq.data.severity import lesion_fraction, severity_from_fraction
from cropiq.models.loading import load_classifier, load_ddpm
from cropiq.utils import ensure_dir, get_device, get_logger, match_label, save_json, seed_everything

log = get_logger("generate_synthetic")


@torch.no_grad()
def extract_features(teacher, frame, class_to_idx, size, device, batch_size=64):
    ds = LeafDataset(frame, class_to_idx, build_transform(size, False))
    feats = []
    for i in range(0, len(ds), batch_size):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).to(device)
        feats.append(F.normalize(teacher.features(x).float(), dim=1).cpu())
    return torch.cat(feats) if feats else torch.zeros(0, teacher.feature_dim)


def main() -> None:
    parser = base_parser("Generate synthetic images with the trained DDPM")
    parser.add_argument("--ddpm", default=None, help="DDPM checkpoint (default: <runs_dir>/ddpm/best.pt)")
    parser.add_argument("--teacher", default=None, help="real-data-only classifier checkpoint used as the quality gate")
    parser.add_argument("--classes", nargs="*", default=None, help="classes to generate (default: the configured rare classes)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    g = cfg.diffusion.generation
    size = cfg.data.image_size

    frames = load_frames(cfg, with_synthetic=False)
    names, c2i = frames.class_names, frames.class_to_idx
    targets = [match_label(c, names, cfg.data.crop) for c in args.classes] if args.classes else frames.rare
    if not targets:
        raise SystemExit("No target classes: set data.rare_classes or pass --classes.")

    ddpm_path = args.ddpm or str(Path(cfg.paths.runs_dir) / "ddpm" / "best.pt")
    model, diffusion, ck = load_ddpm(ddpm_path, device)
    if ck["class_names"] != names:
        raise SystemExit("DDPM was trained with different classes than the current config/manifest.")
    signature = rare_signature(frames)
    if ck.get("rare_signature") not in (None, signature):
        raise SystemExit(
            f"DDPM was trained on a different rare-class subset ({ck['rare_signature']}) than the current config "
            f"({signature}, keep {cfg.data.rare_keep_fraction}). Retrain the DDPM for this keep-fraction."
        )
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    teacher, real_feats = None, None
    if args.teacher:
        teacher, tck = load_classifier(args.teacher, device)
        if tck.get("rare_signature") not in (None, signature):
            raise SystemExit("The teacher classifier was trained on a different rare-class subset; retrain it first.")
        real_feats = extract_features(teacher, frames.train, c2i, size, device)
        log.info("Teacher gate on (min prob %.2f); %d real train features cached", g.teacher_min_prob, len(real_feats))
    else:
        log.warning("No --teacher given: synthetic images are NOT quality-filtered.")

    out_root = ensure_dir(cfg.paths.synthetic_dir)
    tf = build_transform(size, False)
    rows, report = [], {}
    for cls in targets:
        idx = c2i[cls]
        out_dir = ensure_dir(out_root / cls)
        kept, tried, rejected_teacher, rejected_dup = 0, 0, 0, 0
        max_try = int(g.n_per_class * 4)
        while kept < g.n_per_class and tried < max_try:
            bsz = min(32, max_try - tried)
            labels = torch.full((bsz,), idx, device=device, dtype=torch.long)
            imgs = diffusion.ddim_sample(model, labels, cfg.diffusion.image_size, g.ddim_steps, g.guidance_scale, generator=gen)
            tried += bsz
            arr = ((imgs + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            pil = [Image.fromarray(a).resize((size, size), Image.BICUBIC) for a in arr]
            probs = np.ones(bsz)
            sims = np.zeros(bsz)
            if teacher is not None:
                x = torch.stack([tf(image=np.asarray(p))["image"] for p in pil]).to(device)
                with torch.no_grad():
                    probs = torch.softmax(teacher(x)[0] / 1.0, dim=1)[:, idx].cpu().numpy()
                    f = F.normalize(teacher.features(x).float(), dim=1).cpu()
                sims = (f @ real_feats.T).max(1).values.numpy()
            for p, prob, sim in zip(pil, probs, sims):
                if kept >= g.n_per_class:
                    break
                if prob < g.teacher_min_prob:
                    rejected_teacher += 1
                    continue
                if sim >= g.near_duplicate_sim:
                    rejected_dup += 1
                    continue
                frac = lesion_fraction(np.asarray(p))
                healthy = cls.endswith("healthy")
                sev = 0 if healthy else severity_from_fraction(frac, cfg.severity.mild_max, cfg.severity.moderate_max)
                rel = f"{cls}/syn_{kept:05d}.png"
                p.save(out_root / rel)
                rows.append(
                    dict(path=rel, label=cls, is_healthy=healthy, split="train", lesion_frac=round(frac, 4),
                         severity=sev, source="synthetic", teacher_prob=round(float(prob), 4), nn_sim=round(float(sim), 4))
                )
                kept += 1
        report[cls] = dict(kept=kept, tried=tried, rejected_teacher=rejected_teacher, rejected_near_duplicate=rejected_dup,
                           acceptance_rate=round(kept / max(tried, 1), 3))
        log.info("%-30s kept %4d / tried %4d (teacher-rejected %d, duplicate-rejected %d)", cls, kept, tried, rejected_teacher, rejected_dup)

    pd.DataFrame(rows).to_csv(out_root / "manifest.csv", index=False)
    report["_meta"] = {"rare_signature": signature, "rare_keep_fraction": float(cfg.data.rare_keep_fraction),
                       "teacher_gate": bool(args.teacher), "ddpm": ddpm_path}
    save_json(report, out_root / "generation_report.json")
    log.info("Wrote %d synthetic images to %s", len(rows), out_root)


if __name__ == "__main__":
    main()
