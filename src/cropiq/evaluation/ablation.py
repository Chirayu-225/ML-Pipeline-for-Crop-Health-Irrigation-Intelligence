"""Collate classifier runs into markdown tables (the 'does diffusion augmentation help?' ablation).

Single runs:
    python -m cropiq.evaluation.ablation --runs clf_real clf_balanced clf_synthetic
Several seeds per arm (runs named <arm>_s<seed>, e.g. clf_real_s0 ... clf_synthetic_s2):
    python -m cropiq.evaluation.ablation --arms clf_real clf_balanced clf_synthetic --seeds 0 1 2

The seeded table reports mean +- std over seeds, plus the PAIRED per-seed difference of each arm against a
reference arm (default clf_balanced). Same seed = same initialisation and data order, so the paired
difference cancels much of the seed noise; it is the number to quote.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cropiq.config import base_parser, load_config
from cropiq.utils import ensure_dir


def build_table(runs_dir: Path, runs: list[str]) -> str:
    rows, rare_names = [], []
    for r in runs:
        path = runs_dir / r / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rare_names = m.get("rare_classes", rare_names)
        rows.append((r, m))
    if not rows:
        return "No metrics found."
    header = ["run", "synthetic", "balanced sampler", "test acc", "macro-F1", "rare macro-F1", "ECE (T-scaled)"] + [f"F1 {c}" for c in rare_names]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r, m in rows:
        t = m["test"]
        f1s = [f"{t['per_class_f1'].get(c, float('nan')):.3f}" for c in rare_names]
        rare = t["rare_macro_f1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    r,
                    str(m["use_synthetic"]),
                    str(m["balanced_sampler"]),
                    f"{t['accuracy']:.4f}",
                    f"{t['macro_f1']:.4f}",
                    "n/a" if rare is None else f"{rare:.4f}",
                    f"{t['ece']:.4f}",
                    *f1s,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(values: list[float]) -> str:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return "n/a"
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if v.size > 1 else f"{v[0]:.4f}"


def _metric(m: dict, key: str, rare_class: str | None = None):
    t = m["test"]
    if rare_class is not None:
        return t["per_class_f1"].get(rare_class, float("nan"))
    return t[key] if t.get(key) is not None else float("nan")


def build_seed_table(runs_dir: Path, arms: list[str], seeds: list[int], reference: str = "clf_balanced") -> str:
    loaded: dict[str, dict[int, dict]] = {}
    rare_names: list[str] = []
    meta = set()
    for arm in arms:
        for s in seeds:
            path = runs_dir / f"{arm}_s{s}" / "metrics.json"
            if path.exists():
                m = json.loads(path.read_text())
                loaded.setdefault(arm, {})[s] = m
                rare_names = m.get("rare_classes", rare_names)
                meta.add((m.get("rare_keep_fraction"), m.get("rare_signature")))
    if not loaded:
        return "No seeded metrics found."
    cols = [("test acc", "accuracy", None), ("macro-F1", "macro_f1", None), ("rare macro-F1", "rare_macro_f1", None),
            ("ECE (T-scaled)", "ece", None)] + [(f"F1 {c}", None, c) for c in rare_names]
    lines = []
    if len(meta) > 1:
        lines.append(f"> WARNING: runs mix different rare subsets / keep-fractions: {sorted(meta, key=str)}\n")
    else:
        kf, sig = next(iter(meta))
        lines.append(f"Rare keep-fraction {kf}, rare subset {sig}; mean ± std over seeds.\n")
    header = ["arm", "seeds"] + [c[0] for c in cols]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm, by_seed in loaded.items():
        row = [arm, str(len(by_seed))]
        for _, key, rc in cols:
            row.append(_fmt([_metric(m, key, rc) for m in by_seed.values()]))
        lines.append("| " + " | ".join(row) + " |")

    if reference in loaded:
        lines += ["", f"Paired difference vs `{reference}` (same seed), mean ± std:", ""]
        dheader = ["arm", "paired seeds", "Δ macro-F1", "Δ rare macro-F1"] + [f"Δ F1 {c}" for c in rare_names]
        lines += ["| " + " | ".join(dheader) + " |", "|" + "---|" * len(dheader)]
        ref = loaded[reference]
        for arm, by_seed in loaded.items():
            if arm == reference:
                continue
            common = sorted(set(by_seed) & set(ref))
            if not common:
                continue
            row = [arm, str(len(common))]
            for key, rc in [("macro_f1", None), ("rare_macro_f1", None)] + [(None, c) for c in rare_names]:
                row.append(_fmt([_metric(by_seed[s], key, rc) - _metric(ref[s], key, rc) for s in common]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("\nIf a Δ interval comfortably includes 0, the honest conclusion is 'no measurable difference'.")
    return "\n".join(lines)


def main() -> None:
    parser = base_parser("Build the augmentation ablation table")
    parser.add_argument("--runs", nargs="+", default=["clf_real", "clf_balanced", "clf_synthetic"])
    parser.add_argument("--arms", nargs="+", default=None, help="arm names for a seeded ablation (<arm>_s<seed> runs)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--reference", default="clf_balanced", help="arm to compute paired differences against")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    runs_dir = Path(cfg.paths.runs_dir)
    if args.arms and args.seeds:
        table = build_seed_table(runs_dir, args.arms, args.seeds, args.reference)
        out = ensure_dir(runs_dir) / "ablation_seeds.md"
        out.write_text(table + "\n")
        print(table)
        print(f"\nSaved to {out}")
        return
    table = build_table(runs_dir, args.runs)
    out = ensure_dir(runs_dir) / "ablation.md"
    out.write_text(table + "\n")
    print(table)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
