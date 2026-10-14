"""Collate classifier runs into one markdown table (the 'does diffusion augmentation help?' ablation).

    python -m cropiq.evaluation.ablation --runs clf_real clf_balanced clf_synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

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


def main() -> None:
    parser = base_parser("Build the augmentation ablation table")
    parser.add_argument("--runs", nargs="+", default=["clf_real", "clf_balanced", "clf_synthetic"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    runs_dir = Path(cfg.paths.runs_dir)
    table = build_table(runs_dir, args.runs)
    out = ensure_dir(runs_dir) / "ablation.md"
    out.write_text(table + "\n")
    print(table)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
