"""Run the irrigation comparison (SNN, GRU, rule, tree, noise-augmented tree) over several seeds.

    python -m cropiq.evaluation.snn_seeds --seeds 0 1 2
    python -m cropiq.evaluation.snn_seeds --seeds 0 1 2 --skip-train      # only re-aggregate existing runs

Each seed re-simulates the fields AND re-initialises every model (runs are saved as <runs_dir>/snn_s<seed>), so the
spread covers both data and training randomness. Writes <runs_dir>/snn_seeds.md with mean ± std per model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cropiq.config import base_parser, load_config
from cropiq.utils import ensure_dir

METRICS = [("macro_f1", "macro-F1"), ("alert_recall", "ALERT recall"), ("macro_f1_noise_x3", "F1 @ noise x3"),
           ("latency_median_ms", "CPU ms/decision")]


def _fmt(values: list[float], digits: int) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}" if v.size > 1 else f"{v[0]:.{digits}f}"


def aggregate(summaries: dict[int, dict]) -> str:
    """summaries: {seed: metrics['summary']} -> markdown table of mean ± std per model."""
    seeds = sorted(summaries)
    models = list(summaries[seeds[0]])
    lines = [f"Seeds: {', '.join(map(str, seeds))} (mean ± std; each seed = new simulated fields + new initialisation)", "",
             "| model | " + " | ".join(label for _, label in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for m in models:
        cells = []
        for key, _ in METRICS:
            vals = [summaries[s][m][key] for s in seeds if m in summaries[s]]
            cells.append(_fmt(vals, 3 if key == "latency_median_ms" else 4))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = base_parser("Seeded irrigation-model comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-train", action="store_true", help="only aggregate existing snn_s<seed> runs")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    runs_dir = Path(cfg.paths.runs_dir)

    if not args.skip_train:
        for s in args.seeds:
            cmd = [sys.executable, "-m", "cropiq.training.train_snn", "--config", args.config, "--run-name", f"snn_s{s}",
                   "--override", *args.override, f"seed={s}"]
            print(">>>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    summaries = {}
    for s in args.seeds:
        path = runs_dir / f"snn_s{s}" / "metrics.json"
        if path.exists():
            summaries[s] = json.loads(path.read_text())["summary"]
    if not summaries:
        raise SystemExit("No snn_s<seed> runs found.")
    table = aggregate(summaries)
    out = ensure_dir(runs_dir) / "snn_seeds.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
