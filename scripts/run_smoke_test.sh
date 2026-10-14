#!/usr/bin/env bash
# End-to-end CPU smoke test on procedurally generated toy leaves + simulated sensors.
# It verifies that every stage runs and hands artifacts to the next one. Metrics here are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/smoke.yaml
PY=${PY:-python}

rm -rf artifacts_smoke
$PY -m cropiq.data.toy --out artifacts_smoke/toy_plantvillage
$PY -m cropiq.data.manifest --config $CFG
$PY -m cropiq.training.train_classifier --config $CFG --run-name clf_real
$PY -m cropiq.training.train_classifier --config $CFG --run-name clf_balanced --balanced-sampler
$PY -m cropiq.training.train_ddpm --config $CFG
$PY -m cropiq.training.generate_synthetic --config $CFG --teacher artifacts_smoke/runs/clf_real/best.pt
$PY -m cropiq.training.train_classifier --config $CFG --run-name clf_synthetic --use-synthetic --balanced-sampler
$PY -m cropiq.training.train_vae --config $CFG
$PY -m cropiq.training.train_snn --config $CFG
$PY -m cropiq.evaluation.ablation --config $CFG
$PY -m cropiq.edge.export_snn --ckpt artifacts_smoke/runs/snn/best.pt --out artifacts_smoke/edge

# Fused advisory from the command line
$PY - <<'EOF'
import json, pandas as pd
from cropiq.config import load_config
from cropiq.data.sensors import simulate_field, FEATURES
cfg = load_config("configs/smoke.yaml")
df = simulate_field(3, cfg)
df.iloc[200:224][FEATURES].to_csv("artifacts_smoke/sample_sensors.csv", index=False)
EOF
LEAF=$(ls artifacts_smoke/toy_plantvillage/color/Tomato___Late_blight/*.jpg | head -1)
$PY -m cropiq.pipeline.inference --config $CFG --image "$LEAF" --sensors artifacts_smoke/sample_sensors.csv
echo "SMOKE TEST PASSED"
