#!/usr/bin/env bash
# Seeded diffusion-augmentation ablation at a low rare-class keep-fraction.
#
#   bash scripts/run_ablation_seeds.sh                      # defaults below
#   KEEP=0.02 SEEDS="0 1 2 3 4" bash scripts/run_ablation_seeds.sh
#   GATE_ABLATION=1 bash scripts/run_ablation_seeds.sh      # also: synthetic images WITHOUT the teacher gate
#
# What is fixed across seeds: the manifest split and the rare-class subset (data.rare_seed), so the DDPM
# and every classifier see exactly the same real rare images. What varies per seed: classifier init + data order.
# The DDPM and the synthetic set are built ONCE per keep-fraction (they are the expensive part).
# Requires: artifacts/manifest.csv (make data). Each keep-fraction gets its own runs folder.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/default.yaml}
PY=${PY:-python}
KEEP=${KEEP:-0.05}
SEEDS=${SEEDS:-"0 1 2"}
GATE_ABLATION=${GATE_ABLATION:-0}

TAG="keep${KEEP}"
RUNS="artifacts/runs_${TAG}"
SYN="artifacts/synthetic_${TAG}"
SYN_NOGATE="artifacts/synthetic_${TAG}_nogate"
BASE=(paths.runs_dir=$RUNS paths.synthetic_dir=$SYN data.rare_keep_fraction=$KEEP)
FIRST_SEED=${SEEDS%% *}

echo ">>> keep-fraction $KEEP | seeds: $SEEDS | runs -> $RUNS"
for s in $SEEDS; do
  $PY -m cropiq.training.train_classifier --config "$CFG" --run-name clf_real_s$s --override "${BASE[@]}" seed=$s
  $PY -m cropiq.training.train_classifier --config "$CFG" --run-name clf_balanced_s$s --balanced-sampler --override "${BASE[@]}" seed=$s
done

# Diffusion: trained once on the same (fixed) rare subset. Resumable if interrupted.
if [ ! -f "$RUNS/ddpm/best.pt" ]; then
  $PY -m cropiq.training.train_ddpm --config "$CFG" --override "${BASE[@]}" --resume
fi
$PY -m cropiq.training.generate_synthetic --config "$CFG" --teacher "$RUNS/clf_real_s$FIRST_SEED/best.pt" --override "${BASE[@]}"
ARMS="clf_real clf_balanced clf_synthetic"
if [ "$GATE_ABLATION" = "1" ]; then
  $PY -m cropiq.training.generate_synthetic --config "$CFG" --ddpm "$RUNS/ddpm/best.pt" \
      --override "${BASE[@]}" paths.synthetic_dir=$SYN_NOGATE
  ARMS="$ARMS clf_synthetic_nogate"
fi

for s in $SEEDS; do
  $PY -m cropiq.training.train_classifier --config "$CFG" --run-name clf_synthetic_s$s --use-synthetic --balanced-sampler \
      --override "${BASE[@]}" seed=$s
  if [ "$GATE_ABLATION" = "1" ]; then
    $PY -m cropiq.training.train_classifier --config "$CFG" --run-name clf_synthetic_nogate_s$s --use-synthetic --balanced-sampler \
        --override "${BASE[@]}" paths.synthetic_dir=$SYN_NOGATE seed=$s
  fi
done

$PY -m cropiq.evaluation.ablation --config "$CFG" --override "${BASE[@]}" --arms $ARMS --seeds $SEEDS
