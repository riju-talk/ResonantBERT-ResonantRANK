#!/usr/bin/env bash
# Runs the ablation queue sequentially on one GPU. Resumable: finished runs
# (eval_results/ablations/<split>/<variant>_seed<seed>/metrics.json) are skipped.
#   bash run_ablations.sh                # everything
#   SEEDS="42" bash run_ablations.sh     # one seed only
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"
SEEDS="${SEEDS:-42 43}"
mkdir -p runs/ablations/logs

run() {   # split variant seed
  local log="runs/ablations/logs/$1_$2_seed$3.log"
  echo "[$(date +%H:%M)] START $1 $2 seed$3"
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 "$PY" scripts/ablation_study.py --split "$1" --variant "$2" --seed "$3" > "$log" 2>&1 \
    && echo "[$(date +%H:%M)] DONE  $1 $2 seed$3: $(grep -E 'CNN \[(value_sum|scalar)' "$log" | head -1)" \
    || echo "[$(date +%H:%M)] FAIL  $1 $2 seed$3 (see $log)"
}

for seed in $SEEDS; do
  for v in full plain_coral plain_mse no_rank no_interaction no_token; do run random "$v" "$seed"; done
done
for seed in $SEEDS; do
  for v in full plain_coral; do run temporal "$v" "$seed"; done
done
echo "[$(date +%H:%M)] QUEUE FINISHED"
