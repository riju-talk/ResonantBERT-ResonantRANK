#!/usr/bin/env bash
# =============================================================================
# ResonantBERT v3 ablation study — cloud runner (Linux, 1..N NVIDIA GPUs)
#
# Runs 16 trainings (BBC -> CNN zero-shot), spread round-robin over all GPUs:
#   random split  : full, no_rank, no_interaction, no_token, plain_coral, plain_mse  x seeds 42, 43
#   temporal split: full, plain_coral                                               x seeds 42, 43
# then aggregates everything with ablation_report.py and packs the results.
#
# Usage (from a fresh cloud machine):
#   git clone https://github.com/riju-talk/ResonantBERT-ResonantRANK.git && cd ResonantBERT-ResonantRANK
#   bash cloud_run_ablations.sh
#
# Knobs (environment variables):
#   SEEDS="42 43"     seeds to run
#   GPUS="0 1"        GPU ids to use (default: all visible)
#   MICRO=8 ACCUM=1   lists per forward pass x accumulation. Keep MICRO*ACCUM = 8 so every
#                     run matches the reference recipe (8 lists of 8 articles per optimizer step).
#                     24 GB+ GPU: MICRO=8 ACCUM=1 | 16 GB: MICRO=4 ACCUM=2 | 8 GB: MICRO=2 ACCUM=4
#   TRAIN_GROUPS=5000 training lists per run (5000 x 8 = 40k BBC articles, 1 epoch)
#   ONLY="random:full temporal:full"   run just these split:variant pairs
#
# Resumable: a run whose eval_results/ablations/<split>/<variant>_seed<seed>/metrics.json
# exists is skipped, so re-running the script continues where it stopped.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43}"
MICRO="${MICRO:-8}"
ACCUM="${ACCUM:-1}"
TRAIN_GROUPS="${TRAIN_GROUPS:-5000}"
PY="${PY:-python}"
LOG_DIR="runs/ablations/logs"
mkdir -p "$LOG_DIR"

if [ $((MICRO * ACCUM)) -ne 8 ]; then
  echo "WARNING: MICRO*ACCUM=$((MICRO * ACCUM)) (reference recipe uses 8 lists per step)"
fi

# ---------- 1. data (only the LFS files the study needs) ----------
echo "== [1/5] Fetching data"
git lfs install --local >/dev/null
git lfs pull --include="bbc_articles/track_a_all.csv,cnn_mass_dataset/cnn_labels.csv"
for f in bbc_articles/track_a_all.csv cnn_mass_dataset/cnn_labels_with_content.csv; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 1000000 ] || { echo "ERROR: $f missing or still an LFS pointer"; exit 1; }
done

# ---------- 2. split + baseline files produced locally (must be committed) ----------
echo "== [2/5] Checking evaluation splits"
REQUIRED=(
  eval_results/cnn/cnn_split.json
  eval_results/cnn/cnn_split_temporal.json
  eval_results/cnn/test_scores_cnn_baselines.csv
  eval_results/ablations/random/baselines/test_scores.csv
  eval_results/ablations/temporal/baselines/test_scores.csv
)
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "ERROR: $f not found. Commit eval_results/cnn/ and eval_results/ablations/*/baselines/ from the local machine."; exit 1; }
done

# ---------- 3. environment ----------
echo "== [3/5] Python environment"
if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "Installing CUDA PyTorch"
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
fi
$PY -m pip install -q -r scripts/requirements-ablation.txt
$PY - <<'EOF'
import torch, transformers, peft
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/2**30:.0f} GB, bf16={torch.cuda.is_bf16_supported()}")
EOF
# Download ModernBERT once, before parallel workers race for the cache
$PY -c "from transformers import AutoModelForMaskedLM, AutoTokenizer as T; T.from_pretrained('answerdotai/ModernBERT-base'); AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-base')" >/dev/null

# ---------- 4. job queue, round-robin over GPUs ----------
echo "== [4/5] Training"
if [ -z "${GPUS:-}" ]; then GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '); fi
read -r -a GPU_LIST <<< "$GPUS"
JOBS=()
for seed in $SEEDS; do
  for v in full plain_coral plain_mse no_rank no_interaction no_token; do JOBS+=("random:$v:$seed"); done
done
for seed in $SEEDS; do
  for v in full plain_coral; do JOBS+=("temporal:$v:$seed"); done
done
if [ -n "${ONLY:-}" ]; then
  FILTERED=(); for j in "${JOBS[@]}"; do for o in $ONLY; do [[ "$j" == "$o:"* ]] && FILTERED+=("$j"); done; done
  JOBS=("${FILTERED[@]}")
fi
echo "${#JOBS[@]} jobs on ${#GPU_LIST[@]} GPU(s): ${GPU_LIST[*]}"

worker() {   # gpu_id job...
  local gpu=$1; shift
  for job in "$@"; do
    IFS=: read -r split variant seed <<< "$job"
    local tag="${split}_${variant}_seed${seed}"
    if [ -f "eval_results/ablations/$split/${variant}_seed${seed}/metrics.json" ]; then
      echo "[gpu$gpu $(date +%H:%M)] SKIP  $tag (done)"; continue
    fi
    echo "[gpu$gpu $(date +%H:%M)] START $tag"
    if CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY scripts/ablation_study.py --split "$split" --variant "$variant" \
         --seed "$seed" --train-groups "$TRAIN_GROUPS" --micro-groups "$MICRO" --accum "$ACCUM" \
         > "$LOG_DIR/$tag.log" 2>&1; then
      echo "[gpu$gpu $(date +%H:%M)] DONE  $tag  $(grep -E 'CNN \[(value_sum|scalar)' "$LOG_DIR/$tag.log" | head -1)"
    else
      echo "[gpu$gpu $(date +%H:%M)] FAIL  $tag  (tail of $LOG_DIR/$tag.log:)"; tail -5 "$LOG_DIR/$tag.log"
    fi
  done
}

PIDS=()
for i in "${!GPU_LIST[@]}"; do
  MINE=()
  for j in "${!JOBS[@]}"; do [ $((j % ${#GPU_LIST[@]})) -eq "$i" ] && MINE+=("${JOBS[$j]}"); done
  [ ${#MINE[@]} -gt 0 ] && { worker "${GPU_LIST[$i]}" "${MINE[@]}" & PIDS+=($!); }
done
for p in "${PIDS[@]}"; do wait "$p"; done

# ---------- 5. aggregate + pack ----------
echo "== [5/5] Report"
$PY scripts/ablation_report.py | tee runs/ablations/ablation_report.txt
N_DONE=$(find eval_results/ablations -name metrics.json -path '*_seed*' | wc -l)
echo "Finished runs: $N_DONE / ${#JOBS[@]}"
tar -czf ablation_results.tar.gz eval_results/ablations runs/ablations/logs runs/ablations/ablation_report.txt \
    --exclude='eval_results/ablations/cache'
echo "Packed: $(pwd)/ablation_results.tar.gz  (download this; checkpoints stay in runs/ablations/<split>/*/best.pt)"
