#!/usr/bin/env bash
# Full v6 experiment: 4 system configurations x 3 runs (mode full) + 3 soft-recall runs.
# Resumable: every job uses --limit 0, so re-running the script skips finished cases.
#
#   bash evaluation/run_v6.sh            # everything
#   ONLY=bge bash evaluation/run_v6.sh   # skip the MiniLM configurations
#   RUNS="1 2" bash evaluation/run_v6.sh # fewer repetitions
#
# Needs .env with ANTHROPIC_API_KEY (or OPENAI_API_KEY). GPU is optional: query encoding of 600 questions
# on CPU takes a few minutes; only building a new FAISS index is slow on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-ragas_eval_dataset_v6.json}
OUT=${OUT:-evaluation/results/v6}
RUNS=${RUNS:-"1 2 3"}
WORKERS=${WORKERS:-8}
ONLY=${ONLY:-all}
MINILM_MODEL=${MINILM_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}
BGE_MODEL=${BGE_MODEL:-BAAI/bge-m3}
mkdir -p "$OUT"

run_job () {  # label model emb_dir rrf_flag run extra...
  local label=$1 model=$2 emb_dir=$3 rrf=$4 run=$5; shift 5
  local csv="$OUT/${label}_r${run}.csv"
  echo; echo "=== $label run $run -> $csv"
  # RAGAS_AR_EMBEDDING_MODEL stays fixed (BGE) for every configuration, so answer
  # relevancy is measured with the same ruler whichever retriever is under test.
  EMBEDDING_MODEL="$model" EMBEDDINGS_DIR="$emb_dir" RAGAS_AR_EMBEDDING_MODEL="$BGE_MODEL" \
  python -m evaluation.ragas_eval --dataset "$DATASET" --out "$csv" --mode full --limit 0 \
    --workers "$WORKERS" --run-id "r${run}" --seed "$run" --judge-temperature 0 $rrf "$@"
}

if [[ "$ONLY" == "all" || "$ONLY" == "minilm" ]]; then
  # One-time index for MiniLM in its own folder (dim 384). ~10 min on CPU.
  if [[ ! -f data/embeddings_minilm/beans.index ]]; then
    echo "=== building MiniLM index in data/embeddings_minilm"
    EMBEDDING_MODEL="$MINILM_MODEL" EMBEDDINGS_DIR=data/embeddings_minilm python -m src.preprocessing.build_embeddings --force
  fi
  for r in $RUNS; do
    run_job minilm_norrf "$MINILM_MODEL" data/embeddings_minilm --no-use-rrf "$r"
    run_job minilm_rrf   "$MINILM_MODEL" data/embeddings_minilm --use-rrf    "$r"
  done
fi

if [[ "$ONLY" == "all" || "$ONLY" == "bge" ]]; then
  for r in $RUNS; do
    run_job bge_norrf "$BGE_MODEL" data/embeddings --no-use-rrf "$r"
    run_job bge_rrf   "$BGE_MODEL" data/embeddings --use-rrf    "$r"
    # Soft recall on the main configuration (embedding-only, cheap).
    EMBEDDING_MODEL="$BGE_MODEL" EMBEDDINGS_DIR=data/embeddings RAGAS_AR_EMBEDDING_MODEL="$BGE_MODEL" \
    python -m evaluation.ragas_eval --dataset "$DATASET" --out "$OUT/bge_rrf_soft_r${r}.csv" \
      --mode retrieval --soft --metrics context_recall --limit 0 --workers "$WORKERS" \
      --run-id "soft_r${r}" --seed "$r" --use-rrf
  done
fi

echo; echo "=== aggregate"
python -m evaluation.aggregate_runs \
  --run "MiniLM"     $(ls $OUT/minilm_norrf_r*.csv 2>/dev/null) \
  --run "MiniLM+RRF" $(ls $OUT/minilm_rrf_r*.csv   2>/dev/null) \
  --run "BGE-M3"     $(ls $OUT/bge_norrf_r*.csv) \
  --run "BGE-M3+RRF" $(ls $OUT/bge_rrf_r*.csv) \
  --soft-file $(ls $OUT/bge_rrf_soft_r*.csv) --soft-label "BGE-M3+RRF" \
  --baseline "BGE-M3" --metric-table "BGE-M3+RRF" --out "$OUT/tables"

echo; echo "=== annotation sample (from bge_rrf run 1)"
python -m evaluation.annotation.sample_cases --contexts "$OUT/bge_rrf_r1_contexts.jsonl" \
  --out evaluation/annotation/annotation_sample.json --per-stratum 10 --seed 42
echo "Open evaluation/annotation/annotate.html, load annotation_sample.json, label, export labels_<name>.json"
