# Full v6 experiment on Windows PowerShell: 4 configurations x 3 runs + 3 soft-recall runs.
# Resumable (every job uses --limit 0). Needs .env with OPENAI_API_KEY. No GPU needed.
#
#   powershell -ExecutionPolicy Bypass -File evaluation\run_v6.ps1
#   powershell -ExecutionPolicy Bypass -File evaluation\run_v6.ps1 -Only bge -Runs 1,2
param(
  [string]$Dataset = "ragas_eval_dataset_v6.json",
  [string]$Out = "evaluation/results/v6",
  [int[]]$Runs = @(1, 2, 3),
  [int]$Workers = 8,
  [ValidateSet("all", "minilm", "bge")] [string]$Only = "all",
  [string]$MiniLMModel = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  [string]$BgeModel = "BAAI/bge-m3"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Run-Job($label, $model, $embDir, $rrfFlag, $run, [string[]]$extra = @()) {
  $csv = "$Out/${label}_r$run.csv"
  Write-Host "`n=== $label run $run -> $csv"
  # RAGAS_AR_EMBEDDING_MODEL stays fixed (BGE) for every configuration (same ruler for answer relevancy).
  $env:EMBEDDING_MODEL = $model; $env:EMBEDDINGS_DIR = $embDir; $env:RAGAS_AR_EMBEDDING_MODEL = $BgeModel
  python -m evaluation.ragas_eval --dataset $Dataset --out $csv --mode full --limit 0 `
    --workers $Workers --run-id "r$run" --seed $run --judge-temperature 0 $rrfFlag @extra
  if ($LASTEXITCODE -ne 0) { throw "job $label r$run failed" }
}

if ($Only -in @("all", "minilm")) {
  if (-not (Test-Path "data/embeddings_minilm/beans.index")) {
    Write-Host "=== building MiniLM index in data/embeddings_minilm"
    $env:EMBEDDING_MODEL = $MiniLMModel; $env:EMBEDDINGS_DIR = "data/embeddings_minilm"
    python -m src.preprocessing.build_embeddings --force
    if ($LASTEXITCODE -ne 0) { throw "index build failed" }
  }
  foreach ($r in $Runs) {
    Run-Job "minilm_norrf" $MiniLMModel "data/embeddings_minilm" "--no-use-rrf" $r
    Run-Job "minilm_rrf"   $MiniLMModel "data/embeddings_minilm" "--use-rrf"    $r
  }
}

if ($Only -in @("all", "bge")) {
  foreach ($r in $Runs) {
    Run-Job "bge_norrf" $BgeModel "data/embeddings" "--no-use-rrf" $r
    Run-Job "bge_rrf"   $BgeModel "data/embeddings" "--use-rrf"    $r
    Write-Host "`n=== soft recall run $r"
    $env:EMBEDDING_MODEL = $BgeModel; $env:EMBEDDINGS_DIR = "data/embeddings"; $env:RAGAS_AR_EMBEDDING_MODEL = $BgeModel
    python -m evaluation.ragas_eval --dataset $Dataset --out "$Out/bge_rrf_soft_r$r.csv" `
      --mode retrieval --soft --metrics context_recall --limit 0 --workers $Workers `
      --run-id "soft_r$r" --seed $r --use-rrf
    if ($LASTEXITCODE -ne 0) { throw "soft run $r failed" }
  }
}

Write-Host "`n=== aggregate"
$args = @("-m", "evaluation.aggregate_runs")
$minilmNo = Get-ChildItem "$Out/minilm_norrf_r*.csv" -ErrorAction SilentlyContinue | % FullName
$minilmYes = Get-ChildItem "$Out/minilm_rrf_r*.csv" -ErrorAction SilentlyContinue | % FullName
if ($minilmNo) { $args += @("--run", "MiniLM") + $minilmNo }
if ($minilmYes) { $args += @("--run", "MiniLM+RRF") + $minilmYes }
$args += @("--run", "BGE-M3") + (Get-ChildItem "$Out/bge_norrf_r*.csv" | % FullName)
$args += @("--run", "BGE-M3+RRF") + (Get-ChildItem "$Out/bge_rrf_r*.csv" | % FullName)
$args += @("--soft-file") + (Get-ChildItem "$Out/bge_rrf_soft_r*.csv" | % FullName)
$args += @("--soft-label", "BGE-M3+RRF", "--baseline", "BGE-M3", "--metric-table", "BGE-M3+RRF", "--out", "$Out/tables")
python @args

Write-Host "`n=== annotation sample (from bge_rrf run 1)"
python -m evaluation.annotation.sample_cases --contexts "$Out/bge_rrf_r1_contexts.jsonl" `
  --out evaluation/annotation/annotation_sample.json --per-stratum 10 --seed 42
Write-Host "Open evaluation\annotation\annotate.html, load annotation_sample.json, label, export labels_<name>.json"
