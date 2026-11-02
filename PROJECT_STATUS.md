# PROJECT_STATUS — coffee-rag paper

Updated: 2026-11-01

## Target and framing

- Venue: not settled. Realistic first target: a regional applied conference (RIVF, SoICT, KSE). Q2 journal only if the human-validation study (B2) shows the soft metric tracks human judgement better than Ragas context recall.
- Two candidate framings, decided by B2:
  1. *Evaluation paper*: reference-based context recall underestimates retrieval for recommendation-style queries with many equally valid answers; a set-valued benchmark plus an identity-anchored soft recall validated against human labels.
  2. *System paper*: hybrid retrieval (structured filter + dense + RRF + post-RRF prioritisation) for a bilingual coffee catalogue, with an honest evaluation; Ragas used only as a tool.
- Metric name in prose: **identity-anchored soft recall** (not "semantic recall", taken by Kuffo et al., SIGIR 2026).

## Source of truth

- Code and data: `D:\github\coffee-rag` (git, branch main). No manuscript exists yet.
- Benchmark for any published number: `ragas_eval_dataset_v6.json` (template, retriever-independent, `generate_dataset_v6.py --seed 42`). v5 (`ragas_eval_dataset.json`) is retrieval-grounded and circular; keep only for debugging and as an illustration of the bias.
- Results: `evaluation/results/v6/*.csv` (per run), `evaluation/results/v6/tables/` (aggregate), `evaluation/results/tables_v5/` (old v5 numbers, single run each).

## Review verdict (2026-11-01)

Reject-and-resubmit at Q1/Q2 as a "Ragas improvement" paper. Blockers:
- B1 ground truth generated from the system's own retrieval (fixed by v6).
- B2 soft recall has no human validation (tooling ready, labels pending).
- B3 system comparison and metric comparison were mixed in one table (fixed by `aggregate_runs.py`).
Majors: single runs with a non-deterministic judge (M1, fixed by `--run-id/--seed/--judge-temperature 0` and 3 runs), unexplained thresholds (M2, `sweep_soft_recall.py`), metric name collision (M3), stale numbers in README (M4, fixed).

## Done this session

- `evaluation/generate_dataset_v6.py` + `ragas_eval_dataset_v6.json` (600 cases, 300 vi / 300 en, relevance sets stored).
- `evaluation/ragas_eval.py`: ID-based Hit@k / Recall@k / Precision@k / MRR, retrieved-oriented Soft Recall@k when `relevant_ids` exist, `--run-id`, `--seed`, `--judge-temperature` (default 0), `--early-stop` auto-off for v6, `<out>_contexts.jsonl` dump.
- `evaluation/aggregate_runs.py` (system table, metric table, Wilcoxon + bootstrap CI, LaTeX).
- `evaluation/sweep_soft_recall.py`, `evaluation/annotation/{sample_cases.py, annotate.html, analyze_annotations.py}`.
- `evaluation/run_v6.sh`, `evaluation/run_v6.ps1`; `EMBEDDINGS_DIR` override in `semantic_search.py` and `build_embeddings.py` so MiniLM and BGE-M3 indices coexist.
- README and evaluation/README corrected (language split 492/108 for v5, dataset size 600, v12 table marked historical, v6 documented).
- v5 numbers re-tabulated in `evaluation/results/tables_v5/` (paired tests included).

## Provider

- User has only an Anthropic API key (2026-11-01). Pipeline LLM, Ragas judge and embeddings all run without OpenAI: `LLM_PROVIDER=anthropic`, `RAGAS_JUDGE_PROVIDER=anthropic`, judge model `claude-haiku-5-5`, embeddings `BAAI/bge-m3` local. Adapter in `src/generation/llm_client.py`; judge selection in `ragas_eval.py` (`--judge-provider`).
- Consequence for the paper: judge = Claude Haiku for every v6 number; the old v5 numbers (gpt-4o-mini judge) are not comparable and stay out.
- Bug fixed on the way: `entity_extractor.py` read `OLLAMA_MODEL` before `OPENAI_MODEL`, so with `OLLAMA_MODEL` set in `.env` the LLM call failed silently and extraction fell back to rules. Unknown whether the v5 runs were affected.

## Compute

- Runs go on Kaggle (free GPU, 12 h/session): `evaluation/kaggle/coffee_rag_kaggle.ipynb`, data as Kaggle Dataset `coffee-rag-data` (`evaluation/kaggle/pack_data.py`), guide in `evaluation/kaggle/README.md`. Judge `claude-sonnet-5-5`, pipeline `claude-haiku-5-5`, AR embedding fixed to bge-m3 for every configuration.

## Incident 2026-11-02

- First Kaggle run with judge `claude-sonnet-5-5` exhausted the $500 monthly API budget after about 11 of 12 full runs (~5.2 h, stopped at case 534 of a run with exit 1). Cause: my cost estimate ($5/run) was for gpt-4o-mini; a Sonnet judge costs about $45/run. Measured, now documented in evaluation/README.md and the Kaggle README.
- Results of the completed runs are in the failed version's Output under `coffee-rag/evaluation/results/v6/`. That Output also contains `coffee-rag/.env` with the API key (packaging cell never ran): keep the notebook private, delete the file after download, rotate the key if the notebook was ever public.
- Soft recall no longer needs API runs: computed offline from `*_contexts.jsonl` (`SOFT_ONLY=1`).
- Remaining: finish the interrupted run (~$45) when the budget resets, or report with the runs that are complete.

## Open items (owner: user)

1. Create `.env` from `.env.example` with `ANTHROPIC_API_KEY` (and `pip install -r requirements.txt` for the `anthropic` SDK).
2. Run `python -m evaluation.ragas_eval --dataset ragas_eval_dataset_v6.json --mode full --limit 20 --out evaluation/results/v6/smoke.csv` first, check the Anthropic usage console for the cost of 20 cases, multiply by 30 (600 cases) and by 12 runs before launching `bash evaluation/run_v6.sh` (or the `.ps1`). Resumable.
3. Label `evaluation/annotation/annotation_sample.json` with `annotate.html`, two annotators, export `labels_<name>.json` into `evaluation/annotation/`.
4. Run `analyze_annotations.py` and `sweep_soft_recall.py` (commands in evaluation/README.md).
5. Send the resulting `REPORT.md` files back; the framing decision and the manuscript draft follow from them.

## Decisions taken

- v6 questions are template-based, no LLM paraphrase (user choice, 2026-11-01).
- Two annotators (user + one team member).
- Full 3 runs × 4 configurations plus 3 soft runs.
- No GPU needed; MiniLM index built once into `data/embeddings_minilm/`.

## Numbers that changed

- v5 language split: documented 60/40, measured 492 vi / 108 en.
- Dataset size: 600 (README said 500 in one place).
