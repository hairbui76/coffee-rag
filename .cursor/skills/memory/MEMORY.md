# Project Memory — Coffee RAG

> Auto-maintained by the memory skill. Newest entries first.

---

### 2026-04-26 — Added Ragas evaluation harness

- **Category:** progress
- **Details:** Added `evaluation/ragas_eval.py` to run Ragas metrics over `ragas_eval_dataset.json`, with retrieval-only and full RAG modes plus CSV output. Added `ragas>=0.4.3` and evaluator env vars; smoke-tested one retrieval sample (`context_precision=0.303`).
- **Files:** `evaluation/ragas_eval.py`, `requirements.txt`, `.env.example`

### 2026-04-16 — Implement Pydantic structured output for LLM responses

- **Category:** decision
- **Details:** Replaced free-form text LLM output with Pydantic schema (`CoffeeResponse` → `summary`, `products[]`, `articles[]`) using OpenAI `client.beta.chat.completions.parse()`. Guarantees 100% consistent JSON structure. Removed streaming (`generate_stream`) since structured parse doesn't support it; trade-off accepted for format reliability. Streamlit UI now renders structured fields as formatted markdown. Prompt templates simplified (schema replaces format instructions).
- **Files:** `src/generation/schemas.py` (new), `src/generation/llm_client.py`, `src/generation/prompt_templates.py`, `src/pipeline.py`, `app/streamlit_app.py`

### 2026-04-16 — Fix bilingual response inconsistency

- **Category:** bugfix
- **Details:** LLM was mixing Vietnamese and English in responses because system prompt had a weak "respond in same language" instruction and all context data is English. Split into separate VI/EN system prompts + user templates. Added `detect_language()` using Vietnamese diacritics regex to auto-select the right prompt pair. Now queries with Vietnamese characters get fully Vietnamese prompts/instructions, English queries get English ones.
- **Files:** `src/generation/prompt_templates.py`

### 2026-04-16 — Suppress transformers/sentence-transformers warnings

- **Category:** config
- **Details:** Suppressed noisy `Accessing __path__` and `UNEXPECTED key` warnings from `transformers` and `sentence_transformers` libraries in `semantic_search.py` by setting their loggers to ERROR/WARNING level.
- **Files:** `src/retrieval/semantic_search.py`

### 2026-04-16 — Fix numpy array truth-value errors in UI & prompt templates

- **Category:** bugfix
- **Details:** `row.get("flavor_notes_clean") or []` crashes when the column holds a numpy array (ambiguous truth value). Added `_as_list()` and `_as_str()` helpers in `prompt_templates.py` to safely convert numpy arrays, NaN, and None. Reused them in `streamlit_app.py` expander section. Also fixed `NaT.strftime()` crash in `format_news_context`.
- **Files:** `src/generation/prompt_templates.py`, `app/streamlit_app.py`

### 2026-04-16 — Added .env support + debug logging for LLM client

- **Category:** config
- **Details:** Rewrote `llm_client.py` to load config from `.env` via `python-dotenv`. Supports `LLM_PROVIDER=openai` (with `OPENAI_API_KEY`) or `ollama` (with `OLLAMA_BASE_URL`). Added debug logging at every step: client init, request params, response status, error traces. Previously config was hardcoded with empty API key pointing to wrong base URL. Created `.env.example` with all variables documented.
- **Files:** `src/generation/llm_client.py`, `.env.example`, `requirements.txt`

### 2026-04-16 — Fixed ModuleNotFoundError for src package

- **Category:** bugfix
- **Details:** `streamlit run app/streamlit_app.py` failed with `ModuleNotFoundError: No module named 'src'` because the project root wasn't in `sys.path`. Added `sys.path.insert(0, ...)` at the top of `streamlit_app.py` to resolve parent directory.
- **Files:** `app/streamlit_app.py`

### 2026-04-16 — Full RAG system implemented

- **Category:** progress
- **Details:** Built complete RAG pipeline: data cleaning (14,537 beans + 1,947 articles -> 13,427 chunks), multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2, dim=384), FAISS indices, semantic search + structured filter + RRF reranking, intent classification + entity extraction (rule-based + LLM fallback), LLM generation via Gemma 4 E4B (Ollama), Streamlit chatbot UI with streaming. Supports Vietnamese + English queries.
- **Files:** `src/preprocessing/`, `src/retrieval/`, `src/generation/`, `src/query/`, `src/pipeline.py`, `app/streamlit_app.py`, `data/processed/`, `data/embeddings/`, `requirements.txt`

### 2026-04-16 — Removed all evaluation content from AGENTS.md

- **Category:** decision
- **Details:** Repo is purely a RAG system, not a research project. Removed entire section 5 (Evaluation: retrieval metrics, LLM-as-Judge, recommendation quality, ablation study), `evaluation/` from planned directory, evaluation dependencies (rouge-score, scikit-learn, rank-bm25), and "pseudo ground truth" references. Sections renumbered: old 6→5, old 7→6.
- **Files:** `AGENTS.md`

### 2026-04-16 — AGENTS.md updated to match actual structure

- **Category:** progress
- **Details:** Fixed title (removed "AGENTS_RAG.md" reference), corrected data file paths from `output/...` to root-level `coffee_beans.json` and `coffee_news.json`, split section 6 into "Hiện trạng" (actual 5-file structure) and "Kế hoạch" (planned directories). Removed phantom `data/raw/` and `output/` references.
- **Files:** `AGENTS.md`

### 2026-04-16 — Memory skill created

- **Category:** config
- **Details:** Added `.cursor/skills/memory/` with SKILL.md (instructions) and MEMORY.md (storage). This skill ensures project context persists across sessions.
- **Files:** `.cursor/skills/memory/SKILL.md`, `.cursor/skills/memory/MEMORY.md`

### 2026-04-16 — Project initialized with AGENTS.md

- **Category:** note
- **Details:** Project is a Coffee Specialty RAG chatbot. Knowledge base: 14,537 beans + 1,947 news articles. Architecture: Query Understanding → Structured Filter + Semantic Retrieval → Re-Ranking → LLM Generation. Roadmap has 12 phases (P0–P11), estimated 12-15 days. Tech stack: Python 3.10+, sentence-transformers, FAISS, LangChain, Streamlit.
- **Files:** `AGENTS.md`
