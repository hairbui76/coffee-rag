# ☕ Coffee Advisor — Specialty Coffee RAG System

A **Retrieval-Augmented Generation (RAG)** chatbot that recommends specialty coffee products and answers coffee-related questions. Built on a knowledge base of **14,537 coffee beans** and **1,947 news articles**, with bilingual support (Vietnamese + English).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Components](#system-components)
  - [M1: Query Understanding](#m1-query-understanding)
  - [M2A: Structured Filter](#m2a-structured-filter)
  - [M2B: Semantic Retrieval](#m2b-semantic-retrieval)
  - [M2C: Product Name Matching](#m2c-product-name-matching)
  - [M3: Re-Ranking & Fusion](#m3-re-ranking--fusion)
  - [M4: Response Generation](#m4-response-generation)
- [Data Pipeline](#data-pipeline)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## Overview

| Attribute       | Value                                                      |
| --------------- | ---------------------------------------------------------- |
| Problem type    | Information Retrieval + Content-based Recommendation + NLG |
| System          | RAG Chatbot (Retrieval-Augmented Generation)               |
| Data            | Unsupervised / self-supervised (no labeled data)           |
| Languages       | Bilingual — English (primary), Vietnamese (secondary)      |
| Embedding model | BAAI/bge-m3 (1024-dim, multilingual dense retrieval)       |
| LLM             | OpenAI GPT-4o-mini or Ollama (Gemma 4 E4B)                |

### Supported Query Types

| Query type       | Example                                                        |
| ---------------- | -------------------------------------------------------------- |
| Product search   | "Cà phê vị hoa quả, rang nhạt, pha filter"                    |
| Similar search   | "Gợi ý cà phê giống Ethiopia Yirgacheffe"                     |
| Comparison       | "So sánh Arabica Geisha với Typica"                            |
| Knowledge QA     | "Natural process khác gì Washed?"                              |
| News search      | "Tin tức mới nhất về thị trường cà phê Việt Nam"               |
| Exploration      | "Roaster nào ở Việt Nam có nhiều sản phẩm specialty nhất?"     |

### Input / Output

```
INPUT:  Natural language question or request
        e.g. "Tìm cà phê vị chocolate, medium roast, từ Colombia"

OUTPUT: {
          "summary":   "Concise answer in natural language",
          "products":  [{ name, roaster, reason, url }, ...],
          "articles":  [{ title, summary }, ...]
        }
```

---

## Architecture

```
                         ┌──────────────┐
                         │  User Query  │
                         └──────┬───────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │   M1: Query Understanding     │
                 │                               │
                 │  ┌─────────────┐ ┌──────────┐ │
                 │  │  Intent     │ │  Entity  │ │
                 │  │  Classify   │ │  Extract │ │
                 │  │  (regex)    │ │  (LLM)   │ │
                 │  └─────┬───────┘ └────┬─────┘ │
                 └────────┼──────────────┼───────┘
                          │              │
         ┌────────────────┤              ├────────────────┐
         ▼                ▼              ▼                ▼
┌─────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────┐
│ M2A:        │  │ M2B:         │  │ M2C:          │  │ News     │
│ Structured  │  │ Semantic     │  │ Product Name  │  │ Semantic │
│ Filter      │  │ Search       │  │ Matching      │  │ Search   │
│ (metadata)  │  │ (FAISS)      │  │ (fuzzy)       │  │ (FAISS)  │
└──────┬──────┘  └──────┬───────┘  └───────┬───────┘  └────┬─────┘
       │                │                  │                │
       └────────────────┼──────────────────┘                │
                        ▼                                   │
              ┌──────────────────┐                          │
              │ M3: RRF Fusion   │                          │
              │ (re-rank beans)  │                          │
              └────────┬─────────┘                          │
                       │                                    │
                       ▼                                    ▼
              ┌──────────────────────────────────────────────┐
              │   M4: Response Generation                    │
              │                                              │
              │   System prompt (VI/EN auto-detect)          │
              │   + Retrieved beans context                  │
              │   + Retrieved news context                   │
              │   + User query                               │
              │                                              │
              │   → LLM → Pydantic structured output         │
              └──────────────────────────────────────────────┘
```

---

## System Components

### M1: Query Understanding

**Intent Classification** (`src/query/intent_classifier.py`)

Rule-based regex classifier with priority ordering (first match wins):

| Priority | Intent           | Trigger patterns                                                   |
| -------- | ---------------- | ------------------------------------------------------------------ |
| 1        | `edge_case`      | Very short (<15 chars), adversarial, out-of-scope                  |
| 2        | `exploration`    | "bao nhiêu", "top N", "thống kê", "most common"                   |
| 3        | `similar_search` | "giống", "tương tự", "similar", "alternatives"                     |
| 4        | `comparison`     | "so sánh", "compare", "khác gì", "vs"                             |
| 5        | `news_search`    | "tin tức", "news", "thị trường", "market"                         |
| 6        | `product_search` | "tìm", "gợi ý", "recommend", "find", "cà phê có vị"              |
| 7        | `knowledge_qa`   | "là gì", "what is", "how", "tại sao"                              |

Default fallback: `product_search`

**Entity Extraction** (`src/query/entity_extractor.py`)

LLM-first approach with rule-based fallback:

1. **LLM extraction**: Sends structured prompt to LLM, requesting JSON with 7 fields:
   - `flavor` — taste descriptors, auto-translated VI→EN ("sô cô la" → "Chocolate")
   - `origin` — country/region name
   - `roast` — Light, Medium-Light, Medium, Medium-Dark, Dark
   - `processing` — Washed, Natural, Honey, Anaerobic, etc.
   - `typology` — Arabica, Robusta, Liberica
   - `roaster` — validated roaster/brand name only
   - `product` — specific product name

2. **Rule-based fallback**: Keyword matching against curated vocabularies if LLM fails

3. **Post-processing**: Title-casing flavors, Vietnamese processing translation, roaster validation

### M2A: Structured Filter

**File**: `src/retrieval/structured_filter.py`

Metadata-based filtering on the beans DataFrame using extracted entities.

**Algorithm**:
```
mask = True for all beans
if origin   → filter by country OR origin column (regex, case-insensitive)
if roast    → filter by roast_level_clean (with normalization: "Medium Light" → "Medium-Light")
if flavor   → filter by flavor_notes_clean (AND logic: all requested flavors must match)
if typology → filter by species column
if processing → filter by processing_clean column
```

**Progressive relaxation**: If strict AND-filtering returns < 3 results, progressively drop filters in order: `processing → typology → roast` until enough results are found.

After filtering, results are **re-ranked by semantic similarity** with the query vector using precomputed embeddings.

### M2B: Semantic Retrieval

**File**: `src/retrieval/semantic_search.py`

Dense vector search using sentence-transformers + FAISS.

| Component      | Details                                        |
| -------------- | ---------------------------------------------- |
| Embedding      | BAAI/bge-m3 (1024-dim, multilingual)           |
| Index          | FAISS IndexFlatIP (inner product / cosine sim) |
| Bean documents | 14,537 vectors                                 |
| News chunks    | 13,427 vectors                                 |
| Normalization  | L2-normalized embeddings → IP = cosine sim     |

**Document text construction** for beans:
```
"{product_name}. {about_description}. Origin: {origin}. Country: {country}.
 Roast: {roast_level}. Flavor: {flavor_notes}. Processing: {processing}. Type: {species}."
```

**News chunking**: `RecursiveCharacterTextSplitter` with chunk_size=512, overlap=64.

### M2C: Product Name Matching

**File**: `src/retrieval/product_matcher.py`

Handles queries referencing specific products by name.

1. **Exact match**: Substring search on `product_name` column
2. **Fuzzy match**: `SequenceMatcher` with threshold ≥ 0.6
3. **Roaster boost**: Combined score = 0.7 × product_score + 0.3 × roaster_score

### M3: Re-Ranking & Fusion

**File**: `src/retrieval/reranker.py`

**Reciprocal Rank Fusion (RRF)** combines results from multiple retrieval sources:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))    for each list i where d appears
```

With `k = 60` (smoothing constant).

**Fusion inputs** (in priority order):
1. Product name matches (if query contains a specific product name)
2. Semantic search results (top-K × 3 beans)
3. Structured filter results (re-ranked by semantic similarity, top-K × 3)

Output: final top-K beans sorted by fused RRF score.

### M4: Response Generation

**File**: `src/generation/prompt_templates.py`, `src/generation/llm_client.py`, `src/generation/schemas.py`

**Language detection**: Auto-detects Vietnamese via diacritics regex → selects corresponding system prompt.

**Prompt structure**:
```
[System]  "Bạn là Coffee Advisor..." / "You are Coffee Advisor..."
[User]    === RETRIEVED COFFEE BEANS ===
          --- Bean 1 ---
          Name: ... Roaster: ... Origin: ... Flavor: ... URL: ...
          === RELATED ARTICLES ===
          --- Article ---
          Title: ... Content: ...

          CÂU HỎI: {user_query}
```

**Structured output**: Uses OpenAI's `beta.chat.completions.parse()` with Pydantic `CoffeeResponse` schema for guaranteed structured responses.

**LLM providers**: Configurable via `LLM_PROVIDER` env var:
- `openai` → OpenAI API (GPT-4o-mini default)
- `ollama` → Ollama local server (Gemma 4 E4B default)

**Streaming**: `generate_stream()` yields tokens for real-time UI rendering.

---

## Data Pipeline

### Raw Data

| Dataset            | File                | Records | Source                     |
| ------------------ | ------------------- | ------- | -------------------------- |
| Coffee beans       | `coffee_beans.json` | 14,537  | thewaytocoffee.com         |
| Coffee news        | `coffee_news.json`  | 1,947   | Vietnam.vn, World Coffee Portal, etc. |

### Preprocessing

**Beans** (`src/preprocessing/clean_beans.py`):
1. Clean `flavor_notes` — remove stray `\nORIGIN` entries from crawling artifacts
2. Clean `processing` — same decontamination
3. Parse `typology` — split into `species` (Arabica/Robusta) and `cultivars` (Geisha/Typica)
4. Normalize `roast_level` — map to canonical values (Light, Medium-Light, Medium, Medium-Dark, Dark, Unknown)
5. Build `document_text` — composite field for embedding

**News** (`src/preprocessing/clean_news.py`):
1. Clean `tags` — remove navigation junk ("Skip to content", "Login", etc.)
2. Chunk `content_text` — RecursiveCharacterTextSplitter (512 chars, 64 overlap)
3. Preserve metadata per chunk (title, source, datetime, URL)

### Embeddings (`src/preprocessing/build_embeddings.py`)

1. Load cleaned parquet data
2. Encode with `BAAI/bge-m3` (1024-dim, L2-normalized)
3. Build FAISS `IndexFlatIP` indices
4. Auto-detects dimension mismatch if model changes → rebuilds

**Output**:
```
data/embeddings/
├── beans_embeddings.npy   # (14537, 1024) float32
├── news_embeddings.npy    # (13427, 1024) float32
├── beans.index            # FAISS IndexFlatIP
└── news.index             # FAISS IndexFlatIP
```

---

## Evaluation

### Framework: RAGAS

**File**: `evaluation/ragas_eval.py`

Evaluates the RAG pipeline using [RAGAS](https://docs.ragas.io/) metrics with an LLM-as-judge approach.

**Metrics**:

| Metric              | What it measures                                   | Mode       |
| ------------------- | -------------------------------------------------- | ---------- |
| `context_precision` | Are retrieved contexts relevant to the question?   | retrieval  |
| `context_recall`    | Does retrieved context cover the ground truth?     | retrieval  |
| `faithfulness`      | Is the answer grounded in retrieved context?       | full       |
| `answer_relevancy`  | Is the answer relevant to the question?            | full       |

**Intent-aware metric selection**: Context metrics are only evaluated for retrieval-focused intents where they are meaningful:

| Intent           | context_precision | context_recall | faithfulness | answer_relevancy |
| ---------------- | :---------------: | :------------: | :----------: | :--------------: |
| product_search   | ✅                | ✅             | ✅           | ✅               |
| similar_search   | ✅                | ✅             | ✅           | ✅               |
| news_search      | ✅                | ✅             | ✅           | ✅               |
| knowledge_qa     | —                 | —              | ✅           | ✅               |
| comparison       | —                 | —              | ✅           | ✅               |
| exploration      | —                 | —              | ✅           | ✅               |
| edge_case        | —                 | —              | ✅           | ✅               |

*Context metrics are skipped for non-retrieval intents because their ground truth contains conceptual knowledge, aggregate statistics, or out-of-scope responses not present in individual bean/news contexts.*

**Early stopping**: If ≥ 20% of retrieval-intent cases score `context_precision = 0`, evaluation stops early to save API costs.

### Evaluation Dataset

**File**: `ragas_eval_dataset.json` (500 cases)

| Intent           | Cases | Description                                |
| ---------------- | ----: | ------------------------------------------ |
| product_search   |   121 | Find beans by attributes                   |
| knowledge_qa     |   100 | Coffee knowledge questions                 |
| similar_search   |    69 | Find similar beans                         |
| comparison       |    69 | Compare processing/roast/origin            |
| news_search      |    69 | Coffee industry news                       |
| exploration      |    61 | Statistics and data exploration            |
| edge_case        |    11 | Out-of-scope / adversarial queries         |

**Dataset generation** (`evaluation/generate_dataset.py`):
- Samples real beans from the database
- Generates questions via LLM (GPT-4o-mini)
- Runs actual retrieval pipeline to build ground truth contexts
- Supports resume from partial results

### Running Evaluation

```bash
# Retrieval mode (fast, no LLM generation needed for answers)
python -m evaluation.ragas_eval --mode retrieval --limit 50

# Full mode (includes faithfulness + answer_relevancy)
python -m evaluation.ragas_eval --mode full --limit 20

# Filter by intent
python -m evaluation.ragas_eval --mode retrieval --intent product_search

# All cases, resume from previous run
python -m evaluation.ragas_eval --mode retrieval --limit 0
```

---

## Tech Stack

| Category              | Technology                                           |
| --------------------- | ---------------------------------------------------- |
| Language              | Python ≥ 3.10                                        |
| Embedding             | BAAI/bge-m3 via sentence-transformers (1024-dim)     |
| Vector search         | FAISS (IndexFlatIP, cosine similarity)               |
| LLM                   | OpenAI GPT-4o-mini or Ollama (Gemma 4 E4B)          |
| Structured output     | Pydantic v2 + OpenAI structured parsing              |
| Text chunking         | LangChain RecursiveCharacterTextSplitter             |
| Data processing       | pandas + pyarrow (parquet)                           |
| Evaluation            | RAGAS ≥ 0.4.3                                        |
| UI                    | Streamlit                                            |
| Config                | python-dotenv (.env)                                 |

---

## Project Structure

```
coffee-rag/
├── README.md                         # This file
├── AGENTS.md                         # Detailed system design document (Vietnamese)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
│
├── coffee_beans.json                 # Raw data: 14,537 coffee products
├── coffee_news.json                  # Raw data: 1,947 news articles
├── ragas_eval_dataset.json           # Evaluation dataset (500 cases)
│
├── data/
│   ├── processed/
│   │   ├── beans_clean.parquet       # Cleaned beans (14,537 rows)
│   │   ├── news_clean.parquet        # Cleaned news (1,947 rows)
│   │   └── news_chunks.parquet       # Chunked news (13,427 chunks)
│   └── embeddings/
│       ├── beans_embeddings.npy      # Bean vectors (14537 × 1024)
│       ├── news_embeddings.npy       # News vectors (13427 × 1024)
│       ├── beans.index               # FAISS index for beans
│       └── news.index                # FAISS index for news
│
├── src/
│   ├── pipeline.py                   # CoffeeRAG orchestrator (M1→M4)
│   ├── query/
│   │   ├── intent_classifier.py      # M1A: Rule-based intent classification
│   │   └── entity_extractor.py       # M1B: LLM + rule-based entity extraction
│   ├── retrieval/
│   │   ├── structured_filter.py      # M2A: Metadata filtering + relaxation
│   │   ├── semantic_search.py        # M2B: FAISS dense vector search
│   │   ├── product_matcher.py        # M2C: Exact + fuzzy product name matching
│   │   └── reranker.py               # M3: Reciprocal Rank Fusion
│   ├── generation/
│   │   ├── prompt_templates.py       # Bilingual prompts + context formatting
│   │   ├── schemas.py                # Pydantic response schemas
│   │   └── llm_client.py             # OpenAI / Ollama LLM client
│   └── preprocessing/
│       ├── clean_beans.py            # Bean data cleaning pipeline
│       ├── clean_news.py             # News cleaning + chunking
│       └── build_embeddings.py       # Embedding generation + FAISS indexing
│
├── evaluation/
│   ├── ragas_eval.py                 # RAGAS evaluation runner
│   ├── generate_dataset.py           # Eval dataset generator (LLM-powered)
│   ├── fix_ground_truth.py           # Ground truth cleanup utility
│   └── results/                      # CSV evaluation results
│
└── app/
    └── streamlit_app.py              # Chatbot UI (Streamlit)
```

---

## Setup & Usage

### Prerequisites

- Python ≥ 3.10
- (Optional) Ollama for local LLM inference

### Installation

```bash
git clone https://github.com/hairbui76/coffee-rag.git
cd coffee-rag
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings:
#   LLM_PROVIDER=openai          (or ollama)
#   OPENAI_API_KEY=sk-...        (if using OpenAI)
#   EMBEDDING_MODEL=BAAI/bge-m3  (default)
```

### Data Preprocessing (if starting from scratch)

```bash
# Step 1: Clean raw data → parquet
python -m src.preprocessing.clean_beans
python -m src.preprocessing.clean_news

# Step 2: Build embeddings + FAISS indices
python -m src.preprocessing.build_embeddings
# Use --force to rebuild if changing embedding model
```

### Run Chatbot

```bash
streamlit run app/streamlit_app.py
```

### Run Evaluation

```bash
# Quick test (5 cases)
python -m evaluation.ragas_eval --mode retrieval --limit 5

# Full evaluation
python -m evaluation.ragas_eval --mode full --limit 0
```

---

## Key Algorithms

### Reciprocal Rank Fusion (RRF)

Combines multiple ranked lists into a single ranking. For each document `d` appearing in any list:

```
score(d) = Σ 1/(k + rank_i(d))   for all lists i
```

where `k = 60` is a smoothing constant that prevents top-ranked items from dominating.

### Hybrid Retrieval Strategy

The system uses a **3-way hybrid** retrieval approach per query:

1. **Dense retrieval** (semantic): BAAI/bge-m3 embeddings → FAISS cosine search
2. **Sparse retrieval** (structured): Metadata filtering by entities (country, roast, flavor, processing, species)
3. **Name matching** (lexical): Exact + fuzzy product name lookup

Results are fused via RRF, giving precedence to product name matches.

### Progressive Filter Relaxation

When structured filtering is too restrictive (< 3 results), filters are progressively dropped:

```
processing → typology → roast → (country only)
```

This ensures the system always returns results, even for niche queries.

---

## License

This project is for educational and research purposes.
