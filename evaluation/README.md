# Evaluation — RAGAS cho Coffee RAG

Tài liệu chi tiết về cách đánh giá hệ thống Coffee RAG bằng framework [RAGAS](https://docs.ragas.io/).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc evaluation](#kiến-trúc-evaluation)
- [RAGAS metrics](#ragas-metrics)
  - [Context Precision](#context-precision)
  - [Context Recall](#context-recall)
  - [Faithfulness](#faithfulness)
  - [Answer Relevancy](#answer-relevancy)
- [Evaluation dataset](#evaluation-dataset)
  - [Schema](#schema)
  - [Intent distribution](#intent-distribution)
  - [Cách generate dataset](#cách-generate-dataset)
- [Evaluation pipeline](#evaluation-pipeline)
  - [Luồng xử lý 1 case](#luồng-xử-lý-1-case)
  - [Intent-aware metric selection](#intent-aware-metric-selection)
  - [Early stopping](#early-stopping)
  - [Context formatting](#context-formatting)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Generate dataset](#generate-dataset)
  - [Chạy evaluation](#chạy-evaluation)
  - [Validate dataset](#validate-dataset)
  - [Debug cases](#debug-cases)
- [Phân tích kết quả](#phân-tích-kết-quả)
  - [Output files](#output-files)
  - [Đọc kết quả CSV](#đọc-kết-quả-csv)
  - [EDA notebook](#eda-notebook)
- [Các vấn đề đã biết và giải pháp](#các-vấn-đề-đã-biết-và-giải-pháp)
- [Cấu hình](#cấu-hình)

---

## Tổng quan

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   generate_      │     │   ragas_eval.py   │     │   EDA / phân     │
│   dataset.py     │────▶│                  │────▶│   tích kết quả   │
│                  │     │   Chấm điểm từng │     │                  │
│   Tạo bộ câu hỏi│     │   case bằng RAGAS │     │   CSV, notebook  │
│   + ground truth │     │   metrics         │     │   biểu đồ        │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Quy trình 3 bước:**

1. **Generate dataset** — Tạo bộ câu hỏi evaluation từ dữ liệu thật (beans + news)
2. **Run evaluation** — Chạy RAG pipeline cho mỗi câu hỏi, chấm điểm bằng RAGAS
3. **Analyze results** — Phân tích CSV/notebook để tìm điểm yếu

---

## Kiến trúc evaluation

```
                    Evaluation Dataset (JSON)
                    ┌─────────────────────────────────┐
                    │ { question, ground_truth,        │
                    │   ground_truth_contexts,         │
                    │   intent, difficulty, language }  │
                    └───────────────┬─────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │                       │
                  ┌─────▼──────┐         ┌──────▼──────┐
                  │  question   │         │  ground_    │
                  │             │         │  truth      │
                  └─────┬──────┘         │  (reference)│
                        │                └──────┬──────┘
                        ▼                       │
              ┌─────────────────┐               │
              │  CoffeeRAG      │               │
              │  .retrieve()    │               │
              │                 │               │
              │  → intent       │               │
              │  → entities     │               │
              │  → beans (top-K)│               │
              │  → news  (top-K)│               │
              └────────┬────────┘               │
                       │                        │
                       ▼                        │
              ┌─────────────────┐               │
              │ Format contexts │               │
              │                 │               │
              │ "Bean: {name}.  │               │
              │  Roaster: ..."  │               │
              └────────┬────────┘               │
                       │                        │
                       ▼                        ▼
              ┌─────────────────────────────────────┐
              │         RAGAS Metrics                │
              │                                     │
              │  context_precision(question,         │
              │                   reference,         │
              │                   retrieved_contexts)│
              │                                     │
              │  context_recall(question,             │
              │                reference,             │
              │                retrieved_contexts)    │
              │                                     │
              │  faithfulness(question,               │
              │              response,                │
              │              retrieved_contexts)      │
              │                                     │
              │  answer_relevancy(question, response) │
              └─────────────────┬───────────────────┘
                                │
                                ▼
                       Scores (0.0 — 1.0)
```

**Quan trọng:** RAGAS dùng **LLM-as-judge** (mặc định `gpt-4o-mini`) để chấm điểm. Mỗi metric gọi LLM evaluator riêng — KHÔNG phải rule-based.

---

## RAGAS metrics

### Context Precision

```
                    ┌───────────────┐
                    │   question    │
                    │   reference   │──────────────────────┐
                    └───────┬───────┘                      │
                            │                              │
                            ▼                              ▼
                ┌───────────────────────┐     ┌────────────────────┐
                │  retrieved_contexts   │     │  LLM Judge đánh    │
                │  [ctx_1, ctx_2, ...]  │────▶│  giá: ctx_i có     │
                │                       │     │  relevant không?   │
                └───────────────────────┘     └──────────┬─────────┘
                                                         │
                                                         ▼
                                              relevant@rank / rank
                                              = Precision@K (weighted)
```

**Ý nghĩa:** Trong các context được retrieve, bao nhiêu phần trăm thực sự relevant? Context relevant ở rank cao hơn được trọng số lớn hơn.

**Công thức:**

```
CP = (1/K) × Σ (Precision@k × rel(k))

Trong đó:
  K = tổng số contexts
  rel(k) = 1 nếu context ở vị trí k relevant, 0 nếu không
  Precision@k = (số relevant trong top-k) / k
```

**Ví dụ:** Retrieve 5 contexts, relevant = [1, 0, 1, 0, 0]

```
Precision@1 = 1/1 = 1.0    × rel(1) = 1.0
Precision@2 = 1/2 = 0.5    × rel(2) = 0
Precision@3 = 2/3 = 0.667  × rel(3) = 0.667
Precision@4 = 2/4 = 0.5    × rel(4) = 0
Precision@5 = 2/5 = 0.4    × rel(5) = 0
CP = (1.0 + 0 + 0.667 + 0 + 0) / 5 = 0.333
```

**CP = 0 khi nào?** Khi LLM judge cho rằng KHÔNG context nào relevant cho question + reference.

---

### Context Recall

```
                    ┌───────────────┐
                    │   reference   │
                    │   (ground     │
                    │    truth)     │
                    └───────┬───────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Phân tách reference  │
                │  thành các CLAIMS     │
                │                       │
                │  claim_1: "Gute       │
                │   Village từ TENFOLD" │
                │  claim_2: "Origin     │
                │   Sidama, Ethiopia"   │
                │  claim_3: "Rang vừa"  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Với mỗi claim:       │
                │  LLM check claim có  │
                │  thể tìm thấy trong  │
                │  retrieved_contexts?  │
                └───────────┬───────────┘
                            │
                            ▼
              CR = claims_attributed / total_claims
```

**Ý nghĩa:** Bao nhiêu phần trăm thông tin trong ground truth có thể tìm thấy trong các contexts đã retrieve?

**Công thức:**

```
CR = |{claim ∈ reference : ∃ ctx ∈ retrieved_contexts sao cho ctx chứa claim}|
     ─────────────────────────────────────────────────────────────────────────
                          |tổng số claims trong reference|
```

**CR = 0 khi nào?**
1. Ground truth nhắc tên sản phẩm cụ thể mà retrieval không trả về
2. Ground truth chứa kiến thức tổng quát (ví dụ: "Ethiopia nổi tiếng với vị trái cây") mà KHÔNG có trong bean descriptions
3. Ground truth contexts bị fabricated (không phải dữ liệu thật)

**CR cao khi:** Ground truth text chỉ reference thông tin CÓ trong retrieved contexts.

---

### Faithfulness

> Chỉ đánh giá ở mode `full` (cần LLM generate response)

```
CR = |{claim ∈ response : ∃ ctx ∈ retrieved_contexts chứa claim}|
     ───────────────────────────────────────────────────────────
                    |tổng số claims trong response|
```

**Ý nghĩa:** Response có "hallucinate" không? Mọi claim trong response phải tìm được nguồn trong retrieved contexts.

---

### Answer Relevancy

> Chỉ đánh giá ở mode `full`

**Ý nghĩa:** Response có trả lời đúng câu hỏi không? Dùng embedding similarity giữa question và response.

---

## Evaluation dataset

### Schema

Mỗi case trong dataset JSON:

```json
{
  "id": "PS_001",
  "intent": "product_search",
  "difficulty": "easy",
  "language": "vi",
  "question": "Bạn có thể giới thiệu cho tôi một loại cà phê từ Ethiopia, rang vừa, có hương vị cam không?",
  "ground_truth": "Bạn có thể thử loại cà phê Gute Village, Ethiopia từ TENFOLD COFFEE...",
  "ground_truth_contexts": [
    "Bean: Gute Village, Ethiopia. Roaster: TENFOLD COFFEE. Origin: Sidama (Ethiopia)..."
  ],
  "metadata": {
    "expected_entities": { "country": "Ethiopia", "roast": "Medium", "flavor": ["Orange"] },
    "retrieved_product_names": ["Gute Village, Ethiopia", "Ethiopian Sidamo", ...]
  }
}
```

| Trường | Vai trò trong evaluation |
|--------|--------------------------|
| `question` | Input cho RAG pipeline + RAGAS metrics |
| `ground_truth` | `reference` cho context_precision, context_recall |
| `ground_truth_contexts` | Để validate dataset quality (KHÔNG dùng trực tiếp trong RAGAS scoring) |
| `metadata.expected_entities` | Debug: so sánh entities expected vs extracted |
| `metadata.retrieved_product_names` | Debug: so sánh beans expected vs retrieved |
| `intent` | Quyết định metrics nào được đánh giá (intent-aware selection) |
| `difficulty` | Phân tích kết quả theo độ khó |
| `language` | Filter evaluation theo ngôn ngữ |

### Intent distribution

**Dataset v4** (retrieval-grounded, 410 cases):

| Intent | Cases | Metrics đánh giá |
|--------|------:|-------------------|
| product_search | 130 | CP, CR, Faithfulness, Answer Relevancy |
| similar_search | 80 | CP, CR, Faithfulness, Answer Relevancy |
| comparison | 70 | Faithfulness, Answer Relevancy |
| knowledge_qa | 70 | Faithfulness, Answer Relevancy |
| news_search | 60 | CP, CR, Faithfulness, Answer Relevancy |

**Difficulty:** 30% easy, 45% medium, 25% hard

**Language:** 60% Vietnamese, 40% English

### Cách generate dataset

Script: `evaluation/generate_dataset.py`

```
Phase 1: Build specs (data-only, no LLM)
  ├── product_search: Sampling (country × roast × flavor) từ top-12 countries
  ├── similar_search: Random seed bean + find similar beans
  ├── comparison:     Concept pairs (Natural vs Washed, Light vs Dark, ...)
  ├── knowledge_qa:   Topic-based (processing, roast, species, origin, flavor)
  └── news_search:    Sample articles từ news_clean

Phase 2: Generate questions (LLM)
  └── GPT-4o-mini generates natural-sounding questions từ specs

Phase 3: Retrieve actual contexts (RAG pipeline)    ← QUAN TRỌNG
  └── rag.retrieve(question, top_k_beans=10, top_k_news=5)
  └── ground_truth_contexts = top-5 beans + top-3 news từ KẾT QUẢ RETRIEVAL THẬT

Phase 4: Generate ground truth text (LLM)
  └── GPT-4o-mini viết câu trả lời DỰA TRÊN retrieved data từ Phase 3
```

**Phase 3 là then chốt:** Ground truth contexts được lấy từ kết quả retrieval thật, KHÔNG phải pre-selected. Điều này đảm bảo ground truth khớp với những gì retrieval pipeline thực sự trả về → context metrics chính xác hơn.

---

## Evaluation pipeline

### Luồng xử lý 1 case

```python
# File: evaluation/ragas_eval.py → retrieve_one()

def retrieve_one(rag, case, args):
    question = case["question"]

    # 1. Chạy RAG pipeline (M1 → M2 → M3)
    ctx = rag.retrieve(question, top_k_beans=10, top_k_news=5)
    #   → ctx["intent"]   : intent được classify
    #   → ctx["entities"] : entities được extract
    #   → ctx["beans"]    : DataFrame top-K beans (sau RRF)
    #   → ctx["news"]     : DataFrame top-K news chunks

    # 2. Format contexts thành text strings
    retrieved_contexts = build_retrieved_contexts(ctx)
    #   → ["Bean: Gute Village. Roaster: TENFOLD...", ...]

    # 3. (Chỉ mode full) Generate response bằng LLM
    response = None
    if args.mode == "full":
        messages = build_prompt(question, ctx["beans"], ctx["news"])
        response = generate_structured(messages, CoffeeResponse)

    # 4. Return sample dict cho RAGAS scoring
    return {
        "question": question,
        "reference": case["ground_truth"],      # ← từ dataset
        "response": response_to_text(response), # ← từ LLM (hoặc "" nếu retrieval mode)
        "retrieved_contexts": retrieved_contexts, # ← từ pipeline lúc eval
    }
```

### Intent-aware metric selection

```python
RETRIEVAL_INTENTS = {"product_search", "similar_search", "news_search"}
CONTEXT_METRICS = {"context_precision", "context_recall"}

def _metrics_for_intent(intent, all_metrics):
    if intent in RETRIEVAL_INTENTS:
        return all_metrics                    # Đánh giá TẤT CẢ metrics
    return {k: v for k, v in all_metrics.items()
            if k not in CONTEXT_METRICS}      # Bỏ qua context metrics
```

**Tại sao bỏ context metrics cho knowledge_qa, comparison?**

Vì ground truth của các intent này chứa **kiến thức tổng quát** (ví dụ: "Natural processing là phương pháp phơi khô cherry nguyên quả...") — thông tin này KHÔNG có trong các bean descriptions hay news articles. RAGAS sẽ luôn cho CR=0 vì không tìm thấy claim trong contexts → metrics vô nghĩa.

### Early stopping

```python
ZERO_PRECISION_STOP_RATIO = 0.2

# Nếu ≥ 20% retrieval-intent cases có CP=0 → dừng sớm
if zero_precision_count >= ceil(0.2 * retrieval_case_count):
    stop_event.set()
```

Mục đích: tiết kiệm API costs khi pipeline rõ ràng có vấn đề.

### Context formatting

Contexts được format thống nhất giữa dataset generation và evaluation:

**Bean context:**
```
Bean: {product_name}. Roaster: {roaster_name}. Origin: {origin}. Country: {country}.
Roast: {roast_level_clean}. Flavor: {flavor_notes_clean}. Processing: {processing_clean}.
Species: {species}. Description: {about_description}. URL: {product_url}
```

**News context:**
```
Article: {title}. Source: {source}. Date: {publish_datetime}. Content: {text}. URL: {article_url}
```

**Format này phải ĐỒNG NHẤT** giữa `generate_dataset.py::_bean_context_text()` và `ragas_eval.py::_bean_contexts()`. Nếu khác → RAGAS so sánh text khác nhau → CR giảm giả tạo.

---

## Hướng dẫn sử dụng

### Generate dataset

```bash
# Dry run — chỉ build specs, không gọi LLM
python -m evaluation.generate_dataset --dry-run

# Generate dataset v4 (cần OPENAI_API_KEY)
python -m evaluation.generate_dataset --out ragas_eval_dataset_v4.json --workers 16

# Resume từ partial file (nếu bị crash)
# Script tự tìm file .partial.json và tiếp tục
python -m evaluation.generate_dataset --out ragas_eval_dataset_v4.json --workers 16
```

**Yêu cầu:**
- `OPENAI_API_KEY` trong `.env`
- Embeddings đã build (`data/embeddings/beans.index`, `news.index`)
- Dữ liệu đã clean (`data/processed/beans_clean.parquet`, `news_chunks.parquet`, `news_clean.parquet`)

**Output:** File JSON theo schema ở trên

### Chạy evaluation

#### Mode retrieval (chỉ context metrics, nhanh)

```bash
# Chạy 50 cases đầu
python -m evaluation.ragas_eval --mode retrieval --limit 50

# Chạy full dataset
python -m evaluation.ragas_eval --mode retrieval --limit 0

# Chỉ test product_search
python -m evaluation.ragas_eval --mode retrieval --intent product_search --limit 30

# Chỉ test tiếng Việt
python -m evaluation.ragas_eval --mode retrieval --language vi --limit 20

# Dùng dataset khác
python -m evaluation.ragas_eval --mode retrieval --dataset ragas_eval_dataset_v4.json --limit 0

# Verbose mode — hiện entities, context counts, running averages
python -m evaluation.ragas_eval --mode retrieval --limit 10 -v

# Custom output path
python -m evaluation.ragas_eval --mode retrieval --out evaluation/results/v4_retrieval.csv

# Concurrent workers (tăng tốc, mặc định 4)
python -m evaluation.ragas_eval --mode retrieval --workers 8 --limit 0
```

#### Mode full (tất cả metrics, cần LLM generate response)

```bash
# Full eval 20 cases
python -m evaluation.ragas_eval --mode full --limit 20

# Chỉ đánh giá faithfulness
python -m evaluation.ragas_eval --mode full --metrics faithfulness --limit 30

# Chỉ đánh giá answer_relevancy
python -m evaluation.ragas_eval --mode full --metrics answer_relevancy --limit 30
```

#### Resume (chạy tiếp từ lần trước)

```bash
# --limit 0 = chạy tất cả, bỏ qua IDs đã có trong CSV output
python -m evaluation.ragas_eval --mode retrieval --limit 0
```

Script tự đọc file CSV output, skip các case đã đánh giá, chỉ chạy cases còn thiếu.

#### Tất cả arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--dataset` | `ragas_eval_dataset.json` | Đường dẫn dataset JSON |
| `--out` | `evaluation/results/ragas_results.csv` | Output CSV |
| `--mode` | `full` | `retrieval` hoặc `full` |
| `--limit` | `5` | Số cases chạy. `0` = tất cả (+ resume) |
| `--offset` | `0` | Bỏ qua N cases đầu |
| `--intent` | None | Filter theo intent |
| `--language` | None | Filter theo ngôn ngữ (`vi` hoặc `en`) |
| `--metrics` | Theo mode | Chọn metrics cụ thể |
| `--top-k-beans` | `10` | Số beans retrieve |
| `--top-k-news` | `5` | Số news chunks retrieve |
| `--workers` | `4` | Concurrent scoring (RAGAS API calls) |
| `--verbose` / `-v` | False | Hiện chi tiết mỗi case |
| `--evaluator-model` | `gpt-4o-mini` | LLM dùng cho RAGAS judge |
| `--embedding-model` | `text-embedding-3-small` | Embedding cho answer_relevancy |
| `--responses-out` | `<out>_responses.json` | Output JSON cho question-response pairs |

### Validate dataset

```bash
python -m evaluation.validate_dataset
```

Kiểm tra:
- Distribution (intent, difficulty, language)
- Duplicate questions
- Ground truth contexts có grounded trong dữ liệu thật không
- Quality checks (missing fields, short texts)
- ID uniqueness

### Debug cases

```bash
# Debug cases mặc định (hardcoded trong script)
python -m evaluation.ragas_eval_debug

# Debug cases cụ thể
python -m evaluation.ragas_eval_debug --ids PS_003 PS_006 PS_007

# Debug theo intent
python -m evaluation.ragas_eval_debug --intent similar_search --limit 5

# Verbose mode
python -m evaluation.ragas_eval_debug --ids PS_001 --verbose
```

**Debug retrieval overlap** (so sánh GT beans vs retrieved beans):

```bash
python -m evaluation.debug_retrieval_overlap --dataset ragas_eval_dataset_v4.json --limit 30 --intent product_search
```

---

## Phân tích kết quả

### Output files

| File | Nội dung |
|------|----------|
| `evaluation/results/ragas_results.csv` | Scores chi tiết từng case |
| `evaluation/results/ragas_results_responses.json` | Question-response pairs (mode full) |

### Đọc kết quả CSV

```python
import pandas as pd

df = pd.read_csv("evaluation/results/ragas_results.csv")

# Tổng quan
print(df[["context_precision", "context_recall"]].describe())

# Theo intent
print(df.groupby("intent")[["context_precision", "context_recall"]].mean())

# Cases có CR=0
zero_cr = df[df["context_recall"] == 0]
print(f"CR=0: {len(zero_cr)} cases")
print(zero_cr[["id", "intent", "question"]].head(10))

# Theo difficulty
print(df.groupby("difficulty")[["context_precision", "context_recall"]].mean())

# Theo language
print(df.groupby("language")[["context_precision", "context_recall"]].mean())
```

### Ý nghĩa các cột CSV

| Cột | Ý nghĩa |
|-----|---------|
| `id` | ID case trong dataset (PS_001, SS_001, ...) |
| `intent` | Intent được classify (từ dataset, KHÔNG phải từ pipeline) |
| `difficulty` | easy / medium / hard |
| `language` | vi / en |
| `question` | Câu hỏi input |
| `retrieved_context_count` | Số contexts retrieved |
| `bean_count` | Số beans trong retrieval results |
| `news_count` | Số news chunks trong retrieval results |
| `context_precision` | Score 0-1 (empty nếu intent không đánh giá) |
| `context_recall` | Score 0-1 (empty nếu intent không đánh giá) |
| `faithfulness` | Score 0-1 (chỉ có ở mode full) |
| `answer_relevancy` | Score 0-1 (chỉ có ở mode full) |
| `response` | LLM response JSON (chỉ có ở mode full) |
| `reference` | Ground truth text |
| `error` | Error message nếu có |
| `elapsed_s` | Thời gian xử lý (giây) |

### EDA notebook

File: `evaluation/ragas_eda.ipynb`

Notebook phân tích kết quả với biểu đồ: score distributions, breakdown theo intent/difficulty/language, correlation analysis.

---

## Các vấn đề đã biết và giải pháp

### 1. Context Recall thấp cho product_search / similar_search

**Triệu chứng:** CR = 0 cho nhiều cases dù retrieval trả về beans đúng.

**Nguyên nhân:** Ground truth text chứa claims không tìm thấy trong retrieved contexts:
- Kiến thức tổng quát: *"Ethiopia nổi tiếng với profile trái cây"* → không có trong bean descriptions
- Tên sản phẩm cụ thể: ground truth nhắc bean A nhưng retrieval trả về bean B (cùng đúng attributes nhưng khác tên)

**Giải pháp:**
- Dùng **retrieval-grounded dataset** (generate_dataset.py v3+) — ground truth được viết từ kết quả retrieval thật
- Chạy `debug_retrieval_overlap.py` để verify overlap giữa GT beans và retrieved beans

### 2. Response null ở mode retrieval

**Không phải bug.** Mode `retrieval` chỉ đánh giá context_precision và context_recall, KHÔNG generate LLM response → cột `response` trống.

### 3. numpy.ndarray vs list trong filter

**Đã fix.** Parquet lưu array columns dạng `numpy.ndarray`, không phải `list`. Các hàm filter đã được update dùng `isinstance(x, (list, np.ndarray))`.

### 4. Roast normalization

**Đã fix.** Entity extractor trả về "Medium Light" (space) nhưng DB dùng "Medium-Light" (hyphen). Đã thêm normalization map.

### 5. Non-retrieval intents luôn CP=CR=0

**Đã fix.** Intent-aware metric selection bỏ qua context metrics cho knowledge_qa, comparison, exploration, edge_case.

### 6. Fabricated ground_truth_contexts

**Triệu chứng:** ground_truth_contexts chứa placeholder: `"Chunk on Blue Bottle expansion..."` hoặc `"No exact match found for entities: ..."`

**Nguyên nhân:** Dataset cũ (v1/v2) dùng pre-selected beans hoặc fabricated contexts.

**Giải pháp:** Dùng dataset v3+ (retrieval-grounded) hoặc chạy `fix_ground_truth.py` để repair.

---

## Cấu hình

### Biến môi trường (.env)

```bash
# Bắt buộc cho evaluation
OPENAI_API_KEY=sk-...              # Cho RAGAS evaluator LLM + embedding
RAGAS_EVALUATOR_MODEL=gpt-4o-mini  # LLM dùng chấm điểm
RAGAS_EMBEDDING_MODEL=text-embedding-3-small  # Cho answer_relevancy metric

# Cho RAG pipeline (cần cho retrieval)
EMBEDDING_MODEL=BAAI/bge-m3        # PHẢI khớp với FAISS index đã build
LLM_PROVIDER=openai                # hoặc ollama
OPENAI_MODEL=gpt-4o-mini           # Cho generate response (mode full)
```

### Constants trong ragas_eval.py

```python
MAX_EVAL_CONTEXTS = 20              # Giới hạn contexts gửi cho RAGAS
ZERO_PRECISION_STOP_RATIO = 0.2     # Early stop threshold (20%)
RETRIEVAL_INTENTS = {"product_search", "similar_search", "news_search"}
```

### Costs ước tính

| Mode | Cases | Evaluator calls | Ước tính cost (GPT-4o-mini) |
|------|------:|----------------:|----------------------------:|
| retrieval | 100 | ~200 (CP + CR) | ~$0.50 |
| retrieval | 410 | ~820 | ~$2.00 |
| full | 100 | ~400 (all 4 metrics) | ~$1.50 |
| full | 410 | ~1640 | ~$5.00 |

*Cost thực tế phụ thuộc vào độ dài contexts và responses.*
