# Coffee Advisor: Hệ thống tư vấn cà phê Specialty bằng RAG

## 1. Tổng quan bài toán

### Mục tiêu

Xây dựng **hệ thống chatbot tư vấn cà phê specialty** sử dụng kiến trúc **Retrieval-Augmented Generation (RAG)**, kết hợp dữ liệu sản phẩm cà phê (14,537 beans) và tin tức ngành cà phê (1,947 bài báo) làm knowledge base.

### Phân loại bài toán

| Thuộc tính     | Giá trị                                                    |
| -------------- | ---------------------------------------------------------- |
| Loại bài toán  | Information Retrieval + Content-based Recommendation + NLG |
| Dạng hệ thống  | RAG Chatbot (Retrieval-Augmented Generation)               |
| Dữ liệu       | Không đánh nhãn (unsupervised / self-supervised)           |
| Ngôn ngữ       | Đa ngôn ngữ (EN chính, VI phụ)                            |

### Input / Output

```
INPUT:  Câu hỏi hoặc yêu cầu bằng ngôn ngữ tự nhiên
        Ví dụ: "Tôi thích cà phê vị chocolate, medium roast, từ Việt Nam"

OUTPUT: {
          "answer":            "Câu trả lời tổng hợp bằng NL",
          "recommended_beans": [Top-K sản phẩm phù hợp],
          "related_articles":  [Bài báo liên quan nếu có],
          "reasoning":         "Giải thích lý do gợi ý"
        }
```

### Các dạng query hỗ trợ

| Dạng query          | Ví dụ                                                         |
| ------------------- | ------------------------------------------------------------- |
| Tìm theo sở thích   | "Cà phê vị hoa quả, rang nhạt, pha filter"                   |
| Tìm tương tự        | "Gợi ý cà phê giống Ethiopia Yirgacheffe"                    |
| So sánh             | "So sánh Arabica Geisha với Typica"                           |
| Hỏi kiến thức       | "Natural process khác gì Washed?"                             |
| Tin tức             | "Tin tức mới nhất về thị trường cà phê Việt Nam"              |
| Khám phá            | "Roaster nào ở Việt Nam có nhiều sản phẩm specialty nhất?"    |

---

## 2. Dữ liệu (Materials)

### 2.1. Coffee Beans — `coffee_beans.json`

- **Số lượng:** 14,537 sản phẩm
- **Nguồn:** thewaytocoffee.com (Bean Discovery Engine)

**Schema chính được sử dụng:**

| Trường              | Kiểu       | Vai trò trong hệ thống                         |
| ------------------- | ---------- | ----------------------------------------------- |
| `product_name`      | string     | Hiển thị + embedding                            |
| `about_description` | string     | **Trường text chính** để embed và retrieve       |
| `origin`            | string     | Structured filter + embedding                   |
| `country`           | string     | Structured filter (faceted search)              |
| `flavor_notes`      | [string]   | Structured filter + embedding                   |
| `roast_level`       | string     | Structured filter (Light/Medium/Dark/...)       |
| `processing`        | [string]   | Structured filter + embedding                   |
| `typology`          | [string]   | Structured filter (Arabica/Robusta/Liberica)    |
| `roaster_name`      | string     | Metadata hiển thị + filter                      |
| `buy_links`         | [object]   | Hiển thị link mua hàng                          |
| `similar_products`  | [object]   | Sản phẩm tương tự (do website gợi ý)            |
| `product_url`       | string     | Reference link                                  |

**Vấn đề cần xử lý:**
- `flavor_notes` một số record chứa `\n` lẫn lộn (ví dụ `"ORIGIN\nMaejantai (Thailand)"`)
- `processing` một số record bị parse nhầm dữ liệu từ field khác
- `typology` cần tách species (Arabica/Robusta) và cultivar (Geisha/Typica/...)
- `roast_level` cần chuẩn hóa (Light, Medium Light, Medium, Medium Dark, Dark, "—")
### 2.2. Coffee News — `coffee_news.json`

- **Số lượng:** 1,947 bài báo
- **Nguồn:** Vietnam.vn, World Coffee Portal, v.v.

**Schema chính được sử dụng:**

| Trường             | Kiểu       | Vai trò                                 |
| ------------------ | ---------- | --------------------------------------- |
| `title`            | string     | Embedding + hiển thị                    |
| `summary`          | string     | Embedding (ngắn, dense info)            |
| `content_text`     | string     | **Text chính** để embed (chunked)        |
| `tags`             | [string]   | Structured filter                       |
| `section`          | string     | Phân loại chủ đề                        |
| `source`           | string     | Metadata                                |
| `publish_datetime` | datetime   | Sắp xếp theo thời gian                  |
| `article_url`      | string     | Reference link                          |
| `language`         | string     | Filter (en)                             |

**Vấn đề cần xử lý:**
- `tags` nhiều bài chứa navigation items lẫn lộn ("Skip to content", "Login", ...)
- `content_text` có bài rất dài → cần chunking
- Một số bài không liên quan trực tiếp đến coffee beans (tin kinh tế, PR)

---

## 3. Kiến trúc hệ thống (Method)

### 3.1. Pipeline tổng quan

```
                        ┌──────────────┐
                        │  User Query  │
                        └──────┬───────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   M1: Query Understanding     │
                │                              │
                │  ┌────────────┐ ┌──────────┐ │
                │  │  Intent    │ │  Entity  │ │
                │  │  Classify  │ │  Extract │ │
                │  └─────┬──────┘ └────┬─────┘ │
                └────────┼─────────────┼───────┘
                         │             │
            ┌────────────┘             └────────────┐
            ▼                                       ▼
   ┌─────────────────┐                   ┌────────────────────┐
   │ M2A: Structured │                   │ M2B: Semantic      │
   │ Filter          │                   │ Retrieval          │
   │                 │                   │                    │
   │ country="VN"    │                   │ query → embedding  │
   │ roast="Medium"  │                   │ → cosine search    │
   │ flavor∩[choco]  │                   │ top-K from FAISS   │
   └────────┬────────┘                   └────────┬───────────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           ▼
                ┌──────────────────────────────┐
                │   M3: Re-Ranking & Fusion     │
                │                              │
                │   Hybrid = α×struct + β×sem  │
                │   + cross-encoder (optional) │
                │                              │
                │   → Final Top-K documents    │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   M4: Response Generation     │
                │                              │
                │   Prompt = Instruction        │
                │         + Retrieved Context   │
                │         + User Query          │
                │                              │
                │   LLM → structured answer    │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Structured Response         │
                │   - answer (NL text)          │
                │   - recommended_beans []      │
                │   - related_articles []       │
                │   - reasoning                 │
                └──────────────────────────────┘
```

### 3.2. Chi tiết từng module

#### Module 1: Query Understanding

**Mục đích:** Phân tích câu hỏi người dùng để xác định ý định và trích xuất thực thể.

**Intent Classification** — Phân loại ý định bằng rule-based regex, ưu tiên từ trên xuống (match đầu tiên thắng):

| Thứ tự | Intent               | Trigger keywords / patterns                                          |
| ------ | -------------------- | -------------------------------------------------------------------- |
| 1      | `edge_case`          | Query rất ngắn (<15 ký tự), adversarial, out-of-scope               |
| 2      | `exploration`        | "bao nhiêu", "top N", "thống kê", "phổ biến nhất"                  |
| 3      | `similar_search`     | "giống", "tương tự", "similar", "alternatives"                      |
| 4      | `comparison`         | "so sánh", "compare", "khác gì/nhau/biệt", "differ", "vs"          |
| 5      | `news_search`        | "tin tức", "news", "thị trường", "market", "xu hướng"               |
| 6      | `product_search`     | "tìm", "gợi ý", "recommend", "find", "giới thiệu", "cà phê có vị" |
| 7      | `knowledge_qa`       | "là gì", "what is", "how", "tại sao", "phương pháp", "method"      |

> **Lưu ý:** `product_search` được đặt **trước** `knowledge_qa` để các query phức hợp
> (vừa tìm sản phẩm vừa hỏi phương pháp) được phân loại đúng là product_search.

**Entity Extraction** — Trích xuất các thuộc tính từ query:

```python
entities = {
    "flavor":     ["Chocolate", "Fruity"],    # title-cased, từ flavor_notes vocab
    "origin":     "Vietnam",                  # từ country/origin vocab
    "country":    "Vietnam",                  # alias của origin (cho eval compatibility)
    "roast":      "Medium",                   # từ roast_level vocab
    "processing": "Washed",                   # từ processing vocab (EN + VI)
    "typology":   "Arabica",                  # từ typology vocab
    "roaster":    None,                       # chỉ tên roaster thật (đã validate)
    "product":    None                        # tên sản phẩm cụ thể
}
```

**Cách triển khai: LLM-first + rule-based fallback**
- **Bước 1 (LLM):** Gửi structured prompt tới Gemma 4 E4B (qua Ollama), yêu cầu trả JSON.
  Prompt bao gồm ví dụ EN + VI, hướng dẫn dịch flavor VI→EN, processing VI→EN.
- **Bước 2 (Fallback):** Nếu LLM fail, dùng rule-based regex + keyword matching.

**Post-processing sau extraction:**
- **Flavor title-casing:** `"brown sugar"` → `"Brown Sugar"` để khớp data format
- **Non-flavor filtering:** Loại bỏ `"hạt"`, `"bean"` — không phải flavor
- **Roaster validation:** Reject query fragments (chứa country name, "that has", "flavor notes"...)
- **Vietnamese processing map:** `"chế biến ướt"` → `"Washed"`, `"tự nhiên"` → `"Natural"`, `"kỵ khí"` → `"Anaerobic"`
- **Country alias:** `entities["country"] = entities["origin"]`

#### Module 2A: Structured Filter

**Mục đích:** Lọc nhanh candidates từ DB dựa trên metadata có cấu trúc.

```python
def _is_array(x):
    """Parquet stores list columns as numpy.ndarray, not Python list."""
    return isinstance(x, (list, np.ndarray))

def structured_filter(beans_df, entities):
    mask = pd.Series(True, index=beans_df.index)

    if entities.get("origin"):
        origin_pat = re.escape(entities["origin"])
        mask &= (
            beans_df["country"].str.contains(origin_pat, case=False, na=False) |
            beans_df["origin"].str.contains(origin_pat, case=False, na=False)
        )

    if entities.get("roast"):
        # Normalize: "Medium Light" → "Medium-Light" để khớp DB
        roast_val = _ROAST_NORMALIZE.get(entities["roast"].lower(), entities["roast"])
        mask &= beans_df["roast_level_clean"].str.contains(roast_val, case=False, na=False)

    if entities.get("flavor"):
        flat = beans_df["flavor_notes_clean"].apply(
            lambda x: " ".join(x).lower() if _is_array(x) else ""
        )
        for f in entities["flavor"]:
            mask &= flat.str.contains(re.escape(f.lower()), na=False)

    if entities.get("typology"):
        species_flat = beans_df["species"].apply(
            lambda x: " ".join(x).lower() if _is_array(x) else ""
        )
        mask &= species_flat.str.contains(entities["typology"].lower(), na=False)

    if entities.get("processing"):
        proc_flat = beans_df["processing_clean"].apply(
            lambda x: " ".join(x).lower() if _is_array(x) else ""
        )
        mask &= proc_flat.str.contains(entities["processing"].lower(), na=False)

    return beans_df[mask]
```

**Progressive relaxation:** Nếu strict AND filter trả < 3 kết quả, lần lượt bỏ bớt filter
theo thứ tự `processing → typology → roast` cho đến khi đủ kết quả.

#### Module 2B: Semantic Retrieval

**Mục đích:** Tìm kiếm ngữ nghĩa dựa trên embedding similarity.

**Indexing (offline — chạy 1 lần):**

1. Tạo **document text** cho mỗi bean bằng cách nối các trường:
   ```
   "{product_name}. {about_description}.
    Origin: {origin}. Roast: {roast_level}.
    Flavor: {flavor_notes joined}.
    Processing: {processing joined}. Type: {typology joined}."
   ```

2. Tạo **document text** cho mỗi news article:
   - Chunk `content_text` thành đoạn ~512 tokens (overlap 64 tokens)
   - Mỗi chunk kèm metadata: title, source, publish_datetime

3. Embed tất cả documents bằng sentence-transformer:
   ```
   Model hiện tại: BAAI/bge-m3 (dim=1024, multilingual, dense+sparse)
   ```

4. Lưu vectors vào **FAISS** IndexFlatIP (inner product = cosine similarity do L2-normalized).

**Querying (online):**

```python
query_vec = model.encode(user_query)
top_k_beans = faiss_beans.search(query_vec, k=20)
top_k_news  = faiss_news.search(query_vec, k=5)
```

#### Module 3: Re-Ranking & Fusion

**Mục đích:** Kết hợp kết quả từ structured filter và semantic search, re-rank.

**Pipeline thực tế:**

1. **Product name matching** (nếu query chứa tên sản phẩm cụ thể):
   - Exact substring match trước, fuzzy match (SequenceMatcher ≥ 0.6) sau
   - Kết hợp roaster name với weight 0.7×product + 0.3×roaster

2. **Semantic search** (FAISS): top-K×3 beans + top-K news

3. **Structured filter**: lọc toàn bộ beans_df theo entities, sau đó
   **re-rank bằng semantic similarity** với query để chọn top-K×3 từ kết quả filter

4. **RRF fusion**: kết hợp các danh sách trên

```python
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

# Với mỗi document d xuất hiện trong >=1 danh sách kết quả:
# score(d) = Σ rrf_score(rank_of_d_in_list_i)  for all lists i
```

**Structured filter được chạy cho các intent:** `product_search`, `similar_search`,
`comparison`, `knowledge_qa` (khi có entity filters).

#### Module 4: Response Generation

**Mục đích:** Tổng hợp retrieved context thành câu trả lời có cấu trúc (Pydantic structured output).

**Response schema (Pydantic):**

```python
class CoffeeResponse(BaseModel):
    summary: str          # Câu trả lời chính (2-4 câu)
    products: list[RecommendedProduct]  # Sản phẩm gợi ý + lý do
    articles: list[RelatedArticle]      # Bài báo liên quan

class RecommendedProduct(BaseModel):
    name: str             # Tên sản phẩm
    roaster: str          # Tên roaster
    reason: str           # Lý do gợi ý
    url: str              # Product URL
```

**Bilingual system prompts:** Tự động phát hiện ngôn ngữ (Vietnamese diacritics)
và chọn prompt tương ứng:
- **VI:** `"Bạn là Coffee Advisor... Trả lời HOÀN TOÀN bằng TIẾ̀NG VIỆT..."`
- **EN:** `"You are Coffee Advisor... You MUST respond ENTIRELY in ENGLISH..."`

**LLM:** Hỗ trợ 2 provider qua biến `LLM_PROVIDER`:
- `openai` → OpenAI API (GPT-4o-mini mặc định)
- `ollama` → Ollama local (Gemma 4 E4B mặc định)

---

## 4. Data Preprocessing Pipeline

### 4.1. Beans — Cleaning

```python
# Bước 1: Clean flavor_notes (loại bỏ entries chứa "ORIGIN\n...")
def clean_flavor_notes(notes):
    return [n.strip() for n in notes if not n.startswith("ORIGIN")]

# Bước 2: Tách typology → species + cultivar
def parse_typology(typo_list):
    species, cultivars = [], []
    for t in typo_list:
        parts = t.split("\n")
        if len(parts) >= 1:
            species.append(parts[0].strip())
        if len(parts) >= 2:
            cultivars.append(parts[1].strip())
    return list(set(species)), cultivars

# Bước 3: Clean processing (loại entries chứa "ORIGIN\n...")
def clean_processing(proc_list):
    return [p.strip() for p in proc_list if not p.startswith("ORIGIN")]

# Bước 4: Chuẩn hóa roast_level
ROAST_MAP = {
    "Light": "Light", "Medium Light": "Medium-Light",
    "Medium": "Medium", "Medium Dark": "Medium-Dark",
    "Dark": "Dark", "—": "Unknown", "": "Unknown"
}

# Bước 5: Tạo composite text field cho embedding
def build_bean_document(bean):
    parts = [
        bean["product_name"],
        bean.get("about_description", ""),
        f"Origin: {bean.get('origin', '')}",
        f"Country: {bean.get('country', '')}",
        f"Roast: {bean.get('roast_level', '')}",
        f"Flavor: {', '.join(bean.get('flavor_notes_clean', []))}",
        f"Processing: {', '.join(bean.get('processing_clean', []))}",
        f"Type: {', '.join(bean.get('species', []))}",
    ]
    return ". ".join([p for p in parts if p and p.split(": ")[-1]])
```

### 4.2. News — Cleaning & Chunking

```python
# Bước 1: Clean tags (loại navigation items)
NAV_KEYWORDS = ["Skip to content", "Login", "Subscribe", "[has_child]",
                "[subitem]", "Share on", "Privacy Policy", "@"]
def clean_tags(tags):
    return [t for t in tags if not any(kw in t for kw in NAV_KEYWORDS)]

# Bước 2: Chunking content_text
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "]
)

def chunk_article(article):
    chunks = splitter.split_text(article["content_text"])
    return [{
        "text": chunk,
        "title": article["title"],
        "source": article["source"],
        "publish_datetime": article["publish_datetime"],
        "article_url": article["article_url"],
        "chunk_index": i
    } for i, chunk in enumerate(chunks)]
```

---

## 5. Cấu trúc thư mục

```
coffee-rag/
├── README.md                    # Tổng quan hệ thống (English)
├── AGENTS.md                    # Chi tiết thiết kế hệ thống (file này)
├── coffee_beans.json            # Dữ liệu beans gốc (14,537 sản phẩm)
├── coffee_news.json             # Dữ liệu news gốc (1,947 bài báo)
├── ragas_eval_dataset.json      # Dataset đánh giá chính (600 cases, 3 intents: PS/SM/NS)
├── requirements.txt
├── .env.example                 # Template biến môi trường
│
├── data/
│   ├── processed/
│   │   ├── beans_clean.parquet  # Beans sau khi cleaning (14,537 rows)
│   │   ├── news_clean.parquet   # News sau khi cleaning (1,947 rows)
│   │   └── news_chunks.parquet  # News đã chunk (13,427 chunks)
│   └── embeddings/
│       ├── beans_embeddings.npy # Embedding vectors (14537 × 1024, BAAI/bge-m3)
│       ├── news_embeddings.npy  # Embedding vectors (13427 × 1024, BAAI/bge-m3)
│       ├── beans.index          # FAISS IndexFlatIP
│       └── news.index           # FAISS IndexFlatIP
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_beans.py       # Cleaning pipeline cho beans
│   │   ├── clean_news.py        # Cleaning + chunking cho news
│   │   └── build_embeddings.py  # Tạo embeddings + FAISS index
│   ├── retrieval/
│   │   ├── semantic_search.py   # Module 2B: FAISS vector search
│   │   ├── structured_filter.py # Module 2A: Filter theo metadata
│   │   ├── product_matcher.py   # Exact + fuzzy product name matching
│   │   └── reranker.py          # Module 3: RRF fusion
│   ├── generation/
│   │   ├── prompt_templates.py  # Bilingual prompt templates
│   │   ├── schemas.py           # Pydantic response schemas
│   │   └── llm_client.py        # Gemma 4 E4B via Ollama / OpenAI
│   ├── query/
│   │   ├── intent_classifier.py # Module 1A: Rule-based intent
│   │   └── entity_extractor.py  # Module 1B: LLM + rule-based entities
│   └── pipeline.py              # Orchestrate M1→M4
│
├── evaluation/
│   ├── ragas_eval.py            # RAGAS evaluation runner (intent-aware metrics)
│   ├── eda.py                   # EDA charts & statistics (10 chart types)
│   ├── generate_dataset.py      # Tạo eval dataset (LLM-powered, retrieval-grounded)
│   ├── fix_ground_truth.py      # Utility: thay ground_truth_contexts bằng real beans
│   └── results/                 # CSV kết quả đánh giá + charts (PNG)
│
├── app/
│   └── streamlit_app.py         # UI chatbot (Streamlit)
│
└── .cursor/skills/memory/       # Agent memory
```

---

## 6. Tech Stack & Dependencies

```txt
# Core
python>=3.10
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
pydantic>=2.0                   # Structured LLM output schemas

# Embedding & Vector Search
sentence-transformers>=2.7      # model: BAAI/bge-m3 (1024-dim, multilingual)
faiss-cpu>=1.7                  # IndexFlatIP (inner product = cosine similarity)

# Text Processing
langchain-text-splitters>=0.2   # RecursiveCharacterTextSplitter

# LLM (OpenAI GPT-4o-mini hoặc Ollama Gemma 4 E4B)
openai>=1.0                     # OpenAI-compatible client
python-dotenv>=1.0              # .env config loading
ragas>=0.4.3                    # RAG evaluation framework

# UI
streamlit>=1.30
```

## 7. Cách chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Cấu hình
cp .env.example .env
# Chỉnh LLM_PROVIDER, OPENAI_API_KEY, EMBEDDING_MODEL trong .env

# 3. (Nếu chưa có data processed) Chạy preprocessing
python -m src.preprocessing.clean_beans
python -m src.preprocessing.clean_news
python -m src.preprocessing.build_embeddings   # --force nếu đổi model

# 4. Chạy chatbot UI
streamlit run app/streamlit_app.py

# 5. Chạy evaluation
python -m evaluation.ragas_eval --mode retrieval --limit 50
python -m evaluation.ragas_eval --mode full --limit 20
```