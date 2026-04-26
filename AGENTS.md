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

**Intent Classification** — Phân loại ý định bằng rule-based hoặc zero-shot LLM:

| Intent               | Trigger keywords / patterns                        |
| -------------------- | -------------------------------------------------- |
| `product_search`     | "tìm", "gợi ý", "recommend", "cà phê có vị..."   |
| `similar_search`     | "giống", "tương tự", "similar to", tên sản phẩm   |
| `comparison`         | "so sánh", "khác gì", "vs", "compare"             |
| `knowledge_qa`       | "là gì", "what is", "how", "tại sao"              |
| `news_search`        | "tin tức", "news", "thị trường", "market"          |

**Entity Extraction** — Trích xuất các thuộc tính từ query:

```python
entities = {
    "flavor":     ["chocolate", "fruity"],    # từ flavor_notes vocab
    "origin":     ["Vietnam"],                # từ country/origin vocab
    "roast":      "Medium",                   # từ roast_level vocab
    "processing": None,                       # từ processing vocab
    "typology":   "Arabica",                  # từ typology vocab
    "roaster":    None,                       # từ roaster_name vocab
    "product":    None                        # tên sản phẩm cụ thể
}
```

**Cách triển khai:**
- **Phương án A (đơn giản):** Rule-based + keyword matching với vocab trích từ data
- **Phương án B (nâng cao):** Dùng LLM (Gemma/Qwen) extract entities qua structured prompt

#### Module 2A: Structured Filter

**Mục đích:** Lọc nhanh candidates từ DB dựa trên metadata có cấu trúc.

```python
def structured_filter(beans_df, entities):
    mask = pd.Series(True, index=beans_df.index)

    if entities.get("origin"):
        mask &= beans_df["country"].str.contains(entities["origin"], case=False)

    if entities.get("roast"):
        mask &= beans_df["roast_level_clean"] == entities["roast"]

    if entities.get("flavor"):
        for f in entities["flavor"]:
            mask &= beans_df["flavor_notes_flat"].str.contains(f, case=False)

    if entities.get("typology"):
        mask &= beans_df["species"] == entities["typology"]

    return beans_df[mask]
```

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
   Model gợi ý: BAAI/bge-small-en-v1.5 (dim=384, fast, multilingual ok)
   Backup:       all-MiniLM-L6-v2 (dim=384, rất nhẹ)
   Nâng cao:     BAAI/bge-m3 (dim=1024, multilingual tốt nhất)
   ```

4. Lưu vectors vào **FAISS** (IndexFlatIP hoặc IndexIVFFlat) hoặc **ChromaDB**.

**Querying (online):**

```python
query_vec = model.encode(user_query)
top_k_beans = faiss_beans.search(query_vec, k=20)
top_k_news  = faiss_news.search(query_vec, k=5)
```

#### Module 3: Re-Ranking & Fusion

**Mục đích:** Kết hợp kết quả từ structured filter và semantic search, re-rank.

**Phương án Reciprocal Rank Fusion (RRF):**

```python
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

# Với mỗi document d xuất hiện trong >=1 danh sách kết quả:
# score(d) = Σ rrf_score(rank_of_d_in_list_i)  for all lists i
```

**Phương án Weighted Hybrid (đơn giản hơn):**

```python
hybrid_score = alpha * structured_match_score + beta * semantic_similarity
# alpha, beta tunable (mặc định alpha=0.3, beta=0.7)
```

**Phương án Cross-Encoder Re-rank (nâng cao, optional):**

```python
# Lấy top-50 từ bi-encoder, re-rank bằng cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc) for doc in top_50_docs])
# Sắp xếp theo scores, lấy top-K
```

#### Module 4: Response Generation

**Mục đích:** Tổng hợp retrieved context thành câu trả lời tự nhiên.

**Prompt template:**

```
SYSTEM:
Bạn là Coffee Advisor, chuyên gia tư vấn cà phê specialty.
Trả lời dựa HOÀN TOÀN vào thông tin được cung cấp bên dưới.
Nếu không có đủ thông tin, hãy nói rõ.
Trả lời bằng ngôn ngữ phù hợp với câu hỏi (Tiếng Việt hoặc English).

RETRIEVED CONTEXT:
--- Bean 1 ---
Name: {product_name}
Roaster: {roaster_name}
Origin: {origin}
Flavor: {flavor_notes}
Roast: {roast_level}
Processing: {processing}
Description: {about_description}
Buy: {buy_links}
--- Bean 2 ---
...
--- Related Article ---
Title: {title}
Summary: {summary}
...

USER QUERY:
{user_query}

Respond with:
1. Tóm tắt tư vấn (2-3 câu)
2. Danh sách sản phẩm gợi ý (kèm lý do cho mỗi sản phẩm)
3. Bài báo liên quan (nếu có)
```

**LLM options:**

| Model                     | Ưu điểm                      | Nhược điểm          |
| ------------------------- | ----------------------------- | -------------------- |
| OpenAI GPT-4o-mini        | Chất lượng cao, dễ gọi API   | Tốn phí, cần API key |
| Google Gemma 2 9B (local) | Miễn phí, chạy local         | Cần GPU >=16GB       |
| Qwen2.5 7B (local)       | Hỗ trợ tiếng Việt tốt        | Cần GPU >=16GB       |
| Groq API (Llama 3.1 8B)  | Miễn phí tier, rất nhanh     | Rate limit           |

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
├── AGENTS.md                    # Mô tả hệ thống RAG (file này)
├── coffee_beans.json            # Dữ liệu beans gốc (14,537 sản phẩm)
├── coffee_news.json             # Dữ liệu news gốc (1,947 bài báo)
├── requirements.txt
│
├── data/
│   ├── processed/
│   │   ├── beans_clean.parquet  # Beans sau khi cleaning (14,537 rows)
│   │   ├── news_clean.parquet   # News sau khi cleaning (1,947 rows)
│   │   └── news_chunks.parquet  # News đã chunk (13,427 chunks)
│   └── embeddings/
│       ├── beans_embeddings.npy # Embedding vectors (dim=384)
│       ├── news_embeddings.npy  # Embedding vectors (dim=384)
│       ├── beans.index          # FAISS IndexFlatIP
│       └── news.index           # FAISS IndexFlatIP
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_beans.py       # Cleaning pipeline cho beans
│   │   ├── clean_news.py        # Cleaning + chunking cho news
│   │   └── build_embeddings.py  # Tạo embeddings + FAISS index
│   ├── retrieval/
│   │   ├── semantic_search.py   # Module 2B: Vector search
│   │   ├── structured_filter.py # Module 2A: Filter theo metadata
│   │   └── reranker.py          # Module 3: RRF fusion
│   ├── generation/
│   │   ├── prompt_templates.py  # Prompt templates cho các intent
│   │   └── llm_client.py        # Gemma 4 E4B via Ollama
│   ├── query/
│   │   ├── intent_classifier.py # Module 1A: Intent classification
│   │   └── entity_extractor.py  # Module 1B: Entity extraction
│   └── pipeline.py              # Orchestrate M1->M4
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

# Embedding & Vector Search
sentence-transformers>=2.7      # model: paraphrase-multilingual-MiniLM-L12-v2
faiss-cpu>=1.7                  # IndexFlatIP (inner product)

# Text Processing
langchain-text-splitters>=0.2   # RecursiveCharacterTextSplitter

# LLM (Gemma 4 E4B via Ollama)
openai>=1.0                     # OpenAI-compatible client for Ollama

# UI
streamlit>=1.30
```

## 7. Cách chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Chạy Ollama với Gemma 4 E4B
ollama pull gemma4:e4b
ollama serve

# 3. (Nếu chưa có data processed) Chạy preprocessing
python -m src.preprocessing.clean_beans
python -m src.preprocessing.clean_news
python -m src.preprocessing.build_embeddings

# 4. Chạy chatbot UI
streamlit run app/streamlit_app.py
```