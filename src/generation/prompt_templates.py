"""Prompt templates for RAG response generation."""

import re

SYSTEM_PROMPT_VI = """\
Bạn là Coffee Advisor, chuyên gia tư vấn cà phê specialty.
Trả lời HOÀN TOÀN bằng TIẾNG VIỆT, kể cả khi dữ liệu tham khảo bằng tiếng Anh.
Dựa trên thông tin được cung cấp bên dưới để trả lời. Nếu không đủ thông tin, hãy nói rõ.
Trả lời ngắn gọn, hữu ích. Mọi field trong response đều phải bằng tiếng Việt.
Chỉ gợi ý sản phẩm nếu context có sản phẩm phù hợp. Giữ nguyên product URL từ context."""

SYSTEM_PROMPT_EN = """\
You are Coffee Advisor, a specialty coffee expert.
You MUST respond ENTIRELY in ENGLISH.
Answer based on the provided context below. If information is insufficient, say so clearly.
Be concise and helpful. Only recommend products if the context contains relevant ones.
Keep product URLs exactly as provided in the context."""

CONTEXT_TEMPLATE = """\
=== RETRIEVED COFFEE BEANS ===
{beans_context}

=== RELATED ARTICLES ===
{news_context}"""

BEAN_TEMPLATE = """\
--- Bean {i} ---
Name: {product_name}
Roaster: {roaster_name}
Origin: {origin}
Roast: {roast_level_clean}
Flavor: {flavor_notes}
Processing: {processing}
Species: {species}
Description: {about_description}
URL: {product_url}"""

NEWS_TEMPLATE = """\
--- Article ---
Title: {title}
Source: {source}
Date: {publish_datetime}
Content: {text}"""

USER_TEMPLATE_VI = """\
{context}

CÂU HỎI:
{query}

Trả lời bằng TIẾNG VIỆT. Điền đầy đủ các field trong schema."""

USER_TEMPLATE_EN = """\
{context}

USER QUESTION:
{query}

Respond in ENGLISH. Fill all schema fields."""


_VI_PATTERN = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    r"ùúủũụưứừửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Return 'vi' if text contains Vietnamese diacritics, else 'en'."""
    return "vi" if _VI_PATTERN.search(text) else "en"


def _as_list(val) -> list:
    """Safely convert a value (list, numpy array, NaN, None) to a plain list."""
    if val is None:
        return []
    try:
        if hasattr(val, "tolist"):
            return val.tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
    except (TypeError, ValueError):
        pass
    return []


def _as_str(val) -> str:
    if val is None:
        return ""
    try:
        import numpy as np
        if isinstance(val, float) and np.isnan(val):
            return ""
    except (ImportError, TypeError):
        pass
    return str(val)


def format_beans_context(beans_df) -> str:
    if beans_df is None or beans_df.empty:
        return "(No matching beans found)"
    parts = []
    for i, (_, row) in enumerate(beans_df.head(5).iterrows(), 1):
        parts.append(BEAN_TEMPLATE.format(
            i=i,
            product_name=_as_str(row.get("product_name")),
            roaster_name=_as_str(row.get("roaster_name")),
            origin=_as_str(row.get("origin")),
            roast_level_clean=_as_str(row.get("roast_level_clean")),
            flavor_notes=", ".join(_as_list(row.get("flavor_notes_clean"))),
            processing=", ".join(_as_list(row.get("processing_clean"))),
            species=", ".join(_as_list(row.get("species"))),
            about_description=_as_str(row.get("about_description"))[:300],
            product_url=_as_str(row.get("product_url")),
        ))
    return "\n".join(parts)


def format_news_context(news_df) -> str:
    if news_df is None or news_df.empty:
        return "(No related articles)"
    parts = []
    for _, row in news_df.head(3).iterrows():
        dt = row.get("publish_datetime", "")
        if hasattr(dt, "strftime") and not (hasattr(dt, "isnull") and dt.isnull() or str(dt) == "NaT"):
            dt = dt.strftime("%Y-%m-%d")
        else:
            dt = ""
        parts.append(NEWS_TEMPLATE.format(
            title=row.get("title", ""),
            source=row.get("source", ""),
            publish_datetime=dt,
            text=(row.get("text") or row.get("summary") or "")[:400],
        ))
    return "\n".join(parts)


def build_prompt(query: str, beans_df=None, news_df=None) -> list[dict]:
    lang = detect_language(query)
    context = CONTEXT_TEMPLATE.format(
        beans_context=format_beans_context(beans_df),
        news_context=format_news_context(news_df),
    )
    if lang == "vi":
        system_prompt = SYSTEM_PROMPT_VI
        user_content = USER_TEMPLATE_VI.format(context=context, query=query)
    else:
        system_prompt = SYSTEM_PROMPT_EN
        user_content = USER_TEMPLATE_EN.format(context=context, query=query)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
