"""Module 1A: Classify user query intent using keyword rules.

Priority order matters: more specific intents are checked first so that
queries like "Gợi ý cà phê tương tự X" match similar_search (not product_search).
"""

import re

INTENT_PRIORITY = [
    ("edge_case", [
        r"^.{0,15}$",                                         # very short / vague
        r"ignore.*(previous|instructions|above)|forget.*prompt",  # adversarial
        r"book.*(flight|hotel|ticket)|giá cổ phiếu|stock price", # out-of-scope
    ]),
    ("exploration", [
        r"(bao nhiêu|how many|tổng (số|cộng)|total)\b.*\b(hạt|bean|loại|quốc gia|countr|roaster|nhà rang|news|tin|chunk)",
        r"\b(top|xếp hạng|ranking|hàng đầu)\b.*\d+|\d+.*\b(top|hàng đầu)\b",
        r"which.*(countr|roaster|origin).*(most|produce|top)",
        r"\b(distribution|phân bố|tỷ lệ)\b",
        r"(phổ biến nhất|most (common|popular))",
        r"\b(thống kê|statistic|overview|tổng quan)\b",
    ]),
    ("similar_search", [
        r"tương tự|giống|similar\b|like\b",
        r"alternatives|thay thế|comparable",
    ]),
    ("comparison", [
        r"so sánh|compare|khác (gì|nhau|biệt)|differ|vs\b|versus",
    ]),
    ("news_search", [
        r"tin tức|news|thị trường|market|trend|xu hướng|latest|mới nhất",
    ]),
    ("product_search", [
        r"tìm|gợi ý|recommend|suggest|cho tôi|give me|looking for|find|giới thiệu",
        r"cà phê có vị|coffee with|cà phê.*rang|want.*coffee",
        r"pha (filter|espresso|pour.?over|drip)",
    ]),
    ("knowledge_qa", [
        r"là gì|what is|how (does|do|to)|tại sao|why|explain|giải thích",
        r"phương pháp|method",
    ]),
]


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    for intent, patterns in INTENT_PRIORITY:
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "product_search"
