"""Module 1A: Classify user query intent using keyword rules."""

import re

INTENT_PATTERNS = {
    "product_search": [
        r"tìm|gợi ý|recommend|suggest|cho tôi|give me|looking for",
        r"cà phê có vị|coffee with|cà phê.*rang|want.*coffee",
        r"pha (filter|espresso|pour.?over|drip)",
    ],
    "similar_search": [
        r"giống|tương tự|similar|like\b",
        r"alternatives|thay thế|comparable",
    ],
    "comparison": [
        r"so sánh|compare|khác gì|differ|vs\b|versus",
    ],
    "knowledge_qa": [
        r"là gì|what is|how (does|do|to)|tại sao|why|explain|giải thích",
        r"process|phương pháp|method|cách",
    ],
    "news_search": [
        r"tin tức|news|thị trường|market|trend|xu hướng|latest|mới nhất",
    ],
}


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "product_search"
