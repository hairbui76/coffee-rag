"""Module 1A: Classify user query intent using keyword rules.

Priority order matters: more specific intents are checked first so that
queries like "Gợi ý cà phê tương tự X" match similar_search (not product_search).
"""

import re

INTENT_PRIORITY = [
    ("similar_search", [
        r"tương tự|giống|similar\b|like\b",
        r"alternatives|thay thế|comparable",
    ]),
    ("comparison", [
        r"so sánh|compare|khác (gì|nhau)|differ|vs\b|versus",
    ]),
    ("news_search", [
        r"tin tức|news|thị trường|market|trend|xu hướng|latest|mới nhất",
    ]),
    ("knowledge_qa", [
        r"là gì|what is|how (does|do|to)|tại sao|why|explain|giải thích",
        r"phương pháp|method",
    ]),
    ("product_search", [
        r"tìm|gợi ý|recommend|suggest|cho tôi|give me|looking for",
        r"cà phê có vị|coffee with|cà phê.*rang|want.*coffee",
        r"pha (filter|espresso|pour.?over|drip)",
    ]),
]


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    for intent, patterns in INTENT_PRIORITY:
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "product_search"
