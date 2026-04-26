"""Module 1B: Extract structured entities from user query.

Uses LLM (via Ollama) to extract coffee-related entities from free-text queries.
Falls back to rule-based extraction if LLM is unavailable.
"""

import json
import re

from openai import OpenAI

EXTRACT_PROMPT = """\
Extract coffee-related entities from the user query. Return ONLY valid JSON.

Fields to extract (use null if not mentioned):
- "flavor": list of flavor/taste notes (e.g. ["chocolate", "fruity"])
- "origin": country or region name (e.g. "Vietnam", "Ethiopia")
- "roast": roast level (Light, Medium-Light, Medium, Medium-Dark, Dark)
- "processing": processing method (Washed, Natural, Honey, Anaerobic, etc.)
- "typology": coffee species (Arabica, Robusta, Liberica)
- "roaster": roaster/brand name
- "product": specific product name if mentioned

Query: {query}

JSON:"""

ROAST_KEYWORDS = {
    "light": "Light", "nhạt": "Light", "sáng": "Light",
    "medium-light": "Medium-Light", "medium light": "Medium-Light",
    "medium": "Medium", "trung bình": "Medium", "vừa": "Medium",
    "medium-dark": "Medium-Dark", "medium dark": "Medium-Dark",
    "dark": "Dark", "đậm": "Dark", "tối": "Dark",
}

ORIGIN_KEYWORDS = [
    "vietnam", "việt nam", "ethiopia", "colombia", "brazil", "kenya",
    "guatemala", "indonesia", "peru", "costa rica", "honduras",
    "thailand", "myanmar", "laos", "lào", "panama", "jamaica",
    "yemen", "india", "ấn độ", "mexico", "rwanda", "burundi",
    "el salvador", "nicaragua", "tanzania", "uganda", "papua",
]

FLAVOR_KEYWORDS = [
    "chocolate", "fruity", "floral", "nutty", "citrus", "berry",
    "caramel", "honey", "spicy", "herbal", "tropical", "stone fruit",
    "vanilla", "cocoa", "almond", "hazelnut", "peach", "apple",
    "cherry", "blueberry", "raspberry", "strawberry", "mango",
    "jasmine", "rose", "lavender", "tea", "wine", "whiskey",
    "sô cô la", "hoa quả", "trái cây", "hoa", "hạt", "cam",
    "mật ong", "chua", "đắng", "ngọt", "béo", "kem",
]


def _rule_based_extract(query: str) -> dict:
    q = query.lower()
    entities: dict = {
        "flavor": None, "origin": None, "roast": None,
        "processing": None, "typology": None, "roaster": None,
        "product": None,
    }

    flavors = [f for f in FLAVOR_KEYWORDS if f in q]
    if flavors:
        entities["flavor"] = flavors

    for kw, level in ROAST_KEYWORDS.items():
        if kw in q:
            entities["roast"] = level
            break

    for origin in ORIGIN_KEYWORDS:
        if origin in q:
            entities["origin"] = origin.title()
            break

    for species in ["arabica", "robusta", "liberica"]:
        if species in q:
            entities["typology"] = species.title()
            break

    for proc in ["washed", "natural", "honey", "anaerobic", "wet hulled"]:
        if proc in q:
            entities["processing"] = proc.title()
            break

    return entities


def extract_entities(query: str, client: OpenAI | None = None) -> dict:
    """Try LLM extraction first, fall back to rules."""
    if client:
        try:
            resp = client.chat.completions.create(
                model="gemma4:e4b",
                messages=[{"role": "user", "content": EXTRACT_PROMPT.format(query=query)}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

    return _rule_based_extract(query)
