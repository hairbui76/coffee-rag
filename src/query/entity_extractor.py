"""Module 1B: Extract structured entities from user query.

Uses LLM (via OpenAI-compatible API) to extract coffee-related entities.
Falls back to rule-based extraction if LLM is unavailable.
"""

import json
import os
import re

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("OPENAI_MODEL", "gemma4:e4b"))

EXTRACT_PROMPT = """\
You are a coffee entity extractor. Extract coffee-related entities from the user query.

IMPORTANT RULES:
- "product": Extract the EXACT product name if the user mentions a specific coffee by name.
  Do NOT split product names into other fields. For example, "MIDNIGHT CHOCOLATE" is a
  product name, not a flavor.
- "roaster": Extract the roaster/brand name if mentioned (e.g. "Every Half", "Starbucks").
- "flavor": ONLY extract actual taste/aroma descriptors the user is looking for
  (e.g. ["chocolate", "fruity"]), NOT words from a product name.
- "origin": country or region name (e.g. "Vietnam", "Ethiopia")
- "roast": roast level (Light, Medium-Light, Medium, Medium-Dark, Dark)
- "processing": processing method (Washed, Natural, Honey, Anaerobic, etc.)
- "typology": coffee species (Arabica, Robusta, Liberica)

Return ONLY valid JSON with these 7 fields. Use null if not mentioned.

Examples:
Query: "Gợi ý cà phê tương tự 'MIDNIGHT CHOCOLATE' của Every Half."
JSON: {"flavor": null, "origin": null, "roast": null, "processing": null, "typology": null, "roaster": "Every Half", "product": "MIDNIGHT CHOCOLATE"}

Query: "Tìm cà phê vị chocolate, medium roast, từ Việt Nam"
JSON: {"flavor": ["chocolate"], "origin": "Vietnam", "roast": "Medium", "processing": null, "typology": null, "roaster": null, "product": null}

Query: "Compare Ethiopia Yirgacheffe with Kenya AA"
JSON: {"flavor": null, "origin": null, "roast": null, "processing": null, "typology": null, "roaster": null, "product": null}

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

_QUOTED_NAME_RE = re.compile(r"""['""'\u2018\u2019\u201C\u201D]([^'""'\u2018\u2019\u201C\u201D]{2,50})['""'\u2018\u2019\u201C\u201D]""")
_OF_ROASTER_RE = re.compile(r"(?:của|by|from|of)\s+([A-Z][\w\s&'.]+?)(?:\s*[.,;?!]|$)", re.IGNORECASE)


def _rule_based_extract(query: str) -> dict:
    q = query.lower()
    entities: dict = {
        "flavor": None, "origin": None, "roast": None,
        "processing": None, "typology": None, "roaster": None,
        "product": None,
    }

    quoted = _QUOTED_NAME_RE.search(query)
    if quoted:
        entities["product"] = quoted.group(1).strip()

    roaster_match = _OF_ROASTER_RE.search(query)
    if roaster_match:
        entities["roaster"] = roaster_match.group(1).strip()

    product_lower = (entities["product"] or "").lower()
    flavors = [f for f in FLAVOR_KEYWORDS if f in q and f not in product_lower]
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
        if proc in q and proc not in product_lower:
            entities["processing"] = proc.title()
            break

    return entities


def extract_entities(query: str, client: OpenAI | None = None) -> dict:
    """Try LLM extraction first, fall back to rules."""
    if client:
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": EXTRACT_PROMPT.format(query=query)}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                expected_keys = {"flavor", "origin", "roast", "processing", "typology", "roaster", "product"}
                for key in expected_keys:
                    result.setdefault(key, None)
                return result
        except Exception:
            pass

    return _rule_based_extract(query)
