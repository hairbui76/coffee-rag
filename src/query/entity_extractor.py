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
- "roaster": ONLY extract a real roaster/brand name (e.g. "Every Half", "Starbucks").
  A roaster is a company that roasts coffee. Country names, flavor descriptions, and
  phrases like "from Colombia" are NEVER roasters. If no roaster is mentioned, use null.
- "flavor": ONLY extract actual taste/aroma descriptors the user is looking for.
  Always translate to English (e.g. "dâu tây"→"Strawberry", "sô cô la"→"Chocolate",
  "đào"→"Peach", "việt quất"→"Blueberry", "mật ong"→"Honey", "hoa nhài"→"Jasmine",
  "cam"→"Orange", "chanh"→"Citrus Fruit", "táo"→"Apple", "hoa"→"Floral",
  "trái cây"→"Fruity", "kem"→"Creamy", "ca cao"→"Cocoa", "đường nâu"→"Brown Sugar",
  "caramel"→"Caramelized", "brown sugar"→"Brown Sugar").
  Do NOT include words from a product name.
- "origin": country or region name (e.g. "Vietnam", "Ethiopia")
- "roast": roast level (Light, Medium-Light, Medium, Medium-Dark, Dark)
- "processing": processing method (Washed, Natural, Honey, Anaerobic, etc.)
- "typology": coffee species (Arabica, Robusta, Liberica)

Return ONLY valid JSON with these 7 fields. Use null if not mentioned.

Examples:
Query: "Gợi ý cà phê tương tự 'MIDNIGHT CHOCOLATE' của Every Half."
JSON: {"flavor": null, "origin": null, "roast": null, "processing": null, "typology": null, "roaster": "Every Half", "product": "MIDNIGHT CHOCOLATE"}

Query: "Tìm cà phê vị chocolate, medium roast, từ Việt Nam"
JSON: {"flavor": ["Chocolate"], "origin": "Vietnam", "roast": "Medium", "processing": null, "typology": null, "roaster": null, "product": null}

Query: "Can you help me find a light roast coffee from Colombia that has floral flavor notes?"
JSON: {"flavor": ["Floral"], "origin": "Colombia", "roast": "Light", "processing": null, "typology": null, "roaster": null, "product": null}

Query: "Can you recommend a medium roast coffee from Kenya that has brown sugar flavor notes?"
JSON: {"flavor": ["Brown Sugar"], "origin": "Kenya", "roast": "Medium", "processing": null, "typology": null, "roaster": null, "product": null}

Query: "Compare Ethiopia Yirgacheffe with Kenya AA"
JSON: {"flavor": null, "origin": null, "roast": null, "processing": null, "typology": null, "roaster": null, "product": null}

Query: {query}
JSON:"""

ROAST_KEYWORDS = [
    ("vừa-nhẹ", "Medium-Light"), ("vừa nhẹ", "Medium-Light"),
    ("vừa nhạt", "Medium-Light"), ("vừa sáng", "Medium-Light"),
    ("medium-light", "Medium-Light"), ("medium light", "Medium-Light"),
    ("vừa đến tối", "Medium-Dark"), ("vừa đến đậm", "Medium-Dark"),
    ("trung bình đến đậm", "Medium-Dark"),
    ("vừa-dark", "Medium-Dark"), ("vừa đậm", "Medium-Dark"), ("vừa tối", "Medium-Dark"),
    ("trung-cao", "Medium-Dark"), ("trung cao", "Medium-Dark"),
    ("medium-dark", "Medium-Dark"), ("medium dark", "Medium-Dark"),
    ("light", "Light"), ("nhạt", "Light"), ("sáng", "Light"), ("nhẹ", "Light"),
    ("medium", "Medium"), ("trung bình", "Medium"), ("vừa", "Medium"),
    ("dark", "Dark"), ("đậm", "Dark"), ("tối", "Dark"),
]

ORIGIN_KEYWORDS = [
    "vietnam", "việt nam", "ethiopia", "colombia", "brazil", "kenya",
    "guatemala", "indonesia", "peru", "costa rica", "honduras",
    "thailand", "myanmar", "laos", "lào", "panama", "jamaica",
    "yemen", "india", "ấn độ", "mexico", "rwanda", "burundi",
    "el salvador", "nicaragua", "tanzania", "uganda", "papua",
]

FLAVOR_KEYWORDS = [
    "brown sugar", "stone fruit", "dark chocolate", "milk chocolate",
    "black tea", "citrus fruit",
    "chocolate", "fruity", "floral", "nutty", "citrus", "berry",
    "caramel", "honey", "spicy", "herbal", "tropical",
    "vanilla", "cocoa", "almond", "hazelnut", "peach", "apple",
    "cherry", "blueberry", "raspberry", "strawberry", "mango",
    "jasmine", "rose", "lavender", "tea", "wine", "whiskey",
    "plum", "grape", "orange", "lemon", "lime", "grapefruit",
    "sô cô la đen", "sô cô la", "trái cây họ cam",
    "hoa quả", "trái cây", "hoa", "hạt", "cam",
    "mật ong", "chua", "đắng", "ngọt", "béo", "kem", "đường nâu",
    "mâm xôi", "hương nhài",
]

VI_FLAVOR_MAP = {
    "quả mâm xôi": "Raspberry", "mâm xôi": "Raspberry",
    "dâu tây": "Strawberry", "dâu": "Strawberry", "việt quất": "Blueberry",
    "đào": "Peach", "mận": "Plum", "cherry": "Cherry", "anh đào": "Cherry",
    "táo": "Apple", "cam": "Orange", "chanh": "Citrus Fruit", "bưởi": "Grapefruit",
    "xoài": "Mango", "dừa": "Coconut", "chuối": "Banana",
    "sô cô la đen": "Dark Chocolate", "socola đen": "Dark Chocolate",
    "sô cô la": "Chocolate", "socola": "Chocolate", "ca cao": "Cocoa",
    "trái cây họ cam": "Citrus Fruit",
    "caramel": "Caramelized", "mật ong": "Honey", "vani": "Vanilla",
    "hạnh nhân": "Almond", "hạt phỉ": "Hazelnut", "hạt điều": "Cashew",
    "hương nhài": "Jasmine", "hoa nhài": "Jasmine", "hoa hồng": "Rose", "hoa": "Floral",
    "trái cây": "Fruity", "hoa quả": "Fruity", "nhiệt đới": "Tropical",
    "trà đen": "Black Tea", "trà": "Black Tea", "rượu vang": "Wine",
    "kem": "Creamy", "béo": "Creamy", "bơ": "Butterscotch",
    "đường nâu": "Brown Sugar", "mía": "Brown Sugar",
    "cay": "Spicy", "thảo mộc": "Herbal",
}

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
    vi_mapped: set[str] = set()
    for vi_kw, en_flavor in VI_FLAVOR_MAP.items():
        if vi_kw in q:
            vi_mapped.add(vi_kw)
            if en_flavor not in flavors:
                flavors.append(en_flavor)
    flavors = [f for f in flavors if f not in vi_mapped]
    if flavors:
        entities["flavor"] = flavors

    for kw, level in ROAST_KEYWORDS:
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


def _clean_flavors(flavors: list[str] | None) -> list[str] | None:
    """Deduplicate flavors: map Vietnamese to English, remove duplicates."""
    if not flavors:
        return flavors
    cleaned: list[str] = []
    seen_lower: set[str] = set()
    for f in flavors:
        mapped = VI_FLAVOR_MAP.get(f.lower(), f)
        if mapped.lower() not in seen_lower:
            seen_lower.add(mapped.lower())
            cleaned.append(mapped)
    return cleaned or None


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
                result["flavor"] = _clean_flavors(result.get("flavor"))
                return result
        except Exception:
            pass

    return _rule_based_extract(query)
