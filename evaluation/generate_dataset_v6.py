"""Generate evaluation dataset v6: retriever-independent, template-based, no LLM.

Why v6 exists
-------------
Dataset v5 built its ground truth from the output of ``CoffeeRAG.retrieve()``
(generate_dataset.py, Phase 3). Any recall or precision measured against such a
reference is circular: the reference only contains what the retriever already
returned. v6 removes the retriever from the loop entirely.

Relevance definitions (metadata-only, exact rules)
--------------------------------------------------
product_search   R = all beans whose country == C, whose roast == RST (if set),
                 whose flavor list contains F (case-insensitive substring, so
                 "Chocolate" accepts "Dark Chocolate"), and whose processing list
                 contains P (if set).
similar_search   seed bean S. R = all beans b != S with b.country == S.country,
                 b.roast == S.roast and |flavors(b) ∩ flavors(S)| >= 1.
                 The seed itself is excluded from R and from the retrieved list
                 at scoring time (``exclude_ids``).
news_search      easy/medium: R = the one source article (all of its chunks).
                 hard: an organisation name from a title; R = every article whose
                 title contains that name.

Difficulty
----------
product_search   easy = {country, flavor} with |R| >= 15
                 medium = {country, roast, flavor} with |R| >= 4
                 hard = {country, roast, flavor, processing} with |R| >= 2
similar_search   easy |R| >= 20, medium 6 <= |R| <= 19, hard 2 <= |R| <= 5
news_search      easy = title as query, medium = summary sentence as query,
                 hard = organisation name as query (multi-document R)

Each case stores ``relevant_ids`` (product_url or article_url of every member of
R), ``n_relevant``, ``exclude_ids``, a template ``ground_truth`` text that names
two members of R, and ``ground_truth_contexts`` formatted exactly like
``ragas_eval.build_retrieved_contexts`` for up to 8 members of R.

Usage:
    python -m evaluation.generate_dataset_v6 --out ragas_eval_dataset_v6.json
    python -m evaluation.generate_dataset_v6 --dry-run
    python -m evaluation.generate_dataset_v6 --seed 7 --per-intent 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEANS_PATH = ROOT / "data" / "processed" / "beans_clean.parquet"
NEWS_PATH = ROOT / "data" / "processed" / "news_chunks.parquet"
NEWS_CLEAN_PATH = ROOT / "data" / "processed" / "news_clean.parquet"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GENERATOR_VERSION = "v6-template-retriever-independent"

# ── Vocabularies ───────────────────────────────────────────────

ROASTS = ["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"]
ROAST_VI = {
    "Light": "rang nhạt",
    "Medium-Light": "rang vừa nhạt",
    "Medium": "rang vừa",
    "Medium-Dark": "rang vừa đậm",
    "Dark": "rang đậm",
}
ROAST_EN = {
    "Light": "light roast",
    "Medium-Light": "medium-light roast",
    "Medium": "medium roast",
    "Medium-Dark": "medium-dark roast",
    "Dark": "dark roast",
}
FLAVOR_VI = {
    "Chocolate": "sô cô la",
    "Dark Chocolate": "sô cô la đen",
    "Caramelized": "caramel",
    "Floral": "hoa",
    "Citrus Fruit": "trái cây họ cam chanh",
    "Honey": "mật ong",
    "Peach": "đào",
    "Orange": "cam",
    "Strawberry": "dâu tây",
    "Cherry": "anh đào",
    "Jasmine": "hoa nhài",
    "Brown Sugar": "đường nâu",
    "Vanilla": "vani",
    "Grapefruit": "bưởi",
    "Plum": "mận",
    "Raspberry": "mâm xôi",
    "Hazelnut": "hạt phỉ",
    "Black Tea": "trà đen",
    "Cocoa": "ca cao",
    "Blueberry": "việt quất",
    "Almond": "hạnh nhân",
    "Lemon": "chanh",
    "Nutmeg": "nhục đậu khấu",
    "Blackberry": "mâm xôi đen",
}
PROCESSING_VI = {
    "Washed": "chế biến ướt (washed)",
    "Natural": "chế biến khô (natural)",
    "Honey": "chế biến honey",
    "Anaerobic Natural": "lên men yếm khí natural",
}
PROCESSING_EN = {
    "Washed": "washed process",
    "Natural": "natural process",
    "Honey": "honey process",
    "Anaerobic Natural": "anaerobic natural process",
}
PROCESSING_POOL = ["Washed", "Natural", "Honey", "Anaerobic Natural"]

# Question templates. {c}=country {r}=roast phrase {f}=flavor phrase {p}=processing phrase
PS_TEMPLATES = {
    "vi": {
        "easy": [
            "Gợi ý cho tôi cà phê {c} có hương {f}.",
            "Tôi muốn tìm cà phê từ {c} với hương vị {f}, bạn có loại nào không?",
            "Có loại cà phê {c} nào vị {f} không?",
            "Bạn tư vấn giúp tôi một loại cà phê {c}, hương {f}.",
            "Tìm giúp tôi cà phê {c} mang hương {f}.",
        ],
        "medium": [
            "Gợi ý cho tôi cà phê {c}, {r}, có hương {f}.",
            "Tôi muốn mua cà phê từ {c}, {r}, vị {f}. Có loại nào phù hợp không?",
            "Bạn có cà phê {c} {r} với hương {f} không?",
            "Tư vấn giúp tôi một loại cà phê {c}, {r}, hương vị {f}.",
            "Cà phê {c} nào {r} và có nốt hương {f}?",
        ],
        "hard": [
            "Tôi tìm cà phê {c}, {r}, hương {f}, {p}. Bạn gợi ý giúp nhé.",
            "Có loại cà phê {c} nào {r}, vị {f} và được {p} không?",
            "Bạn tư vấn cà phê từ {c}, {p}, {r}, có hương {f} giúp tôi.",
            "Tìm giúp tôi cà phê {c} {p}, {r}, mang nốt hương {f}.",
            "Tôi thích cà phê {c} {p}, {r}, có hương {f}. Có sản phẩm nào không?",
        ],
    },
    "en": {
        "easy": [
            "Recommend me a coffee from {c} with {f} notes.",
            "I am looking for a {c} coffee that tastes of {f}. Any options?",
            "Do you have any {c} coffee with {f} flavor?",
            "Suggest a coffee from {c} with {f} in the cup.",
            "Find me a {c} coffee with {f} notes.",
        ],
        "medium": [
            "Recommend a {r} coffee from {c} with {f} notes.",
            "I want to buy a {c} coffee, {r}, with {f} flavor. What do you have?",
            "Do you have a {r} {c} coffee with {f} notes?",
            "Suggest a {c} coffee, {r}, tasting of {f}.",
            "Which {c} coffees are {r} with {f} in the flavor profile?",
        ],
        "hard": [
            "I am looking for a {p} coffee from {c}, {r}, with {f} notes. Any suggestions?",
            "Is there a {c} coffee that is {r}, {p}, with {f} flavor?",
            "Recommend a {p} {c} coffee, {r}, with {f} notes.",
            "Find me a {c} coffee, {p}, {r}, with {f} in the cup.",
            "I like {p} coffees from {c}, {r}, with {f} notes. What do you have?",
        ],
    },
}

SS_TEMPLATES = {
    "vi": [
        "Gợi ý cho tôi cà phê tương tự '{n}' của {ro}.",
        "Tôi thích '{n}' của {ro}. Có loại nào giống vậy không?",
        "Có cà phê nào tương tự như '{n}' ({ro}) không?",
        "Bạn tìm giúp tôi các loại cà phê giống '{n}' của {ro}.",
        "Cà phê nào có hương vị gần với '{n}' của {ro}?",
    ],
    "en": [
        "Recommend coffees similar to '{n}' by {ro}.",
        "I like '{n}' from {ro}. What else is like it?",
        "Are there coffees similar to '{n}' ({ro})?",
        "Find me alternatives to '{n}' by {ro}.",
        "Which coffees taste close to '{n}' from {ro}?",
    ],
}

NS_TEMPLATES = {
    "vi": {
        "easy": [
            "Cho tôi biết tin tức: {t}",
            "Có bài viết nào về \"{t}\" không?",
            "Tôi muốn đọc tin \"{t}\".",
        ],
        "medium": [
            "Tìm bài viết nói về: {s}",
            "Có bài báo nào nói rằng {s} không?",
            "Cho tôi thông tin về chuyện này: {s}",
        ],
        "hard": [
            "{e} gần đây có tin gì mới?",
            "Tin tức mới nhất về {e}?",
            "Có bài viết nào về {e} không?",
        ],
    },
    "en": {
        "easy": [
            "Tell me about the news: {t}",
            "Is there an article on \"{t}\"?",
            "I want to read the news \"{t}\".",
        ],
        "medium": [
            "Find the article saying: {s}",
            "Is there a story reporting that {s}?",
            "Give me the news on this: {s}",
        ],
        "hard": [
            "What is the latest news about {e}?",
            "Any recent news on {e}?",
            "Are there articles about {e}?",
        ],
    },
}

# ── Helpers ────────────────────────────────────────────────────


def _arr(val: Any) -> list[str]:
    if isinstance(val, np.ndarray):
        return [str(x) for x in val]
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def _bean_context_text(row: pd.Series) -> str:
    """Identical to ragas_eval._bean_contexts() formatting."""
    name = str(row.get("product_name", ""))
    roaster = str(row.get("roaster_name", ""))
    origin = str(row.get("origin", ""))
    country = str(row.get("country", ""))
    roast = str(row.get("roast_level_clean", "") or row.get("roast_level", ""))
    flavors = ", ".join(_arr(row.get("flavor_notes_clean", [])))
    processing = ", ".join(_arr(row.get("processing_clean", [])))
    species = ", ".join(_arr(row.get("species", [])))
    desc = str(row.get("about_description", ""))
    url = str(row.get("product_url", ""))
    return (
        f"Bean: {name}. Roaster: {roaster}. Origin: {origin}. Country: {country}. "
        f"Roast: {roast}. Flavor: {flavors}. Processing: {processing}. "
        f"Species: {species}. Description: {desc}. URL: {url}"
    )


def _news_context(row: pd.Series) -> str:
    title = row.get("title", "")
    source = row.get("source", "")
    date = str(row.get("publish_datetime", ""))[:10]
    text = str(row.get("text", ""))
    url = row.get("article_url", "")
    return f"Article: {title}. Source: {source}. Date: {date}. Content: {text}. URL: {url}"


def _eligible_beans(beans: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (beans["country"] != "")
        & (beans["roast_level_clean"].isin(ROASTS))
        & (beans["flavor_notes_clean"].apply(lambda x: len(_arr(x)) >= 2))
        & (beans["about_description"].str.len() > 100)
        & (beans["origin"] != "")
        & (beans["product_url"] != "")
    )
    out = beans[mask].drop_duplicates(subset="product_url").copy()
    out["_flavors_lc"] = out["flavor_notes_clean"].apply(lambda x: [f.lower() for f in _arr(x)])
    out["_proc_lc"] = out["processing_clean"].apply(lambda x: [p.lower() for p in _arr(x)])
    return out.reset_index(drop=True)


def _flavor_mask(df: pd.DataFrame, flavor: str) -> pd.Series:
    f = flavor.lower()
    return df["_flavors_lc"].apply(lambda xs: any(f in x for x in xs))


def _proc_mask(df: pd.DataFrame, proc: str) -> pd.Series:
    p = proc.lower()
    return df["_proc_lc"].apply(lambda xs: any(p == x or p in x for x in xs))


def _quota(n: int) -> dict[str, int]:
    easy = round(n * 0.30)
    hard = round(n * 0.25)
    return {"easy": easy, "medium": n - easy - hard, "hard": hard}


def _lang_cycle(n: int) -> list[str]:
    return ["vi" if i % 2 == 0 else "en" for i in range(n)]


def _flavor_phrase(flavor: str, lang: str) -> str:
    if lang == "vi":
        return FLAVOR_VI.get(flavor, flavor.lower())
    return flavor.lower()


def _join_names(names: list[str], lang: str) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    sep = " và " if lang == "vi" else " and "
    return ", ".join(names[:-1]) + sep + names[-1]


def _bean_gt_fragment(row: pd.Series, lang: str, prefer: list[str] | None = None) -> str:
    """One clause naming a bean. Flavors that match the query come first."""
    all_flavors = _arr(row["flavor_notes_clean"])
    prefer_lc = [p.lower() for p in (prefer or [])]
    first = [f for f in all_flavors if any(p in f.lower() for p in prefer_lc)]
    rest = [f for f in all_flavors if f not in first]
    flavors = (first + rest)[:3]
    roast = str(row["roast_level_clean"])
    if lang == "vi":
        return (
            f"{row['product_name']} của {row['roaster_name']} ({ROAST_VI.get(roast, roast)}, "
            f"hương {', '.join(_flavor_phrase(f, 'vi') for f in flavors)})"
        )
    return (
        f"{row['product_name']} by {row['roaster_name']} ({ROAST_EN.get(roast, roast)}, "
        f"{', '.join(f.lower() for f in flavors)} notes)"
    )


# ── product_search ─────────────────────────────────────────────


def build_product_search(good: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    quota = _quota(n)
    top_countries = good["country"].value_counts().head(14).index.tolist()
    flavor_counter: Counter = Counter()
    for fl in good["flavor_notes_clean"]:
        for f in _arr(fl):
            flavor_counter[f] += 1
    top_flavors = [f for f, _ in flavor_counter.most_common(30) if f != "—"][:20]

    cases: list[dict] = []
    used: set[tuple] = set()
    idx = 0
    for diff in ["easy", "medium", "hard"]:
        made = 0
        attempts = 0
        while made < quota[diff] and attempts < 20000:
            attempts += 1
            country = rng.choice(top_countries)
            flavor = rng.choice(top_flavors)
            roast = rng.choice(ROASTS) if diff != "easy" else ""
            proc = rng.choice(PROCESSING_POOL) if diff == "hard" else ""
            key = (country, roast, flavor, proc)
            if key in used:
                continue
            mask = (good["country"] == country) & _flavor_mask(good, flavor)
            if roast:
                mask &= good["roast_level_clean"] == roast
            if proc:
                mask &= _proc_mask(good, proc)
            rel = good[mask]
            min_rel = {"easy": 15, "medium": 4, "hard": 2}[diff]
            if len(rel) < min_rel:
                continue
            used.add(key)
            lang = "vi" if idx % 2 == 0 else "en"
            tmpl = rng.choice(PS_TEMPLATES[lang][diff])
            question = tmpl.format(
                c=country,
                r=(ROAST_VI if lang == "vi" else ROAST_EN).get(roast, roast),
                f=_flavor_phrase(flavor, lang),
                p=(PROCESSING_VI if lang == "vi" else PROCESSING_EN).get(proc, proc),
            )
            named = rel.sample(min(2, len(rel)), random_state=rng.randrange(10**9))
            others = rel.drop(named.index)
            extra = others.sample(min(6, len(others)), random_state=rng.randrange(10**9))
            ctx_rows = pd.concat([named, extra])
            frags = [_bean_gt_fragment(r, lang, prefer=[flavor]) for _, r in named.iterrows()]
            if lang == "vi":
                gt = (
                    f"Bạn có thể thử {_join_names(frags, 'vi')}. "
                    f"Các loại này đều từ {country}"
                    + (f", {ROAST_VI[roast]}" if roast else "")
                    + f" và có hương {_flavor_phrase(flavor, 'vi')}"
                    + (f", {PROCESSING_VI[proc]}" if proc else "")
                    + f". Tổng cộng có {len(rel)} sản phẩm phù hợp trong danh mục."
                )
            else:
                gt = (
                    f"You could try {_join_names(frags, 'en')}. "
                    f"Both come from {country}"
                    + (f", {ROAST_EN[roast]}" if roast else "")
                    + f", with {flavor.lower()} notes"
                    + (f", {PROCESSING_EN[proc]}" if proc else "")
                    + f". The catalogue holds {len(rel)} matching products in total."
                )
            idx += 1
            made += 1
            cases.append({
                "id": f"PS_{idx:03d}",
                "intent": "product_search",
                "difficulty": diff,
                "language": lang,
                "question": question,
                "ground_truth": gt,
                "ground_truth_contexts": [_bean_context_text(r) for _, r in ctx_rows.iterrows()],
                "relevant_ids": rel["product_url"].tolist(),
                "n_relevant": int(len(rel)),
                "exclude_ids": [],
                "metadata": {
                    "expected_entities": {
                        "country": country,
                        "roast": roast or None,
                        "flavor": [flavor],
                        "processing": proc or None,
                    },
                    "named_products": named["product_name"].tolist(),
                    "relevance_rule": "country == C and flavor contains F"
                    + (" and roast == R" if roast else "")
                    + (" and processing contains P" if proc else ""),
                },
            })
        if made < quota[diff]:
            print(f"  [product_search/{diff}] only {made}/{quota[diff]} combos found")
    return cases


# ── similar_search ─────────────────────────────────────────────


def build_similar_search(good: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    quota = _quota(n)
    bands = {"easy": (20, 10**9), "medium": (6, 19), "hard": (2, 5)}
    cases: list[dict] = []
    used_seed: set[str] = set()
    counts = {d: 0 for d in quota}
    idx = 0
    order = list(range(len(good)))
    rng.shuffle(order)
    for pos in order:
        if all(counts[d] >= quota[d] for d in quota):
            break
        seed = good.iloc[pos]
        if seed["product_url"] in used_seed or len(str(seed["product_name"])) < 4:
            continue
        seed_flavors = set(seed["_flavors_lc"])
        mask = (
            (good["product_url"] != seed["product_url"])
            & (good["country"] == seed["country"])
            & (good["roast_level_clean"] == seed["roast_level_clean"])
            & good["_flavors_lc"].apply(lambda xs: len(seed_flavors & set(xs)) >= 1)
        )
        rel = good[mask]
        diff = None
        for d, (lo, hi) in bands.items():
            if lo <= len(rel) <= hi and counts[d] < quota[d]:
                diff = d
                break
        if diff is None:
            continue
        used_seed.add(seed["product_url"])
        lang = "vi" if idx % 2 == 0 else "en"
        question = rng.choice(SS_TEMPLATES[lang]).format(n=seed["product_name"], ro=seed["roaster_name"])
        named = rel.sample(min(2, len(rel)), random_state=rng.randrange(10**9))
        others = rel.drop(named.index)
        extra = others.sample(min(6, len(others)), random_state=rng.randrange(10**9))
        ctx_rows = pd.concat([named, extra])
        shared = sorted({f for f in _arr(seed["flavor_notes_clean"])
                         if any(f.lower() in set(r["_flavors_lc"]) for _, r in named.iterrows())})[:3]
        frags = [_bean_gt_fragment(r, lang, prefer=shared) for _, r in named.iterrows()]
        roast = str(seed["roast_level_clean"])
        if lang == "vi":
            gt = (
                f"Nếu bạn thích {seed['product_name']} của {seed['roaster_name']}, hãy thử "
                f"{_join_names(frags, 'vi')}. Các loại này cùng đến từ {seed['country']}, "
                f"cùng {ROAST_VI.get(roast, roast)}"
                + (f" và có chung hương {_join_names([_flavor_phrase(f, 'vi') for f in shared], 'vi')}" if shared else "")
                + f". Có {len(rel)} sản phẩm tương tự trong danh mục."
            )
        else:
            gt = (
                f"If you like {seed['product_name']} by {seed['roaster_name']}, try "
                f"{_join_names(frags, 'en')}. They share the same origin ({seed['country']}) "
                f"and the same {ROAST_EN.get(roast, roast)}"
                + (f", with {_join_names([f.lower() for f in shared], 'en')} notes in common" if shared else "")
                + f". The catalogue holds {len(rel)} similar products."
            )
        idx += 1
        counts[diff] += 1
        cases.append({
            "id": f"SS_{idx:03d}",
            "intent": "similar_search",
            "difficulty": diff,
            "language": lang,
            "question": question,
            "ground_truth": gt,
            "ground_truth_contexts": [_bean_context_text(r) for _, r in ctx_rows.iterrows()],
            "relevant_ids": rel["product_url"].tolist(),
            "n_relevant": int(len(rel)),
            "exclude_ids": [seed["product_url"]],
            "metadata": {
                "reference_product": seed["product_name"],
                "reference_roaster": seed["roaster_name"],
                "reference_url": seed["product_url"],
                "seed_country": seed["country"],
                "seed_roast": roast,
                "seed_flavors": _arr(seed["flavor_notes_clean"]),
                "named_products": named["product_name"].tolist(),
                "relevance_rule": "same country and same roast and >=1 shared flavor, seed excluded",
            },
        })
    for d in quota:
        if counts[d] < quota[d]:
            print(f"  [similar_search/{d}] only {counts[d]}/{quota[d]} seeds found")
    # Re-number so IDs are contiguous by difficulty order (easy, medium, hard)
    cases.sort(key=lambda c: ({"easy": 0, "medium": 1, "hard": 2}[c["difficulty"]], c["id"]))
    for i, c in enumerate(cases, 1):
        c["id"] = f"SS_{i:03d}"
    return cases


# ── news_search ────────────────────────────────────────────────

_ORG_SUFFIX = r"(Coffee|Coffees|Roasters|Roastery|Café|Cafe|Group|Holdings|Inc|Corp|Company|Co\.|Brands|Foods)"
_ORG_RE = re.compile(
    r"\b((?:[A-Z][\w’'&\-]*\s){0,3}" + _ORG_SUFFIX + r")\b"
)
_GENERIC_ORGS = {
    "coffee", "the coffee", "coffee company", "specialty coffee", "world coffee",
    "british coffee", "american coffee", "australian coffee", "chinese coffee",
    "coffee group", "coffee roasters", "coffee chain", "coffee brands",
}
_ORG_LEAD_STOP = {"how", "why", "the", "a", "an", "as", "with", "from", "for", "and", "but",
                  "new", "us", "uk", "eu", "british", "american", "australian", "indian",
                  "join", "inaugural", "california-based", "nebraska-based"}
_MAX_HARD_NEWS_HITS = 40
_MAX_SUMMARY_QUERY_CHARS = 140


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    m = re.split(r"(?<=[.!?])\s+", text)
    s = m[0] if m else text
    return s.rstrip(" .!?")


def _summary_query_fragment(summary: str) -> str:
    s = _first_sentence(summary)
    if not s or s.startswith("TT -") or len(s) > _MAX_SUMMARY_QUERY_CHARS:
        return ""
    return s


def _org_from_title(title: str) -> str | None:
    for m in _ORG_RE.finditer(title):
        tokens = m.group(1).strip().split()
        # Drop everything up to the last possessive ("South Korea’s Mega Coffee" -> "Mega Coffee"),
        # then leading function words.
        poss = [i for i, t in enumerate(tokens) if re.search(r"[’']s$", t)]
        if poss:
            tokens = tokens[poss[-1] + 1:]
        while tokens and (tokens[0].lower() in _ORG_LEAD_STOP or tokens[0].lower().endswith("-based")):
            tokens = tokens[1:]
        cand = " ".join(tokens)
        if len(tokens) < 2 or cand.lower() in _GENERIC_ORGS:
            continue
        return cand
    return None


def build_news_search(news_clean: pd.DataFrame, news_chunks: pd.DataFrame, n: int,
                      rng: random.Random) -> list[dict]:
    quota = _quota(n)
    usable = news_clean[
        (news_clean["title"].str.len() > 20)
        & (news_clean["summary"].str.len() >= 50)
        & (news_clean["content_text"].str.len() >= 400)
    ].drop_duplicates(subset="article_url").reset_index(drop=True)
    chunks_by_url = {u: g.sort_values("chunk_index") for u, g in news_chunks.groupby("article_url")}
    title_lc = usable["title"].str.lower()

    cases: list[dict] = []
    used_urls: set[str] = set()
    used_orgs: set[str] = set()
    idx = 0
    order = list(range(len(usable)))
    rng.shuffle(order)

    def _ctx_for(urls: list[str], per_article: int) -> list[str]:
        out = []
        for u in urls:
            g = chunks_by_url.get(u)
            if g is None or g.empty:
                continue
            out += [_news_context(r) for _, r in g.head(per_article).iterrows()]
        return out

    for diff in ["easy", "medium", "hard"]:
        made = 0
        for pos in order:
            if made >= quota[diff]:
                break
            art = usable.iloc[pos]
            url = art["article_url"]
            if url in used_urls or url not in chunks_by_url:
                continue
            lang = "vi" if idx % 2 == 0 else "en"
            if diff == "easy":
                question = rng.choice(NS_TEMPLATES[lang]["easy"]).format(t=art["title"].rstrip("."))
                rel_urls = [url]
                gt = str(art["summary"]).strip()
                meta_extra = {"query_source": "title"}
            elif diff == "medium":
                frag = _summary_query_fragment(art["summary"])
                if len(frag) < 30:
                    continue
                question = rng.choice(NS_TEMPLATES[lang]["medium"]).format(s=frag)
                rel_urls = [url]
                gt = str(art["summary"]).strip()
                meta_extra = {"query_source": "summary_first_sentence"}
            else:
                org = _org_from_title(str(art["title"]))
                if not org:
                    continue
                org_lc = org.lower()
                if any(org_lc in u or u in org_lc for u in used_orgs):
                    continue
                hits = usable[title_lc.str.contains(re.escape(org_lc), regex=True)]
                if len(hits) < 1 or len(hits) > _MAX_HARD_NEWS_HITS:
                    continue
                used_orgs.add(org_lc)
                question = rng.choice(NS_TEMPLATES[lang]["hard"]).format(e=org)
                rel_urls = hits["article_url"].tolist()
                top = hits.sort_values("publish_datetime", ascending=False).head(2)
                gt = " ".join(str(s).strip() for s in top["summary"].tolist())
                meta_extra = {"query_source": "organisation_in_title", "organisation": org}
            used_urls.add(url)
            idx += 1
            made += 1
            per_article = 3 if len(rel_urls) == 1 else 2
            cases.append({
                "id": f"NS_{idx:03d}",
                "intent": "news_search",
                "difficulty": diff,
                "language": lang,
                "question": question,
                "ground_truth": gt,
                "ground_truth_contexts": _ctx_for(rel_urls[:4], per_article),
                "relevant_ids": rel_urls,
                "n_relevant": int(len(rel_urls)),
                "exclude_ids": [],
                "metadata": {
                    "source_article": url,
                    "source_title": art["title"],
                    "source_name": art["source"],
                    "relevance_rule": "article-level: retrieved chunk belongs to a relevant article",
                    **meta_extra,
                },
            })
        if made < quota[diff]:
            print(f"  [news_search/{diff}] only {made}/{quota[diff]} articles found")
    return cases


# ── Main ───────────────────────────────────────────────────────


def _summarise(cases: list[dict]) -> None:
    print(f"\nTotal cases: {len(cases)}")
    print("  intent:", dict(Counter(c["intent"] for c in cases)))
    print("  difficulty:", dict(Counter(c["difficulty"] for c in cases)))
    print("  language:", dict(Counter(c["language"] for c in cases)))
    for intent in ["product_search", "similar_search", "news_search"]:
        sizes = [c["n_relevant"] for c in cases if c["intent"] == intent]
        if sizes:
            arr = np.array(sizes)
            print(f"  |R| {intent}: median={np.median(arr):.0f} min={arr.min()} max={arr.max()} "
                  f"p25={np.percentile(arr, 25):.0f} p75={np.percentile(arr, 75):.0f}")
    dup = len(cases) - len({c["question"] for c in cases})
    print(f"  duplicate questions: {dup}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retriever-independent evaluation dataset v6.")
    parser.add_argument("--out", type=Path, default=ROOT / "ragas_eval_dataset_v6.json")
    parser.add_argument("--per-intent", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("Loading data...")
    beans = pd.read_parquet(BEANS_PATH)
    news_chunks = pd.read_parquet(NEWS_PATH)
    news_clean = pd.read_parquet(NEWS_CLEAN_PATH)
    good = _eligible_beans(beans)
    print(f"  beans={len(beans)} eligible={len(good)} news={len(news_clean)} chunks={len(news_chunks)}")

    print("Building product_search...")
    ps = build_product_search(good, args.per_intent, rng)
    print("Building similar_search...")
    ss = build_similar_search(good, args.per_intent, rng)
    print("Building news_search...")
    ns = build_news_search(news_clean, news_chunks, args.per_intent, rng)
    cases = ps + ss + ns
    for c in cases:
        c["generator"] = GENERATOR_VERSION
        c["seed"] = args.seed
    _summarise(cases)

    if args.dry_run:
        for c in cases[:3] + ss[:2] + ns[:3]:
            print(f"\n[{c['id']}/{c['difficulty']}/{c['language']}] {c['question']}\n   GT: {c['ground_truth'][:160]}")
        return
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
