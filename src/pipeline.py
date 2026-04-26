"""Full RAG pipeline: Query → Understand → Retrieve → Re-rank → Generate."""

from src.query.intent_classifier import classify_intent
from src.query.entity_extractor import extract_entities
from src.retrieval.semantic_search import SemanticSearcher
from src.retrieval.structured_filter import structured_filter
from src.retrieval.reranker import reciprocal_rank_fusion
from src.generation.prompt_templates import build_prompt
from src.generation.llm_client import get_client, generate_structured
from src.generation.schemas import CoffeeResponse


class CoffeeRAG:
    def __init__(self):
        self.searcher = SemanticSearcher()
        self.llm_client = get_client()

    def retrieve(self, query: str, top_k_beans: int = 10, top_k_news: int = 5):
        intent = classify_intent(query)
        entities = extract_entities(query, client=self.llm_client)

        sem_beans = self.searcher.search_beans(query, top_k=top_k_beans * 2)
        sem_news = self.searcher.search_news(query, top_k=top_k_news)

        struct_beans = None
        has_filters = any(entities.get(k) for k in ("origin", "roast", "flavor", "typology", "processing"))
        if intent in ("product_search", "similar_search") and has_filters:
            struct_beans = structured_filter(self.searcher.beans, entities)
            if not struct_beans.empty:
                struct_beans = struct_beans.head(top_k_beans * 2)

        if struct_beans is not None and not struct_beans.empty:
            beans = reciprocal_rank_fusion(sem_beans, struct_beans, top_k=top_k_beans)
        else:
            beans = sem_beans.head(top_k_beans)

        return {
            "intent": intent,
            "entities": entities,
            "beans": beans,
            "news": sem_news if intent in ("news_search", "knowledge_qa") else sem_news.head(2),
        }

    def ask(self, query: str) -> CoffeeResponse:
        ctx = self.retrieve(query)
        messages = build_prompt(query, ctx["beans"], ctx["news"])
        return generate_structured(messages, CoffeeResponse, client=self.llm_client)
