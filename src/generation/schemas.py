"""Pydantic response schemas for structured LLM output."""

from pydantic import BaseModel, Field


class RecommendedProduct(BaseModel):
    name: str = Field(description="Product name")
    roaster: str = Field(default="", description="Roaster name")
    reason: str = Field(description="Why this product is recommended")
    url: str = Field(default="", description="Product URL")


class RelatedArticle(BaseModel):
    title: str = Field(description="Article title")
    summary: str = Field(description="Brief summary of relevance")


class CoffeeResponse(BaseModel):
    summary: str = Field(description="Main answer in 2-4 sentences")
    products: list[RecommendedProduct] = Field(
        default_factory=list,
        description="Recommended coffee products with reasons",
    )
    articles: list[RelatedArticle] = Field(
        default_factory=list,
        description="Related articles if applicable",
    )
