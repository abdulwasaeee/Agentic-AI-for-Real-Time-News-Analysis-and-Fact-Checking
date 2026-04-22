"""Article and Source data models."""

from datetime import datetime

from pydantic import BaseModel, Field


class Source(BaseModel):
    source_id: str
    name: str
    domain: str
    category: str  # "wire_service", "social_media", "news_outlet", etc.
    base_credibility: float = Field(ge=0.0, le=1.0)


class Article(BaseModel):
    article_id: str
    title: str
    url: str
    source_id: str
    published_at: datetime
    ingested_at: datetime
    content_hash: str
    body_snippet: str = ""
