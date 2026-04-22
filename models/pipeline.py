"""Composite output model for the Preprocessing Agent pipeline."""

from typing import Optional

from pydantic import BaseModel

from models.article import Article, Source
from models.caption import ImageCaption
from models.claim import Claim


class PreprocessingOutput(BaseModel):
    """Contract between Preprocessing Agent and Memory Agent."""

    source: Source
    article: Article
    claims: list[Claim]
    image_caption: Optional[ImageCaption] = None
