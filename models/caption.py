"""Image caption data model."""

from pydantic import BaseModel


class ImageCaption(BaseModel):
    caption_id: str
    article_id: str
    image_url: str
    vlm_caption: str
