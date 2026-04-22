"""Cross-modal consistency tool — checks for conflicts between claim text and image."""
import base64
import io
import json
import logging
from functools import lru_cache
from typing import Optional

from openai import OpenAI

import llm_factory as _llm_factory
from config import settings
from prompts import CROSS_MODAL_PROMPT, CROSS_MODAL_VISION_PROMPT

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_siglip(model_name: str):
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _decode_image(image_url: str):
    from PIL import Image as PILImage
    if image_url.startswith("data:"):
        header, b64 = image_url.split(",", 1)
        return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    import urllib.request
    with urllib.request.urlopen(image_url, timeout=10) as r:
        return PILImage.open(io.BytesIO(r.read())).convert("RGB")


def _siglip_check(claim_text: str, image_url: str) -> dict:
    import torch
    try:
        processor, model = _load_siglip(settings.siglip_model)
        image = _decode_image(image_url)
        inputs = processor(text=[claim_text], images=[image], return_tensors="pt",
                           padding="max_length", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.sigmoid(outputs.logits_per_image[0, 0]).item()
        conflict = prob < settings.siglip_threshold
        return {"conflict": conflict,
                "explanation": f"SigLIP score {prob:.3f} below threshold" if conflict else None,
                "siglip_score": prob}
    except Exception as e:
        logger.error("SigLIP check failed: %s", e)
        return {"conflict": False, "explanation": None, "siglip_score": None}


def _ensure_base64_uri(image_url: str) -> Optional[str]:
    if image_url.startswith("data:"):
        return image_url
    try:
        import urllib.request
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ct = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            raw = r.read()
        return f"data:{ct};base64,{base64.b64encode(raw).decode()}"
    except Exception as e:
        logger.warning("Could not fetch image %s: %s", image_url, e)
        return None


def _vision_check(claim_text: str, image_url: str) -> Optional[dict]:
    image_data_uri = _ensure_base64_uri(image_url)
    if image_data_uri is None:
        return None
    client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    prompt = CROSS_MODAL_VISION_PROMPT.format(claim_text=claim_text)
    try:
        response = client.chat.completions.create(
            model=settings.ollama_llm_model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": prompt},
            ]}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("Vision cross-modal check failed (%s) — falling back to caption mode", e)
        return None


def _llm_check(claim_text: str, image_caption: str, api_key: str, model: str) -> dict:
    client = _llm_factory.make_llm_client()
    prompt = CROSS_MODAL_PROMPT.format(claim_text=claim_text, image_caption=image_caption)
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Cross-modal LLM check failed: %s", e)
        return {"conflict": False, "explanation": None}


def check_cross_modal(claim_text: str, image_caption: Optional[str], api_key: str,
                       model: str, image_url: Optional[str] = None) -> dict:
    if not image_url and not image_caption:
        return {"flag": False, "explanation": None, "siglip_score": None}

    siglip_score = None
    if image_url and settings.use_siglip:
        result = _siglip_check(claim_text, image_url)
        siglip_score = result.get("siglip_score")
    elif image_url and settings.llm_provider == "ollama":
        result = _vision_check(claim_text, image_url)
        if result is None:
            result = _llm_check(claim_text, image_caption or "", api_key, model)
    else:
        result = _llm_check(claim_text, image_caption or "", api_key, model)

    return {
        "flag":         result.get("conflict", False),
        "explanation":  result.get("explanation"),
        "siglip_score": siglip_score,
    }
