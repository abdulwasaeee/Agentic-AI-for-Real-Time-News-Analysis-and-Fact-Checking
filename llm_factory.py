"""Factory for LLM and embedding clients.

Supports OpenAI (cloud) and Ollama (local) providers via a unified interface.
"""
from openai import OpenAI

from config import settings


def make_llm_client() -> OpenAI:
    if settings.llm_provider == "ollama":
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    return OpenAI(api_key=settings.openai_api_key)


def llm_model_name() -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_llm_model
    return settings.llm_model


def make_embedding_client() -> OpenAI:
    if settings.embedding_provider == "ollama":
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    return OpenAI(api_key=settings.openai_api_key)


def embedding_model_name() -> str:
    if settings.embedding_provider == "ollama":
        return settings.ollama_embedding_model
    return settings.embedding_model
