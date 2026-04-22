"""Unified application configuration for all agents.

Merges Task 1 (Data & Memory Engineer) and Task 2 (Core Agentic AI Engineer) settings
into a single config loaded from a .env file in the project root.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Neo4j ────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "fakenews123"

    # ── ChromaDB Cloud (leave blank to use local Docker) ─────────────────
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    chroma_host: str = ""          # empty = in-memory (no server needed)
    chroma_port: int = 8000

    # ── OpenAI ───────────────────────────────────────────────────────────
    openai_api_key: str = ""

    # ── Tavily Search ────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── News scraping ────────────────────────────────────────────────────
    newsapi_key: str = ""
    telegram_scraper_api_url: str = ""
    telegram_scraper_api_key: str = ""

    # ── Model settings ───────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    llm_provider: str = "openai"       # "openai" | "ollama"
    embedding_provider: str = "openai"

    # ── Ollama (only used when *_provider = "ollama") ────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_llm_model: str = "gemma4:e2b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_vlm_model: str = ""

    # ── Fact-Check Agent — retrieval enhancements ────────────────────────
    use_graph_rag: bool = False
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── Cross-modal (SigLIP) ─────────────────────────────────────────────
    use_siglip: bool = False
    siglip_model: str = "google/siglip-base-patch16-224"
    siglip_threshold: float = 0.10

    # ── SOTA flags ───────────────────────────────────────────────────────
    use_retrieval_gate: bool = False
    use_claim_decomposition: bool = False
    use_debate: bool = False
    debate_confidence_threshold: int = 70
    use_freshness_react: bool = False

    # ── Benchmark / evaluation ───────────────────────────────────────────
    dry_run: bool = False
    offline_mode: bool = False

    # ── Observability ────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "fakenews-factcheck"


settings = Settings()
