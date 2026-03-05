'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Environment helpers and .env-driven resolvers for autonomous-ai-platform."
'''

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

## ============================================================
## TYPE ALIASES
## ============================================================
ChatMode = str
ChatProvider = str
ProviderName = str
EmbeddingProvider = str

## ============================================================
## ENV HELPERS
## ============================================================
def _get_env_str(key: str, default: str = "") -> str:
    """
        Read string environment variable

        Args:
            key: Env var name
            default: Default value

        Returns:
            String value
    """
    
    return os.getenv(key, default)

def _get_env_int(key: str, default: int) -> int:
    """
        Read int environment variable

        Args:
            key: Env var name
            default: Default value

        Returns:
            Integer value
    """

    ## Read raw environment variable
    raw = os.getenv(key)

    ## If missing, return default
    if raw is None:
        return default

    ## Convert to int with fallback
    try:
        return int(raw)
    except ValueError:
        return default

def _get_env_bool(key: str, default: bool = False) -> bool:
    """
        Read bool environment variable

        Args:
            key: Env var name
            default: Default value

        Returns:
            Boolean value
    """

    ## Read raw environment variable
    raw = os.getenv(key)

    ## If missing, return default
    if raw is None:
        return default

    ## Normalize to boolean
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_timeout_seconds() -> int:
    """
        Resolve HTTP timeout

        Env variables:
            HTTP_TIMEOUT_SEC

        Returns:
            Timeout seconds
    """

    ## Read timeout as string
    raw = _get_env_str("HTTP_TIMEOUT_SEC", "60").strip()

    ## Convert to int with fallback
    try:
        return int(raw)
    except ValueError:
        return 60

## ============================================================
## ROUTING & VECTOR DECISION
## ============================================================
def _resolve_chat_mode() -> ChatMode:
    """
        Resolve chat routing mode from environment

        Env variables:
            CHAT_MODE

        Returns:
            ChatMode
    """

    raw = _get_env_str("CHAT_MODE", "auto").strip().lower()
    if raw not in {"auto", "local", "api"}:
        return "auto"

    return raw  # type: ignore[return-value]

def _resolve_api_provider() -> ChatProvider:
    """
        Resolve API provider

        Env variables:
            API_PROVIDER

        Returns:
            Provider name
    """

    raw = _get_env_str("API_PROVIDER", "openai").strip().lower()
    if raw not in {"openai", "xai", "gemini", "generic_oai"}:
        return "openai"

    return raw  # type: ignore[return-value]

def _local_is_configured() -> bool:
    """
        Check if local model configuration exists

        Returns:
            Boolean
    """

    ## Either explicit path OR HF repo + filename
    return bool(
        _get_env_str("LOCAL_MODEL_PATH", "").strip()
        or (
            _get_env_str("HF_MODEL_ID", "").strip()
            and _get_env_str("HF_MODEL_FILENAME", "").strip()
        )
    )

def _api_is_configured(provider: ChatProvider) -> bool:
    """
        Check if API provider is configured

        Args:
            provider: Provider name

        Returns:
            Boolean
    """

    from src.core.config import config

    if provider == "openai":
        return bool(config.api_keys.openai_api_key)

    if provider == "xai":
        return bool(config.api_keys.xai_api_key)

    if provider == "gemini":
        return bool(_get_env_str("GEMINI_API_KEY", "").strip())

    if provider == "generic_oai":
        return bool(_get_env_str("GENERIC_OAI_BASE_URL", "").strip())

    return False

def _resolve_vector_store_backend() -> str:
    """
        Resolve vector store backend

        Env variables:
            VECTOR_STORE_BACKEND

        Returns:
            Backend name
    """

    backend = _get_env_str("VECTOR_STORE_BACKEND", "faiss").strip().lower()
    if backend not in {"faiss", "chroma"}:
        backend = "faiss"
    
    return backend

def _resolve_vector_store_dir() -> Path:
    """
        Resolve vector store persistence directory

        Env variables:
            VECTOR_STORE_DIR

        Returns:
            Path
    """
    from src.core.config import config
    
    raw = _get_env_str("VECTOR_STORE_DIR", str(config.paths.artifacts_vector_store_dir))
    
    return Path(raw).expanduser().resolve()

def _resolve_index_name() -> str:
    """
        Resolve default index or collection name

        Env variables:
            VECTOR_INDEX_NAME

        Returns:
            Name string
    """

    return _get_env_str("VECTOR_INDEX_NAME", "default").strip() or "default"

def _now_unix() -> int:
    """
        Current unix time

        Returns:
            Integer unix timestamp
    """

    return int(time.time())

def _resolve_exports_dir() -> Path:
    """
        Resolve exports dir

        Returns:
            Path
    """

    """
        Resolve exports directory without depending on global config.
        Priority:
          1) EXPORTS_DIR env var
          2) default: artifacts/exports
    """
    
    raw = os.getenv("EXPORTS_DIR", "").strip()
    base = Path(raw) if raw else Path("artifacts/exports")
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    
    return base
    

## ============================================================
## PROVIDER RESOLVERS
## ============================================================
def _resolve_model(provider: ProviderName, model_override: Optional[str]) -> str:
    """
        Resolve model for a provider

        Env variables:
            OPENAI_MODEL
            XAI_MODEL
            GEMINI_MODEL
            GENERIC_OAI_MODEL

        Args:
            provider: Provider name
            model_override: Optional explicit model

        Returns:
            Model name
    """

    ## Use explicit override if provided
    if model_override:
        return model_override

    ## OpenAI default model string is controlled by env
    if provider == "openai":
        return _get_env_str("OPENAI_MODEL", "gpt-4.1-mini").strip()

    ## xAI default model string is controlled by env
    if provider == "xai":
        return _get_env_str("XAI_MODEL", "grok-2-latest").strip()

    ## Gemini model string is controlled by env
    if provider == "gemini":
        return _get_env_str("GEMINI_MODEL", "gemini-2.5-flash").strip()

    ## Generic provider model must be configured or provided
    return _get_env_str("GENERIC_OAI_MODEL", "model").strip()
    
def _resolve_api_key(provider: ProviderName) -> str:
    """
        Resolve API key for provider

        Env variables:
            OPENAI_API_KEY
            XAI_API_KEY
            GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY
            GENERIC_OAI_API_KEY

        Args:
            provider: Provider name

        Returns:
            API key
    """

    from src.core.config import config

    ## Prefer config object (if you already load .env into config)
    if provider == "openai":
        return config.api_keys.openai_api_key or _get_env_str("OPENAI_API_KEY", "").strip()

    if provider == "xai":
        return config.api_keys.xai_api_key or _get_env_str("XAI_API_KEY", "").strip()

    if provider == "gemini":
        ## Support multiple common env names for Gemini key
        return (
            _get_env_str("GEMINI_API_KEY", "").strip()
            or _get_env_str("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
            or _get_env_str("GOOGLE_API_KEY", "").strip()
        )

    return _get_env_str("GENERIC_OAI_API_KEY", "").strip()

def _resolve_base_url(provider: EmbeddingProvider) -> str:
    """
        Resolve API base url for provider

        Env variables:
            OPENAI_BASE_URL
            XAI_BASE_URL
            GEMINI_BASE_URL
            GENERIC_OAI_BASE_URL

        Args:
            provider: Provider name

        Returns:
            Base URL without trailing slash
    """

    ## Resolve per provider with sensible defaults
    if provider == "openai":
        return _get_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")

    if provider == "xai":
        return _get_env_str("XAI_BASE_URL", "https://api.x.ai/v1").strip().rstrip("/")

    if provider == "gemini":
        return _get_env_str(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).strip().rstrip("/")

    return _get_env_str("GENERIC_OAI_BASE_URL", "").strip().rstrip("/")

def _resolve_local_embedding_model(model_override: Optional[str]) -> str:
    """
        Resolve local embedding model name

        Env variables:
            LOCAL_EMBEDDING_MODEL

        Args:
            model_override: Optional model override

        Returns:
            Model name
    """

    ## Prefer explicit override
    if model_override:
        return model_override

    ## Default sentence-transformers model
    return _get_env_str(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()

def _resolve_embedding_model(provider: EmbeddingProvider, model_override: Optional[str]) -> str:
    """
        Resolve embedding model name for provider

        Env variables:
            OPENAI_EMBEDDING_MODEL
            XAI_EMBEDDING_MODEL
            GEMINI_EMBEDDING_MODEL
            GENERIC_OAI_EMBEDDING_MODEL

        Args:
            provider: Provider name
            model_override: Optional override

        Returns:
            Model name
    """

    ## Prefer override
    if model_override:
        return model_override

    ## Defaults per provider
    if provider == "openai":
        return _get_env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

    if provider == "xai":
        return _get_env_str("XAI_EMBEDDING_MODEL", "grok-embedding").strip()

    if provider == "gemini":
        return _get_env_str("GEMINI_EMBEDDING_MODEL", "text-embedding-004").strip()

    return _get_env_str("GENERIC_OAI_EMBEDDING_MODEL", "embedding-model").strip()