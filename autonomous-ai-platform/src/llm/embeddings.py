'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Embeddings layer supporting local SentenceTransformers (CPU/GPU) and provider APIs (OpenAI, xAI/Grok, Gemini, generic) with structured errors."
'''

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import requests

from src.core.config import config
from src.core.errors import DependencyError, LlmProviderError
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str
from src.utils.env_utils import (
    _get_timeout_seconds,
    _get_env_int,
    _resolve_embedding_model,
    _resolve_api_key,
    _resolve_local_embedding_model,
    _resolve_base_url,
)
from src.utils.llm_utils import (
    _extract_vectors_gemini_embeddings,
    _extract_vectors_openai_embeddings,
    _load_sentence_transformer,
)
from src.utils.request_utils import _build_headers_embeddings

logger = get_logger(__name__)

EmbeddingProvider = Literal["local", "openai", "xai", "gemini", "generic_oai"]

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class EmbeddingsResult:
    """
        Embeddings result

        Args:
            provider: Provider used
            model: Model used
            vectors: List of vectors
            dims: Vector dimension
            metadata: Runtime metadata
    """

    provider: str
    model: str
    vectors: List[List[float]]
    dims: int
    metadata: Dict[str, Any]


@log_execution_time
def embed_local(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    use_gpu: bool = False,
    normalize: bool = True,
    batch_size: Optional[int] = None,
) -> EmbeddingsResult:
    """
        Compute embeddings locally using SentenceTransformers

        Args:
            texts: Input texts
            model: Optional model override
            use_gpu: Whether GPU is allowed
            normalize: Whether to L2-normalize vectors
            batch_size: Optional batch size override

        Returns:
            EmbeddingsResult
    """

    ## Resolve model name
    model_name = _resolve_local_embedding_model(model)

    ## Resolve batch size
    bs = batch_size if batch_size is not None else _get_env_int("EMBED_BATCH_SIZE", 32)

    ## Handle empty input safely
    if not texts:
        return EmbeddingsResult(
            provider="local",
            model=model_name,
            vectors=[],
            dims=0,
            metadata={
                "backend": "sentence_transformers",
                "device": "cuda" if use_gpu else "cpu",
                "batch_size": bs,
                "normalized": normalize,
            },
        )

    ## Load embedding model
    st_model = _load_sentence_transformer(model_name=model_name, use_gpu=use_gpu)

    ## Compute embeddings
    try:
        start = time.perf_counter()

        try:
            vectors_np = st_model.encode(
                list(texts),
                batch_size=bs,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        except TypeError:
            vectors_np = st_model.encode(
                list(texts),
                normalize_embeddings=normalize,
            )
            
        duration = time.perf_counter() - start

        ## Ensure numpy float32
        vectors_np = np.asarray(vectors_np, dtype=np.float32)

        ## Extract dimension
        dims = int(vectors_np.shape[1]) if vectors_np.ndim == 2 else 0

        ## Convert to python list
        vectors = vectors_np.tolist()

        return EmbeddingsResult(
            provider="local",
            model=model_name,
            vectors=vectors,
            dims=dims,
            metadata={
                "backend": "sentence_transformers",
                "device": "cuda" if use_gpu else "cpu",
                "batch_size": bs,
                "normalized": normalize,
                "duration_sec": duration,
            },
        )

    except Exception as exc:
        raise LlmProviderError(
            message="Local embeddings computation failed",
            error_code="llm_provider_error",
            details={"model_name": model_name, "device": "cuda" if use_gpu else "cpu"},
            origin="embeddings",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc


@log_execution_time
def embed_api(
    texts: Sequence[str],
    *,
    provider: EmbeddingProvider,
    model: Optional[str] = None,
) -> EmbeddingsResult:
    """
        Compute embeddings through provider APIs

        Providers:
            - openai: /embeddings
            - xai: /embeddings (OpenAI-compatible)
            - generic_oai: /embeddings (OpenAI-compatible)
            - gemini: /models/{model}:embedContent or :batchEmbedContents

        Args:
            texts: Input texts
            provider: Provider name
            model: Optional model override

        Returns:
            EmbeddingsResult
    """

    ## Validate provider
    if provider not in {"openai", "xai", "gemini", "generic_oai"}:
        raise DependencyError(
            message="Invalid embeddings provider for API mode",
            error_code="validation_error",
            details={"provider": provider},
            origin="embeddings",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Resolve config
    base_url = _resolve_base_url(provider)
    api_key = _resolve_api_key(provider)
    model_name = _resolve_embedding_model(provider, model)
    timeout_sec = _get_timeout_seconds()

    ## Validate base url for generic
    if provider == "generic_oai" and not base_url:
        raise DependencyError(
            message="GENERIC_OAI_BASE_URL is not configured",
            error_code="configuration_error",
            details={"provider": provider},
            origin="embeddings",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Validate keys for providers that require it
    if provider in {"openai", "xai", "gemini"} and not api_key:
        raise DependencyError(
            message="API key is missing for embeddings provider",
            error_code="configuration_error",
            details={"provider": provider},
            origin="embeddings",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Handle empty input safely
    if not texts:
        return EmbeddingsResult(
            provider=provider,
            model=model_name,
            vectors=[],
            dims=0,
            metadata={"base_url": base_url},
        )

    ## Build headers
    headers = _build_headers_embeddings(provider, api_key)

    ## Build URL and payload depending on provider
    if provider == "gemini":
        ## Gemini supports single and batch endpoints
        if len(texts) == 1:
            url = f"{base_url}/models/{model_name}:embedContent"
            payload: Dict[str, Any] = {
                "content": {"parts": [{"text": str(texts[0])}]},
            }
        else:
            url = f"{base_url}/models/{model_name}:batchEmbedContents"
            payload = {
                "requests": [{"content": {"parts": [{"text": str(t)}]}} for t in texts],
            }
    else:
        ## OpenAI-compatible embeddings endpoint
        url = f"{base_url}/embeddings"
        payload = {
            "model": model_name,
            "input": [str(t) for t in texts],
        }

    ## Execute HTTP call
    try:
        start = time.perf_counter()

        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)

        duration = time.perf_counter() - start

        ## Handle auth and rate limiting
        if resp.status_code == 401:
            raise DependencyError(
                message="Unauthorized (invalid API key or missing permissions)",
                error_code="unauthorized",
                details={"provider": provider, "status_code": resp.status_code},
                origin="embeddings",
                cause=None,
                http_status=401,
                is_retryable=False,
            )

        if resp.status_code == 429:
            raise DependencyError(
                message="Rate limited by provider",
                error_code="rate_limit",
                details={"provider": provider, "status_code": resp.status_code},
                origin="embeddings",
                cause=None,
                http_status=429,
                is_retryable=True,
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} | body={resp.text[:800]}")

        data = resp.json()

        ## Extract vectors per provider shape
        if provider == "gemini":
            vectors = _extract_vectors_gemini_embeddings(data)
        else:
            vectors = _extract_vectors_openai_embeddings(data)

        ## Resolve dims
        dims = len(vectors[0]) if vectors else 0

        return EmbeddingsResult(
            provider=provider,
            model=model_name,
            vectors=vectors,
            dims=dims,
            metadata={
                "base_url": base_url,
                "status_code": resp.status_code,
                "duration_sec": duration,
                "endpoint": url,
            },
        )

    except DependencyError:
        raise

    except requests.Timeout as exc:
        raise LlmProviderError(
            message="Embeddings request timed out",
            error_code="timeout",
            details={"provider": provider, "url": url, "timeout_sec": timeout_sec},
            origin="embeddings",
            cause=exc,
            http_status=504,
            is_retryable=True,
        ) from exc

    except Exception as exc:
        raise LlmProviderError(
            message="Embeddings request failed",
            error_code="llm_provider_error",
            details={
                "provider": provider,
                "url": url,
                "model": model_name,
                "cause": _safe_str(exc),
            },
            origin="embeddings",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc


## ============================================================
## TEST-COMPATIBILITY HELPERS (DO NOT CHANGE PUBLIC API)
## ============================================================
def _embed_local(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    use_gpu: Optional[bool] = None,
) -> List[List[float]]:
    """
        Local embeddings wrapper returning raw vectors (used by unit tests).

        Args:
            texts: Input texts
            model: Optional local model override
            use_gpu: None=auto, True/False explicit

        Returns:
            List of embeddings (one vector per input text)
    """

    result = embed_local(
        texts=texts,
        model=model,
        use_gpu=bool(use_gpu),
        normalize=True,
        batch_size=None,
    )

    return result.vectors


def _embed_api(
    texts: Sequence[str],
    *,
    provider: Optional[EmbeddingProvider] = None,
    model: Optional[str] = None,
) -> List[List[float]]:
    """
        API embeddings wrapper returning raw vectors (used by unit tests).

        Notes:
            - Provider can be passed explicitly, otherwise resolved from env/API_PROVIDER.
            - This wrapper keeps the original embed_api() logic unchanged.

        Args:
            texts: Input texts
            provider: Optional provider override
            model: Optional model override

        Returns:
            List of embeddings (one vector per input text)
    """

    prov: EmbeddingProvider
    if provider is not None:
        prov = provider
    else:
        env_provider = str(os.environ.get("API_PROVIDER", "")).strip().lower()
        if env_provider in {"openai", "xai", "gemini", "generic_oai"}:
            prov = env_provider  # type: ignore[assignment]
        else:
            prov = "openai"

    result = embed_api(
        texts=texts,
        provider=prov,
        model=model,
    )

    return result.vectors


## ============================================================
## PUBLIC DISPATCH
## ============================================================
def embed_texts(
    texts: Sequence[str],
    *,
    prefer_local: Optional[bool] = None,
    provider: EmbeddingProvider = "local",
    model: Optional[str] = None,
    use_gpu: bool = False,
    normalize: bool = True,
    batch_size: Optional[int] = None,
) -> Any:
    """
        Unified embeddings entrypoint

        Args:
            texts: Input texts
            provider: local|openai|xai|gemini|generic_oai
            model: Optional model override
            use_gpu: GPU for local embeddings
            normalize: Normalize local vectors
            batch_size: Local embeddings batch size override

        Returns:
            EmbeddingsResult
    """

    ## Unit-test / legacy compatibility
    if prefer_local is not None:
        return embed_texts_legacy(
            texts=list(texts),
            prefer_local=bool(prefer_local),
            use_gpu=use_gpu,
        )
        
    ## Route local embeddings to SentenceTransformers
    if provider == "local":
        return embed_local(
            texts=texts,
            model=model,
            use_gpu=use_gpu,
            normalize=normalize,
            batch_size=batch_size,
        )

    ## Route provider embeddings to HTTP APIs
    return embed_api(
        texts=texts,
        provider=provider,
        model=model,
    )


def embed_texts_legacy(
    texts: List[str],
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> List[List[float]]:
    """
        Public embedding entrypoint used by the platform and unit tests.

        Args:
            texts: Input texts
            prefer_local: If True, use local embeddings; otherwise use API embeddings
            use_gpu: None=auto, True/False explicit

        Returns:
            List of embeddings (one vector per input text)
    """

    ## Route to the correct backend
    if prefer_local:
        return _embed_local(texts=texts, use_gpu=use_gpu)
    return _embed_api(texts=texts)