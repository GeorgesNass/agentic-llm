'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic embeddings clients (OpenAI/Gemini/XAI) with pass-through parameters."
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Union

import requests

from src.core.errors import (
    log_and_raise_provider_error,
    log_and_raise_validation_error,
)
from src.utils.logging_utils import get_logger, log_execution_time_and_path
from src.utils.http_utils import (
    require_env,
    build_openai_headers,
    normalize_input_texts

)

## Initialize module-level logger
LOGGER = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class EmbeddingsProviderResponse:
    """
        Normalized provider response for embeddings

        Args:
            provider: Provider identifier
            model: Model name
            vectors: List of embedding vectors
            raw: Raw provider response payload
            usage: Token usage if returned
    """

    provider: str
    model: str
    vectors: List[List[float]]
    raw: dict[str, Any]
    usage: dict[str, int]

## ============================================================
## PROVIDER CALLS
## ============================================================
@log_execution_time_and_path
def call_openai_embeddings(
    model: str,
    inp: Union[str, List[str]],
    extra: dict[str, Any],
    timeout_s: int = 90,
) -> EmbeddingsProviderResponse:
    """
        Call OpenAI-compatible embeddings endpoint

        Args:
            model: Embeddings model name
            inp: Input string or list of strings
            extra: Provider pass-through fields
            timeout_s: HTTP timeout

        Returns:
            EmbeddingsProviderResponse
    """
    
    ## Resolve credentials and endpoint
    api_key = require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    url = f"{base_url}/embeddings"

    ## Normalize inputs
    inputs = normalize_input_texts(inp)
    if not inputs:
        log_and_raise_validation_error(
            reason="Embeddings input is empty",
            context="openai_embeddings_input",
        )

    ## Build request headers and payload
    headers = build_openai_headers(api_key)

    payload: dict[str, Any] = {"model": model, "input": inputs}

    ## Merge pass-through parameters (best-effort)
    for k, v in extra.items():
        payload[k] = v

    LOGGER.info("OpenAI embeddings request | url=%s | n_inputs=%s", url, len(inputs))

    ## Execute HTTP request
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except Exception as exc:
        log_and_raise_provider_error(
            provider="openai",
            reason=f"HTTP request failed: {str(exc)}",
        )

    ## Handle HTTP errors
    if response.status_code >= 400:
        log_and_raise_provider_error(
            provider="openai",
            reason=f"HTTP {response.status_code} | body={response.text[:2000]}",
        )

    data = response.json()

    ## Extract vectors (OpenAI-like format)
    vectors: List[List[float]] = []
    try:
        for item in data.get("data", []):
            vectors.append(item.get("embedding", []))
    except Exception:
        vectors = []

    ## Extract usage if present
    usage_raw = data.get("usage", {}) if isinstance(data, dict) else {}
    usage = {
        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
        "completion_tokens": 0,
        "total_tokens": int(usage_raw.get("total_tokens", 0)),
    }

    return EmbeddingsProviderResponse(
        provider="openai",
        model=model,
        vectors=vectors,
        raw=data,
        usage=usage,
    )

@log_execution_time_and_path
def call_gemini_embeddings(
    model: str,
    inp: Union[str, List[str]],
    extra: dict[str, Any],
    timeout_s: int = 90,
) -> EmbeddingsProviderResponse:
    """
        Call Gemini embeddings endpoint (REST) in a simplified way

        Notes:
            - This uses a best-effort REST call without SDK
            - Endpoint naming can vary by API version

        Args:
            model: Embeddings model name
            inp: Input string or list of strings
            extra: Provider pass-through fields
            timeout_s: HTTP timeout

        Returns:
            EmbeddingsProviderResponse
    """
    
    ## Resolve credentials and endpoint base
    api_key = require_env("GOOGLE_API_KEY")
    base_url = os.getenv(
        "GOOGLE_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    ).strip()

    ## Normalize inputs
    inputs = normalize_input_texts(inp)
    if not inputs:
        log_and_raise_validation_error(
            reason="Embeddings input is empty",
            context="gemini_embeddings_input",
        )

    ## Build Gemini endpoint (best-effort)
    url = f"{base_url}/models/{model}:batchEmbedContents?key={api_key}"

    ## Build request payload in batch format
    requests_payload: list[dict[str, Any]] = []
    for text in inputs:
        requests_payload.append({"content": {"parts": [{"text": text}]}})

    payload: dict[str, Any] = {"requests": requests_payload}

    ## Merge pass-through parameters (best-effort)
    for k, v in extra.items():
        payload[k] = v

    LOGGER.info("Gemini embeddings request | url=%s | n_inputs=%s", url, len(inputs))

    ## Execute HTTP request
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
    except Exception as exc:
        log_and_raise_provider_error(
            provider="google",
            reason=f"HTTP request failed: {str(exc)}",
        )

    ## Handle HTTP errors
    if response.status_code >= 400:
        log_and_raise_provider_error(
            provider="google",
            reason=f"HTTP {response.status_code} | body={response.text[:2000]}",
        )

    data = response.json()

    ## Extract vectors (Gemini formats may differ)
    vectors: List[List[float]] = []
    try:
        for item in data.get("embeddings", []):
            vectors.append(item.get("values", []))
    except Exception:
        vectors = []

    ## Usage fields vary across Gemini endpoints
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return EmbeddingsProviderResponse(
        provider="google",
        model=model,
        vectors=vectors,
        raw=data,
        usage=usage,
    )

@log_execution_time_and_path
def call_xai_embeddings(
    model: str,
    inp: Union[str, List[str]],
    extra: dict[str, Any],
    timeout_s: int = 90,
) -> EmbeddingsProviderResponse:
    """
        Call xAI embeddings endpoint in an OpenAI-compatible way

        Notes:
            - This implementation assumes /v1/embeddings

        Args:
            model: Embeddings model name
            inp: Input string or list of strings
            extra: Provider pass-through fields
            timeout_s: HTTP timeout

        Returns:
            EmbeddingsProviderResponse
    """
    
    ## Resolve credentials and endpoint
    api_key = require_env("XAI_API_KEY")
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip()
    url = f"{base_url}/embeddings"

    ## Normalize inputs
    inputs = normalize_input_texts(inp)
    if not inputs:
        log_and_raise_validation_error(
            reason="Embeddings input is empty",
            context="xai_embeddings_input",
        )

    ## Build headers and payload (OpenAI-like)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {"model": model, "input": inputs}

    ## Merge pass-through parameters (best-effort)
    for k, v in extra.items():
        payload[k] = v

    LOGGER.info("xAI embeddings request | url=%s | n_inputs=%s", url, len(inputs))

    ## Execute HTTP request
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except Exception as exc:
        log_and_raise_provider_error(
            provider="xai",
            reason=f"HTTP request failed: {str(exc)}",
        )

    ## Handle HTTP errors
    if response.status_code >= 400:
        log_and_raise_provider_error(
            provider="xai",
            reason=f"HTTP {response.status_code} | body={response.text[:2000]}",
        )

    data = response.json()

    ## Extract vectors (OpenAI-like format)
    vectors: List[List[float]] = []
    try:
        for item in data.get("data", []):
            vectors.append(item.get("embedding", []))
    except Exception:
        vectors = []

    ## Extract usage if present
    usage_raw = data.get("usage", {}) if isinstance(data, dict) else {}
    usage = {
        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
        "completion_tokens": 0,
        "total_tokens": int(usage_raw.get("total_tokens", 0)),
    }

    return EmbeddingsProviderResponse(
        provider="xai",
        model=model,
        vectors=vectors,
        raw=data,
        usage=usage,
    )

## ============================================================
## DISPATCH
## ============================================================
@log_execution_time_and_path
def run_embeddings(
    provider: str,
    model: str,
    inp: Union[str, List[str]],
    extra: dict[str, Any],
) -> EmbeddingsProviderResponse:
    """
        Dispatch an embeddings request to a provider

        Args:
            provider: Provider name
            model: Model name
            inp: Input string or list of strings
            extra: Pass-through parameters

        Returns:
            EmbeddingsProviderResponse
    """
    
    ## Normalize provider name
    normalized = provider.strip().lower()

    ## Route to the appropriate provider client
    if normalized == "openai":
        return call_openai_embeddings(model=model, inp=inp, extra=extra)

    if normalized in {"google", "gemini"}:
        return call_gemini_embeddings(model=model, inp=inp, extra=extra)

    if normalized in {"xai", "grok"}:
        return call_xai_embeddings(model=model, inp=inp, extra=extra)

    ## Fail fast on unsupported providers
    log_and_raise_validation_error(
        reason=f"Unsupported provider={provider}",
        context="run_embeddings",
    )

    raise RuntimeError("Unreachable")