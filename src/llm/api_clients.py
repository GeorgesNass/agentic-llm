'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "External LLM API clients (ChatGPT/OpenAI, xAI/Grok, Google Gemini) + generic OpenAI-compatible with structured errors."
'''

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests

from src.core.config import config
from src.core.errors import DependencyError, LlmProviderError
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str, _safe_json_parse
from src.utils.env_utils import (
    _get_timeout_seconds,
    _resolve_base_url,
    _resolve_api_key,
    _resolve_model,
)
from src.utils.request_utils import (
    _extract_usage,
    _extract_text_from_openai_chat,
    _build_headers,
)
from src.utils.llm_utils import _openai_messages_to_gemini_contents, _extract_text_from_gemini_generate_content, _split_system_and_chat

logger = get_logger(__name__)

ProviderName = Literal["openai", "xai", "gemini", "generic_oai"]

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class ApiChatResult:
    """
        API chat completion result

        Args:
            provider: Provider used
            model: Model used
            text: Assistant text
            usage: Optional usage dict
            metadata: Optional metadata dict
    """

    provider: str
    model: str
    text: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]

## ============================================================
## OPENAI-COMPATIBLE CHAT (OPENAI + GROK + GENERIC)
## ============================================================
@log_execution_time
def chat_completion_openai_compatible(
    provider: ProviderName,
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> ApiChatResult:
    """
        Perform chat completion using an OpenAI-compatible endpoint

        Args:
            provider: Provider name (openai, xai, generic_oai)
            messages: Chat messages list
            model: Optional model override
            temperature: Sampling temperature
            top_p: Nucleus sampling top_p
            max_tokens: Maximum output tokens
            extra_payload: Optional provider-specific payload fields

        Returns:
            ApiChatResult
    """

    if provider not in {"openai", "xai", "generic_oai"}:
        raise DependencyError(
            message="Invalid provider for OpenAI-compatible chat client",
            error_code="validation_error",
            details={"provider": provider},
            origin="api_clients",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    api_key = _resolve_api_key(provider)
    base_url = _resolve_base_url(provider)
    model_name = _resolve_model(provider, model)
    timeout_sec = _get_timeout_seconds()

    ## Validate generic base URL
    if provider == "generic_oai" and not base_url:
        raise DependencyError(
            message="GENERIC_OAI_BASE_URL is not configured",
            error_code="configuration_error",
            details={"provider": provider},
            origin="api_clients",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Validate API keys for providers that require them
    if provider in {"openai", "xai"} and not api_key:
        raise DependencyError(
            message="API key is missing for provider",
            error_code="configuration_error",
            details={"provider": provider},
            origin="api_clients",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Build headers using Bearer auth
    headers = _build_headers(api_key=api_key, auth_style="bearer")

    ## Build payload for /chat/completions
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    ## Merge extra payload if provided
    if extra_payload:
        for k, v in extra_payload.items():
            payload[k] = v

    ## Build URL
    url = f"{base_url}/chat/completions"

    try:
        ## Measure latency
        start = time.perf_counter()

        ## Execute HTTP call
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)

        ## Compute duration
        duration = time.perf_counter() - start

        ## Handle explicit auth and rate-limit errors
        if resp.status_code == 401:
            raise DependencyError(
                message="Unauthorized (invalid API key or missing permissions)",
                error_code="unauthorized",
                details={"provider": provider, "status_code": resp.status_code},
                origin="api_clients",
                cause=None,
                http_status=401,
                is_retryable=False,
            )

        if resp.status_code == 429:
            raise DependencyError(
                message="Rate limited by provider",
                error_code="rate_limit",
                details={"provider": provider, "status_code": resp.status_code},
                origin="api_clients",
                cause=None,
                http_status=429,
                is_retryable=True,
            )

        ## Raise for any other HTTP error
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} | body={resp.text[:800]}")

        ## Parse response JSON
        data = _safe_json_parse(resp)

        ## Extract text and usage
        text = _extract_text_from_openai_chat(data)
        usage = _extract_usage(data)

        return ApiChatResult(
            provider=provider,
            model=model_name,
            text=text,
            usage=usage,
            metadata={
                "base_url": base_url,
                "duration_sec": duration,
                "status_code": resp.status_code,
                "endpoint": "/chat/completions",
            },
        )

    except DependencyError:
        raise

    except requests.Timeout as exc:
        raise LlmProviderError(
            message="Provider request timed out",
            error_code="timeout",
            details={"provider": provider, "url": url, "timeout_sec": timeout_sec},
            origin="api_clients",
            cause=exc,
            http_status=504,
            is_retryable=True,
        ) from exc

    except Exception as exc:
        raise LlmProviderError(
            message="Provider chat completion failed",
            error_code="llm_provider_error",
            details={
                "provider": provider,
                "url": url,
                "model": model_name,
                "cause": _safe_str(exc),
            },
            origin="api_clients",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## GEMINI CHAT (generateContent)
## ============================================================
@log_execution_time
def chat_completion_gemini(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> ApiChatResult:
    """
        Perform chat completion using Gemini generateContent endpoint

        Env variables:
            GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY
            GEMINI_BASE_URL
            GEMINI_MODEL

        Args:
            messages: OpenAI-style chat messages list
            model: Optional Gemini model override
            temperature: Sampling temperature
            top_p: Nucleus sampling top_p
            max_tokens: Maximum output tokens

        Returns:
            ApiChatResult
    """

    api_key = _resolve_api_key("gemini")
    base_url = _resolve_base_url("gemini")
    model_name = _resolve_model("gemini", model)
    timeout_sec = _get_timeout_seconds()

    if not api_key:
        raise DependencyError(
            message="Gemini API key is missing",
            error_code="configuration_error",
            details={"provider": "gemini"},
            origin="api_clients",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Convert OpenAI messages to Gemini payload
    system_instruction, contents = _openai_messages_to_gemini_contents(messages)

    ## Build generation config
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "topP": top_p,
        "maxOutputTokens": max_tokens,
    }

    ## Build request payload
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    ## Include system instruction if present
    if system_instruction is not None:
        payload["systemInstruction"] = system_instruction

    ## Build headers with x-goog-api-key
    headers = _build_headers(api_key=api_key, auth_style="x-goog-api-key")

    ## Build Gemini endpoint
    url = f"{base_url}/models/{model_name}:generateContent"

    try:
        ## Measure latency
        start = time.perf_counter()

        ## Execute request
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)

        ## Compute duration
        duration = time.perf_counter() - start

        ## Explicit error handling for common status codes
        if resp.status_code == 401:
            raise DependencyError(
                message="Unauthorized (invalid Gemini API key or missing permissions)",
                error_code="unauthorized",
                details={"provider": "gemini", "status_code": resp.status_code},
                origin="api_clients",
                cause=None,
                http_status=401,
                is_retryable=False,
            )

        if resp.status_code == 429:
            raise DependencyError(
                message="Rate limited by Gemini API",
                error_code="rate_limit",
                details={"provider": "gemini", "status_code": resp.status_code},
                origin="api_clients",
                cause=None,
                http_status=429,
                is_retryable=True,
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} | body={resp.text[:800]}")

        ## Parse response JSON
        data = _safe_json_parse(resp)

        ## Extract assistant text
        text = _extract_text_from_gemini_generate_content(data)

        ## Gemini may include usageMetadata
        usage = {}
        usage_meta = data.get("usageMetadata")
        if isinstance(usage_meta, dict):
            usage = usage_meta

        return ApiChatResult(
            provider="gemini",
            model=model_name,
            text=text,
            usage=usage,
            metadata={
                "base_url": base_url,
                "duration_sec": duration,
                "status_code": resp.status_code,
                "endpoint": ":generateContent",
            },
        )

    except DependencyError:
        raise

    except requests.Timeout as exc:
        raise LlmProviderError(
            message="Gemini request timed out",
            error_code="timeout",
            details={"provider": "gemini", "url": url, "timeout_sec": timeout_sec},
            origin="api_clients",
            cause=exc,
            http_status=504,
            is_retryable=True,
        ) from exc

    except Exception as exc:
        raise LlmProviderError(
            message="Gemini chat completion failed",
            error_code="llm_provider_error",
            details={"provider": "gemini", "url": url, "model": model_name, "cause": _safe_str(exc)},
            origin="api_clients",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## DISPATCH
## ============================================================
def chat_completion_dispatch(
    provider: ProviderName,
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> ApiChatResult:
    """
        Dispatch chat completion to provider client

        Args:
            provider: Provider name (openai, xai, gemini, generic_oai)
            messages: Chat messages list
            model: Optional model override
            temperature: Sampling temperature
            top_p: Nucleus sampling top_p
            max_tokens: Maximum output tokens

        Returns:
            ApiChatResult
    """

    ## Route Gemini to its native endpoint
    if provider == "gemini":
        return chat_completion_gemini(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    ## Route OpenAI/xAI/generic to OpenAI-compatible endpoint
    return chat_completion_openai_compatible(
        provider=provider,
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )