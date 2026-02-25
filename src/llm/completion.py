'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Generic chat completion clients (OpenAI/Gemini/XAI) with pass-through parameters."
'''

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from src.core.errors import (
    log_and_raise_provider_error,
    log_and_raise_validation_error,
)
from src.utils.logging_utils import get_logger
from src.utils.logging_utils import log_execution_time_and_path
from src.utils.http_utils import (
    require_env,
    build_openai_headers,
    build_openai_chat_payload,
    safe_json,
)

## Initialize module-level logger
LOGGER = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class ProviderResponse:
    """
        Normalized provider response for chat completions
    """

    provider: str
    model: str
    text: str
    raw: dict[str, Any]
    usage: dict[str, int]

## ============================================================
## PROVIDER CALLS
## ============================================================
@log_execution_time_and_path
def call_openai_chat_completions(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stream: bool,
    extra: dict[str, Any],
    timeout_s: int = 90,
) -> ProviderResponse:
    """
        Call OpenAI-compatible chat completions endpoint
    """
    ## Resolve credentials and endpoint dynamically
    api_key = require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    url = f"{base_url}/chat/completions"

    ## Build standardized OpenAI-compatible payload
    headers = build_openai_headers(api_key)
    payload = build_openai_chat_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        extra=extra,
    )

    LOGGER.info("OpenAI request | url=%s | payload=%s", url, safe_json(payload))

    ## Streaming not supported in MVP for simplicity
    if stream:
        log_and_raise_validation_error(
            reason="stream=true not supported in MVP",
            context="openai_stream",
        )

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
    except Exception as exc:
        ## Network-level failure
        log_and_raise_provider_error(
            provider="openai",
            reason=f"HTTP request failed: {str(exc)}",
        )

    ## Handle provider HTTP errors explicitly
    if response.status_code >= 400:
        log_and_raise_provider_error(
            provider="openai",
            reason=f"HTTP {response.status_code} | body={response.text[:2000]}",
        )

    data = response.json()

    ## Extract assistant message (best-effort defensive parsing)
    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = ""

    ## Extract usage metadata if returned by provider
    usage_raw = data.get("usage", {}) if isinstance(data, dict) else {}
    usage = {
        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
        "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
        "total_tokens": int(usage_raw.get("total_tokens", 0)),
    }

    return ProviderResponse(
        provider="openai",
        model=model,
        text=text,
        raw=data,
        usage=usage,
    )

@log_execution_time_and_path
def call_gemini_chat_completions(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    extra: dict[str, Any],
    timeout_s: int = 90,
) -> ProviderResponse:
    """
        Call Gemini chat completions (REST) in a simplified way
    """
    
    ## Resolve Gemini credentials and endpoint
    api_key = require_env("GOOGLE_API_KEY")
    base_url = os.getenv(
        "GOOGLE_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    ).strip()

    ## Convert multi-turn messages into a single prompt (MVP simplification)
    prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])

    url = f"{base_url}/models/{model}:generateContent?key={api_key}"

    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }

    ## Merge optional pass-through parameters
    for k, v in extra.items():
        payload[k] = v

    LOGGER.info("Gemini request | url=%s | payload=%s", url, safe_json(payload))

    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
    except Exception as exc:
        log_and_raise_provider_error(
            provider="google",
            reason=f"HTTP request failed: {str(exc)}",
        )

    if response.status_code >= 400:
        log_and_raise_provider_error(
            provider="google",
            reason=f"HTTP {response.status_code} | body={response.text[:2000]}",
        )

    data = response.json()

    ## Extract generated text (structure varies across API versions)
    text = ""
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = ""

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return ProviderResponse(
        provider="google",
        model=model,
        text=text,
        raw=data,
        usage=usage,
    )

@log_execution_time_and_path
def call_xai_chat_completions(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    extra: dict[str, Any],
    timeout_s: int = 90,
) -> ProviderResponse:
    """
        Call xAI (Grok) chat completions endpoint in an OpenAI-compatible way
    """
    
    ## Resolve xAI credentials
    api_key = require_env("XAI_API_KEY")
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip()
    url = f"{base_url}/chat/completions"

    headers = build_openai_headers(api_key)

    ## Reuse OpenAI-compatible payload builder for Grok
    payload = build_openai_chat_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False,
        extra=extra,
    )

    LOGGER.info("xAI request | url=%s | payload=%s", url, safe_json(payload))

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
    except Exception as exc:
        log_and_raise_provider_error(
            provider="xai",
            reason=f"HTTP request failed: {str(exc)}",
        )

    if response.status_code >= 400:
        log_and_raise_provider_error(
            provider="xai",
            reason=f"HTTP {response.status_code} | body={response.text[:2000]}",
        )

    data = response.json()

    ## Extract assistant output safely
    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = ""

    usage_raw = data.get("usage", {}) if isinstance(data, dict) else {}
    usage = {
        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
        "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
        "total_tokens": int(usage_raw.get("total_tokens", 0)),
    }

    return ProviderResponse(
        provider="xai",
        model=model,
        text=text,
        raw=data,
        usage=usage,
    )

## ============================================================
## DISPATCH
## ============================================================
@log_execution_time_and_path
def run_chat_completion(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stream: bool,
    extra: dict[str, Any],
) -> ProviderResponse:
    """
        Dispatch a chat completion request to a provider
    """
    
    ## Normalize provider name to avoid case issues
    normalized = provider.strip().lower()

    if normalized == "openai":
        return call_openai_chat_completions(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            extra=extra,
        )

    if normalized in {"google", "gemini"}:
        if stream:
            log_and_raise_validation_error(
                reason="stream=true not supported for gemini in MVP",
                context="gemini_stream",
            )
        return call_gemini_chat_completions(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra=extra,
        )

    if normalized in {"xai", "grok"}:
        if stream:
            log_and_raise_validation_error(
                reason="stream=true not supported for xai in MVP",
                context="xai_stream",
            )
        return call_xai_chat_completions(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra=extra,
        )

    log_and_raise_validation_error(
        reason=f"Unsupported provider={provider}",
        context="run_chat_completion",
    )

    raise RuntimeError("Unreachable")