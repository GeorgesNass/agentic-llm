'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "HTTP/provider helpers: env requirements, safe JSON logging, OpenAI-like headers/payload builders, and embeddings input normalization."
'''

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Union

from src.core.errors import log_and_raise_validation_error
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

## ============================================================
## ENV HELPERS
## ============================================================
def require_env(key: str, context: str = "env") -> str:
    """
        Read a required environment variable or raise

        Args:
            key: Environment variable name
            context: Error context label

        Returns:
            Environment variable value
    """
    ## Read and validate required environment variable
    value = os.getenv(key, "").strip()
    if value == "":
        log_and_raise_validation_error(
            reason=f"Missing required environment variable: {key}",
            context=context,
        )
    return value

## ============================================================
## SAFE JSON HELPERS
## ============================================================
def safe_json(obj: Any, max_chars: int = 2000) -> str:
    """
        Safely serialize an object to JSON for logging

        Args:
            obj: Any serializable object
            max_chars: Max output chars

        Returns:
            JSON string (possibly truncated)
    """
    ## Serialize best-effort (never fail logging)
    try:
        return json.dumps(obj, ensure_ascii=False)[:max_chars]
    except Exception:
        return "<json_serialization_failed>"

## ============================================================
## OPENAI-COMPATIBLE HELPERS
## ============================================================
def build_openai_headers(api_key: str) -> Dict[str, str]:
    """
        Build OpenAI-compatible HTTP headers

        Args:
            api_key: Provider API key

        Returns:
            Headers dictionary
    """
    
    ## Standard OpenAI-compatible headers
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def build_openai_chat_payload(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stream: bool,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """
        Build an OpenAI-like chat completion payload

        Args:
            model: Model name
            messages: OpenAI-like messages
            temperature: Temperature
            max_tokens: Maximum output tokens
            top_p: Top-p sampling
            stream: Stream flag
            extra: Provider pass-through parameters

        Returns:
            Payload dictionary
    """
    
    ## Base OpenAI-compatible payload
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
        "stream": bool(stream),
    }

    ## Merge pass-through fields (best-effort)
    for k, v in extra.items():
        payload[k] = v

    return payload

def build_openai_embeddings_payload(
    model: str,
    inp: Union[str, List[str]],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """
        Build an OpenAI-like embeddings payload

        Args:
            model: Embeddings model name
            inp: Input string or list of strings
            extra: Provider pass-through parameters

        Returns:
            Payload dictionary
    """
    
    ## Normalize input list
    inputs = normalize_input_texts(inp)
    if not inputs:
        log_and_raise_validation_error(
            reason="Embeddings input is empty after normalization",
            context="build_openai_embeddings_payload",
        )

    payload: Dict[str, Any] = {"model": model, "input": inputs}

    ## Merge pass-through fields (best-effort)
    for k, v in extra.items():
        payload[k] = v

    return payload

## ============================================================
## INPUT NORMALIZATION
## ============================================================
def normalize_input_texts(inp: Union[str, List[str]]) -> List[str]:
    """
        Normalize embeddings input to a list of non-empty strings

        Args:
            inp: Input string or list of strings

        Returns:
            List of cleaned strings
    """
    
    ## Single string case
    if isinstance(inp, str):
        cleaned = inp.strip()
        if cleaned == "":
            return []
        return [inp]

    ## List case
    if isinstance(inp, list):
        out: List[str] = []
        for x in inp:
            s = str(x).strip()
            if s != "":
                out.append(str(x))
        return out

    ## Unsupported input types
    return []