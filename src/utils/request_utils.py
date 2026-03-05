'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "HTTP request helpers: headers builders, response extraction and request correlation utilities."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Request

## ============================================================
## TYPE ALIASES
## ============================================================
ProviderName = str
EmbeddingProvider = str

## ============================================================
## REQUEST HELPERS
## ============================================================
def _build_headers(
    api_key: str,
    *,
    auth_style: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
        Build standard HTTP headers

        Args:
            api_key: API key string
            auth_style: Auth strategy (bearer or x-goog-api-key)
            extra_headers: Additional headers

        Returns:
            Headers dict
    """

    ## Always send JSON content type
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    ## Apply authentication header strategy
    if api_key:
        if auth_style == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_style == "x-goog-api-key":
            headers["x-goog-api-key"] = api_key

    ## Merge extra headers if any
    if extra_headers:
        for k, v in extra_headers.items():
            headers[k] = v

    return headers

def _build_headers_embeddings(provider: EmbeddingProvider, api_key: str) -> Dict[str, str]:
    """
        Build HTTP headers

        Args:
            provider: Provider name
            api_key: API key

        Returns:
            Headers dict
    """

    ## Default JSON headers
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    ## Auth strategy differs for Gemini
    if api_key:
        if provider == "gemini":
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    return headers

def _extract_text_from_openai_chat(data: Dict[str, Any]) -> str:
    """
        Extract assistant text from OpenAI-style chat response

        Args:
            data: Response JSON dict

        Returns:
            Assistant text
    """

    ## OpenAI-like shape: { choices: [ { message: { content: "..." } } ] }
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    msg = first.get("message", {})
    if not isinstance(msg, dict):
        return ""

    content = msg.get("content", "")
    return str(content) if content is not None else ""

def _extract_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Extract usage from response JSON

        Args:
            data: Response JSON dict

        Returns:
            Usage dict
    """

    ## Many providers return usage dict, some do not
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else {}

def _get_request_id(request: Optional[Request]) -> str:
    """
        Extract a request id if available

        Args:
            request: FastAPI request

        Returns:
            Request id string
    """

    if request is None:
        return "n/a"

    ## Try common correlation header names
    return (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or "n/a"
    )