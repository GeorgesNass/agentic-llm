'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Routing layer: decide local vs API LLM backend (CPU/GPU) and dispatch chat calls with structured errors."
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from src.core.config import config
from src.core.errors import (
    ConfigurationError,
    OrchestrationError,
    ValidationError,
)
from src.llm.api_clients import chat_completion_dispatch
from src.llm.local_runtime import local_chat_completion
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str
from src.utils.env_utils import (
    _api_is_configured,
    _local_is_configured,
    _resolve_api_provider,
    _resolve_chat_mode,
)

logger = get_logger(__name__)

ChatProvider = Literal["local", "openai", "xai", "gemini", "generic_oai"]
ChatMode = Literal["auto", "local", "api"]

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class RoutingDecision:
    """
        Routing decision for a chat request

        Args:
            mode: local or api
            provider: Provider name
            model: Optional model override
            use_gpu: GPU usage flag
            metadata: Extra routing metadata
    """

    mode: str
    provider: str
    model: str
    use_gpu: bool
    metadata: Dict[str, Any]
    
## ============================================================
## MESSAGE VALIDATION
## ============================================================
def _validate_messages(messages: List[Dict[str, Any]]) -> None:
    """
        Validate chat messages structure

        Args:
            messages: List of chat messages

        Returns:
            None
    """

    ## Ensure list structure
    if not isinstance(messages, list) or not messages:
        raise ValidationError(
            message="messages must be a non-empty list",
            error_code="validation_error",
            details={"messages_type": str(type(messages))},
            origin="routing",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Validate each message
    for i, msg in enumerate(messages):

        ## Ensure dict
        if not isinstance(msg, dict):
            raise ValidationError(
                message="Each message must be a dict",
                error_code="validation_error",
                details={"index": i, "msg_type": str(type(msg))},
                origin="routing",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        role = msg.get("role")
        content = msg.get("content")

        ## Validate role
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValidationError(
                message="Invalid message role",
                error_code="validation_error",
                details={"index": i, "role": role},
                origin="routing",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

        ## Validate content
        if content is None or str(content).strip() == "":
            raise ValidationError(
                message="Message content cannot be empty",
                error_code="validation_error",
                details={"index": i, "role": role},
                origin="routing",
                cause=None,
                http_status=400,
                is_retryable=False,
            )

## ============================================================
## PROMPT CONVERSION (LOCAL MODELS)
## ============================================================
def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """
        Convert structured chat messages to a prompt for local models

        Strategy:
            - System block at top
            - Role-based formatting
            - Force final assistant turn

        Args:
            messages: Chat messages

        Returns:
            Prompt string
    """

    system_lines: List[str] = []
    turns: List[str] = []

    ## Separate system messages
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()

        if role == "system":
            system_lines.append(content)
        elif role == "user":
            turns.append(f"User: {content}")
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:
            turns.append(f"Tool: {content}")

    ## Build system block
    system_block = ""
    if system_lines:
        system_block = "System:\n" + "\n".join(system_lines).strip() + "\n\n"

    ## Assemble final prompt
    prompt = system_block + "\n".join(turns).strip()

    ## Ensure assistant is next speaker
    if not prompt.endswith("Assistant:"):
        prompt = prompt + "\nAssistant:"

    return prompt

def decide_chat_routing(
    *,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> RoutingDecision:
    """
        Decide routing strategy

        Args:
            prefer_local: Prefer local if both available
            use_gpu: Optional GPU flag

        Returns:
            RoutingDecision
    """

    mode = _resolve_chat_mode()
    provider = _resolve_api_provider()

    local_ok = _local_is_configured()
    api_ok = _api_is_configured(provider)

    ## Explicit LOCAL mode
    if mode == "local":
        if not local_ok:
            raise ConfigurationError(
                message="CHAT_MODE=local but no local model configured",
                error_code="configuration_error",
                details={},
                origin="routing",
                cause=None,
                http_status=500,
                is_retryable=False,
            )

        return RoutingDecision(
            mode="local",
            provider="local",
            model="",
            use_gpu=bool(use_gpu),
            metadata={"forced_mode": "local"},
        )

    ## Explicit API mode
    if mode == "api":
        if not api_ok:
            raise ConfigurationError(
                message="CHAT_MODE=api but API provider not configured",
                error_code="configuration_error",
                details={"provider": provider},
                origin="routing",
                cause=None,
                http_status=500,
                is_retryable=False,
            )

        return RoutingDecision(
            mode="api",
            provider=provider,
            model="",
            use_gpu=False,
            metadata={"forced_mode": "api"},
        )

    ## AUTO MODE
    if prefer_local and local_ok:
        return RoutingDecision(
            mode="local",
            provider="local",
            model="",
            use_gpu=bool(use_gpu),
            metadata={"mode": "auto_local"},
        )

    if api_ok:
        return RoutingDecision(
            mode="api",
            provider=provider,
            model="",
            use_gpu=False,
            metadata={"mode": "auto_api"},
        )

    raise ConfigurationError(
        message="No backend configured",
        error_code="configuration_error",
        details={"local": local_ok, "api": api_ok},
        origin="routing",
        cause=None,
        http_status=500,
        is_retryable=False,
    )

## ============================================================
## PUBLIC DISPATCH
## ============================================================
@log_execution_time
def route_chat_completion(
    messages: List[Dict[str, Any]],
    *,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
        Route chat completion to appropriate backend

        Args:
            messages: Chat messages
            prefer_local: Prefer local backend
            use_gpu: GPU flag
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max tokens

        Returns:
            Dict response
    """

    ## Validate message structure
    _validate_messages(messages)

    ## Decide routing
    decision = decide_chat_routing(prefer_local=prefer_local, use_gpu=use_gpu)

    try:
        ## LOCAL BACKEND
        if decision.mode == "local":

            ## Convert chat to prompt
            prompt = _messages_to_prompt(messages)

            ## Call local runtime
            result = local_chat_completion(
                prompt=prompt,
                use_gpu=decision.use_gpu,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            return {
                "provider": result.provider,
                "model": result.model,
                "text": result.text,
                "usage": result.usage,
                "metadata": result.metadata | decision.metadata,
            }

        ## API BACKEND
        api_result = chat_completion_dispatch(
            provider=decision.provider,  # type: ignore[arg-type]
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return {
            "provider": api_result.provider,
            "model": api_result.model,
            "text": api_result.text,
            "usage": api_result.usage,
            "metadata": api_result.metadata | decision.metadata,
        }

    except Exception as exc:
        raise OrchestrationError(
            message="Chat routing failed",
            error_code="orchestration_error",
            details={"cause": _safe_str(exc)},
            origin="routing",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc