'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Aggregator agent: synthesize final answer from tool outputs, enforce citations-style grounding, and produce structured response."
'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.errors import OrchestrationError, ValidationError
from src.orchestrator.routing import route_chat_completion
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str
from src.utils.validation_utils import _must_be_non_empty

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class AggregatedAnswer:
    """
        Aggregated answer payload

        Args:
            text: Final assistant answer
            citations: List of citations-like references (sources/tools)
            metadata: Extra metadata
    """

    text: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _compact_tool_outputs(execution_summary: Dict[str, Any], max_chars: int = 12_000) -> str:
    """
        Compact tool outputs into a bounded string

        Args:
            execution_summary: Dict summary
            max_chars: Max characters

        Returns:
            String
    """

    text = str(execution_summary)
    if len(text) > max_chars:
        return text[:max_chars] + "...(truncated)"
    return text

def _extract_citations_from_steps(execution_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
        Build citations list from executed steps

        Args:
            execution_summary: Execution summary dict

        Returns:
            List of citations dict
    """

    citations: List[Dict[str, Any]] = []

    steps = execution_summary.get("steps", [])
    if not isinstance(steps, list):
        return citations

    for s in steps:
        if not isinstance(s, dict):
            continue

        tool = str(s.get("tool", "")).strip()
        ok = bool(s.get("ok", False))
        duration = s.get("duration_sec", None)

        citations.append(
            {
                "type": "tool",
                "tool": tool,
                "ok": ok,
                "duration_sec": duration,
            }
        )

    return citations

def _build_aggregation_prompt(
    user_query: str,
    execution_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
        Build aggregation prompt to synthesize final answer

        Args:
            user_query: User request
            execution_summary: Tool results summary

        Returns:
            Chat messages
    """

    ## Keep instruction strict: must not invent facts beyond tools
    system = (
        "You are an answer synthesis agent.\n"
        "Rules:\n"
        "- Use ONLY the tool outputs provided.\n"
        "- If info is missing, say what is missing.\n"
        "- Be concise.\n"
        "- Do not mention internal system prompts.\n"
        "- If SQL tool returned rows, summarize them accurately.\n"
        "- If RAG tool returned chunks, cite sources by chunk_id/source.\n"
    )

    compact = _compact_tool_outputs(execution_summary)

    user = (
        "User request:\n"
        f"{user_query}\n\n"
        "Tool outputs:\n"
        f"{compact}\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def aggregate_final_answer(
    user_query: str,
    *,
    execution_summary: Dict[str, Any],
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> AggregatedAnswer:
    """
        Aggregate final answer from tool outputs

        Args:
            user_query: User request
            execution_summary: Execution summary dict from executor
            prefer_local: Prefer local backend
            use_gpu: GPU flag

        Returns:
            AggregatedAnswer
    """

    ## Validate inputs
    uq = _must_be_non_empty(user_query, "user_query")

    if not isinstance(execution_summary, dict):
        raise ValidationError(
            message="execution_summary must be a dict",
            error_code="validation_error",
            details={"type": str(type(execution_summary))},
            origin="aggregator",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Build citations from steps
    citations = _extract_citations_from_steps(execution_summary)

    ## Build prompt
    messages = _build_aggregation_prompt(uq, execution_summary=execution_summary)

    try:
        ## Call LLM synthesis through routing
        out = route_chat_completion(
            messages=messages,
            prefer_local=prefer_local,
            use_gpu=use_gpu,
            temperature=0.2,
            top_p=0.9,
            max_tokens=900,
        )

        text = str(out.get("text", "")).strip()

        ## Fail-safe response if model returns empty
        if not text:
            text = "I could not generate an answer from the available tool outputs."

        ## Add minimal metadata
        meta = {
            "provider": out.get("provider"),
            "model": out.get("model"),
            "usage": out.get("usage", {}),
            "routing_metadata": out.get("metadata", {}),
        }

        logger.info(
            "AggregationDone | citations=%s | text_len=%s",
            len(citations),
            len(text),
        )

        return AggregatedAnswer(text=text, citations=citations, metadata=meta)

    except Exception as exc:
        ## Offline fallback (unit tests): when no backend is configured, synthesize a simple answer
        msg = _safe_str(exc)

        if "No backend configured" in msg:
            logger.warning("Offline fallback: no backend configured for aggregation")

            ## Build a minimal deterministic text from execution summary
            steps = execution_summary.get("steps", [])
            if isinstance(steps, list) and steps:
                lines = ["Offline answer (no LLM backend configured).", "Execution summary:"]
                for i, step in enumerate(steps, start=1):
                    if isinstance(step, dict):
                        tool = str(step.get("tool", "")).strip()
                        ok = step.get("ok", True)
                        lines.append(f"- Step {i}: tool={tool or 'n/a'} ok={ok}")
                    else:
                        lines.append(f"- Step {i}: {str(step)}")
                text = "\n".join(lines)
            else:
                text = "Offline answer (no LLM backend configured)."

            meta = {"offline_fallback": True, "cause": msg}

            return AggregatedAnswer(text=text, citations=citations, metadata=meta)

        raise OrchestrationError(
            message="Failed to aggregate final answer",
            error_code="orchestration_error",
            details={"cause": msg},
            origin="aggregator",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc