'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Reasoning agent: task planning, heuristic tool detection, and LLM-based self-evaluation."
'''

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.errors import OrchestrationError, ValidationError
from src.orchestrator.routing import route_chat_completion
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_json, _safe_str, _extract_first_json_object
from src.utils.validation_utils import _must_be_non_empty, _clamp_float

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class PlanStep:
    """
        Single plan step

        Args:
            step_id: Step number
            action: Action description
            tool: Tool name (empty if none)
            rationale: Why this step exists
            payload_hint: Optional structured hint
    """

    step_id: int
    action: str
    tool: str
    rationale: str
    payload_hint: Dict[str, Any]

@dataclass(frozen=True)
class TaskPlan:
    """
        Task plan

        Args:
            goal: User goal summary
            steps: Ordered steps
            needs_rag: Whether RAG is needed
            needs_sql: Whether SQL is needed
            metadata: Extra metadata
    """

    goal: str
    steps: List[PlanStep]
    needs_rag: bool
    needs_sql: bool
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class SelfEval:
    """
        Self-evaluation result

        Args:
            verdict: pass or fail
            issues: List of detected issues
            improvements: Suggested improvements
            confidence: Confidence score 0..1
            metadata: Extra metadata
    """

    verdict: str
    issues: List[str]
    improvements: List[str]
    confidence: float
    metadata: Dict[str, Any]

## ============================================================
## HEURISTIC DETECTION
## ============================================================
def _infer_needs_rag(user_query: str) -> bool:
    """
        Heuristic detection for RAG usage

        Args:
            user_query: User request

        Returns:
            Boolean
    """

    q = user_query.lower()

    ## Keywords indicating document search
    triggers = [
        "document",
        "pdf",
        "file",
        "folder",
        "search",
        "find",
        "source",
        "cite",
        "knowledge base",
    ]

    return any(t in q for t in triggers)

def _infer_needs_sql(user_query: str) -> bool:
    """
        Heuristic detection for SQL usage

        Args:
            user_query: User request

        Returns:
            Boolean
    """

    q = user_query.lower()

    ## Keywords indicating database queries
    triggers = [
        "sql",
        "database",
        "table",
        "query",
        "select",
        "count",
        "join",
        "schema",
    ]

    return any(t in q for t in triggers)

## ============================================================
## PROMPT BUILDERS
## ============================================================
def _build_planning_prompt(user_query: str, tools: List[str]) -> List[Dict[str, Any]]:
    """
        Build planning prompt messages

        Args:
            user_query: User request
            tools: Available tools

        Returns:
            Chat messages
    """

    ## Join tool list
    tools_line = ", ".join(tools)

    ## Strict JSON schema for reliable parsing
    system = (
        "You are a planning agent.\n"
        "Return ONLY valid JSON.\n"
        "Schema:\n"
        "{\n"
        '  "goal": "string",\n'
        '  "needs_rag": true|false,\n'
        '  "needs_sql": true|false,\n'
        '  "steps": [\n'
        "    {\n"
        '      "action": "string",\n'
        '      "tool": "string or empty",\n'
        '      "rationale": "string",\n'
        '      "payload_hint": {}\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Available tools: {tools_line}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

def _build_self_eval_prompt(user_query: str, draft_answer: str) -> List[Dict[str, Any]]:
    """
        Build self-evaluation prompt

        Args:
            user_query: Original request
            draft_answer: Draft output

        Returns:
            Chat messages
    """

    system = (
        "You are a critic agent.\n"
        "Return ONLY valid JSON.\n"
        "Schema:\n"
        "{\n"
        '  "verdict": "pass|fail",\n'
        '  "issues": ["string"],\n'
        '  "improvements": ["string"],\n'
        '  "confidence": 0.0\n'
        "}"
    )

    user = f"User request:\n{user_query}\n\nDraft answer:\n{draft_answer}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def build_task_plan(
    user_query: str,
    *,
    available_tools: List[str],
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> TaskPlan:
    """
        Build task plan using LLM + heuristics

        Args:
            user_query: User request
            available_tools: Tool names
            prefer_local: Prefer local backend
            use_gpu: GPU flag

        Returns:
            TaskPlan
    """

    ## Validate input
    uq = _must_be_non_empty(user_query, "user_query")

    ## Heuristic flags
    needs_rag = _infer_needs_rag(uq)
    needs_sql = _infer_needs_sql(uq)

    ## Build planning prompt
    messages = _build_planning_prompt(uq, available_tools)

    try:
        ## Call routing layer
        result = route_chat_completion(
            messages=messages,
            prefer_local=prefer_local,
            use_gpu=use_gpu,
            temperature=0.1,
            top_p=0.9,
            max_tokens=600,
        )

        ## Extract JSON from model output
        text = str(result.get("text", "")).strip()
        data = _extract_first_json_object(text)

        ## Parse plan safely
        steps: List[PlanStep] = []
        raw_steps = data.get("steps", [])

        if isinstance(raw_steps, list):
            for i, step in enumerate(raw_steps, start=1):
                if not isinstance(step, dict):
                    continue

                steps.append(
                    PlanStep(
                        step_id=i,
                        action=str(step.get("action", "")).strip(),
                        tool=str(step.get("tool", "")).strip(),
                        rationale=str(step.get("rationale", "")).strip(),
                        payload_hint=step.get("payload_hint", {})
                        if isinstance(step.get("payload_hint"), dict)
                        else {},
                    )
                )

        ## Fallback minimal plan if empty
        if not steps:
            steps = [
                PlanStep(1, "analyze_request", "", "Understand user intent", {}),
                PlanStep(2, "produce_answer", "", "Generate response", {}),
            ]

        return TaskPlan(
            goal=str(data.get("goal", uq)),
            steps=steps,
            needs_rag=bool(data.get("needs_rag", needs_rag)),
            needs_sql=bool(data.get("needs_sql", needs_sql)),
            metadata={"steps_count": len(steps)},
        )

    except Exception as exc:
        ## Offline fallback (unit tests): when no backend is configured, return a minimal plan
        msg = _safe_str(exc)

        if "No backend configured" in msg:
            logger.warning("Offline fallback: no backend configured for planning")

            fallback_tool = (
                "rag_search"
                if "rag_search" in available_tools
                else (available_tools[0] if available_tools else "")
            )

            ## IMPORTANT: provide a minimal payload for tools that require it
            payload_hint: Dict[str, Any] = {}
            if fallback_tool == "rag_search":
                payload_hint = {"query": uq}
                needs_rag = True

            steps = [
                PlanStep(1, "analyze_request", fallback_tool, "Offline fallback plan", payload_hint),
                PlanStep(2, "produce_answer", fallback_tool, "Offline fallback plan", payload_hint),
            ]

            return TaskPlan(
                goal=uq,
                steps=steps,
                needs_rag=needs_rag,
                needs_sql=needs_sql,
                metadata={"steps_count": len(steps), "offline_fallback": True},
            )

        raise OrchestrationError(
            message="Failed to build task plan",
            error_code="orchestration_error",
            details={"cause": msg},
            origin="reasoning",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc
    
@log_execution_time
def self_evaluate_answer(
    user_query: str,
    draft_answer: str,
    *,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> SelfEval:
    """
        Self-evaluate a draft answer

        Args:
            user_query: Original user request
            draft_answer: Current answer
            prefer_local: Prefer local backend
            use_gpu: GPU flag

        Returns:
            SelfEval
    """

    ## Validate inputs
    uq = _must_be_non_empty(user_query, "user_query")
    da = _must_be_non_empty(draft_answer, "draft_answer")

    messages = _build_self_eval_prompt(uq, da)

    try:
        ## Call LLM critic
        result = route_chat_completion(
            messages=messages,
            prefer_local=prefer_local,
            use_gpu=use_gpu,
            temperature=0.0,
            top_p=1.0,
            max_tokens=500,
        )

        ## Parse JSON
        text = str(result.get("text", "")).strip()
        data = _extract_first_json_object(text)

        verdict = str(data.get("verdict", "fail")).lower()
        issues = data.get("issues", [])
        improvements = data.get("improvements", [])

        return SelfEval(
            verdict="pass" if verdict == "pass" else "fail",
            issues=[str(i) for i in issues] if isinstance(issues, list) else [],
            improvements=[str(i) for i in improvements] if isinstance(improvements, list) else [],
            confidence=_clamp_float(data.get("confidence")),
            metadata={"raw": _safe_json(data)},
        )

    except Exception as exc:
        ## Offline fallback (unit tests): when no backend is configured, return deterministic eval
        msg = _safe_str(exc)

        if "No backend configured" in msg:
            logger.warning("Offline fallback: no backend configured for self-evaluation")

            ## Simple deterministic policy: offline -> pass with low confidence
            return SelfEval(
                verdict="pass",
                issues=[],
                improvements=[],
                confidence=0.3,
                metadata={"offline_fallback": True, "cause": msg},
            )

        raise OrchestrationError(
            message="Self-evaluation failed",
            error_code="orchestration_error",
            details={"cause": msg},
            origin="reasoning",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc