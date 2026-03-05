'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Executor agent: runs planned tool calls, validates payloads, captures structured outputs, and enforces safety limits."
'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.errors import OrchestrationError, ToolExecutionError, ValidationError
from src.orchestrator.tools import ToolResult, _resolve_tool_names, run_tool
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_json, _safe_str
from src.utils.validation_utils import _must_be_dict, _enforce_budget

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class ExecutedStep:
    """
        Executed step result

        Args:
            step_id: Step index
            tool: Tool name
            ok: Execution status
            output: Tool output payload
            duration_sec: Execution duration
            metadata: Extra metadata
    """

    step_id: int
    tool: str
    ok: bool
    output: Dict[str, Any]
    duration_sec: float
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class ExecutionSummary:
    """
        Execution summary

        Args:
            ok: Global status
            steps: Executed steps
            errors: List of error messages
            metadata: Extra metadata
    """

    ok: bool
    steps: List[ExecutedStep]
    errors: List[str]
    metadata: Dict[str, Any]

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def execute_plan_steps(
    steps: List[Dict[str, Any]],
    *,
    max_steps: int = 8,
    stop_on_error: bool = True,
) -> ExecutionSummary:
    """
        Execute a list of tool steps

        Expected step schema:
            {
              "step_id": int,
              "tool": "tool_name",
              "payload": {...}
            }

        Args:
            steps: List of step dicts
            max_steps: Max steps allowed
            stop_on_error: Stop execution at first error

        Returns:
            ExecutionSummary
    """

    ## Validate list structure
    if not isinstance(steps, list):
        raise ValidationError(
            message="steps must be a list",
            error_code="validation_error",
            details={"type": str(type(steps))},
            origin="executor",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Enforce max step budget
    _enforce_budget(max_steps=max_steps, planned_steps=len(steps))

    ## Resolve available tools
    available_tools = set(_resolve_tool_names())

    executed: List[ExecutedStep] = []
    errors: List[str] = []

    ## Execute each step sequentially
    for i, step in enumerate(steps, start=1):

        start = time.perf_counter()

        ## Validate each step object
        if not isinstance(step, dict):
            msg = f"Invalid step type at index={i}"
            errors.append(msg)
            if stop_on_error:
                break
            continue

        step_id = step.get("step_id", i)
        tool_name = str(step.get("tool", "")).strip()
        payload = _must_be_dict(step.get("payload", {}), "payload")

        ## Check tool existence
        if tool_name not in available_tools:
            msg = f"Unknown tool '{tool_name}'"
            errors.append(msg)
            executed.append(
                ExecutedStep(
                    step_id=int(step_id) if str(step_id).isdigit() else i,
                    tool=tool_name,
                    ok=False,
                    output={"error": msg},
                    duration_sec=time.perf_counter() - start,
                    metadata={"available_tools": sorted(list(available_tools))},
                )
            )
            if stop_on_error:
                break
            continue

        ## Execute tool
        try:
            result: ToolResult = run_tool(tool_name, payload)

            executed.append(
                ExecutedStep(
                    step_id=int(step_id) if str(step_id).isdigit() else i,
                    tool=tool_name,
                    ok=bool(result.ok),
                    output=result.output,
                    duration_sec=result.duration_sec,
                    metadata=result.metadata,
                )
            )

            ## Stop early if tool failed
            if not result.ok:
                errors.append(f"Tool failed: {tool_name}")
                if stop_on_error:
                    break

        except (ValidationError, ToolExecutionError) as exc:
            ## Keep errors structured and non-raw
            msg = f"{tool_name} execution error: {_safe_str(exc)}"
            errors.append(msg)

            executed.append(
                ExecutedStep(
                    step_id=int(step_id) if str(step_id).isdigit() else i,
                    tool=tool_name,
                    ok=False,
                    output={"error": msg},
                    duration_sec=time.perf_counter() - start,
                    metadata={"payload": _safe_json(payload)},
                )
            )

            if stop_on_error:
                break

        except Exception as exc:
            ## Unexpected exceptions are wrapped
            raise OrchestrationError(
                message="Executor failed with unexpected error",
                error_code="orchestration_error",
                details={"tool": tool_name, "cause": _safe_str(exc)},
                origin="executor",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    ## Compute global ok status
    ok = len(errors) == 0

    logger.info(
        "ExecutionSummary | ok=%s | executed_steps=%s | errors=%s",
        ok,
        len(executed),
        len(errors),
    )

    return ExecutionSummary(
        ok=ok,
        steps=executed,
        errors=errors,
        metadata={"max_steps": max_steps, "stop_on_error": stop_on_error},
    )