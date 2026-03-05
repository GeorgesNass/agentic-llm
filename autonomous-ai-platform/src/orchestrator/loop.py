'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Autonomous orchestration loop: plan -> tool execution -> aggregation -> self-evaluation -> optional refinement."
'''

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from src.agents.aggregator import AggregatedAnswer, aggregate_final_answer
from src.agents.executor import ExecutionSummary, execute_plan_steps
from src.agents.reasoning import SelfEval, TaskPlan, build_task_plan, self_evaluate_answer
from src.core.errors import OrchestrationError, ValidationError
from src.orchestrator.tools import get_tools_registry
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str

logger = get_logger(__name__)

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _plan_to_step_payloads(plan: TaskPlan) -> List[Dict[str, Any]]:
    """
        Convert TaskPlan to executor step payloads

        Args:
            plan: Task plan

        Returns:
            List of step dicts for executor
    """

    steps: List[Dict[str, Any]] = []

    ## Convert each plan step to executor schema
    for s in plan.steps:
        steps.append(
            {
                "step_id": s.step_id,
                "tool": s.tool,
                "payload": s.payload_hint or {},
            }
        )

    return steps

def _should_refine(eval_result: SelfEval, *, min_confidence: float) -> bool:
    """
        Decide if we should refine based on critic evaluation

        Args:
            eval_result: SelfEval
            min_confidence: Minimum confidence threshold

        Returns:
            Boolean
    """

    ## Refine if explicit fail
    if eval_result.verdict != "pass":
        return True

    ## Refine if confidence too low
    if eval_result.confidence < min_confidence:
        return True

    return False

def _build_refinement_steps(
    user_query: str,
    last_answer: str,
    eval_result: SelfEval,
) -> List[Dict[str, Any]]:
    """
        Create a refinement plan for a second pass

        Strategy:
            - If critic says missing RAG, run rag_search
            - If critic says missing SQL evidence, run sql_query if possible
            - Otherwise, re-run aggregation with a stricter instruction

        Args:
            user_query: User query
            last_answer: Previous answer text
            eval_result: SelfEval result

        Returns:
            Executor step dicts
    """

    issues_text = " | ".join(eval_result.issues).lower()

    ## Pull available tools
    tools = get_tools_registry()
    steps: List[Dict[str, Any]] = []

    ## Try to recover missing retrieval
    if "rag" in issues_text or "retrieval" in issues_text or "source" in issues_text:
        if "rag_search" in tools:
            steps.append(
                {
                    "step_id": 1,
                    "tool": "rag_search",
                    "payload": {"query": user_query, "top_k": 6},
                }
            )

    ## Try to recover SQL issues
    if "sql" in issues_text or "database" in issues_text or "sqlite" in issues_text:
        if "sql_query" in tools:
            ## We cannot guess db_path here, but tool layer will validate
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "tool": "sql_query",
                    "payload": {"db_path": "", "query": "SELECT 1;", "limit": 1},
                }
            )

    ## If nothing added, return empty (meaning refine via better synthesis prompt only)
    return steps

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def run_autonomous_loop(
    query: str,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
    export: bool = False,
    max_steps: int = 8,
    max_iterations: int = 2,
    min_confidence: float = 0.6,
    stop_on_tool_error: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
        Run full autonomous loop

        High-level workflow:
            1) Build task plan (planner)
            2) Execute tools (executor)
            3) Aggregate final answer (aggregator)
            4) Self-evaluate (critic)
            5) Optionally refine once

        Args:
            query: User request
            prefer_local: Prefer local backend
            use_gpu: GPU flag for local
            export: Export artifacts (reserved for future use)
            max_steps: Max tool steps per iteration
            max_iterations: Max full loop iterations
            min_confidence: Confidence threshold for passing
            stop_on_tool_error: Stop execution at first tool error
            kwargs: Forward-compat extra args

        Returns:
            Dict with final answer, evaluation, and traces
    """

    ## Keep for forward-compat
    _ = export
    _ = kwargs

    ## Validate input early
    query = str(query).strip()
    if not query:
        raise ValidationError(
            message="query cannot be empty",
            error_code="validation_error",
            details={},
            origin="loop",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    ## Resolve tools once for planning
    registry = get_tools_registry()
    available_tools = sorted(list(registry.keys()))

    ## Initialize traces
    traces: List[Dict[str, Any]] = []
    last_answer_text = ""
    last_eval: Optional[SelfEval] = None

    ## Loop through iterations
    for iteration in range(1, max_iterations + 1):

        iter_start = time.perf_counter()

        try:
            ## Build plan
            plan = build_task_plan(
                query,
                available_tools=available_tools,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
            )

            ## Convert to executor steps
            step_payloads = _plan_to_step_payloads(plan)

            ## Execute tools
            exec_summary: ExecutionSummary = execute_plan_steps(
                step_payloads,
                max_steps=max_steps,
                stop_on_error=stop_on_tool_error,
            )

            ## Aggregate final answer from tool outputs
            aggregated: AggregatedAnswer = aggregate_final_answer(
                query,
                execution_summary={
                    "ok": exec_summary.ok,
                    "steps": [asdict(s) for s in exec_summary.steps],
                    "errors": exec_summary.errors,
                    "metadata": exec_summary.metadata,
                },
                prefer_local=prefer_local,
                use_gpu=use_gpu,
            )

            last_answer_text = aggregated.text

            ## Self-evaluate
            eval_result: SelfEval = self_evaluate_answer(
                query,
                last_answer_text,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
            )
            last_eval = eval_result

            ## Store iteration trace
            iter_duration = time.perf_counter() - iter_start
            traces.append(
                {
                    "iteration": iteration,
                    "duration_sec": iter_duration,
                    "plan": {
                        "goal": plan.goal,
                        "needs_rag": plan.needs_rag,
                        "needs_sql": plan.needs_sql,
                        "steps": [asdict(s) for s in plan.steps],
                    },
                    "execution": {
                        "ok": exec_summary.ok,
                        "errors": exec_summary.errors,
                        "steps": [asdict(s) for s in exec_summary.steps],
                    },
                    "answer": {
                        "text": aggregated.text,
                        "citations": aggregated.citations,
                        "metadata": aggregated.metadata,
                    },
                    "self_eval": asdict(eval_result),
                }
            )

            ## Decide if we stop
            if not _should_refine(eval_result, min_confidence=min_confidence):
                break

            ## Optional refinement: attempt extra tool calls based on critic issues
            if iteration < max_iterations:
                refine_steps = _build_refinement_steps(query, last_answer_text, eval_result)

                if refine_steps:
                    ## Execute refinement tools only
                    refine_exec = execute_plan_steps(
                        refine_steps,
                        max_steps=max_steps,
                        stop_on_error=True,
                    )

                    ## Re-aggregate using combined tool outputs
                    combined_steps = [asdict(s) for s in exec_summary.steps] + [
                        asdict(s) for s in refine_exec.steps
                    ]
                    combined_errors = exec_summary.errors + refine_exec.errors

                    aggregated2 = aggregate_final_answer(
                        query,
                        execution_summary={
                            "ok": exec_summary.ok and refine_exec.ok,
                            "steps": combined_steps,
                            "errors": combined_errors,
                            "metadata": {"iteration": iteration, "refinement": True},
                        },
                        prefer_local=prefer_local,
                        use_gpu=use_gpu,
                    )

                    last_answer_text = aggregated2.text

                    ## Re-run self-eval on refined answer
                    eval2 = self_evaluate_answer(
                        query,
                        last_answer_text,
                        prefer_local=prefer_local,
                        use_gpu=use_gpu,
                    )
                    last_eval = eval2

                    ## Append a refinement trace record
                    traces.append(
                        {
                            "iteration": iteration,
                            "refinement": True,
                            "execution": {
                                "ok": refine_exec.ok,
                                "errors": refine_exec.errors,
                                "steps": [asdict(s) for s in refine_exec.steps],
                            },
                            "answer": {"text": aggregated2.text},
                            "self_eval": asdict(eval2),
                        }
                    )

                    ## If refined passed, stop
                    if not _should_refine(eval2, min_confidence=min_confidence):
                        break

        except (ValidationError, OrchestrationError):
            raise

        except Exception as exc:
            raise OrchestrationError(
                message="Autonomous loop failed",
                error_code="orchestration_error",
                details={"cause": _safe_str(exc), "iteration": iteration},
                origin="loop",
                cause=exc,
                http_status=500,
                is_retryable=True,
            ) from exc

    ## Final output bundle
    eval_payload = asdict(last_eval) if last_eval is not None else {}

    return {
        "query": query,
        "answer": last_answer_text,
        "self_eval": eval_payload,
        "traces": traces,
        "metadata": {
            "iterations_used": len({t.get("iteration") for t in traces if "iteration" in t}),
            "max_iterations": max_iterations,
        },
    }