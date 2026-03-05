'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Main pipeline orchestration: manual run (chat/loop/eval), ingestion hooks, exports, and integration with monitoring."
'''

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import config
from src.core.errors import (
    ConfigurationError,
    OrchestrationError,
    PlatformError,
    ValidationError,
)
from src.monitoring.evaluation import evaluate_answer, report_to_dict
from src.monitoring.metrics import record_loop_run
from src.monitoring.tracing import end_trace, save_trace_to_file, start_trace, trace_span
from src.orchestrator.loop import run_autonomous_loop
from src.orchestrator.routing import route_chat_completion
from src.utils.env_utils import _get_env_str, _now_unix, _resolve_exports_dir
from src.utils.io_utils import _write_json
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str
from src.utils.validation_utils import _must_be_non_empty

logger = get_logger(__name__)

## ============================================================
## CHAT MODE
## ============================================================
@log_execution_time
def run_chat(
    prompt: str,
    *,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
) -> Dict[str, Any]:
    """
        Run a simple chat completion

        Args:
            prompt: User prompt string
            prefer_local: Prefer local backend
            use_gpu: GPU flag

        Returns:
            Dict response
    """

    p = _must_be_non_empty(prompt, "prompt")

    ## Build messages
    messages = [{"role": "user", "content": p}]

    ## Route completion
    result = route_chat_completion(
        messages=messages,
        prefer_local=prefer_local,
        use_gpu=use_gpu,
        temperature=0.2,
        top_p=0.95,
        max_tokens=600,
    )

    return result


## ============================================================
## LOOP MODE
## ============================================================
@log_execution_time
def run_loop(
    query: str,
    *,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
    export: bool = True,
) -> Dict[str, Any]:
    """
        Run the full autonomous loop

        Args:
            query: User query
            prefer_local: Prefer local backend
            use_gpu: GPU flag
            export: Whether to export artifacts

        Returns:
            Dict loop result
    """

    q = _must_be_non_empty(query, "query")

    ## Create trace session
    trace = start_trace(name="pipeline_loop", metadata={"query": q})

    start_time = time.perf_counter()

    try:
        with trace_span("run_autonomous_loop"):
            ## IMPORTANT: run_autonomous_loop expects 'query' (not user_query)
            result = run_autonomous_loop(
                query=q,
                prefer_local=prefer_local,
                use_gpu=use_gpu,
                export=export,
                max_steps=8,
                max_iterations=2,
            )

        duration = time.perf_counter() - start_time

        ## Record loop metrics
        self_eval = result.get("self_eval", {})
        verdict = "pass"
        if isinstance(self_eval, dict):
            verdict = str(self_eval.get("verdict", "pass"))

        record_loop_run(duration_sec=duration, result=verdict, error_code="")

        ## Export if enabled
        if export:
            exports_dir = _resolve_exports_dir()
            ts = _now_unix()

            out_path = exports_dir / f"loop_result_{ts}.json"
            _write_json(out_path, result)

            ## Save trace JSON
            trace_path = exports_dir / f"trace_{ts}.json"
            save_trace_to_file(trace, trace_path)

            result["exports"] = {
                "result_json": str(out_path),
                "trace_json": str(trace_path),
            }

        return result

    except (ValidationError, OrchestrationError):
        duration = time.perf_counter() - start_time
        record_loop_run(duration_sec=duration, result="fail", error_code="orchestration_error")
        raise

    except Exception as exc:
        duration = time.perf_counter() - start_time
        record_loop_run(duration_sec=duration, result="fail", error_code="internal_error")

        raise OrchestrationError(
            message="Pipeline loop failed",
            error_code="orchestration_error",
            details={"cause": _safe_str(exc)},
            origin="pipeline",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

    finally:
        end_trace()


@log_execution_time
def run_ingest(
    folder: Optional[str] = None,
    *,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
    export: bool = False,
) -> Dict[str, Any]:
    """
    Run document ingestion pipeline.

    This loads raw documents and builds the vector index.

    Args:
        folder: Folder containing documents (txt, md, etc.)
        prefer_local: Prefer local embeddings
        use_gpu: GPU usage
        export: Export ingestion report

    Returns:
        Dict containing ingestion summary
    """

    from src.orchestrator import retrieval

    try:
        logger.info("Starting ingestion pipeline")

        ## Default folder
        if not folder:
            folder = "data/raw"

        logger.info(
            "Ingestion config | folder=%s | prefer_local=%s | use_gpu=%s",
            folder,
            prefer_local,
            use_gpu,
        )

        embedding_provider = _get_env_str("EMBEDDING_PROVIDER", "local")
        embedding_model = _get_env_str(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        result = retrieval.ingest_folder_to_vector_store(
            folder=folder,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=use_gpu,
        )

        summary = {
            "status": "success",
            "folder": folder,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "ingest_result": result,
        }

        if export:
            exports_dir = _resolve_exports_dir()
            ts = _now_unix()
            export_file = exports_dir / f"ingestion_report_{ts}.json"

            _write_json(export_file, summary)

            logger.info("Ingestion report exported | path=%s", export_file)

            summary["exports"] = {"report_json": str(export_file)}

        logger.info("Ingestion completed successfully")

        return summary

    except Exception as exc:
        raise PlatformError(
            message="Ingestion pipeline failed",
            error_code="pipeline_error",
            details={"cause": _safe_str(exc)},
            origin="pipeline",
            cause=exc,
            http_status=500,
            is_retryable=False,
        ) from exc


## ============================================================
## EVALUATION MODE
## ============================================================
@log_execution_time
def run_evaluation(
    query: str,
    answer: str,
    *,
    use_llm_judge: bool = True,
    prefer_local: bool = True,
    use_gpu: Optional[bool] = None,
    export: bool = True,
) -> Dict[str, Any]:
    """
        Run evaluation for a given query/answer

        Args:
            query: Query string
            answer: Answer string
            use_llm_judge: Whether to run judge
            prefer_local: Prefer local judge
            use_gpu: GPU flag for judge
            export: Export report to JSON

        Returns:
            Dict report
    """

    q = _must_be_non_empty(query, "query")
    a = _must_be_non_empty(answer, "answer")

    report = evaluate_answer(
        user_query=q,
        answer=a,
        use_llm_judge=use_llm_judge,
        prefer_local=prefer_local,
        use_gpu=use_gpu,
    )
    payload = report_to_dict(report)

    if export:
        exports_dir = _resolve_exports_dir()
        ts = _now_unix()
        out_path = exports_dir / f"evaluation_{ts}.json"
        _write_json(out_path, payload)
        payload["exports"] = {"report_json": str(out_path)}

    return payload