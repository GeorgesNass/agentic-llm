'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI MCP server exposing chat, autonomous loop, evaluation and health endpoints with structured errors, metrics and tracing."
'''

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager

from src.core.errors import (
    AutonomousAIPlatformError,
    ERROR_CODE_INTERNAL,
    generic_exception_handler,
    platform_exception_handler,
)
from src.core.schema import (
    ChatRequest,
    ChatResponse,
    LoopRequest,
    LoopResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from src.monitoring.metrics import (
    record_healthcheck,
    record_llm_call,
    record_loop_run,
    start_metrics_exporter,
)
from src.monitoring.tracing import (
    end_trace,
    start_trace,
    trace_span,
    trace_to_dict,
)

from src.orchestrator.loop import run_autonomous_loop
from src.orchestrator.routing import route_chat_completion
from src.monitoring.evaluation import evaluate_answer, report_to_dict
from src.utils.logging_utils import get_logger
from src.utils.safe_utils import _safe_str

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ## ============================================================
    ## STARTUP
    ## ============================================================
    ## TODO: init resources here (db, vector store, models, etc.)
    yield
    ## ============================================================
    ## SHUTDOWN
    ## ============================================================
    ## TODO: cleanup resources here
    
## ============================================================
## APP FACTORY
## ============================================================
def create_app() -> FastAPI:
    """
        Create and configure FastAPI MCP server

        Returns:
            FastAPI app
    """

    app = FastAPI(
        title="Autonomous AI Platform MCP",
        version="1.0.0",
        description="Multi-capability platform: LLM routing, RAG, Text-to-SQL, autonomous loop, evaluation, monitoring.",
    )

    ## Middleware (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ## Exception handlers
    app.add_exception_handler(AutonomousAIPlatformError, platform_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    ## Health endpoint
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """
            Basic health endpoint

            Returns:
                Dict with status
        """

        record_healthcheck()

        return {
            "status": "ok",
            "service": "autonomous-ai-platform",
            "version": "1.0.0",
        }

    ## Chat endpoint
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest, request: Request) -> ChatResponse:
        """
            Simple chat endpoint (no autonomous loop)

            Flow:
                1) Route chat completion
                2) Record metrics
                3) Return structured response
        """

        start_time = time.perf_counter()

        with trace_span("chat_request", {"mode": "chat"}):
            result = route_chat_completion(
                messages=req.messages,
                prefer_local=req.prefer_local,
                use_gpu=req.use_gpu,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
            )

        duration = time.perf_counter() - start_time

        ## Record LLM metrics
        record_llm_call(
            provider=result.get("provider", "unknown"),
            model=result.get("model", "unknown"),
            mode="chat",
            duration_sec=duration,
            usage=result.get("usage", {}),
        )

        return ChatResponse(
            text=result.get("text", ""),
            provider=result.get("provider", ""),
            model=result.get("model", ""),
            usage=result.get("usage", {}),
            metadata=result.get("metadata", {}),
        )

    ## Autonomous loop endpoint
    @app.post("/loop", response_model=LoopResponse)
    async def loop_endpoint(req: LoopRequest, request: Request) -> LoopResponse:
        """
            Full autonomous loop endpoint

            Flow:
                1) Start trace
                2) Run loop
                3) Record metrics
                4) Return structured result
        """

        trace = start_trace(name="autonomous_loop", metadata={"query": req.query})
        start_time = time.perf_counter()

        try:
            with trace_span("autonomous_loop_run"):
                result = run_autonomous_loop(
                    user_query=req.query,
                    prefer_local=req.prefer_local,
                    use_gpu=req.use_gpu,
                    max_steps=req.max_steps,
                    max_iterations=req.max_iterations,
                )

            duration = time.perf_counter() - start_time

            ## Extract verdict if present
            verdict = "pass"
            self_eval = result.get("self_eval", {})
            if isinstance(self_eval, dict):
                verdict = self_eval.get("verdict", "pass")

            record_loop_run(
                duration_sec=duration,
                result=verdict,
                error_code="",
            )

            trace_data = trace_to_dict(trace)

            return LoopResponse(
                answer=result.get("answer", ""),
                self_eval=self_eval,
                traces=result.get("traces", []),
                metadata={
                    "loop_metadata": result.get("metadata", {}),
                    "trace": trace_data,
                },
            )

        except AutonomousAIPlatformError as exc:
            duration = time.perf_counter() - start_time

            record_loop_run(
                duration_sec=duration,
                result="fail",
                error_code=exc.error_code,
            )

            raise

        except Exception as exc:
            duration = time.perf_counter() - start_time

            record_loop_run(
                duration_sec=duration,
                result="fail",
                error_code=ERROR_CODE_INTERNAL,
            )

            raise

        finally:
            end_trace()

    ## Evaluation endpoint
    @app.post("/evaluate", response_model=EvaluationResponse)
    async def evaluate_endpoint(req: EvaluationRequest) -> EvaluationResponse:
        """
            Evaluate a given answer independently

            Flow:
                1) Compute basic metrics
                2) Optionally run LLM judge
                3) Return structured report
        """

        with trace_span("evaluation"):
            report = evaluate_answer(
                user_query=req.query,
                answer=req.answer,
                use_llm_judge=req.use_llm_judge,
                prefer_local=req.prefer_local,
                use_gpu=req.use_gpu,
            )

        return EvaluationResponse(
            report=report_to_dict(report),
        )

    return app

## ============================================================
## APP INSTANCE (FOR UVICORN)
## ============================================================
app = create_app()