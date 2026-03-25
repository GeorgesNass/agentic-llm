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

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi.security import OAuth2PasswordBearer

from contextlib import asynccontextmanager

## JWT / SECURITY IMPORTS
from core.auth import (
    login_user,
    refresh_access_token,
    logout_user,
    get_current_active_user,
)
from core.security import (
    JWTMiddleware,
    require_roles,
)

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

## Fake DB (TO REPLACE)
fake_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$examplehash",
        "roles": ["admin"],
        "scopes": ["all"],
        "is_active": True,
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Application lifecycle manager

        Args:
            app: FastAPI application

        Yields:
            None
    """
    ## ============================================================
    ## STARTUP
    ## ============================================================
    ## TODO: init resources (db, models, vector store)
    yield
    ## ============================================================
    ## SHUTDOWN
    ## ============================================================
    ## TODO: cleanup resources

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

    ## CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ## JWT middleware (attach user to request)
    app.add_middleware(JWTMiddleware)

    ## EXCEPTION HANDLERS
    app.add_exception_handler(AutonomousAIPlatformError, platform_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    ## AUTH ENDPOINTS
    @app.post("/login")
    async def login(data: dict):
        """
            Authenticate user and return JWT tokens

            Args:
                data: Dict containing username and password

            Returns:
                Access and refresh tokens
        """
        
        return login_user(data["username"], data["password"], fake_db)

    @app.post("/refresh")
    async def refresh(data: dict):
        """
            Refresh access token

            Args:
                data: Dict containing refresh_token

            Returns:
                New token pair
        """
        
        return refresh_access_token(data["refresh_token"])

    @app.post("/logout")
    async def logout(token: str = Depends(oauth2_scheme)):
        """
            Logout user by revoking token

            Args:
                token: JWT token

            Returns:
                Logout status
        """
        
        ## Revoke token
        logout_user(token)
        return {"status": "logged_out"}

    ## HEALTH
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """
            Basic health endpoint

            Returns:
                Service status
        """

        record_healthcheck()

        return {
            "status": "ok",
            "service": "autonomous-ai-platform",
            "version": "1.0.0",
        }

    ## CHAT
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(
        req: ChatRequest,
        request: Request,
        user=Depends(get_current_active_user),
    ) -> ChatResponse:
        """
            Simple chat endpoint (secured)

            Args:
                req: Chat request
                request: FastAPI request
                user: Authenticated user

            Returns:
                Chat response
        """

        start_time = time.perf_counter()

        ## Execute chat
        with trace_span("chat_request", {"mode": "chat"}):
            result = route_chat_completion(
                messages=req.messages,
                prefer_local=req.prefer_local,
                use_gpu=req.use_gpu,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
            )

        ## Compute duration
        duration = time.perf_counter() - start_time

        ## Record LLM metrics
        record_llm_call(
            provider=result.get("provider", "unknown"),
            model=result.get("model", "unknown"),
            mode="chat",
            duration_sec=duration,
            usage=result.get("usage", {}),
        )

        ## Return response
        return ChatResponse(
            text=result.get("text", ""),
            provider=result.get("provider", ""),
            model=result.get("model", ""),
            usage=result.get("usage", {}),
            metadata=result.get("metadata", {}),
        )

    ## LOOP
    @app.post("/loop", response_model=LoopResponse)
    async def loop_endpoint(
        req: LoopRequest,
        request: Request,
        user=Depends(require_roles(["admin", "service"])),
    ) -> LoopResponse:
        """
            Full autonomous loop endpoint (RBAC protected)

            Args:
                req: Loop request
                request: FastAPI request
                user: Authenticated user

            Returns:
                Loop response
        """

        ## Start trace
        trace = start_trace(name="autonomous_loop", metadata={"query": req.query})
        start_time = time.perf_counter()

        try:
            ## Run loop
            with trace_span("autonomous_loop_run"):
                result = run_autonomous_loop(
                    user_query=req.query,
                    prefer_local=req.prefer_local,
                    use_gpu=req.use_gpu,
                    max_steps=req.max_steps,
                    max_iterations=req.max_iterations,
                )

            duration = time.perf_counter() - start_time

            ## Extract verdict
            verdict = "pass"
            self_eval = result.get("self_eval", {})
            if isinstance(self_eval, dict):
                verdict = self_eval.get("verdict", "pass")

            ## Record metrics
            record_loop_run(
                duration_sec=duration,
                result=verdict,
                error_code="",
            )

            ## Convert trace
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
            ## Record failure
            duration = time.perf_counter() - start_time

            record_loop_run(
                duration_sec=duration,
                result="fail",
                error_code=exc.error_code,
            )

            raise

        except Exception:
            ## Record internal error
            duration = time.perf_counter() - start_time

            record_loop_run(
                duration_sec=duration,
                result="fail",
                error_code=ERROR_CODE_INTERNAL,
            )

            raise

        finally:
            ## End trace
            end_trace()

    ## EVALUATION
    @app.post("/evaluate", response_model=EvaluationResponse)
    async def evaluate_endpoint(
        req: EvaluationRequest,
        user=Depends(get_current_active_user),
    ) -> EvaluationResponse:
        """
            Evaluate answer (secured endpoint)

            Args:
                req: Evaluation request
                user: Authenticated user

            Returns:
                Evaluation report
        """

        ## Run evaluation
        with trace_span("evaluation"):
            report = evaluate_answer(
                user_query=req.query,
                answer=req.answer,
                use_llm_judge=req.use_llm_judge,
                prefer_local=req.prefer_local,
                use_gpu=req.use_gpu,
            )

        ## Return report
        return EvaluationResponse(
            report=report_to_dict(report),
        )

    return app

## ============================================================
## APP INSTANCE (FOR UVICORN)
## ============================================================
app = create_app()