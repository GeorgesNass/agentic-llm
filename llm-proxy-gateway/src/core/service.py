'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI service layer: healthcheck, cost simulation, chat completion, embeddings, and evaluation endpoints."
'''

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.core.errors import (
    log_and_raise_missing_path,
    log_and_raise_validation_error,
)
from src.llm.completion import run_chat_completion
from src.llm.costing import load_catalogs, simulate_cost
from src.llm.embeddings import run_embeddings
from src.llm.evaluation import evaluate_batch
from src.utils.logging_utils import get_logger

## ------------------------------------------------------------
## Optional typed schemas (keep service usable even if schemas evolve)
## ------------------------------------------------------------
try:
    from src.core.schema import (  # type: ignore
        ChatCompletionRequest,
        CostSimulateRequest,
        EmbeddingsRequest,
        EvaluationRequest,
    )
except Exception:  ## pragma: no cover
    ChatCompletionRequest = Any  ## type: ignore
    CostSimulateRequest = Any  ## type: ignore
    EmbeddingsRequest = Any  ## type: ignore
    EvaluationRequest = Any  ## type: ignore

LOGGER = get_logger(__name__)

## ============================================================
## SETTINGS AND CATALOGS
## ============================================================
def _default_models_catalog_path() -> Path:
    """
        Resolve default models catalog path

        High-level workflow:
            1) Read env override if present
            2) Fallback to artifacts/resources/models_catalog.json

        Args:
            None

        Returns:
            Absolute Path to models_catalog.json
    """
    
    raw = os.getenv(
        "MODELS_CATALOG_PATH",
        str(Path("artifacts") / "resources" / "models_catalog.json"),
    )
    
    return Path(raw).expanduser().resolve()

def _default_pricing_catalog_path() -> Path:
    """
        Resolve default pricing catalog path

        High-level workflow:
            1) Read env override if present
            2) Fallback to artifacts/resources/pricing_catalog.json

        Args:
            None

        Returns:
            Absolute Path to pricing_catalog.json
    """
    
    raw = os.getenv(
        "PRICING_CATALOG_PATH",
        str(Path("artifacts") / "resources" / "pricing_catalog.json"),
    )
    
    return Path(raw).expanduser().resolve()

def _load_service_catalogs() -> tuple[dict[str, Any], dict[str, Any]]:
    """
        Load catalogs for service usage

        High-level workflow:
            1) Resolve default paths
            2) Validate existence
            3) Load JSON catalogs using llm.costing helper

        Args:
            None

        Returns:
            Tuple(models_catalog, pricing_catalog)
    """
    
    models_path = _default_models_catalog_path()
    pricing_path = _default_pricing_catalog_path()

    if not models_path.exists():
        log_and_raise_missing_path(models_path, context="service_models_catalog")

    if not pricing_path.exists():
        log_and_raise_missing_path(pricing_path, context="service_pricing_catalog")

    return load_catalogs(models_path, pricing_path)

## ============================================================
## APP FACTORY
## ============================================================
def create_app() -> FastAPI:
    """
        Build the FastAPI application

        High-level workflow:
            1) Create FastAPI instance
            2) Register exception handlers
            3) Register routes

        Args:
            None

        Returns:
            Configured FastAPI app
    """
    app = FastAPI(
        title="llm-proxy-gateway",
        version="1.0.0",
        description="Unified gateway for cost simulation, chat completions, embeddings, and evaluation.",
    )

    ## Attach a global exception handler for safe JSON responses
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        ## Log once at service layer to keep debugging simple
        LOGGER.error("Unhandled exception | path=%s | error=%s", request.url.path, str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc)},
        )

    ## --------------------------------------------------------
    ## HEALTHCHECK
    ## --------------------------------------------------------
    @app.get("/healthcheck")
    async def healthcheck() -> dict[str, Any]:
        """
            Healthcheck endpoint

            High-level workflow:
                1) Return a minimal status payload

            Args:
                None

            Returns:
                {"status": "ok"}
        """
        
        return {"status": "ok"}

    ## --------------------------------------------------------
    ## COST SIMULATION
    ## --------------------------------------------------------
    @app.post("/cost/simulate")
    async def cost_simulate(req: CostSimulateRequest) -> dict[str, Any]:
        """
            Simulate LLM costs for chat or embeddings

            High-level workflow:
                1) Load catalogs (models + pricing)
                2) Validate user input (text XOR path)
                3) Run cost simulation for selected providers/models

            Args:
                req: Cost simulation request payload

            Returns:
                Cost simulation response payload
        """
        
        models_catalog, pricing_catalog = _load_service_catalogs()

        ## Read values defensively (schemas may evolve)
        mode = getattr(req, "mode", "chat")
        providers = getattr(req, "providers", [])
        requested_model = getattr(req, "model", None)

        text = getattr(req, "text", None)
        path = getattr(req, "path", None)
        recursive = bool(getattr(req, "recursive", True))
        include_per_file = bool(getattr(req, "include_per_file", False))

        max_chars_per_file = int(getattr(req, "max_chars_per_file", 200_000))
        expected_output_tokens = int(getattr(req, "expected_output_tokens", 512))

        chunk_size = int(getattr(req, "chunk_size", 1000))
        chunk_overlap = int(getattr(req, "chunk_overlap", 200))

        if not isinstance(providers, list) or len(providers) == 0:
            log_and_raise_validation_error(
                reason="providers must be a non-empty list",
                context="/cost/simulate",
            )

        ## Delegate to costing layer (pure simulation)
        return simulate_cost(
            mode=str(mode),
            providers=[str(p) for p in providers],
            requested_model=requested_model if requested_model is None else str(requested_model),
            text=text if text is None else str(text),
            path=path if path is None else str(path),
            recursive=recursive,
            max_chars_per_file=max_chars_per_file,
            expected_output_tokens=expected_output_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            models_catalog=models_catalog,
            pricing_catalog=pricing_catalog,
            include_per_file=include_per_file,
        )

    ## --------------------------------------------------------
    ## CHAT COMPLETIONS
    ## --------------------------------------------------------
    @app.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        """
            Run a generic chat completion request through a provider

            High-level workflow:
                1) Read provider/model/messages and sampling params
                2) Pass-through extra parameters to provider client
                3) Return normalized response (text + usage + raw)

            Args:
                req: Chat completion request payload

            Returns:
                Provider response payload
        """
        
        provider = getattr(req, "provider", None)
        model = getattr(req, "model", None)
        messages = getattr(req, "messages", None)

        if provider is None or str(provider).strip() == "":
            raise HTTPException(status_code=400, detail="Missing provider")

        if model is None or str(model).strip() == "":
            raise HTTPException(status_code=400, detail="Missing model")

        if not isinstance(messages, list) or len(messages) == 0:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        temperature = float(getattr(req, "temperature", 0.0))
        max_tokens = int(getattr(req, "max_tokens", 512))
        top_p = float(getattr(req, "top_p", 1.0))
        stream = bool(getattr(req, "stream", False))

        ## Pass-through provider-specific parameters (best-effort)
        extra = getattr(req, "extra", {}) or {}
        if not isinstance(extra, dict):
            extra = {}

        resp = run_chat_completion(
            provider=str(provider),
            model=str(model),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            extra=extra,
        )

        return {
            "provider": resp.provider,
            "model": resp.model,
            "text": resp.text,
            "usage": resp.usage,
            "raw": resp.raw,
        }

    ## --------------------------------------------------------
    ## EMBEDDINGS
    ## --------------------------------------------------------

    @app.post("/embeddings")
    async def embeddings(req: EmbeddingsRequest) -> dict[str, Any]:
        """
            Run a generic embeddings request through a provider

            High-level workflow:
                1) Read provider/model/input
                2) Pass-through extra parameters to provider client
                3) Return normalized response (vectors + usage + raw)

            Args:
                req: Embeddings request payload

            Returns:
                Provider response payload
        """
        
        provider = getattr(req, "provider", None)
        model = getattr(req, "model", None)
        inp = getattr(req, "input", None)

        if provider is None or str(provider).strip() == "":
            raise HTTPException(status_code=400, detail="Missing provider")

        if model is None or str(model).strip() == "":
            raise HTTPException(status_code=400, detail="Missing model")

        if inp is None:
            raise HTTPException(status_code=400, detail="Missing input")

        extra = getattr(req, "extra", {}) or {}
        if not isinstance(extra, dict):
            extra = {}

        resp = run_embeddings(
            provider=str(provider),
            model=str(model),
            inp=inp,
            extra=extra,
        )

        return {
            "provider": resp.provider,
            "model": resp.model,
            "vectors": resp.vectors,
            "usage": resp.usage,
            "raw": resp.raw,
        }

    ## --------------------------------------------------------
    ## EVALUATION
    ## --------------------------------------------------------
    @app.post("/evaluation/completions")
    async def evaluation_completions(req: EvaluationRequest) -> dict[str, Any]:
        """
            Evaluate completion outputs against references

            High-level workflow:
                1) Read predictions + references
                2) Compute metrics batch-wise
                3) Return aggregated results

            Args:
                req: Evaluation request payload

            Returns:
                Dict of metrics
        """
        
        predictions = getattr(req, "predictions", None)
        references = getattr(req, "references", None)

        if not isinstance(predictions, list) or not isinstance(references, list):
            raise HTTPException(status_code=400, detail="predictions and references must be lists")

        if len(predictions) != len(references):
            raise HTTPException(status_code=400, detail="predictions and references must have same length")

        ## Compute metrics (MVP: exact-match, contains, rouge/bleu optional in evaluation module)
        return evaluate_batch(
            predictions=[str(x) for x in predictions],
            references=[str(x) for x in references],
        )

    return app

## ============================================================
## APP INSTANCE
## ============================================================
app = create_app()