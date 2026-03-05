'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Local LLM runtime with HuggingFace download support and backend routing (llama-cpp-python GGUF CPU/GPU or vLLM server)."
'''

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.core.config import config
from src.core.errors import DependencyError, LlmProviderError
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str
from src.utils.io_utils import ensure_directory, _hf_download_model_file
from src.utils.env_utils import _get_env_str, _get_env_int, _get_env_bool
        
logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class LocalModelSpec:
    """
        Local model specification

        Args:
            backend: Local backend name (llama_cpp or vllm)
            model_id: HuggingFace repo id or local identifier
            filename: Model filename (e.g. GGUF filename)
            local_path: Resolved local model path
    """

    backend: str
    model_id: str
    filename: str
    local_path: Path

@dataclass(frozen=True)
class LocalCompletionResult:
    """
        Local completion result

        Args:
            provider: Provider used
            model: Model used
            text: Generated text
            usage: Optional usage dict
            metadata: Optional metadata dict
    """

    provider: str
    model: str
    text: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]

## ============================================================
## MODEL RESOLUTION
## ============================================================
def resolve_or_download_local_model() -> Optional[LocalModelSpec]:
    """
        Resolve local model path or download it from HuggingFace if requested

        Env variables:
            HF_MODEL_ID: HuggingFace repo id
            HF_MODEL_FILENAME: Filename inside repo (GGUF)
            HF_TOKEN: HuggingFace token
            HF_MODELS_DIR: Download directory
            LOCAL_BACKEND: auto|llama_cpp|vllm
            LOCAL_MODEL_PATH: Explicit local file path override

        Returns:
            LocalModelSpec or None
    """

    ## Read backend preference from environment
    backend = _get_env_str("LOCAL_BACKEND", "auto").strip().lower()

    ## Ensure backend value is valid
    if backend not in {"auto", "llama_cpp", "vllm"}:
        backend = "auto"

    ## Check if explicit local model path is provided
    local_path_override = _get_env_str("LOCAL_MODEL_PATH", "").strip()

    if local_path_override:
        resolved = Path(local_path_override).expanduser().resolve()

        ## If model exists locally, return immediately
        if resolved.exists():
            return LocalModelSpec(
                backend=backend,
                model_id="local_path",
                filename=resolved.name,
                local_path=resolved,
            )

        logger.warning("LOCAL_MODEL_PATH set but file not found | path=%s", resolved)

    ## Read HuggingFace configuration from environment
    repo_id = _get_env_str("HF_MODEL_ID", "").strip()
    filename = _get_env_str("HF_MODEL_FILENAME", "").strip()
    hf_token = _get_env_str("HF_TOKEN", "").strip()

    ## Resolve target download directory
    models_dir = Path(
        _get_env_str("HF_MODELS_DIR", str(config.paths.artifacts_models_dir))
    ).expanduser().resolve()

    ## If HF repo and filename are defined, trigger download
    if repo_id and filename:
        local_path = _hf_download_model_file(
            repo_id=repo_id,
            filename=filename,
            target_dir=models_dir,
            hf_token=hf_token,
        )

        return LocalModelSpec(
            backend=backend,
            model_id=repo_id,
            filename=filename,
            local_path=local_path,
        )

    return None

## ============================================================
## BACKEND SELECTION
## ============================================================
def _select_backend(requested_backend: str, use_gpu: bool) -> str:
    """
        Select runtime backend based on env and GPU availability

        Args:
            requested_backend: auto|llama_cpp|vllm
            use_gpu: Whether GPU usage is allowed

        Returns:
            backend name
    """

    ## If backend explicitly requested, use it
    if requested_backend in {"llama_cpp", "vllm"}:
        return requested_backend

    ## If GPU allowed and vLLM endpoint configured, prefer vLLM
    if use_gpu and _get_env_str("VLLM_BASE_URL", "").strip():
        return "vllm"

    return "llama_cpp"

## ============================================================
## LLAMA-CPP BACKEND
## ============================================================
def _llama_cpp_complete(
    prompt: str,
    model_path: Path,
    use_gpu: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> LocalCompletionResult:
    """
        Run completion using llama-cpp-python

        Env variables:
            LLAMA_CTX: Context length
            LLAMA_THREADS: CPU threads
            LLAMA_N_GPU_LAYERS: GPU layers (-1 means full offload if supported)
            LLAMA_SEED: Random seed
            LLAMA_VERBOSE: Verbose llama.cpp logs

        Args:
            prompt: Input prompt
            model_path: Path to GGUF model
            use_gpu: Whether GPU offload is allowed
            max_tokens: Max new tokens
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling

        Returns:
            LocalCompletionResult
    """

    ## Import llama-cpp-python runtime
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise DependencyError(
            message="llama-cpp-python dependency not available",
            error_code="dependency_error",
            details={"pip_package": "llama-cpp-python"},
            origin="local_runtime",
            cause=exc,
            http_status=500,
            is_retryable=False,
        ) from exc

    ## Read runtime config from env
    ctx = _get_env_int("LLAMA_CTX", 4096)
    threads = _get_env_int("LLAMA_THREADS", 4)
    seed = _get_env_int("LLAMA_SEED", 42)
    verbose = _get_env_bool("LLAMA_VERBOSE", False)

    ## Decide how many layers to offload on GPU
    n_gpu_layers_env = os.getenv("LLAMA_N_GPU_LAYERS")
    if use_gpu:
        n_gpu_layers = int(n_gpu_layers_env) if n_gpu_layers_env is not None else -1
    else:
        n_gpu_layers = 0

    ## Instantiate model and run inference
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=ctx,
            n_threads=threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )

        ## Measure inference latency
        start = time.perf_counter()

        ## Execute completion
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=[],
        )

        ## Compute duration in seconds
        duration = time.perf_counter() - start

        ## Extract response text safely
        text = ""
        usage: Dict[str, Any] = {}
        if isinstance(out, dict):
            choices = out.get("choices", [])
            if choices and isinstance(choices[0], dict):
                text = str(choices[0].get("text", ""))

            usage = out.get("usage", {}) if isinstance(out.get("usage"), dict) else {}

        return LocalCompletionResult(
            provider="local_llama_cpp",
            model=str(model_path),
            text=text,
            usage=usage,
            metadata={
                "backend": "llama_cpp",
                "n_ctx": ctx,
                "n_threads": threads,
                "n_gpu_layers": n_gpu_layers,
                "duration_sec": duration,
            },
        )

    except Exception as exc:
        raise LlmProviderError(
            message="llama.cpp completion failed",
            error_code="llm_provider_error",
            details={"model_path": str(model_path), "use_gpu": use_gpu},
            origin="local_runtime",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## vLLM BACKEND
## ============================================================
def _vllm_complete(
    prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> LocalCompletionResult:
    """
        Run completion using vLLM server (OpenAI-compatible endpoint)

        Env variables:
            VLLM_BASE_URL: e.g. http://vllm:8000/v1
            VLLM_API_KEY: optional
            VLLM_MODEL: optional default model name

        Args:
            prompt: Prompt string
            model_name: Model served by vLLM
            max_tokens: Max new tokens
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling

        Returns:
            LocalCompletionResult
    """

    base_url = _get_env_str("VLLM_BASE_URL", "").strip().rstrip("/")

    ## Validate configuration
    if not base_url:
        raise DependencyError(
            message="VLLM_BASE_URL is not configured",
            error_code="configuration_error",
            details={},
            origin="local_runtime",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Optional API key support
    api_key = _get_env_str("VLLM_API_KEY", "").strip()

    ## Build headers
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ## Build OpenAI-style payload for /completions
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    ## Execute HTTP request
    try:
        ## Measure latency
        start = time.perf_counter()

        ## Send request to vLLM server
        resp = requests.post(
            f"{base_url}/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        ## Compute duration
        duration = time.perf_counter() - start

        ## Raise on HTTP failure
        if resp.status_code >= 400:
            raise RuntimeError(f"vLLM HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()

        ## Extract generated text
        choices = data.get("choices", [])
        text = ""
        if choices and isinstance(choices[0], dict):
            text = str(choices[0].get("text", ""))

        ## Extract token usage if available
        usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}

        return LocalCompletionResult(
            provider="local_vllm",
            model=model_name,
            text=text,
            usage=usage,
            metadata={"backend": "vllm", "duration_sec": duration, "base_url": base_url},
        )

    except Exception as exc:
        raise LlmProviderError(
            message="vLLM completion failed",
            error_code="llm_provider_error",
            details={"base_url": base_url, "model": model_name, "cause": _safe_str(exc)},
            origin="local_runtime",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## PUBLIC API
## ============================================================
@log_execution_time
def local_chat_completion(
    prompt: str,
    *,
    use_gpu: bool,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> LocalCompletionResult:
    """
        Generate a completion using the configured local backend

        High-level workflow:
            1) Resolve local model (path or HF download)
            2) Select backend (llama_cpp or vllm)
            3) Run completion with selected backend
            4) Return structured result

        Args:
            prompt: Input prompt string
            use_gpu: Whether GPU backends are allowed
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling top_p

        Returns:
            LocalCompletionResult

        Raises:
            DependencyError: If model or backend dependencies are missing
            LlmProviderError: If inference fails
    """

    ## Resolve model configuration (local or HF)
    model_spec = resolve_or_download_local_model()

    ## Fail early if no model is configured
    if model_spec is None:
        raise DependencyError(
            message="No local model configured (set LOCAL_MODEL_PATH or HF_MODEL_ID + HF_MODEL_FILENAME)",
            error_code="configuration_error",
            details={},
            origin="local_runtime",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

    ## Select backend based on env and GPU settings
    requested_backend = model_spec.backend
    backend = _select_backend(requested_backend=requested_backend, use_gpu=use_gpu)

    ## Route to vLLM if selected
    if backend == "vllm":
        model_name = _get_env_str("VLLM_MODEL", "").strip() or model_spec.model_id

        return _vllm_complete(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    ## Default to llama.cpp backend
    return _llama_cpp_complete(
        prompt=prompt,
        model_path=model_spec.local_path,
        use_gpu=use_gpu,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )