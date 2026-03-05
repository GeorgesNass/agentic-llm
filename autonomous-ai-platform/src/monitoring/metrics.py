'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Prometheus metrics exporter: counters, histograms, gauges and helpers to record LLM/tool/loop performance."
'''

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.errors import MonitoringError, ValidationError
from src.utils.logging_utils import get_logger, log_execution_time
from src.utils.safe_utils import _safe_str, _safe_int
from src.utils.env_utils import _get_env_bool, _get_env_int, _get_env_str

logger = get_logger(__name__)

## ============================================================
## METRICS REGISTRY (SINGLETON STYLE)
## ============================================================
_METRICS_STARTED = False
_METRICS_AVAILABLE = False

## LLM metrics
LLM_REQUESTS_TOTAL = None
LLM_ERRORS_TOTAL = None
LLM_LATENCY_SECONDS = None
LLM_TOKENS_IN_TOTAL = None
LLM_TOKENS_OUT_TOTAL = None

## Tool metrics
TOOLS_REQUESTS_TOTAL = None
TOOLS_ERRORS_TOTAL = None
TOOLS_LATENCY_SECONDS = None

## Loop metrics
LOOP_ITERATIONS_TOTAL = None
LOOP_FAILURES_TOTAL = None
LOOP_LATENCY_SECONDS = None

## Generic
LAST_HEALTHCHECK_UNIX = None

## ============================================================
## OPTIONAL DEPENDENCY (PROMETHEUS CLIENT)
## ============================================================
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
except Exception:  ## pragma: no cover
    Counter = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    Summary = None  # type: ignore[assignment]
    start_http_server = None  # type: ignore[assignment]

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(frozen=True)
class MetricsConfig:
    """
        Metrics configuration

        Args:
            enabled: Whether metrics are enabled
            port: Exporter port
            addr: Bind address
            namespace: Metric namespace prefix
    """

    enabled: bool
    port: int
    addr: str
    namespace: str

## ============================================================
## INTERNAL HELPERS
## ============================================================
def load_metrics_config() -> MetricsConfig:
    """
        Load metrics configuration from environment

        Env variables:
            METRICS_ENABLED
            METRICS_PORT
            METRICS_ADDR
            METRICS_NAMESPACE

        Returns:
            MetricsConfig
    """

    enabled = _get_env_bool("METRICS_ENABLED", default=True)
    port = _get_env_int("METRICS_PORT", 9109)
    addr = _get_env_str("METRICS_ADDR", "0.0.0.0").strip() or "0.0.0.0"
    namespace = _get_env_str("METRICS_NAMESPACE", "autonomous_ai_platform").strip()
    if not namespace:
        namespace = "autonomous_ai_platform"

    return MetricsConfig(enabled=enabled, port=port, addr=addr, namespace=namespace)

def _require_prometheus_dependency() -> None:
    """
        Ensure prometheus_client is installed

        Returns:
            None
    """

    if Counter is None or Gauge is None or Histogram is None or start_http_server is None:
        raise MonitoringError(
            message="prometheus_client is not installed",
            error_code="dependency_error",
            details={"pip_package": "prometheus-client"},
            origin="metrics",
            cause=None,
            http_status=500,
            is_retryable=False,
        )

def _init_metrics_objects(namespace: str) -> None:
    """
        Initialize metric objects (idempotent)

        Args:
            namespace: Metric namespace prefix

        Returns:
            None
    """

    global _METRICS_AVAILABLE
    global LLM_REQUESTS_TOTAL, LLM_ERRORS_TOTAL, LLM_LATENCY_SECONDS
    global LLM_TOKENS_IN_TOTAL, LLM_TOKENS_OUT_TOTAL
    global TOOLS_REQUESTS_TOTAL, TOOLS_ERRORS_TOTAL, TOOLS_LATENCY_SECONDS
    global LOOP_ITERATIONS_TOTAL, LOOP_FAILURES_TOTAL, LOOP_LATENCY_SECONDS
    global LAST_HEALTHCHECK_UNIX

    ## If already initialized, do nothing
    if _METRICS_AVAILABLE:
        return

    _require_prometheus_dependency()

    ## LLM metrics
    LLM_REQUESTS_TOTAL = Counter(
        f"{namespace}_llm_requests_total",
        "Total number of LLM requests",
        ["provider", "model", "mode"],
    )
    LLM_ERRORS_TOTAL = Counter(
        f"{namespace}_llm_errors_total",
        "Total number of LLM errors",
        ["provider", "model", "mode", "error_code"],
    )
    LLM_LATENCY_SECONDS = Histogram(
        f"{namespace}_llm_latency_seconds",
        "Latency of LLM requests in seconds",
        ["provider", "model", "mode"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
    )
    LLM_TOKENS_IN_TOTAL = Counter(
        f"{namespace}_llm_tokens_in_total",
        "Total input tokens",
        ["provider", "model", "mode"],
    )
    LLM_TOKENS_OUT_TOTAL = Counter(
        f"{namespace}_llm_tokens_out_total",
        "Total output tokens",
        ["provider", "model", "mode"],
    )

    ## Tool metrics
    TOOLS_REQUESTS_TOTAL = Counter(
        f"{namespace}_tool_requests_total",
        "Total number of tool calls",
        ["tool"],
    )
    TOOLS_ERRORS_TOTAL = Counter(
        f"{namespace}_tool_errors_total",
        "Total number of tool call errors",
        ["tool", "error_code"],
    )
    TOOLS_LATENCY_SECONDS = Histogram(
        f"{namespace}_tool_latency_seconds",
        "Latency of tool calls in seconds",
        ["tool"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    )

    ## Loop metrics
    LOOP_ITERATIONS_TOTAL = Counter(
        f"{namespace}_loop_iterations_total",
        "Total number of autonomous loop iterations",
        ["result"],
    )
    LOOP_FAILURES_TOTAL = Counter(
        f"{namespace}_loop_failures_total",
        "Total loop failures",
        ["error_code"],
    )
    LOOP_LATENCY_SECONDS = Histogram(
        f"{namespace}_loop_latency_seconds",
        "Latency of full autonomous runs",
        ["result"],
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120),
    )

    ## Generic health signals
    LAST_HEALTHCHECK_UNIX = Gauge(
        f"{namespace}_last_healthcheck_unix",
        "Unix timestamp of last healthcheck",
    )

    _METRICS_AVAILABLE = True

## ============================================================
## EXPORTER STARTUP
## ============================================================
@log_execution_time
def start_metrics_exporter() -> None:
    """
        Start Prometheus exporter server (idempotent)

        Returns:
            None
    """

    global _METRICS_STARTED

    ## No-op if disabled
    if not load_metrics_config().enabled():
        logger.info("Metrics disabled via env | METRICS_ENABLED=false")
        return

    ## Idempotent start
    if _METRICS_STARTED:
        return

    cfg = load_metrics_config()

    ## Initialize metric objects
    _init_metrics_objects(cfg.namespace)

    ## Start HTTP server
    try:
        start_http_server(port=cfg.port, addr=cfg.addr)
        _METRICS_STARTED = True
        logger.info("Prometheus exporter started | addr=%s | port=%s", cfg.addr, cfg.port)
    except Exception as exc:
        raise MonitoringError(
            message="Failed to start Prometheus exporter",
            error_code="monitoring_error",
            details={"addr": cfg.addr, "port": cfg.port, "cause": _safe_str(exc)},
            origin="metrics",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## RECORDING HELPERS
## ============================================================
def record_llm_call(
    *,
    provider: str,
    model: str,
    mode: str,
    duration_sec: float,
    usage: Optional[Dict[str, Any]] = None,
    error_code: str = "",
) -> None:
    """
        Record an LLM call metric

        Args:
            provider: Provider name
            model: Model name
            mode: local or api
            duration_sec: Duration seconds
            usage: Optional usage dict (input_tokens/output_tokens)
            error_code: Optional error code if failed

        Returns:
            None
    """

    ## Skip if disabled or not ready
    if not _get_env_bool("METRICS_ENABLED", default=True):
        return

    _init_metrics_objects(load_metrics_config().namespace)

    ## Guard against missing prometheus client
    if LLM_REQUESTS_TOTAL is None or LLM_LATENCY_SECONDS is None:
        return

    p = provider or "unknown"
    m = model or "unknown"
    md = mode or "unknown"

    ## Record request count and latency
    LLM_REQUESTS_TOTAL.labels(provider=p, model=m, mode=md).inc()
    LLM_LATENCY_SECONDS.labels(provider=p, model=m, mode=md).observe(max(duration_sec, 0.0))

    ## Record tokens if present
    if usage and isinstance(usage, dict):
        in_tokens = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0, default=0)
        out_tokens = _safe_int(usage.get("completion_tokens") or usage.get("output_tokens") or 0, default=0)

        if LLM_TOKENS_IN_TOTAL is not None:
            LLM_TOKENS_IN_TOTAL.labels(provider=p, model=m, mode=md).inc(in_tokens)
        if LLM_TOKENS_OUT_TOTAL is not None:
            LLM_TOKENS_OUT_TOTAL.labels(provider=p, model=m, mode=md).inc(out_tokens)

    ## Record error if provided
    if error_code:
        if LLM_ERRORS_TOTAL is not None:
            LLM_ERRORS_TOTAL.labels(provider=p, model=m, mode=md, error_code=error_code).inc()

def record_tool_call(
    *,
    tool: str,
    duration_sec: float,
    error_code: str = "",
) -> None:
    """
        Record a tool call metric

        Args:
            tool: Tool name
            duration_sec: Duration
            error_code: Optional error code if failed

        Returns:
            None
    """

    ## Skip if disabled
    if not _get_env_bool("METRICS_ENABLED", default=True):
        return

    _init_metrics_objects(load_metrics_config().namespace)

    if TOOLS_REQUESTS_TOTAL is None or TOOLS_LATENCY_SECONDS is None:
        return

    t = tool or "unknown"

    ## Record request and latency
    TOOLS_REQUESTS_TOTAL.labels(tool=t).inc()
    TOOLS_LATENCY_SECONDS.labels(tool=t).observe(max(duration_sec, 0.0))

    ## Record errors
    if error_code and TOOLS_ERRORS_TOTAL is not None:
        TOOLS_ERRORS_TOTAL.labels(tool=t, error_code=error_code).inc()

def record_loop_run(
    *,
    duration_sec: float,
    result: str,
    error_code: str = "",
) -> None:
    """
        Record an autonomous loop run metric

        Args:
            duration_sec: Duration
            result: pass or fail
            error_code: Optional error code if failed

        Returns:
            None
    """

    ## Skip if disabled
    if not _get_env_bool("METRICS_ENABLED", default=True):
        return

    _init_metrics_objects(load_metrics_config().namespace)

    if LOOP_ITERATIONS_TOTAL is None or LOOP_LATENCY_SECONDS is None:
        return

    r = result if result in {"pass", "fail"} else "fail"

    ## Record iteration and latency
    LOOP_ITERATIONS_TOTAL.labels(result=r).inc()
    LOOP_LATENCY_SECONDS.labels(result=r).observe(max(duration_sec, 0.0))

    ## Record failures
    if r == "fail" and error_code and LOOP_FAILURES_TOTAL is not None:
        LOOP_FAILURES_TOTAL.labels(error_code=error_code).inc()

def record_healthcheck() -> None:
    """
        Record healthcheck timestamp

        Returns:
            None
    """

    if not _get_env_bool("METRICS_ENABLED", default=True):
        return

    _init_metrics_objects(load_metrics_config().namespace)

    if LAST_HEALTHCHECK_UNIX is None:
        return

    LAST_HEALTHCHECK_UNIX.set(int(time.time()))

## ============================================================
## PUBLIC UTILS
## ============================================================
@log_execution_time
def validate_metrics_ready() -> None:
    """
        Validate that metrics system is available if enabled

        Returns:
            None
    """

    cfg = load_metrics_config()
    if not cfg.enabled:
        return

    ## Ensure prometheus client is present
    _require_prometheus_dependency()

    ## Ensure objects are initialized
    _init_metrics_objects(cfg.namespace)

    if not _METRICS_AVAILABLE:
        raise MonitoringError(
            message="Metrics not available after initialization",
            error_code="monitoring_error",
            details={},
            origin="metrics",
            cause=None,
            http_status=500,
            is_retryable=True,
        )