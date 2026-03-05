'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Structured tracing utilities: in-memory trace collector, span context manager and JSON export helpers."
'''

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.core.errors import MonitoringError, ValidationError
from src.utils.logging_utils import get_logger
from src.utils.safe_utils import _safe_str
from src.utils.io_utils import ensure_directory

logger = get_logger(__name__)

## ============================================================
## DATA MODELS
## ============================================================
@dataclass
class TraceSpan:
    """
        A single trace span

        Args:
            span_id: Unique span id
            name: Span name
            start_ts: Start timestamp (epoch seconds)
            end_ts: End timestamp (epoch seconds)
            duration_sec: Duration in seconds
            status: ok or error
            metadata: Arbitrary metadata
    """

    span_id: str
    name: str
    start_ts: float
    end_ts: float
    duration_sec: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraceSession:
    """
        A trace session grouping multiple spans

        Args:
            trace_id: Unique trace id
            name: Optional name
            started_at: Start timestamp
            spans: List of TraceSpan
            metadata: Global metadata
    """

    trace_id: str
    name: str
    started_at: float
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

## ============================================================
## TRACE MANAGER (SINGLETON STYLE)
## ============================================================
_CURRENT_SESSION: Optional[TraceSession] = None

def _generate_id() -> str:
    """
        Generate unique id

        Returns:
            String id
    """

    return uuid.uuid4().hex

def get_current_trace() -> Optional[TraceSession]:
    """
        Get current trace session

        Returns:
            TraceSession or None
    """

    return _CURRENT_SESSION

## ============================================================
## TRACE SESSION CONTROL
## ============================================================
def start_trace(name: str = "autonomous_run", metadata: Optional[Dict[str, Any]] = None) -> TraceSession:
    """
        Start a new trace session

        Args:
            name: Trace name
            metadata: Optional metadata

        Returns:
            TraceSession
    """

    global _CURRENT_SESSION

    if _CURRENT_SESSION is not None:
        logger.warning("Trace already active, overwriting previous trace")

    trace = TraceSession(
        trace_id=_generate_id(),
        name=str(name),
        started_at=time.time(),
        spans=[],
        metadata=metadata or {},
    )

    _CURRENT_SESSION = trace
    logger.info("Trace started | trace_id=%s | name=%s", trace.trace_id, trace.name)

    return trace

def end_trace() -> Optional[TraceSession]:
    """
        End current trace session

        Returns:
            TraceSession or None
    """

    global _CURRENT_SESSION

    trace = _CURRENT_SESSION
    if trace is None:
        return None

    logger.info("Trace ended | trace_id=%s | spans=%s", trace.trace_id, len(trace.spans))
    _CURRENT_SESSION = None
    return trace

## ============================================================
## SPAN CONTEXT MANAGER
## ============================================================
@contextmanager
def trace_span(name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
    """
        Context manager to record a span

        Usage:
            with trace_span("llm_call", {"provider": "openai"}):
                ...

        Args:
            name: Span name
            metadata: Optional metadata

        Returns:
            Generator
    """

    session = _CURRENT_SESSION
    if session is None:
        ## No active session, just run block
        yield
        return

    span_id = _generate_id()
    start_ts = time.time()

    status = "ok"
    span_meta = metadata.copy() if metadata else {}

    try:
        yield
    except Exception as exc:
        status = "error"
        span_meta["error"] = _safe_str(exc)
        raise
    finally:
        end_ts = time.time()
        duration = max(end_ts - start_ts, 0.0)

        span = TraceSpan(
            span_id=span_id,
            name=str(name),
            start_ts=start_ts,
            end_ts=end_ts,
            duration_sec=duration,
            status=status,
            metadata=span_meta,
        )

        session.spans.append(span)

        logger.debug(
            "TraceSpan | trace_id=%s | span=%s | status=%s | duration=%.4f",
            session.trace_id,
            name,
            status,
            duration,
        )

## ============================================================
## TRACE EXPORT
## ============================================================
def trace_to_dict(trace: TraceSession) -> Dict[str, Any]:
    """
        Convert TraceSession to dict

        Args:
            trace: TraceSession

        Returns:
            Dict
    """

    return {
        "trace_id": trace.trace_id,
        "name": trace.name,
        "started_at": trace.started_at,
        "metadata": trace.metadata,
        "spans": [asdict(s) for s in trace.spans],
    }

def save_trace_to_file(trace: TraceSession, path: str | Path) -> Path:
    """
        Save trace session as JSON file

        Args:
            trace: TraceSession
            path: Target file path

        Returns:
            Resolved Path
    """

    if trace is None:
        raise ValidationError(
            message="Cannot save empty trace",
            error_code="validation_error",
            details={},
            origin="tracing",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    target = Path(path).expanduser().resolve()
    ensure_directory(target.parent)

    payload = trace_to_dict(trace)

    try:
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise MonitoringError(
            message="Failed to save trace file",
            error_code="monitoring_error",
            details={"path": str(target), "cause": _safe_str(exc)},
            origin="tracing",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

    logger.info("Trace saved | path=%s | spans=%s", target, len(trace.spans))
    return target

## ============================================================
## HIGH-LEVEL HELPERS
## ============================================================
def add_trace_metadata(key: str, value: Any) -> None:
    """
        Add metadata to current trace

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            None
    """

    session = _CURRENT_SESSION
    if session is None:
        return

    session.metadata[str(key)] = value

def trace_summary(trace: TraceSession) -> Dict[str, Any]:
    """
        Compute basic summary stats for a trace

        Args:
            trace: TraceSession

        Returns:
            Dict
    """

    if trace is None:
        raise ValidationError(
            message="Trace is None",
            error_code="validation_error",
            details={},
            origin="tracing",
            cause=None,
            http_status=400,
            is_retryable=False,
        )

    total_spans = len(trace.spans)
    total_duration = sum(s.duration_sec for s in trace.spans)
    error_spans = sum(1 for s in trace.spans if s.status == "error")

    return {
        "trace_id": trace.trace_id,
        "name": trace.name,
        "total_spans": total_spans,
        "error_spans": error_spans,
        "total_span_duration_sec": total_duration,
    }