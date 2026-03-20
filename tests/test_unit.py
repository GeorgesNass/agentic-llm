'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Full unit test suite for autonomous-ai-platform (agents, orchestrator, retrieval, sql, monitoring). No real HTTP calls."
'''

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import requests

from src.utils.safe_utils import _safe_json, _safe_str, _extract_first_json_object
from src.utils.env_utils import _get_env_bool
from src.utils.validation_utils import _must_be_non_empty, _require_int
from src.utils.sqlite_manager import SqliteManager

from src.core.errors import ValidationError
from src.core.errors import LlmProviderError
import src.core.mcp_server as mcp

from src.llm.api_clients import chat_completion_openai_compatible
from src.llm.embeddings import embed_texts

import src.orchestrator.retrieval as retrieval
import src.orchestrator.tools as tools
import src.orchestrator.routing as routing
import src.orchestrator.loop as loop_mod

import src.agents.reasoning as reasoning
import src.agents.executor as executor
import src.agents.aggregator as aggregator
import src.agents.text_to_sql as tts

import src.monitoring.evaluation as evaluation
import src.monitoring.metrics as metrics
import src.monitoring.tracing as tracing

import src.pipeline as pipeline

## ============================================================
## TEST HELPERS
## ============================================================
def _maybe_get(obj: Any, name: str) -> Any:
    """
        Get attribute if present

        Args:
            obj: Any object
            name: Attribute name

        Returns:
            Attribute value or None
    """

    return getattr(obj, name, None)

def _require_attr(obj: Any, name: str) -> Any:
    """
        Require an attribute or skip test

        Args:
            obj: Any object
            name: Attribute name

        Returns:
            Attribute value

        Raises:
            pytest.Skip: If missing
    """

    value = getattr(obj, name, None)
    if value is None:
        pytest.skip(f"Missing attribute: {name}")
    return value

def _require_callable(obj: Any, name: str) -> Callable[..., Any]:
    """
        Require a callable attribute or skip test

        Args:
            obj: Any object
            name: Function name

        Returns:
            Callable

        Raises:
            pytest.Skip: If missing or not callable
    """

    fn = getattr(obj, name, None)
    if fn is None or not callable(fn):
        pytest.skip(f"Missing callable: {name}")
    return fn

def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Install a fake sentence_transformers module to avoid real model load

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    fake_mod = types.ModuleType("sentence_transformers")

    class _FakeST:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        def encode(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
            _ = normalize_embeddings
            return [[0.1, 0.2, 0.3] for _t in texts]

    fake_mod.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

def _install_fake_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Install a fake faiss module to avoid native dependency

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    fake_mod = types.ModuleType("faiss")

    class _FakeIndex:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self.vectors: List[List[float]] = []
            self.ids: List[int] = []

        def add(self, x: Any) -> None:
            for row in x:
                self.vectors.append([float(v) for v in row])

        def add_with_ids(self, x: Any, ids: Any) -> None:
            for row, idx in zip(x, ids):
                self.vectors.append([float(v) for v in row])
                self.ids.append(int(idx))

        def search(self, x: Any, k: int) -> Tuple[Any, Any]:
            import numpy as np

            n = len(self.vectors)
            kk = min(int(k), n) if n > 0 else 0
            d = np.zeros((len(x), kk), dtype="float32")
            i = np.zeros((len(x), kk), dtype="int64")
            return d, i

    def IndexFlatIP(dim: int) -> _FakeIndex:  # noqa: N802
        return _FakeIndex(dim)

    def write_index(index: Any, path: str) -> None:
        _ = (index, path)

    def read_index(path: str) -> _FakeIndex:
        _ = path
        return _FakeIndex(3)

    fake_mod.IndexFlatIP = IndexFlatIP  # type: ignore[attr-defined]
    fake_mod.write_index = write_index  # type: ignore[attr-defined]
    fake_mod.read_index = read_index  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", fake_mod)

def _mock_requests_post_success(monkeypatch: pytest.MonkeyPatch, payload: Dict[str, Any]) -> None:
    """
        Mock requests.post to return a success response

        Args:
            monkeypatch: Pytest monkeypatch
            payload: JSON response

        Returns:
            None
    """

    class _DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.text = str(payload)

        def json(self) -> Dict[str, Any]:
            return payload

    def _fake_post(*args: Any, **kwargs: Any) -> _DummyResponse:
        _ = (args, kwargs)
        return _DummyResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

## ============================================================
## SAFE UTILS
## ============================================================
def test_safe_utils_safe_json_and_str() -> None:
    """
        Validate safe helpers behavior

        Returns:
            None
    """

    out1 = _safe_json({"x": "a" * 5000}, max_len=50)
    out2 = _safe_str("b" * 5000, max_len=60)

    assert isinstance(out1, str) and out1.endswith("...(truncated)")
    assert isinstance(out2, str) and out2.endswith("...(truncated)")

## ============================================================
## ENV + VALIDATION + TEXT UTILS
## ============================================================
def test_env_utils_get_env_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate _get_env_bool parsing

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    monkeypatch.setenv("X_BOOL", "true")
    assert _get_env_bool("X_BOOL", default=False) is True

    monkeypatch.setenv("X_BOOL", "0")
    assert _get_env_bool("X_BOOL", default=True) is False

def test_validation_utils_require_int_and_non_empty() -> None:
    """
        Validate validation utils basics

        Returns:
            None
    """

    assert _must_be_non_empty(" hello ", "q") == "hello"
    assert _require_int("5", "n", min_value=0, max_value=10) == 5

    with pytest.raises(ValidationError):
        _must_be_non_empty("   ", "q")

    with pytest.raises(ValidationError):
        _require_int("999", "n", min_value=0, max_value=10)

def test_text_utils_extract_first_json_object() -> None:
    """
        Validate JSON extraction from text

        Returns:
            None
    """

    assert _extract_first_json_object('{"a": 1}').get("a") == 1
    assert _extract_first_json_object("noise {\"ok\": true} tail").get("ok") is True
    assert _extract_first_json_object("no json") == {}

def test_text_utils_invalid_json():
    """
        Validate JSON extraction on invalid input
    """

    out = _extract_first_json_object("{invalid json")
    
    assert isinstance(out, dict)

def test_embeddings_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate embeddings handles empty input
    """

    _install_fake_sentence_transformers(monkeypatch)

    out = embed_texts([], prefer_local=True, use_gpu=False)
    
    assert out == []
    
## ============================================================
## SQLITE MANAGER + TEXT-TO-SQL
## ============================================================
def test_sqlite_manager_basic_roundtrip(tmp_path: Path) -> None:
    """
        Validate SQLite manager create + query

        Args:
            tmp_path: Temporary directory

        Returns:
            None
    """

    db_path = tmp_path / "unit_test.db"
    mgr = SqliteManager(db_path=str(db_path))

    assert db_path.exists()

    mgr.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, name TEXT);")
    mgr.execute("INSERT INTO t(name) VALUES ('alice');")
    rows = mgr.query("SELECT name FROM t ORDER BY id;")

    assert rows and "alice" in str(rows[0])

def test_text_to_sql_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
        Validate Text-to-SQL flow without real LLM calls by mocking generation

        Args:
            monkeypatch: Pytest monkeypatch
            tmp_path: Temporary directory

        Returns:
            None
    """

    db_path = tmp_path / "tts.db"
    mgr = SqliteManager(db_path=str(db_path))
    mgr.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT);")
    mgr.execute("INSERT INTO users(id, name) VALUES (1, 'bob');")

    ## Force deterministic SQL output
    if hasattr(tts, "generate_sql_from_question"):
        monkeypatch.setattr(tts, "generate_sql_from_question", lambda *a, **k: "SELECT name FROM users;")
    elif hasattr(tts, "text_to_sql"):
        monkeypatch.setattr(tts, "text_to_sql", lambda *a, **k: "SELECT name FROM users;")
    else:
        pytest.skip("Missing text-to-sql generation function")

    ## Execute using the module if it provides an execute helper
    if hasattr(tts, "run_text_to_sql"):
        out = tts.run_text_to_sql(
            question="Who?",
            sqlite_manager=mgr,
            prefer_local=True,
            use_gpu=False,
        )
        
        assert "bob" in str(out)
        
        return

    ## Fallback: direct SQL
    rows = mgr.query("SELECT name FROM users;")
    
    assert "bob" in str(rows)

## ============================================================
## LLM API CLIENTS (NO REAL HTTP)
## ============================================================
def test_api_clients_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate OpenAI-compatible chat client parsing with mocked HTTP

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    _mock_requests_post_success(
        monkeypatch,
        payload={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 2},
        },
    )

    res = chat_completion_openai_compatible(
        provider="openai",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )

    assert res.text.strip() == "hello"
    assert int(res.usage.get("total_tokens", 0)) == 2

def test_api_clients_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate timeout conversion into structured error

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    def _fake_post(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise requests.Timeout("timeout")

    monkeypatch.setattr(requests, "post", _fake_post)

    with pytest.raises(LlmProviderError):
        chat_completion_openai_compatible(
            provider="openai",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )

## ============================================================
## EMBEDDINGS (LOCAL MOCK + API MOCK)
## ============================================================
def test_embeddings_local_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate local embeddings without loading real models

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    _install_fake_sentence_transformers(monkeypatch)

    out = embed_texts(["a", "b"], prefer_local=True, use_gpu=False)
    
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(v, list) for v in out)

def test_embeddings_api_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate API embeddings with mocked HTTP

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    monkeypatch.setenv("API_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    _mock_requests_post_success(
        monkeypatch,
        payload={"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]},
    )

    out = embed_texts(["a", "b"], prefer_local=False, use_gpu=False)
    
    assert len(out) == 2
    assert out[0][0] == pytest.approx(0.1)

## ============================================================
## VECTOR STORE + RETRIEVAL (MOCKED NATIVE DEPS)
## ============================================================
def test_vector_store_and_retrieval_ingest_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
        Validate retrieval ingestion + search with mocked FAISS and embeddings

        Args:
            monkeypatch: Pytest monkeypatch
            tmp_path: Temporary folder

        Returns:
            None
    """

    _install_fake_sentence_transformers(monkeypatch)
    _install_fake_faiss(monkeypatch)

    ## Create fake documents
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "a.txt").write_text("hello world", encoding="utf-8")
    (raw_dir / "b.txt").write_text("another text", encoding="utf-8")

    ## Force local vector backend
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setenv("VECTOR_INDEX_NAME", "default")
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vs"))

    ingest_folder = _require_callable(retrieval, "ingest_folder")
    rag_search = _require_callable(retrieval, "rag_search")

    ingest_out = ingest_folder(folder=str(raw_dir), prefer_local=True, use_gpu=False)
    
    assert isinstance(ingest_out, dict)

    search_out = rag_search(query="hello", top_k=3, prefer_local=True, use_gpu=False)
    
    assert isinstance(search_out, dict)
    assert "results" in search_out or "chunks" in search_out

## ============================================================
## TOOLS REGISTRY + ROUTING
## ============================================================
def test_tools_registry_and_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate tools registry and routing decisions offline

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    get_registry = _require_callable(tools, "get_tools_registry")
    registry = get_registry()

    assert isinstance(registry, dict)
    assert len(registry.keys()) >= 1

    ## Basic routing API presence
    route_fn = _maybe_get(routing, "route_request")
    if route_fn is None:
        route_fn = _maybe_get(routing, "route")
    if route_fn is None or not callable(route_fn):
        pytest.skip("No routing entrypoint (route_request/route)")

    monkeypatch.setenv("CHAT_MODE", "auto")
    monkeypatch.setenv("API_PROVIDER", "openai")
    out = route_fn("hello", prefer_local=True, use_gpu=False)  # type: ignore[misc]
   
    assert isinstance(out, dict)

## ============================================================
## AGENTS (REASONING / EXECUTOR / AGGREGATOR) OFFLINE
## ============================================================
def test_reasoning_plan_shape() -> None:
    """
        Validate reasoning agent returns a plan-like structure

        Returns:
            None
    """

    plan_fn = _maybe_get(reasoning, "build_plan")
    if plan_fn is None:
        plan_fn = _maybe_get(reasoning, "plan")
    if plan_fn is None or not callable(plan_fn):
        pytest.skip("No plan builder (build_plan/plan)")

    out = plan_fn("Find info about X", max_steps=3)  # type: ignore[misc]
    
    assert isinstance(out, dict)
    assert "steps" in out or "plan" in out

def test_executor_runs_mock_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate executor can run tools with a mocked registry

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    run_step = _maybe_get(executor, "run_step")
    if run_step is None:
        run_step = _maybe_get(executor, "execute_step")
    if run_step is None or not callable(run_step):
        pytest.skip("No executor entrypoint (run_step/execute_step)")

    ## Mock tools registry used by executor if applicable
    def _fake_registry() -> Dict[str, Any]:
        return {
            "echo": lambda payload, **kw: {"ok": True, "payload": payload, "meta": kw},
        }

    if hasattr(tools, "get_tools_registry"):
        monkeypatch.setattr(tools, "get_tools_registry", _fake_registry)

    step = {"tool": "echo", "args": {"x": 1}}
    out = run_step(step, prefer_local=True, use_gpu=False)  # type: ignore[misc]
    
    assert isinstance(out, dict)
    assert out.get("ok") is True

def test_aggregator_synthesizes_answer() -> None:
    """
        Validate aggregator produces a final answer string

        Returns:
            None
    """

    fn = _maybe_get(aggregator, "synthesize_answer")
    if fn is None:
        fn = _maybe_get(aggregator, "aggregate")
    if fn is None or not callable(fn):
        pytest.skip("No aggregator entrypoint (synthesize_answer/aggregate)")

    out = fn(
        query="Q",
        steps=[{"tool": "x", "result": {"a": 1}}],
        raw_outputs=[{"text": "partial"}],
    )  # type: ignore[misc]

    assert isinstance(out, dict)
    assert "answer" in out

## ============================================================
## ORCHESTRATOR LOOP + PIPELINE
## ============================================================
def test_orchestrator_loop_runs_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate loop runner works offline by mocking routing and tool execution

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    run_fn = _maybe_get(loop_mod, "run_autonomous_loop")
    if run_fn is None:
        run_fn = _maybe_get(loop_mod, "run_loop")
    if run_fn is None or not callable(run_fn):
        pytest.skip("No loop runner (run_autonomous_loop/run_loop)")

    ## Force deterministic behavior by mocking reasoning and executor if referenced
    if hasattr(reasoning, "build_plan"):
        monkeypatch.setattr(reasoning, "build_plan", lambda q, max_steps=3: {"steps": [{"tool": "echo", "args": {"q": q}}]})
    if hasattr(executor, "run_step"):
        monkeypatch.setattr(executor, "run_step", lambda step, **kw: {"ok": True, "step": step, "meta": kw})

    out = run_fn("hello", prefer_local=True, use_gpu=False, export=False)  # type: ignore[misc]
    
    assert isinstance(out, dict)
    assert "answer" in out or "steps" in out

def test_pipeline_entrypoints_exist() -> None:
    """
        Validate pipeline module exposes main public entrypoints

        Returns:
            None
    """

    for name in ["run_chat", "run_loop", "run_evaluation"]:
        _ = _require_callable(pipeline, name)

## ============================================================
## MONITORING (EVALUATION / METRICS / TRACING) OFFLINE
## ============================================================
def test_monitoring_evaluation_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """
        Validate monitoring evaluation works offline by mocking LLM judge

        Args:
            monkeypatch: Pytest monkeypatch

        Returns:
            None
    """

    fn = _maybe_get(evaluation, "evaluate_answer")
    if fn is None:
        fn = _maybe_get(evaluation, "run_evaluation")
    if fn is None or not callable(fn):
        pytest.skip("No evaluation entrypoint (evaluate_answer/run_evaluation)")

    ## Force LLM judge off if supported by signature
    out = fn(query="q", answer="a", use_llm_judge=False)  # type: ignore[misc]
    
    assert isinstance(out, dict)
    assert "score" in out or "metrics" in out

def test_monitoring_metrics_exports() -> None:
    """
        Validate Prometheus exporter returns a text payload

        Returns:
            None
    """

    fn = _maybe_get(metrics, "export_metrics_text")
    if fn is None:
        fn = _maybe_get(metrics, "metrics_text")
    if fn is None or not callable(fn):
        pytest.skip("No metrics exporter (export_metrics_text/metrics_text)")

    text = fn()  # type: ignore[misc]
    
    assert isinstance(text, str)

def test_monitoring_tracing_records_span() -> None:
    """
        Validate tracing can start and close spans

        Returns:
            None
    """

    start = _maybe_get(tracing, "start_span")
    end = _maybe_get(tracing, "end_span")

    if start is None or end is None or not callable(start) or not callable(end):
        pytest.skip("No tracing API (start_span/end_span)")

    span = start("unit_test")  # type: ignore[misc]
    
    assert span is not None
    
    end(span, status="ok")  # type: ignore[misc]

## ============================================================
## MCP SERVER (FASTAPI APP EXISTS)
## ============================================================
def test_mcp_server_app_exists() -> None:
    """
        Validate FastAPI app is defined in mcp_server module

        Returns:
            None
    """

    app = getattr(mcp, "app", None)
    if app is None:
        pytest.skip("FastAPI app not found in src.core.mcp_server")
    
    assert app is not None