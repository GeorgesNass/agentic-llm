'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Streamlit UI (chat + loop + evaluation + status) for autonomous-ai-platform."
'''

from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st

from src.core.config import config
from src.core.errors import AutonomousAIPlatformError
from src.pipeline import run_chat, run_evaluation, run_loop
from src.utils.safe_utils import _safe_json, _safe_str
from src.utils.validation_utils import _must_be_non_empty

## ============================================================
## SESSION STATE
## ============================================================
def _init_session_state() -> None:
    """
        Initialize Streamlit session state

        Returns:
            None
    """

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "prefer_local" not in st.session_state:
        st.session_state.prefer_local = True

    if "use_gpu" not in st.session_state:
        st.session_state.use_gpu = False

    if "last_loop_result" not in st.session_state:
        st.session_state.last_loop_result = {}

    if "last_eval_result" not in st.session_state:
        st.session_state.last_eval_result = {}

## ============================================================
## UI HELPERS
## ============================================================
def _render_sidebar() -> None:
    """
        Render sidebar controls

        Returns:
            None
    """

    st.sidebar.header("Runtime")
    st.session_state.prefer_local = st.sidebar.toggle("Prefer local", value=bool(st.session_state.prefer_local))
    st.session_state.use_gpu = st.sidebar.toggle("Use GPU", value=bool(st.session_state.use_gpu))

    st.sidebar.divider()

    if st.sidebar.button("Clear chat"):
        st.session_state.chat_history = []

    st.sidebar.caption(f"models_dir: {config.paths.artifacts_models_dir}")
    st.sidebar.caption(f"vector_store_dir: {config.paths.artifacts_vector_store_dir}")
    st.sidebar.caption(f"reports_dir: {config.paths.artifacts_reports_dir}")

def _ui_error(exc: Exception) -> None:
    """
        Show structured errors in Streamlit

        Args:
            exc: Exception

        Returns:
            None
    """

    if isinstance(exc, AutonomousAIPlatformError):
        payload = exc.to_payload()
        st.error(f"{payload.error_code} | {payload.message}")
        st.caption(f"origin={payload.origin}")
        if payload.details:
            st.code(_safe_json(payload.details), language="json")
        return

    st.error("internal_error | An unexpected error occurred")
    st.code(_safe_str(exc))

def _render_json_block(title: str, payload: Dict[str, Any]) -> None:
    """
        Render a JSON dict

        Args:
            title: Title
            payload: Dict payload

        Returns:
            None
    """

    st.subheader(title)
    st.code(_safe_json(payload), language="json")

## ============================================================
## CHAT TAB
## ============================================================
def _render_chat_tab() -> None:
    """
        Render chat tab

        Returns:
            None
    """

    st.subheader("Chat")

    ## Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

    prompt = st.chat_input("Type your message")
    if not prompt:
        return

    try:
        user_text = _must_be_non_empty(prompt, "prompt")
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        result = run_chat(
            user_text,
            prefer_local=bool(st.session_state.prefer_local),
            use_gpu=bool(st.session_state.use_gpu),
        )

        answer = str(result.get("text", "")).strip() or "(empty response)"
        meta = f"\n\n---\nprovider=`{result.get('provider','')}` | model=`{result.get('model','')}`"

        st.session_state.chat_history.append({"role": "assistant", "content": answer + meta})
        st.rerun()

    except Exception as exc:
        _ui_error(exc)

## ============================================================
## LOOP TAB
## ============================================================
def _render_loop_tab() -> None:
    """
        Render loop tab

        Returns:
            None
    """

    st.subheader("Autonomous Loop")

    query = st.text_area("Query", height=120)
    export = st.toggle("Export artifacts", value=True)

    if st.button("Run loop"):
        try:
            q = _must_be_non_empty(query, "query")
            result = run_loop(
                q,
                prefer_local=bool(st.session_state.prefer_local),
                use_gpu=bool(st.session_state.use_gpu),
                export=bool(export),
            )
            st.session_state.last_loop_result = result

        except Exception as exc:
            _ui_error(exc)

    if st.session_state.last_loop_result:
        _render_json_block("Last loop result", st.session_state.last_loop_result)

        answer = str(st.session_state.last_loop_result.get("answer", "")).strip()
        if answer:
            st.subheader("Answer")
            st.write(answer)

## ============================================================
## EVALUATION TAB
## ============================================================
def _render_evaluation_tab() -> None:
    """
        Render evaluation tab

        Returns:
            None
    """

    st.subheader("Evaluation")

    query = st.text_area("Query", height=100)
    answer = st.text_area("Answer", height=140)
    export = st.toggle("Export report", value=True)
    use_llm_judge = st.toggle("Use LLM judge", value=True)

    if st.button("Run evaluation"):
        try:
            q = _must_be_non_empty(query, "query")
            a = _must_be_non_empty(answer, "answer")

            report = run_evaluation(
                q,
                a,
                use_llm_judge=bool(use_llm_judge),
                prefer_local=bool(st.session_state.prefer_local),
                use_gpu=bool(st.session_state.use_gpu),
                export=bool(export),
            )
            st.session_state.last_eval_result = report

        except Exception as exc:
            _ui_error(exc)

    if st.session_state.last_eval_result:
        _render_json_block("Last evaluation report", st.session_state.last_eval_result)

## ============================================================
## STATUS TAB
## ============================================================
def _render_status_tab() -> None:
    """
        Render status tab

        Returns:
            None
    """

    st.subheader("Status")

    payload: Dict[str, Any] = {
        "paths": {
            "models_dir": str(config.paths.artifacts_models_dir),
            "vector_store_dir": str(config.paths.artifacts_vector_store_dir),
            "reports_dir": str(config.paths.artifacts_reports_dir),
            "raw_dir": str(config.paths.data_raw_dir),
            "sqlite_dir": str(config.paths.data_sqlite_dir),
        },
        "runtime": {
            "prefer_local": bool(st.session_state.prefer_local),
            "use_gpu": bool(st.session_state.use_gpu),
        },
    }

    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

## ============================================================
## APP ENTRY POINT
## ============================================================
def run_streamlit_app() -> None:
    """
        Streamlit entry point

        Returns:
            None
    """

    st.set_page_config(page_title="Autonomous AI Platform", layout="wide")

    _init_session_state()
    _render_sidebar()

    st.title("Autonomous AI Platform")

    tabs = st.tabs(["Chat", "Loop", "Evaluate", "Status"])
    with tabs[0]:
        _render_chat_tab()
    with tabs[1]:
        _render_loop_tab()
    with tabs[2]:
        _render_evaluation_tab()
    with tabs[3]:
        _render_status_tab()