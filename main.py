'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Prod"
__desc__ = "Main Streamlit entrypoint for the rag-drive-gcp project (Drive → OCR → GCS → RAG chat)."
'''

from typing import Dict

import streamlit as st

from src.model.settings import get_settings
from src.pipelines import run_drive_ingestion_pipeline, run_rag_query_pipeline
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("main")


## ============================================================
## UI INITIALIZATION
## ============================================================
def _init_page() -> None:
    """
        Initialize Streamlit page configuration
    """
    
    st.set_page_config(
        page_title="rag-drive-gcp",
        page_icon="🧠",
        layout="wide",
    )

def _ensure_session_state() -> None:
    """
        Ensure required Streamlit session state variables exist
    """
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "last_ingestion_status" not in st.session_state:
        st.session_state["last_ingestion_status"] = None

def _render_sidebar(settings) -> Dict:
    """
        Render sidebar configuration controls

        Args:
            settings: Application settings loaded from environment

        Returns:
            Dict: User-selected configuration parameters
    """
    
    st.sidebar.header("Configuration")

    folder_id = st.sidebar.text_input(
        "Google Drive folder ID",
        value=settings.drive_folder_id or "",
        help="ID of the Google Drive folder to ingest.",
    )

    run_ocr = st.sidebar.checkbox(
        "Enable OCR (PDFs / images)",
        value=True,
        help="Run OCR locally when documents are not text-based.",
    )

    top_k = st.sidebar.slider(
        "Top-K retrieved chunks",
        min_value=1,
        max_value=30,
        value=int(settings.top_k),
        step=1,
    )

    chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=256,
        max_value=2048,
        value=int(settings.chunk_size),
        step=64,
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk overlap",
        min_value=0,
        max_value=512,
        value=int(settings.chunk_overlap),
        step=32,
    )

    keep_local = st.sidebar.checkbox(
        "Keep local files (debug)",
        value=bool(settings.keep_local),
        help="If disabled, local temporary files are deleted after successful upload to GCS.",
    )

    return {
        "folder_id": folder_id.strip(),
        "run_ocr": run_ocr,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "keep_local": keep_local,
    }

## ============================================================
## MAIN STREAMLIT APPLICATION
## ============================================================
def main() -> None:
    """
        Main Streamlit application entrypoint

        Features:
            - Google Drive ingestion
            - Optional OCR (local docker or remote microservice)
            - Upload TXT + embeddings artifacts to GCS
            - RAG-based chat using Vertex AI
    """
    
    _init_page()
    _ensure_session_state()

    settings = get_settings()
    params = _render_sidebar(settings)

    st.title("rag-drive-gcp")
    st.caption(
        "Drive API → OCR (local) → GCS (text + embeddings) → Vertex AI → RAG Chat"
    )

    ## ========================================================
    ## INGESTION SECTION
    ## ========================================================
    st.subheader("1) Ingest documents from Google Drive")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        ingest_clicked = st.button(
            "Run ingestion",
            use_container_width=True,
        )

    with col_right:
        st.info(
            "Downloads and exports Drive files, applies OCR if needed, "
            "generates embeddings with Vertex AI, uploads artifacts to GCS, "
            "and cleans up local traces."
        )

    if ingest_clicked:
        if not params["folder_id"]:
            st.error("Please provide a Google Drive folder ID.")
        else:
            try:
                logger.info("Starting ingestion pipeline from Streamlit UI.")
                status = run_drive_ingestion_pipeline(
                    drive_folder_id=params["folder_id"],
                    run_ocr=params["run_ocr"],
                    chunk_size=params["chunk_size"],
                    chunk_overlap=params["chunk_overlap"],
                    keep_local=params["keep_local"],
                )
                st.session_state["last_ingestion_status"] = status
                st.success("Ingestion completed successfully.")
                st.json(status)
            except Exception as exc:
                logger.exception("Ingestion pipeline failed.")
                st.error("Ingestion failed. Check logs for details.")
                st.exception(exc)

    if st.session_state.get("last_ingestion_status"):
        with st.expander("Last ingestion status", expanded=False):
            st.json(st.session_state["last_ingestion_status"])

    st.divider()

    ## ========================================================
    ## RAG CHAT SECTION
    ## ========================================================
    st.subheader("2) RAG Chat")

    user_question = st.text_input(
        "Ask a question",
        value="",
        placeholder="e.g. Summarize the key medical insights from the documents",
    )

    ask_clicked = st.button(
        "Ask",
        use_container_width=True,
    )

    if ask_clicked and user_question.strip():
        try:
            logger.info("Executing RAG query.")
            answer = run_rag_query_pipeline(
                question=user_question.strip(),
                top_k=params["top_k"],
            )

            st.session_state["chat_history"].append(
                {"role": "user", "content": user_question.strip()}
            )
            
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
        except Exception as exc:
            logger.exception("RAG query failed.")
            st.error("Query failed. Check logs for details.")
            st.exception(exc)

    ## Render chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

## ============================================================
## SCRIPT ENTRY POINT
## ============================================================
if __name__ == "__main__":
    main()