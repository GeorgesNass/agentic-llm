'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "End-to-end pipelines for rag-drive-gcp: Drive → OCR → chunking → embeddings → GCS → cleanup, and RAG query."
'''

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.ocr import ocr_document
from src.core.rag import RAGIndex, build_rag_index_from_texts, run_rag_query
from src.core.persistence import load_rag_index_from_gcs
from src.io.drive import download_drive_folder
from src.io.gcs import upload_embeddings_artifact, upload_text_artifact
from src.model.settings import IngestionStatus, RagAnswer, get_settings
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_directories

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("pipelines")


## ============================================================
## GLOBAL IN-MEMORY INDEX (SIMPLE MVP)
## ============================================================

## NOTE:
## For a first MVP, we keep an in-memory index that is rebuilt after ingestion.
## In production, you may load embeddings from GCS and build a persistent vector store.
GLOBAL_RAG_INDEX: Optional[RAGIndex] = None

## ============================================================
## HELPERS
## ============================================================
def _read_text_file(path: Path) -> str:
    """
        Read a text file safely

        Args:
            path (Path): Text file path

        Returns:
            str: File content
    """
    
    return path.read_text(encoding="utf-8", errors="ignore")

def _cleanup_local_traces(settings) -> None:
    """
        Remove local traces after successful pipeline execution

        Args:
            settings: Application settings
    """
    
    if settings.keep_local:
        logger.info("KEEP_LOCAL is enabled. Skipping local cleanup.")
        return

    ## Remove local directories used by the pipeline
    for p in [settings.raw_dir, settings.tmp_dir, settings.traces_dir]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            logger.info(f"Deleted local directory: {p}")

    ## Recreate base directories to keep project structure intact
    ensure_directories(settings.raw_dir, settings.tmp_dir, settings.traces_dir)

def _persist_ingestion_trace(trace_path: Path, payload: Dict) -> None:
    """
        Save a JSON trace to local storage

        Args:
            trace_path (Path): Trace file path
            payload (Dict): JSON-serializable payload
    """
    
    ensure_directories(trace_path.parent)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved ingestion trace: {trace_path}")

## ============================================================
## INGESTION PIPELINE
## ============================================================
def run_drive_ingestion_pipeline(
    drive_folder_id: str,
    run_ocr: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    keep_local: Optional[bool] = None,
) -> Dict:
    """
        Run ingestion pipeline from a Google Drive folder

        Steps:
            1) Download/export Drive files locally
            2) OCR them locally or via remote service (if enabled)
            3) Build chunks and compute embeddings using Vertex AI
            4) Upload TXT and embeddings artifacts to GCS
            5) Cleanup local traces on success (unless KEEP_LOCAL=true)

        Args:
            drive_folder_id (str): Google Drive folder ID
            run_ocr (bool): Whether OCR should be applied to non-text files
            chunk_size (Optional[int]): Override chunk size for this run
            chunk_overlap (Optional[int]): Override overlap for this run
            keep_local (Optional[bool]): Override cleanup behavior for this run

        Returns:
            Dict: Serializable ingestion status (for Streamlit)
    """
    
    global GLOBAL_RAG_INDEX  # noqa: WPS420

    settings = get_settings()

    ## Apply runtime overrides
    if chunk_size is not None:
        settings.chunk_size = int(chunk_size)
    if chunk_overlap is not None:
        settings.chunk_overlap = int(chunk_overlap)
    if keep_local is not None:
        settings.keep_local = bool(keep_local)

    if not drive_folder_id:
        raise ValueError("drive_folder_id is empty.")

    ## Ensure local directories exist
    ensure_directories(settings.raw_dir, settings.tmp_dir, settings.traces_dir)

    logger.info(f"Starting ingestion for Drive folder: {drive_folder_id}")

    ## 1) Download/export all files
    download_results = download_drive_folder(folder_id=drive_folder_id, output_dir=settings.raw_dir)
    downloaded_files_count = len(download_results)

    ## 2) OCR or pass-through text
    text_outputs: List[Tuple[str, str]] = []
    ocr_count = 0

    for res in download_results:
        local_path = res.local_path

        ## If user disabled OCR, only accept text files
        if not run_ocr and local_path.suffix.lower() != ".txt":
            logger.warning(f"Skipping non-text file (OCR disabled): {local_path}")
            continue

        ## OCR step
        ocr_res = ocr_document(input_path=local_path, output_dir=settings.tmp_dir)
        if ocr_res.mode in {"local_docker", "remote_service"}:
            ocr_count += 1

        text = _read_text_file(ocr_res.text_path)
        if text.strip():
            text_outputs.append((local_path.name, text))
        else:
            logger.warning(f"Empty text after OCR: {local_path.name}")

    if not text_outputs:
        raise RuntimeError("No usable text documents after download/OCR.")

    ## 3) Build in-memory RAG index
    GLOBAL_RAG_INDEX = build_rag_index_from_texts(text_outputs)

    ## 4) Upload TXT + embeddings artifacts to GCS
    uploaded_text = 0
    uploaded_emb = 0

    ## Upload text artifacts: one text file per document
    ## NOTE: for MVP, we store them as individual .txt files in tmp/
    for source_name, text in text_outputs:
        local_txt = settings.tmp_dir / f"{source_name}.txt"
        local_txt.write_text(text, encoding="utf-8", errors="ignore")
        upload_text_artifact(local_txt)
        uploaded_text += 1

    ## Upload embeddings as a single artifact for MVP
    ## - embeddings.npy for vectors
    ## - chunks.json for chunk metadata
    emb_dir = settings.traces_dir / "rag_index"
    ensure_directories(emb_dir)

    ## Save embeddings + chunks locally, then upload
    from src.core.rag import save_rag_index  # noqa: WPS433

    save_rag_index(GLOBAL_RAG_INDEX, emb_dir)

    emb_file = emb_dir / "embeddings.npy"
    meta_file = emb_dir / "chunks.json"

    upload_embeddings_artifact(emb_file)
    uploaded_emb += 1

    upload_embeddings_artifact(meta_file)
    uploaded_emb += 1

    ## 5) Save trace + cleanup
    status = IngestionStatus(
        drive_folder_id=drive_folder_id,
        downloaded_files=downloaded_files_count,
        ocr_processed_files=ocr_count,
        uploaded_text_files=uploaded_text,
        uploaded_embedding_files=uploaded_emb,
        message="ok",
        details={
            "downloaded_files": [r.local_path.name for r in download_results],
            "indexed_documents": [s for s, _ in text_outputs],
        },
    )

    trace = status.model_dump()
    _persist_ingestion_trace(settings.traces_dir / "last_ingestion_status.json", trace)

    _cleanup_local_traces(settings)

    logger.info("Ingestion pipeline completed successfully.")
    return trace

## ============================================================
## QUERY PIPELINE (STREAMLIT ENTRY)
## ============================================================
def run_rag_query_pipeline(
    question: str,
    top_k: Optional[int] = None,
) -> RagAnswer:
    """
        Run a RAG query using the current in-memory index

        Args:
            question (str): User question.
            top_k (Optional[int]): Optional override for top-k retrieval

        Returns:
            RagAnswer: Generated answer with sources

        Raises:
            RuntimeError: If ingestion has not been run yet
    """
    
    global GLOBAL_RAG_INDEX  # noqa: WPS420

    settings = get_settings()

    if GLOBAL_RAG_INDEX is None:
        GLOBAL_RAG_INDEX = load_rag_index_from_gcs(
            settings.traces_dir / "rag_index"
        )

    if top_k is not None:
        settings.top_k = int(top_k)

    return run_rag_query(index=GLOBAL_RAG_INDEX, question=question)
