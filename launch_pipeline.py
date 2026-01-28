'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "CLI menu to run ingestion and RAG query for rag-drive-gcp."
'''

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.model.settings import get_settings
from src.pipelines import run_drive_ingestion_pipeline, run_rag_query_pipeline
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER INITIALIZATION
## ============================================================
logger = get_logger("launch_pipeline")

## ============================================================
## CLI HELPERS
## ============================================================
def _print_json(payload: dict) -> None:
    """
        Print JSON to stdout

        Args:
            payload (dict): JSON-serializable payload
    """
    
    print(json.dumps(payload, indent=2, ensure_ascii=False))

def _validate_folder_id(folder_id: Optional[str]) -> str:
    """
        Validate Drive folder ID

        Args:
            folder_id (Optional[str]): Folder ID from args or settings

        Returns:
            str: Valid folder ID

        Raises:
            ValueError: If folder ID is missing
    """
    
    if folder_id and folder_id.strip():
        return folder_id.strip()

    settings = get_settings()
    if settings.drive_folder_id and settings.drive_folder_id.strip():
        return settings.drive_folder_id.strip()

    raise ValueError(
        "Missing Drive folder ID. Provide --drive-folder-id or set DRIVE_FOLDER_ID in .env."
    )

def _parse_args() -> argparse.Namespace:
    """
        Parse CLI arguments

        Returns:
            argparse.Namespace: Parsed args
    """
    
    parser = argparse.ArgumentParser(
        description="rag-drive-gcp CLI: ingest Drive data and query RAG."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["ingest", "query"],
        help="Mode to run: ingest | query",
    )

    parser.add_argument(
        "--drive-folder-id",
        type=str,
        default=None,
        help="Google Drive folder ID to ingest (optional if set in .env).",
    )

    parser.add_argument(
        "--run-ocr",
        action="store_true",
        help="Enable OCR for non-text files (default: false).",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size for this run.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap for this run.",
    )

    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep local traces (default: false).",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for RAG query mode.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k retrieval for query mode.",
    )

    return parser.parse_args()

## ============================================================
## CLI MODES
## ============================================================
def run_ingest(args: argparse.Namespace) -> None:
    """
        Run ingestion mode

        Args:
            args (argparse.Namespace): CLI args
    """
    
    folder_id = _validate_folder_id(args.drive_folder_id)

    logger.info("Starting ingestion from CLI.")
    status = run_drive_ingestion_pipeline(
        drive_folder_id=folder_id,
        run_ocr=bool(args.run_ocr),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        keep_local=bool(args.keep_local),
    )
    _print_json(status)

def run_query(args: argparse.Namespace) -> None:
    """
        Run query mode

        Notes:
            - Query mode requires that ingestion has been run in the same process
              (in-memory index)
            - For production, you would load index artifacts from GCS

        Args:
            args (argparse.Namespace): CLI args
    """
    
    if not args.question or not args.question.strip():
        raise ValueError("Missing --question for query mode.")

    logger.info("Starting RAG query from CLI.")
    answer = run_rag_query_pipeline(
        question=args.question.strip(),
        top_k=args.top_k,
    )

    ## Print only the answer text for CLI readability
    print(answer.answer)

## ============================================================
## MAIN ENTRY POINT
## ============================================================
def main() -> None:
    """
        Main CLI entrypoint
    """
    
    args = _parse_args()

    try:
        if args.mode == "ingest":
            run_ingest(args)
        elif args.mode == "query":
            run_query(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as exc:
        logger.exception(f"CLI execution failed: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()