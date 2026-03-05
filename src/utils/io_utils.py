'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Filesystem and file discovery utilities for autonomous-ai-platform."
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.errors import OrchestrationError, StorageError
from src.utils.safe_utils import _safe_str

## ============================================================
## FILESYSTEM HELPERS
## ============================================================
def ensure_directory(path: str | Path) -> Path:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            Resolved Path
    """

    resolved = Path(path).expanduser().resolve()

    ## Create directory if missing
    resolved.mkdir(parents=True, exist_ok=True)

    return resolved

def safe_read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """
        Safely read a text file

        Args:
            path: File path
            encoding: File encoding

        Returns:
            File content as string
    """

    file_path = Path(path).expanduser().resolve()

    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise StorageError(
            message="Failed to read text file",
            error_code="storage_error",
            details={"path": str(path)},
            origin="retrieval",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

def safe_write_text_file(
    path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """
        Safely write text to file

        Args:
            path: File path
            content: Text content
            encoding: File encoding

        Returns:
            None
    """

    file_path = Path(path).expanduser().resolve()

    ## Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    ## Write content
    file_path.write_text(content, encoding=encoding)

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
        Write JSON file safely

        Args:
            path: File path
            payload: Dict payload

        Returns:
            None
    """

    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        safe_write_text_file(path, text)
    except Exception as exc:
        raise OrchestrationError(
            message="Failed to write JSON export",
            error_code="storage_error",
            details={"path": str(path), "cause": _safe_str(exc)},
            origin="pipeline",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## FILE DISCOVERY
## ============================================================
def _list_text_files(folder: Path) -> List[Path]:
    """
        List input files to ingest

        Args:
            folder: Root folder

        Returns:
            File list
    """

    if not folder.exists():
        return []

    ## Whitelist extensions only
    exts = {".txt", ".md", ".csv", ".json", ".log"}

    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    return files
    
## ============================================================
## HUGGINGFACE DOWNLOAD
## ============================================================
def _hf_download_model_file(
    repo_id: str,
    filename: str,
    target_dir: Path,
    hf_token: str = "",
) -> Path:
    """
        Download a single file from HuggingFace Hub into target directory

        Strategy:
            1) Try python package huggingface_hub
            2) Fallback to hf CLI if installed

        Args:
            repo_id: HuggingFace repo id
            filename: File name inside repo
            target_dir: Download target directory
            hf_token: Optional HF token for gated/private repos

        Returns:
            Local file path

        Raises:
            DependencyError: If download fails
    """

    ## Ensure target directory exists
    ensure_directory(target_dir)

    ## Try HuggingFace Hub Python client first
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        ## Download file into local_dir
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token if hf_token else None,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )

        ## Normalize local path
        resolved = Path(local_path).resolve()

        ## Validate file exists
        if not resolved.exists():
            raise FileNotFoundError(f"Downloaded file not found: {resolved}")

        return resolved

    except ModuleNotFoundError:
        logger.warning("huggingface_hub not installed, using hf CLI fallback")

    except Exception as exc:
        raise DependencyError(
            message="HuggingFace model download failed (huggingface_hub)",
            error_code="dependency_error",
            details={"repo_id": repo_id, "filename": filename},
            origin="local_runtime",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

    ## Fallback to HuggingFace CLI if available
    try:
        ## Prepare environment variables for CLI
        env = dict(os.environ)
        if hf_token:
            env["HF_TOKEN"] = hf_token

        ## Build hf CLI command
        cmd = ["hf", "download", repo_id, filename, "--local-dir", str(target_dir)]

        ## Execute command and capture output for debugging
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)

        ## Resolve file path after download
        resolved = (target_dir / filename).resolve()

        ## Validate file exists
        if not resolved.exists():
            raise FileNotFoundError(f"Downloaded file not found: {resolved}")

        return resolved

    except Exception as exc:
        raise DependencyError(
            message="HuggingFace model download failed (hf CLI fallback)",
            error_code="dependency_error",
            details={
                "repo_id": repo_id,
                "filename": filename,
                "target_dir": str(target_dir),
            },
            origin="local_runtime",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc