'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Google Drive I/O for rag-drive-gcp: authenticate with service account and list/download/export files."
'''

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.model.settings import DriveFileMeta, get_settings
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_directories

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("drive")

## ============================================================
## CONSTANTS
## ============================================================

## Google Drive scope for read-only access
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

## Common Google Docs export MIME types
EXPORT_MIME_TEXT = "text/plain"
EXPORT_MIME_PDF = "application/pdf"

## ============================================================
## DATA CLASSES
## ============================================================
@dataclass
class DriveDownloadResult:
    """
        Download/export result for a single Drive file

        Attributes:
            file_meta (DriveFileMeta): Drive file metadata
            local_path (Path): Local file path where content was saved
            exported_mime_type (Optional[str]): Exported MIME type if exported
    """
    
    file_meta: DriveFileMeta
    local_path: Path
    exported_mime_type: Optional[str] = None

## ============================================================
## AUTHENTICATION
## ============================================================
def _get_service_account_credentials() -> service_account.Credentials:
    """
        Build service account credentials for Google Drive API

        This function expects either:
            - GCP_SA_JSON_PATH: path to the service account JSON file

        Returns:
            service_account.Credentials: Google credentials with Drive scopes

        Raises:
            ValueError: If credentials environment variable is missing
            FileNotFoundError: If JSON path does not exist
    """
    
    settings = get_settings()

    ## NOTE: We keep this explicit and strict on purpose
    ## to avoid silently using the wrong credentials.
    sa_path = os.getenv("GCP_SA_JSON_PATH")

    if not sa_path:
        raise ValueError(
            "Missing service account configuration. "
            "Please set GCP_SA_JSON_PATH in your .env."
        )

    sa_file = Path(sa_path)
    if not sa_file.exists():
        raise FileNotFoundError(f"Service account JSON not found: {sa_file}")

    credentials = service_account.Credentials.from_service_account_file(
        filename=str(sa_file),
        scopes=DRIVE_SCOPES,
    )

    logger.info("Service account credentials loaded for Drive API.")
    logger.debug(f"GCP project (from settings): {settings.gcp_project_id}")

    return credentials

def get_drive_service():
    """
        Create a Google Drive API service client

        Returns:
            Resource: Google Drive API service instance
    """
    
    credentials = _get_service_account_credentials()

    ## Build Drive API client
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service

## ============================================================
## LIST FILES
## ============================================================
def list_drive_files(
    folder_id: str,
    max_results: int = 200,
    query_extra: Optional[str] = None,
) -> List[DriveFileMeta]:
    """
        List files from a Google Drive folder

        Args:
            folder_id (str): Google Drive folder ID
            max_results (int): Maximum results to return
            query_extra (Optional[str]): Optional extra query (Drive query language)

        Returns:
            List[DriveFileMeta]: List of file metadata
    """
    
    service = get_drive_service()

    ## Build query: files directly under folder
    query = f"'{folder_id}' in parents and trashed = false"
    if query_extra:
        query = f"{query} and ({query_extra})"

    fields = "files(id, name, mimeType, modifiedTime)"

    logger.info(f"Listing Drive files in folder: {folder_id}")

    response = service.files().list(
        q=query,
        pageSize=max_results,
        fields=fields,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = response.get("files", [])
    results: List[DriveFileMeta] = []

    for f in files:
        results.append(
            DriveFileMeta(
                file_id=f["id"],
                name=f.get("name", ""),
                mime_type=f.get("mimeType", ""),
                modified_time=f.get("modifiedTime"),
            )
        )

    logger.info(f"Found {len(results)} file(s) in Drive folder.")
    return results

## ============================================================
## DOWNLOAD / EXPORT
## ============================================================
def _download_binary_file(
    service,
    file_id: str,
    destination_path: Path,
) -> Path:
    """
        Download a binary file from Google Drive

        Args:
            service: Drive API service
            file_id (str): Drive file ID
            destination_path (Path): Local destination path

        Returns:
            Path: Saved local path
    """
    
    ensure_directories(destination_path.parent)

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.FileIO(destination_path, mode="wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            logger.debug(f"Download progress: {int(status.progress() * 100)}%")

    logger.info(f"Downloaded file to: {destination_path}")
    return destination_path

def _export_google_doc(
    service,
    file_id: str,
    export_mime_type: str,
    destination_path: Path,
) -> Path:
    """
        Export a Google Workspace document to a given format

        Args:
            service: Drive API service
            file_id (str): Drive file ID
            export_mime_type (str): Target MIME type for export
            destination_path (Path): Local destination path

        Returns:
            Path: Saved local path
    """
    
    ensure_directories(destination_path.parent)

    request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
    fh = io.FileIO(destination_path, mode="wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            logger.debug(f"Export progress: {int(status.progress() * 100)}%")

    logger.info(f"Exported file to: {destination_path}")
    return destination_path

def infer_drive_export_strategy(file_meta: DriveFileMeta) -> Tuple[bool, Optional[str], str]:
    """
        Infer whether a Drive file should be exported (Google Docs) or downloaded as-is

        Args:
            file_meta (DriveFileMeta): Drive file metadata

        Returns:
            Tuple[bool, Optional[str], str]:
                - is_export (bool): True if export is needed
                - export_mime (Optional[str]): Export MIME type if export
                - file_extension (str): Extension used for local file
    """
    
    mime = (file_meta.mime_type or "").lower()

    ## Google Workspace types
    if mime == "application/vnd.google-apps.document":
        return True, EXPORT_MIME_TEXT, ".txt"
    if mime == "application/vnd.google-apps.spreadsheet":
        ## Keep it simple: export as PDF by default
        return True, EXPORT_MIME_PDF, ".pdf"
    if mime == "application/vnd.google-apps.presentation":
        return True, EXPORT_MIME_PDF, ".pdf"

    ## Otherwise: download as-is
    ## We do not attempt to guess extension from mime here (kept minimal).
    return False, None, ""

def download_or_export_drive_file(
    file_meta: DriveFileMeta,
    output_dir: Optional[Path] = None,
) -> DriveDownloadResult:
    """
        Download or export a Drive file to local storage

        Args:
            file_meta (DriveFileMeta): Drive file metadata
            output_dir (Optional[Path]): Local output directory. Defaults to settings.raw_dir

        Returns:
            DriveDownloadResult: Download/export outcome
    """
    
    settings = get_settings()
    service = get_drive_service()

    ## Resolve output directory
    local_output_dir = output_dir if output_dir else settings.raw_dir
    ensure_directories(local_output_dir)

    is_export, export_mime, ext = infer_drive_export_strategy(file_meta)

    ## Build a safe filename
    safe_name = file_meta.name.replace("/", "_").replace("\\", "_").strip()
    if not safe_name:
        safe_name = file_meta.file_id

    ## If ext is empty, keep original name as-is
    if ext and not safe_name.lower().endswith(ext):
        safe_name = f"{safe_name}{ext}"

    destination_path = local_output_dir / safe_name

    logger.info(f"Processing Drive file: {file_meta.name} | mime={file_meta.mime_type}")

    if is_export and export_mime:
        _export_google_doc(
            service=service,
            file_id=file_meta.file_id,
            export_mime_type=export_mime,
            destination_path=destination_path,
        )
        return DriveDownloadResult(
            file_meta=file_meta,
            local_path=destination_path,
            exported_mime_type=export_mime,
        )

    _download_binary_file(
        service=service,
        file_id=file_meta.file_id,
        destination_path=destination_path,
    )
    return DriveDownloadResult(
        file_meta=file_meta,
        local_path=destination_path,
        exported_mime_type=None,
    )


def download_drive_folder(
    folder_id: str,
    output_dir: Optional[Path] = None,
    max_results: int = 200,
    query_extra: Optional[str] = None,
) -> List[DriveDownloadResult]:
    """
        Download/export all files from a Drive folder

        Args:
            folder_id (str): Google Drive folder ID
            output_dir (Optional[Path]): Local output directory. Defaults to settings.raw_dir
            max_results (int): Maximum results to fetch
            query_extra (Optional[str]): Optional Drive query filter

        Returns:
            List[DriveDownloadResult]: List of download/export results
    """
    
    files = list_drive_files(
        folder_id=folder_id,
        max_results=max_results,
        query_extra=query_extra,
    )

    results: List[DriveDownloadResult] = []
    for meta in files:
        try:
            res = download_or_export_drive_file(meta, output_dir=output_dir)
            results.append(res)
        except Exception as exc:
            logger.exception(f"Failed to download/export file: {meta.name} | {exc}")

    return results
