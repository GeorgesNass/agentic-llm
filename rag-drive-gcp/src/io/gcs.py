'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Google Cloud Storage I/O for rag-drive-gcp: authenticate with service account and upload/download artifacts."
'''

import base64
import os
from pathlib import Path
from typing import Optional, Tuple

from google.cloud import storage
from google.oauth2 import service_account

from src.model.settings import get_settings
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_directories

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("gcs")


## ============================================================
## AUTHENTICATION
## ============================================================
def _load_service_account_credentials() -> service_account.Credentials:
    """
        Load service account credentials for GCS

        Supported options:
            - GCP_SA_JSON_PATH: path to the service account JSON file
            - GCP_SA_JSON_BASE64: base64-encoded service account JSON content

        Returns:
            service_account.Credentials: Credentials for Google Cloud client usage

        Raises:
            ValueError: If neither credential option is provided
            FileNotFoundError: If JSON path does not exist
    """
    
    sa_path = os.getenv("GCP_SA_JSON_PATH")
    sa_b64 = os.getenv("GCP_SA_JSON_BASE64")

    if sa_path:
        sa_file = Path(sa_path)
        if not sa_file.exists():
            raise FileNotFoundError(f"Service account JSON not found: {sa_file}")

        logger.info("Loading service account credentials from JSON file path.")
        return service_account.Credentials.from_service_account_file(str(sa_file))

    if sa_b64:
        logger.info("Loading service account credentials from base64 content.")
        raw = base64.b64decode(sa_b64.encode("utf-8"))
        info = raw.decode("utf-8")
        return service_account.Credentials.from_service_account_info(eval(info))  # noqa: S307

    raise ValueError(
        "Missing service account configuration for GCS. "
        "Please set GCP_SA_JSON_PATH or GCP_SA_JSON_BASE64 in your .env."
    )

def get_gcs_client() -> storage.Client:
    """
        Create a Google Cloud Storage client using service account credentials

        Returns:
            storage.Client: Configured GCS client
    """
    
    settings = get_settings()
    credentials = _load_service_account_credentials()

    ## Create client with explicit project if provided
    if settings.gcp_project_id:
        return storage.Client(project=settings.gcp_project_id, credentials=credentials)

    return storage.Client(credentials=credentials)

## ============================================================
## PATH HELPERS
## ============================================================
def build_gcs_object_path(prefix: str, filename: str) -> str:
    """
        Build a clean GCS object path using prefix + filename

        Args:
            prefix (str): GCS prefix (e.g., 'texts/')
            filename (str): Object filename

        Returns:
            str: Full object path
    """
    
    clean_prefix = (prefix or "").strip()
    if clean_prefix and not clean_prefix.endswith("/"):
        clean_prefix = f"{clean_prefix}/"

    clean_name = filename.strip().lstrip("/")
    return f"{clean_prefix}{clean_name}"

## ============================================================
## UPLOAD / DOWNLOAD
## ============================================================
def upload_file_to_gcs(
    local_path: Path,
    bucket_name: str,
    object_path: str,
    content_type: Optional[str] = None,
) -> str:
    """
        Upload a local file to Google Cloud Storage

        Args:
            local_path (Path): Local file path
            bucket_name (str): Target GCS bucket name
            object_path (str): Target object path (prefix + filename)
            content_type (Optional[str]): Optional MIME type

        Returns:
            str: gs:// URI for the uploaded object

        Raises:
            FileNotFoundError: If local file does not exist
    """
    
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    logger.info(f"Uploading to GCS: {local_path} -> gs://{bucket_name}/{object_path}")

    if content_type:
        blob.content_type = content_type

    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{object_path}"

def download_file_from_gcs(
    bucket_name: str,
    object_path: str,
    local_path: Path,
) -> Path:
    """
        Download a GCS object to a local file path

        Args:
            bucket_name (str): Source GCS bucket name
            object_path (str): Source object path
            local_path (Path): Local destination file path

        Returns:
            Path: Downloaded local file path
    """
    
    ensure_directories(local_path.parent)

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    logger.info(f"Downloading from GCS: gs://{bucket_name}/{object_path} -> {local_path}")
    blob.download_to_filename(str(local_path))
    return local_path

## ============================================================
## HIGH-LEVEL HELPERS (TEXT / EMBEDDINGS)
## ============================================================
def upload_text_artifact(local_path: Path) -> str:
    """
        Upload a text artifact to the configured GCS text location

        Args:
            local_path (Path): Local text file path

        Returns:
            str: gs:// URI of the uploaded text artifact
    """
    
    settings = get_settings()
    if not settings.gcs_bucket_text:
        raise ValueError("GCS_BUCKET_TEXT is not set in .env.")

    object_path = build_gcs_object_path(settings.gcs_prefix_text, local_path.name)
    return upload_file_to_gcs(
        local_path=local_path,
        bucket_name=settings.gcs_bucket_text,
        object_path=object_path,
        content_type="text/plain",
    )

def upload_embeddings_artifact(local_path: Path) -> str:
    """
        Upload an embeddings artifact to the configured GCS embeddings location

        Args:
            local_path (Path): Local embeddings file path

        Returns:
            str: gs:// URI of the uploaded embeddings artifact
    """
    
    settings = get_settings()
    if not settings.gcs_bucket_embeddings:
        raise ValueError("GCS_BUCKET_EMB is not set in .env.")

    object_path = build_gcs_object_path(settings.gcs_prefix_embeddings, local_path.name)
    return upload_file_to_gcs(
        local_path=local_path,
        bucket_name=settings.gcs_bucket_embeddings,
        object_path=object_path,
        content_type="application/octet-stream",
    )

def resolve_text_and_embeddings_targets(filename: str) -> Tuple[str, str]:
    """
        Resolve the target GCS URIs for text and embeddings artifacts given a base filename

        Args:
            filename (str): Base filename (e.g., 'doc1.txt' or 'doc1.parquet')

        Returns:
            Tuple[str, str]: (text_object_path, embeddings_object_path)
    """
    
    settings = get_settings()

    text_object = build_gcs_object_path(settings.gcs_prefix_text, filename)
    emb_object = build_gcs_object_path(settings.gcs_prefix_embeddings, filename)

    return text_object, emb_object
