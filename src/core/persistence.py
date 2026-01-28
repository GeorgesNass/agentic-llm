'''
__author__ = "Georges Nassopoulos"
__version__ = "1.0.0"
__status__ = "Dev"
__desc__ = "Minimal persistence layer for loading a RAG index from GCS."
'''

from pathlib import Path

from src.core.rag import RAGIndex, load_rag_index
from src.io.gcs import download_file_from_gcs
from src.model.settings import get_settings
from src.utils.logging_utils import get_logger
from src.utils.utils import ensure_directories

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("persistence")

## ============================================================
## PUBLIC API
## ============================================================
def load_rag_index_from_gcs(local_dir: Path) -> RAGIndex:
    """
        Load RAG index artifacts from Google Cloud Storage and rebuild the index

        This function downloads:
            - embeddings.npy
            - chunks.json

        Args:
            local_dir (Path): Local directory where artifacts will be downloaded

        Returns:
            RAGIndex: Reconstructed RAG index

        Raises:
            ValueError: If GCS bucket configuration is missing
    """
    
    settings = get_settings()
    ensure_directories(local_dir)

    if not settings.gcs_bucket_embeddings:
        raise ValueError("GCS_BUCKET_EMB is not set in environment variables.")

    ## Download embeddings matrix
    download_file_from_gcs(
        bucket_name=settings.gcs_bucket_embeddings,
        object_path=f"{settings.gcs_prefix_embeddings}embeddings.npy",
        local_path=local_dir / "embeddings.npy",
    )

    ## Download chunk metadata
    download_file_from_gcs(
        bucket_name=settings.gcs_bucket_embeddings,
        object_path=f"{settings.gcs_prefix_embeddings}chunks.json",
        local_path=local_dir / "chunks.json",
    )

    logger.info("RAG index artifacts downloaded from GCS.")
    return load_rag_index(local_dir)