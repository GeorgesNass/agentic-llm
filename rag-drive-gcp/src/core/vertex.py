'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Vertex AI utilities for rag-drive-gcp: service account auth, embeddings, and LLM calls."
'''

import base64
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from google.oauth2 import service_account

from src.model.settings import get_settings
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("vertex")

## ============================================================
## DATA CLASSES
## ============================================================
@dataclass
class VertexTextGenerationResult:
    """
        Vertex text generation result container

        Attributes:
            prompt (str): Input prompt
            response_text (str): Output response
            model_name (str): Vertex model used
    """

    prompt: str
    response_text: str
    model_name: str

## ============================================================
## AUTHENTICATION
## ============================================================
def load_service_account_credentials() -> service_account.Credentials:
    """
        Load service account credentials for GCP clients

        Supported options:
            - GCP_SA_JSON_PATH: path to the service account JSON file
            - GCP_SA_JSON_BASE64: base64-encoded service account JSON content

        Returns:
            service_account.Credentials: Service account credentials

        Raises:
            ValueError: If neither option is provided
            FileNotFoundError: If JSON path does not exist
    """
    
    sa_path = os.getenv("GCP_SA_JSON_PATH")
    sa_b64 = os.getenv("GCP_SA_JSON_BASE64")

    if sa_path:
        if not os.path.exists(sa_path):
            raise FileNotFoundError(f"Service account JSON not found: {sa_path}")

        logger.info("Loading service account credentials from JSON file path.")
        return service_account.Credentials.from_service_account_file(sa_path)

    if sa_b64:
        logger.info("Loading service account credentials from base64 content.")
        raw = base64.b64decode(sa_b64.encode("utf-8"))
        info = json.loads(raw.decode("utf-8"))
        return service_account.Credentials.from_service_account_info(info)

    raise ValueError(
        "Missing service account configuration. "
        "Please set GCP_SA_JSON_PATH or GCP_SA_JSON_BASE64 in your .env."
    )

## ============================================================
## VERTEX INITIALIZATION
## ============================================================
def init_vertex_ai() -> None:
    """
        Initialize Vertex AI SDK with explicit project and region

        Notes:
            - Uses service account credentials explicitly
            - This must be called before using embeddings or LLM calls
    """
    
    settings = get_settings()
    credentials = load_service_account_credentials()

    if not settings.gcp_project_id or not settings.gcp_region:
        raise ValueError("GCP_PROJECT_ID and GCP_REGION must be set in .env.")

    ## Lazy import to avoid forcing Vertex deps for modules that do not need it
    import vertexai  # noqa: WPS433

    vertexai.init(
        project=settings.gcp_project_id,
        location=settings.gcp_region,
        credentials=credentials,
    )

    logger.info("Vertex AI initialized successfully.")

## ============================================================
## EMBEDDINGS
## ============================================================
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
        Compute embeddings for a list of texts using Vertex AI

        Required env vars:
            - VERTEX_EMBED_MODEL

        Args:
            texts (List[str]): List of strings to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            ValueError: If model name is missing or texts is empty
    """
    
    settings = get_settings()

    if not texts:
        raise ValueError("No texts provided for embeddings.")

    if not settings.vertex_embed_model:
        raise ValueError("VERTEX_EMBED_MODEL must be set in .env.")

    init_vertex_ai()

    ## Lazy import to keep module import light
    from vertexai.language_models import TextEmbeddingModel  # noqa: WPS433

    model = TextEmbeddingModel.from_pretrained(settings.vertex_embed_model)

    ## Vertex returns objects with .values
    embeddings = model.get_embeddings(texts)
    vectors = [e.values for e in embeddings]

    logger.info(f"Generated embeddings for {len(texts)} text(s).")
    return vectors

## ============================================================
## LLM TEXT GENERATION
## ============================================================
def generate_text(
    prompt: str,
    max_output_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> VertexTextGenerationResult:
    """
        Generate text using a Vertex AI text generation model

        Required env vars:
            - VERTEX_LLM_MODEL

        Args:
            prompt (str): Prompt to send to the model
            max_output_tokens (int): Max tokens in the output
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter

        Returns:
            VertexTextGenerationResult: Container with response text

        Raises:
            ValueError: If model name is missing or prompt is empty
    """
    
    settings = get_settings()

    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty.")

    if not settings.vertex_llm_model:
        raise ValueError("VERTEX_LLM_MODEL must be set in .env.")

    init_vertex_ai()

    ## Lazy import to keep module import light
    from vertexai.language_models import TextGenerationModel  # noqa: WPS433

    model = TextGenerationModel.from_pretrained(settings.vertex_llm_model)

    response = model.predict(
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    text = getattr(response, "text", str(response))

    logger.info("Vertex LLM generation completed.")
    return VertexTextGenerationResult(
        prompt=prompt,
        response_text=text,
        model_name=settings.vertex_llm_model,
    )

def build_rag_prompt(
    question: str,
    context_chunks: List[str],
) -> str:
    """
        Build a minimal RAG prompt for the LLM

        Args:
            question (str): User question
            context_chunks (List[str]): Retrieved context chunks

        Returns:
            str: Prompt string.
    """
    
    ## Keep prompt simple and deterministic for now
    context = "\n\n---\n\n".join(context_chunks)

    prompt = (
        "You are a helpful assistant. Use the provided context to answer.\n\n"
        "## Context\n"
        f"{context}\n\n"
        "## Question\n"
        f"{question}\n\n"
        "## Answer\n"
        "Answer using only the context when possible. If not enough information, say so."
    )
    return prompt