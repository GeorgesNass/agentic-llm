'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "LLM-related helpers: chunk deduplication, local embedding loading and embedding response parsing."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.errors import DependencyError
from src.utils.env_utils import _resolve_local_embedding_model

## ============================================================
## TYPE ALIASES
## ============================================================
EmbeddingProvider = str

## ============================================================
## CHUNK HELPERS
## ============================================================
def _dedupe_chunks(chunks: List[Any]) -> List[Any]:
    """
        Remove duplicate chunks by normalized content

        Args:
            chunks: Input chunks (objects with .text attribute)

        Returns:
            Deduped chunks
    """

    seen: set[str] = set()
    out: List[Any] = []

    for ch in chunks:
        key = str(getattr(ch, "text", "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ch)

    return out

## ============================================================
## LOCAL EMBEDDINGS (SENTENCE-TRANSFORMERS)
## ============================================================
def _load_sentence_transformer(model_name: Optional[str], use_gpu: bool) -> Any:
    """
        Load SentenceTransformer model

        Args:
            model_name: SentenceTransformers model id
            use_gpu: Whether GPU is allowed

        Returns:
            SentenceTransformer instance
    """

    ## Resolve model name if not provided
    resolved_model = _resolve_local_embedding_model(model_name)

    ## Import dependency
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise DependencyError(
            message="sentence-transformers dependency not available",
            error_code="dependency_error",
            details={"pip_package": "sentence-transformers"},
            origin="embeddings",
            cause=exc,
            http_status=500,
            is_retryable=False,
        ) from exc

    ## Resolve device
    device = "cuda" if use_gpu else "cpu"

    ## Load model with selected device
    try:
        model = SentenceTransformer(resolved_model, device=device)
        return model
    except Exception as exc:
        raise DependencyError(
            message="Failed to load local embedding model",
            error_code="dependency_error",
            details={"model_name": resolved_model, "device": device},
            origin="embeddings",
            cause=exc,
            http_status=500,
            is_retryable=True,
        ) from exc

## ============================================================
## EMBEDDING RESPONSE PARSERS
## ============================================================
def _extract_vectors_openai_embeddings(data: Dict[str, Any]) -> List[List[float]]:
    """
        Extract vectors from OpenAI-style embeddings response

        Args:
            data: JSON response dict

        Returns:
            vectors list
    """

    ## Response shape: { data: [ { embedding: [...] }, ... ] }
    items = data.get("data", [])
    vectors: List[List[float]] = []

    if not isinstance(items, list):
        return vectors

    for item in items:
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if isinstance(emb, list):
            vectors.append([float(x) for x in emb])

    return vectors

def _extract_vectors_gemini_embeddings(data: Dict[str, Any]) -> List[List[float]]:
    """
        Extract vectors from Gemini embeddings response

        Supported shapes:
            - { embedding: { values: [...] } }
            - { embeddings: [ { values: [...] }, ... ] }

        Args:
            data: JSON response dict

        Returns:
            vectors list
    """

    ## Single embedding shape
    if isinstance(data.get("embedding"), dict):
        values = data["embedding"].get("values")
        if isinstance(values, list):
            return [[float(x) for x in values]]

    ## Batch embeddings shape
    emb_list = data.get("embeddings")
    if isinstance(emb_list, list):
        out: List[List[float]] = []
        for item in emb_list:
            if not isinstance(item, dict):
                continue
            values = item.get("values")
            if isinstance(values, list):
                out.append([float(x) for x in values])
        return out

    return []

## ============================================================
## GEMINI MAPPING HELPERS
## ============================================================
def _split_system_and_chat(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
        Split system messages from user/assistant messages

        Args:
            messages: OpenAI-style messages list

        Returns:
            system_text and filtered messages
    """

    ## Aggregate system messages into one string
    system_parts: List[str] = []
    filtered: List[Dict[str, Any]] = []

    for msg in messages:
        ## Validate basic shape
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")

        ## Normalize content to string
        content_text = str(content) if content is not None else ""

        ## Separate system role
        if role == "system":
            system_parts.append(content_text)
        else:
            filtered.append(msg)

    system_text = "\n".join([p for p in system_parts if p.strip()])
    
    return system_text, filtered

def _openai_messages_to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
        Convert OpenAI messages to Gemini generateContent payload

        Notes:
            - Gemini uses contents[] with role user/model
            - System instructions are provided via systemInstruction

        Args:
            messages: OpenAI-style messages list

        Returns:
            systemInstruction dict or None, and contents list
    """

    ## Split system message(s) first
    system_text, chat_messages = _split_system_and_chat(messages)

    ## Build system instruction if present
    system_instruction: Optional[Dict[str, Any]] = None
    if system_text.strip():
        system_instruction = {"parts": [{"text": system_text}]}

    ## Build Gemini contents
    contents: List[Dict[str, Any]] = []

    for msg in chat_messages:
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")

        ## Normalize content to string
        content_text = str(content) if content is not None else ""

        ## Gemini roles: user / model
        if role == "assistant":
            gemini_role = "model"
        else:
            gemini_role = "user"

        ## Append Gemini content item
        contents.append(
            {
                "role": gemini_role,
                "parts": [{"text": content_text}],
            }
        )

    return system_instruction, contents

def _extract_text_from_gemini_generate_content(data: Dict[str, Any]) -> str:
    """
        Extract text from Gemini generateContent response

        Args:
            data: Gemini response JSON

        Returns:
            Assistant text
    """

    ## Gemini shape typically includes candidates[0].content.parts[0].text
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ""

    first = candidates[0]
    if not isinstance(first, dict):
        return ""

    content = first.get("content", {})
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts", [])
    if not isinstance(parts, list) or not parts:
        return ""

    ## Concatenate all text parts
    texts: List[str] = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(str(p.get("text", "")))

    return "".join(texts)
