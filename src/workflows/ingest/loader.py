"""
Document ingestion utilities using LangChain loaders (v3.0) with fallback to legacy unstructured.

v3.0: Multi-engine PDF extraction, image OCR, rich metadata
v2.1: Legacy unstructured library (fallback only)
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import fsspec

from unstructured.partition.auto import partition

from src.workflows.io.schema import Document
from src.workflows.ingest.langchain_loader import (
    load_document_langchain,
    DocumentExtractionError,
    ALL_SUPPORTED_EXTENSIONS,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    # Web
    ".html",
    ".htm",
    # Text
    ".txt",
    ".md",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
}


def detect_content_type(path_str: str) -> str:
    mime_type, _ = mimetypes.guess_type(path_str)
    return mime_type or "application/octet-stream"


def _partition_file(
    path_str: str,
    file_obj,
    *,
    ocr_languages: Iterable[str],
) -> List:
    """
    Wrapper around `unstructured.partition.auto.partition` with guarded OCR arguments.
    """
    kwargs: Dict[str, object] = {
        "filename": path_str,  # Use filename instead of file object
        "metadata_filename": Path(path_str).name,
    }
    if ocr_languages:
        kwargs["languages"] = "+".join(ocr_languages)  # Use 'languages' instead of 'ocr_languages'

    try:
        return partition(**kwargs)
    except TypeError:
        LOGGER.debug("partition() rejected language parameters, retrying without them")
        kwargs.pop("languages", None)
        return partition(**kwargs)


def ingest_file(
    path_str: str,
    *,
    ocr_languages: Optional[Iterable[str]] = None,
    include_metadata: Optional[Dict[str, str]] = None,
    use_langchain: bool = True,
) -> Document:
    """
    Ingest a single file from a local or remote path and convert it to a `Document` instance.

    v3.0: Uses LangChain loaders by default with multi-engine fallback and rich metadata.
    v2.1: Falls back to legacy unstructured library if LangChain fails.

    Args:
        path_str: Path to file (local or remote)
        ocr_languages: Languages for OCR (default: ["eng", "fra"])
        include_metadata: Additional metadata to include
        use_langchain: Use LangChain loaders (default: True, recommended)

    Returns:
        Document instance with extracted text and metadata

    Raises:
        FileNotFoundError: If file not found
        Exception: If extraction fails with all engines
    """
    path = Path(path_str)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        LOGGER.warning("Unsupported file extension for %s", path_str)

    ocr_langs = list(ocr_languages) if ocr_languages else ["eng", "fra"]

    # Strategy 1: LangChain loaders (v3.0) - RECOMMENDED
    if use_langchain:
        try:
            LOGGER.info(f"[v3.0] Using LangChain loader for {path.name}")
            document = load_document_langchain(
                path,
                ocr_languages=ocr_langs,
                additional_metadata=include_metadata,
            )
            LOGGER.info(
                f"✓ [v3.0] LangChain extraction successful: {len(document.text)} chars, "
                f"engine={document.metadata.get('extraction_engine', 'unknown')}"
            )
            return document

        except DocumentExtractionError as e:
            LOGGER.warning(f"[v3.0] LangChain extraction failed: {e}")
            LOGGER.info("[v2.1] Falling back to legacy unstructured loader")
        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.warning(f"[v3.0] Unexpected error with LangChain: {e}", exc_info=True)
            else:
                LOGGER.warning(f"[v3.0] Erreur avec LangChain (fallback en cours)")
            LOGGER.info("[v2.1] Falling back to legacy unstructured loader")

    # Strategy 2: Legacy unstructured loader (v2.1) - FALLBACK
    try:
        LOGGER.info(f"[v2.1] Using legacy unstructured loader for {path.name}")
        # Use filename directly instead of file object
        elements = _partition_file(path_str, None, ocr_languages=ocr_langs)
        text_content = "\n".join(element.text for element in elements if getattr(element, "text", None))
        metadata: Dict[str, str] = {
            "source_name": path.name,
            "content_type": detect_content_type(path_str),
            "extraction_engine": "unstructured_legacy",
        }
        if include_metadata:
            metadata.update(include_metadata)

        document = Document(
            source_path=path,
            text=text_content,
            metadata=metadata,
            content_type=metadata.get("content_type"),
        )
        LOGGER.info(f"✓ [v2.1] Legacy extraction successful: {len(text_content)} chars")
        return document

    except FileNotFoundError:
        LOGGER.error("File not found at path: %s", path_str)
        raise
    except Exception as e:
        # Dev mode: show full traceback, Prod mode: simple message
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.error("Failed to ingest file %s with all methods: %s", path_str, e, exc_info=True)
        else:
            LOGGER.error("Échec de l'extraction pour: %s", path.name)
        raise
