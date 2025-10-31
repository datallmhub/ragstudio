"""
LangChain-based document loaders (v3.1) with intelligent OCR orchestration.

This module provides intelligent document extraction that automatically
selects the optimal extraction strategy based on document type.

Features:
- Intelligent document type detection
- Auto-selection of extraction engine (PyMuPDF, EasyOCR, etc.)
- Text-based PDFs: Fast extraction without OCR
- Scanned PDFs: High-quality EasyOCR extraction
- Images: EasyOCR with excellent multi-language support
- DOCX, HTML, TXT: Native loaders
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    from langchain.schema import Document as LangChainDocument

from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)

from src.workflows.io.schema import Document
from src.workflows.ingest.intelligent_orchestrator import IntelligentDocumentOrchestrator

LOGGER = logging.getLogger(__name__)

# File type categories
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
DOCX_EXTENSIONS = {".docx", ".doc"}
HTML_EXTENSIONS = {".html", ".htm"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

ALL_SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | IMAGE_EXTENSIONS | DOCX_EXTENSIONS | HTML_EXTENSIONS | TEXT_EXTENSIONS
)


class DocumentExtractionError(Exception):
    """Raised when document extraction fails."""
    pass


def load_docx(docx_path: Path) -> tuple[List[LangChainDocument], Dict]:
    """Load DOCX document."""
    try:
        LOGGER.info(f"Loading DOCX {docx_path.name}")
        loader = UnstructuredWordDocumentLoader(str(docx_path), mode="elements")
        docs = loader.load()

        metadata = {
            "engine": "unstructured_docx",
            "total_chars": sum(len(doc.page_content) for doc in docs),
            "total_elements": len(docs),
            "extraction_success": True,
        }

        LOGGER.info(f"✓ Extracted {metadata['total_chars']} chars from DOCX")
        return docs, metadata

    except Exception as e:
        LOGGER.error(f"DOCX extraction failed for {docx_path.name}: {e}")
        raise DocumentExtractionError(str(e))


def load_html(html_path: Path) -> tuple[List[LangChainDocument], Dict]:
    """Load HTML document."""
    try:
        LOGGER.info(f"Loading HTML {html_path.name}")
        loader = UnstructuredHTMLLoader(str(html_path), mode="elements")
        docs = loader.load()

        metadata = {
            "engine": "unstructured_html",
            "total_chars": sum(len(doc.page_content) for doc in docs),
            "total_elements": len(docs),
            "extraction_success": True,
        }

        LOGGER.info(f"✓ Extracted {metadata['total_chars']} chars from HTML")
        return docs, metadata

    except Exception as e:
        LOGGER.error(f"HTML extraction failed for {html_path.name}: {e}")
        raise DocumentExtractionError(str(e))


def load_text(text_path: Path) -> tuple[List[LangChainDocument], Dict]:
    """Load plain text document."""
    try:
        LOGGER.info(f"Loading text file {text_path.name}")
        loader = TextLoader(str(text_path), encoding="utf-8")
        docs = loader.load()

        metadata = {
            "engine": "text_loader",
            "total_chars": sum(len(doc.page_content) for doc in docs),
            "extraction_success": True,
        }

        LOGGER.info(f"✓ Loaded {metadata['total_chars']} chars from text file")
        return docs, metadata

    except Exception as e:
        LOGGER.error(f"Text loading failed for {text_path.name}: {e}")
        raise DocumentExtractionError(str(e))


def load_document_langchain(
    file_path: Union[str, Path],
    ocr_languages: Optional[List[str]] = None,
    additional_metadata: Optional[Dict] = None,
    use_gpu: bool = False,
    force_ocr: bool = False,
) -> Document:
    """
    Intelligently load any supported document type.

    This function automatically:
    1. Detects document type (text-based, scanned, hybrid, image)
    2. Selects optimal extraction strategy
    3. Extracts content with appropriate engine
    4. Returns rich metadata

    Supported formats:
    - PDF: Intelligent routing (PyMuPDF for text, EasyOCR for scanned)
    - Images: EasyOCR with excellent multi-language support
    - DOCX: Unstructured loader
    - HTML: Unstructured loader
    - TXT, MD: Text loader

    Args:
        file_path: Path to document
        ocr_languages: Languages for OCR (default: ['en', 'fr'])
        additional_metadata: Extra metadata to include
        use_gpu: Use GPU for OCR if available (default: False)
        force_ocr: Force OCR even for text-based PDFs (default: False)

    Returns:
        Document instance with extracted text and rich metadata

    Raises:
        ValueError: If file type not supported
        DocumentExtractionError: If extraction fails
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in ALL_SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
        )

    ocr_langs = ocr_languages or ['en', 'fr']

    try:
        # Route to appropriate loader
        if extension in PDF_EXTENSIONS or extension in IMAGE_EXTENSIONS:
            # Use intelligent orchestrator for PDFs and images
            orchestrator = IntelligentDocumentOrchestrator(
                ocr_languages=ocr_langs,
                use_gpu=use_gpu,
                enable_fallback=True,
            )

            langchain_docs, extraction_meta = orchestrator.load_document(
                path,
                force_ocr=force_ocr,
            )

        elif extension in DOCX_EXTENSIONS:
            langchain_docs, extraction_meta = load_docx(path)

        elif extension in HTML_EXTENSIONS:
            langchain_docs, extraction_meta = load_html(path)

        elif extension in TEXT_EXTENSIONS:
            langchain_docs, extraction_meta = load_text(path)

        else:
            raise ValueError(f"Unsupported extension: {extension}")

        # Combine all pages/elements into single text
        full_text = "\n\n".join(doc.page_content for doc in langchain_docs)

        # Build rich metadata
        metadata = {
            "source_name": path.name,
            "file_extension": extension,
            "total_chars": len(full_text),
            **extraction_meta,
        }

        if ocr_languages:
            metadata["ocr_languages"] = "+".join(ocr_langs)

        # Add page-level metadata if available
        page_metadata = []
        for doc in langchain_docs:
            if doc.metadata:
                page_metadata.append(doc.metadata)
        if page_metadata:
            metadata["page_metadata"] = page_metadata

        if additional_metadata:
            metadata.update(additional_metadata)

        # Create Atlas Document
        atlas_doc = Document(
            source_path=path,
            text=full_text,
            metadata=metadata,
            content_type=f"application/{extension[1:]}",
        )

        LOGGER.info(
            f"✅ Successfully loaded {path.name} using {metadata.get('extraction_strategy', 'unknown')} "
            f"({len(full_text)} chars, OCR={metadata.get('ocr_used', False)})"
        )

        return atlas_doc

    except Exception as e:
        # Dev mode: show full traceback, Prod mode: simple message
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.error(f"Failed to load document {path.name}: {e}", exc_info=True)
            raise DocumentExtractionError(f"Extraction failed: {e}")
        else:
            LOGGER.warning(f"Échec du chargement: {path.name}")
            raise DocumentExtractionError(f"Document non supporté ou inaccessible")
