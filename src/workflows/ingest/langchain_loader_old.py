"""
LangChain-based document loaders (v3.0) with multi-engine support and OCR.

This module provides robust document extraction using LangChain's document loaders
with fallback strategies and quality validation.

Features:
- Multi-engine PDF extraction (Unstructured → PyMuPDF → PDFPlumber)
- Image OCR support (PNG, JPG, TIFF, etc.)
- Rich metadata extraction (page numbers, bounding boxes, etc.)
- Quality validation and metrics
- Support for PDF, DOCX, images, HTML, TXT, MD
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyMuPDFLoader,
    PDFPlumberLoader,
    UnstructuredImageLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    from langchain.schema import Document as LangChainDocument

from src.workflows.io.schema import Document

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
    """Raised when all extraction engines fail."""
    pass


class LoaderQualityMetrics:
    """Track quality metrics for document extraction."""

    def __init__(self):
        self.engine_used: str = "unknown"
        self.total_chars: int = 0
        self.total_pages: int = 0
        self.total_elements: int = 0
        self.ocr_used: bool = False
        self.extraction_success: bool = False
        self.error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "engine_used": self.engine_used,
            "total_chars": self.total_chars,
            "total_pages": self.total_pages,
            "total_elements": self.total_elements,
            "ocr_used": self.ocr_used,
            "extraction_success": self.extraction_success,
            "error_message": self.error_message,
        }


def load_pdf_multi_engine(
    pdf_path: Path,
    ocr_languages: Optional[List[str]] = None,
) -> Tuple[List[LangChainDocument], LoaderQualityMetrics]:
    """
    Load PDF using multiple engines with fallback strategy.

    Strategy:
    1. UnstructuredPDFLoader - Best for complex layouts + OCR
    2. PyMuPDFLoader - Fast, good for text-based PDFs
    3. PDFPlumberLoader - Good for tables

    Args:
        pdf_path: Path to PDF file
        ocr_languages: Languages for OCR (e.g., ["eng", "fra"])

    Returns:
        Tuple of (documents, quality_metrics)

    Raises:
        DocumentExtractionError: If all engines fail
    """
    ocr_langs = ocr_languages or ["eng", "fra"]
    metrics = LoaderQualityMetrics()

    # Strategy 1: Unstructured (best for complex PDFs + OCR)
    try:
        LOGGER.info(f"Trying UnstructuredPDFLoader on {pdf_path.name}")
        loader = UnstructuredPDFLoader(
            str(pdf_path),
            mode="elements",  # Extract elements with metadata
        )
        docs = loader.load()

        if docs and any(doc.page_content.strip() for doc in docs):
            metrics.engine_used = "unstructured"
            metrics.total_chars = sum(len(doc.page_content) for doc in docs)
            metrics.total_elements = len(docs)
            metrics.extraction_success = True
            metrics.ocr_used = True  # Unstructured uses OCR when needed
            LOGGER.info(f"✓ UnstructuredPDFLoader extracted {metrics.total_chars} chars from {len(docs)} elements")
            return docs, metrics
    except Exception as e:
        LOGGER.warning(f"UnstructuredPDFLoader failed: {e}")
        metrics.error_message = f"Unstructured: {str(e)}"

    # Strategy 2: PyMuPDF (fast, good for text-based PDFs)
    try:
        LOGGER.info(f"Trying PyMuPDFLoader on {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        if docs and any(doc.page_content.strip() for doc in docs):
            metrics.engine_used = "pymupdf"
            metrics.total_chars = sum(len(doc.page_content) for doc in docs)
            metrics.total_pages = len(docs)
            metrics.extraction_success = True
            LOGGER.info(f"✓ PyMuPDFLoader extracted {metrics.total_chars} chars from {len(docs)} pages")
            return docs, metrics
    except Exception as e:
        LOGGER.warning(f"PyMuPDFLoader failed: {e}")
        if metrics.error_message:
            metrics.error_message += f" | PyMuPDF: {str(e)}"
        else:
            metrics.error_message = f"PyMuPDF: {str(e)}"

    # Strategy 3: PDFPlumber (good for tables)
    try:
        LOGGER.info(f"Trying PDFPlumberLoader on {pdf_path.name}")
        loader = PDFPlumberLoader(str(pdf_path))
        docs = loader.load()

        if docs and any(doc.page_content.strip() for doc in docs):
            metrics.engine_used = "pdfplumber"
            metrics.total_chars = sum(len(doc.page_content) for doc in docs)
            metrics.total_pages = len(docs)
            metrics.extraction_success = True
            LOGGER.info(f"✓ PDFPlumberLoader extracted {metrics.total_chars} chars from {len(docs)} pages")
            return docs, metrics
    except Exception as e:
        LOGGER.warning(f"PDFPlumberLoader failed: {e}")
        if metrics.error_message:
            metrics.error_message += f" | PDFPlumber: {str(e)}"
        else:
            metrics.error_message = f"PDFPlumber: {str(e)}"

    # All engines failed
    error_msg = f"All PDF extraction engines failed for {pdf_path.name}"
    if metrics.error_message:
        error_msg += f": {metrics.error_message}"
    LOGGER.error(error_msg)
    raise DocumentExtractionError(error_msg)


def load_image_with_ocr(
    image_path: Path,
    ocr_languages: Optional[List[str]] = None,
) -> Tuple[List[LangChainDocument], LoaderQualityMetrics]:
    """
    Load image and extract text using OCR.

    Uses UnstructuredImageLoader with Tesseract OCR.

    Args:
        image_path: Path to image file
        ocr_languages: Languages for OCR (e.g., ["eng", "fra"])

    Returns:
        Tuple of (documents, quality_metrics)

    Raises:
        DocumentExtractionError: If OCR fails
    """
    ocr_langs = ocr_languages or ["eng", "fra"]
    metrics = LoaderQualityMetrics()

    try:
        LOGGER.info(f"Extracting text from image {image_path.name} with OCR ({'+'.join(ocr_langs)})")

        # UnstructuredImageLoader uses Tesseract OCR
        loader = UnstructuredImageLoader(
            str(image_path),
            mode="elements",
        )
        docs = loader.load()

        if docs:
            metrics.engine_used = "unstructured_image"
            metrics.total_chars = sum(len(doc.page_content) for doc in docs)
            metrics.total_elements = len(docs)
            metrics.ocr_used = True
            metrics.extraction_success = True
            LOGGER.info(f"✓ OCR extracted {metrics.total_chars} chars from {image_path.name}")
            return docs, metrics
        else:
            raise DocumentExtractionError(f"No text extracted from image {image_path.name}")

    except Exception as e:
        error_msg = f"Image OCR failed for {image_path.name}: {e}"
        LOGGER.error(error_msg)
        metrics.error_message = str(e)
        raise DocumentExtractionError(error_msg)


def load_docx(
    docx_path: Path,
) -> Tuple[List[LangChainDocument], LoaderQualityMetrics]:
    """Load DOCX document."""
    metrics = LoaderQualityMetrics()

    try:
        LOGGER.info(f"Loading DOCX {docx_path.name}")
        loader = UnstructuredWordDocumentLoader(str(docx_path), mode="elements")
        docs = loader.load()

        metrics.engine_used = "unstructured_docx"
        metrics.total_chars = sum(len(doc.page_content) for doc in docs)
        metrics.total_elements = len(docs)
        metrics.extraction_success = True
        LOGGER.info(f"✓ Extracted {metrics.total_chars} chars from DOCX")
        return docs, metrics

    except Exception as e:
        error_msg = f"DOCX extraction failed for {docx_path.name}: {e}"
        LOGGER.error(error_msg)
        metrics.error_message = str(e)
        raise DocumentExtractionError(error_msg)


def load_html(
    html_path: Path,
) -> Tuple[List[LangChainDocument], LoaderQualityMetrics]:
    """Load HTML document."""
    metrics = LoaderQualityMetrics()

    try:
        LOGGER.info(f"Loading HTML {html_path.name}")
        loader = UnstructuredHTMLLoader(str(html_path), mode="elements")
        docs = loader.load()

        metrics.engine_used = "unstructured_html"
        metrics.total_chars = sum(len(doc.page_content) for doc in docs)
        metrics.total_elements = len(docs)
        metrics.extraction_success = True
        LOGGER.info(f"✓ Extracted {metrics.total_chars} chars from HTML")
        return docs, metrics

    except Exception as e:
        error_msg = f"HTML extraction failed for {html_path.name}: {e}"
        LOGGER.error(error_msg)
        metrics.error_message = str(e)
        raise DocumentExtractionError(error_msg)


def load_text(
    text_path: Path,
) -> Tuple[List[LangChainDocument], LoaderQualityMetrics]:
    """Load plain text document."""
    metrics = LoaderQualityMetrics()

    try:
        LOGGER.info(f"Loading text file {text_path.name}")
        loader = TextLoader(str(text_path), encoding="utf-8")
        docs = loader.load()

        metrics.engine_used = "text_loader"
        metrics.total_chars = sum(len(doc.page_content) for doc in docs)
        metrics.extraction_success = True
        LOGGER.info(f"✓ Loaded {metrics.total_chars} chars from text file")
        return docs, metrics

    except Exception as e:
        error_msg = f"Text loading failed for {text_path.name}: {e}"
        LOGGER.error(error_msg)
        metrics.error_message = str(e)
        raise DocumentExtractionError(error_msg)


def load_document_langchain(
    file_path: Union[str, Path],
    ocr_languages: Optional[List[str]] = None,
    additional_metadata: Optional[Dict] = None,
) -> Document:
    """
    Load any supported document type using LangChain loaders.

    Supports:
    - PDF (multi-engine fallback: Unstructured → PyMuPDF → PDFPlumber)
    - Images with OCR (PNG, JPG, TIFF, etc.)
    - DOCX
    - HTML
    - TXT, MD

    Args:
        file_path: Path to document
        ocr_languages: Languages for OCR (default: ["eng", "fra"])
        additional_metadata: Extra metadata to include

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

    # Route to appropriate loader
    if extension in PDF_EXTENSIONS:
        langchain_docs, metrics = load_pdf_multi_engine(path, ocr_languages)
    elif extension in IMAGE_EXTENSIONS:
        langchain_docs, metrics = load_image_with_ocr(path, ocr_languages)
    elif extension in DOCX_EXTENSIONS:
        langchain_docs, metrics = load_docx(path)
    elif extension in HTML_EXTENSIONS:
        langchain_docs, metrics = load_html(path)
    elif extension in TEXT_EXTENSIONS:
        langchain_docs, metrics = load_text(path)
    else:
        raise ValueError(f"Unsupported extension: {extension}")

    # Combine all pages/elements into single text
    full_text = "\n\n".join(doc.page_content for doc in langchain_docs)

    # Build rich metadata
    metadata = {
        "source_name": path.name,
        "file_extension": extension,
        "extraction_engine": metrics.engine_used,
        "total_chars": metrics.total_chars,
        "ocr_used": metrics.ocr_used,
        "extraction_success": metrics.extraction_success,
    }

    if metrics.total_pages > 0:
        metadata["total_pages"] = metrics.total_pages
    if metrics.total_elements > 0:
        metadata["total_elements"] = metrics.total_elements
    if ocr_languages:
        metadata["ocr_languages"] = "+".join(ocr_languages)

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
        content_type=f"application/{extension[1:]}",  # e.g., "application/pdf"
    )

    LOGGER.info(
        f"✓ Successfully loaded {path.name} using {metrics.engine_used} "
        f"({metrics.total_chars} chars, OCR={metrics.ocr_used})"
    )

    return atlas_doc
