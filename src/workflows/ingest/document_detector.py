"""
Document type detection module for intelligent OCR orchestration.

Detects whether a document is text-based, scanned, or hybrid to choose
the optimal extraction strategy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from enum import Enum

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

LOGGER = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification."""
    TEXT_BASED = "text_based"      # >80% extractible text, no OCR needed
    SCANNED = "scanned"              # <20% extractible text, OCR required
    HYBRID = "hybrid"                # 20-80% extractible, selective OCR
    IMAGE = "image"                  # Image file, OCR required
    UNKNOWN = "unknown"              # Cannot determine


class DocumentAnalysis:
    """Result of document type detection."""

    def __init__(
        self,
        doc_type: DocumentType,
        extractible_ratio: float,
        total_pages: int,
        has_text: bool,
        has_images: bool,
        confidence: float = 1.0,
        recommendation: Optional[str] = None,
    ):
        self.doc_type = doc_type
        self.extractible_ratio = extractible_ratio
        self.total_pages = total_pages
        self.has_text = has_text
        self.has_images = has_images
        self.confidence = confidence
        self.recommendation = recommendation or self._get_default_recommendation()

    def _get_default_recommendation(self) -> str:
        """Get default extraction strategy recommendation."""
        if self.doc_type == DocumentType.TEXT_BASED:
            return "Use PyMuPDF (fast, no OCR)"
        elif self.doc_type == DocumentType.SCANNED:
            return "Use EasyOCR (high quality OCR)"
        elif self.doc_type == DocumentType.HYBRID:
            return "Use PyMuPDF + EasyOCR on missing zones"
        elif self.doc_type == DocumentType.IMAGE:
            return "Use EasyOCR direct"
        else:
            return "Use fallback strategy"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "doc_type": self.doc_type.value,
            "extractible_ratio": round(self.extractible_ratio, 2),
            "total_pages": self.total_pages,
            "has_text": self.has_text,
            "has_images": self.has_images,
            "confidence": round(self.confidence, 2),
            "recommendation": self.recommendation,
        }

    def __repr__(self) -> str:
        return (
            f"DocumentAnalysis(type={self.doc_type.value}, "
            f"extractible={self.extractible_ratio:.1%}, "
            f"pages={self.total_pages}, "
            f"recommendation='{self.recommendation}')"
        )


class DocumentTypeDetector:
    """
    Intelligent document type detector.

    Analyzes PDFs to determine if they are text-based, scanned, or hybrid.
    """

    def __init__(
        self,
        text_threshold: float = 0.8,
        scanned_threshold: float = 0.2,
        min_chars_per_page: int = 100,
    ):
        """
        Initialize detector.

        Args:
            text_threshold: Ratio above which document is considered text-based (default: 0.8)
            scanned_threshold: Ratio below which document is considered scanned (default: 0.2)
            min_chars_per_page: Minimum chars per page to consider it has text (default: 100)
        """
        self.text_threshold = text_threshold
        self.scanned_threshold = scanned_threshold
        self.min_chars_per_page = min_chars_per_page

        if fitz is None:
            LOGGER.warning("PyMuPDF not installed, document detection will be limited")

    def detect_pdf_type(self, pdf_path: Path) -> DocumentAnalysis:
        """
        Detect PDF document type.

        Analyzes text extractibility to determine optimal extraction strategy.

        Args:
            pdf_path: Path to PDF file

        Returns:
            DocumentAnalysis with type detection results
        """
        if fitz is None:
            LOGGER.warning("PyMuPDF not available, cannot detect PDF type accurately")
            return DocumentAnalysis(
                doc_type=DocumentType.UNKNOWN,
                extractible_ratio=0.0,
                total_pages=0,
                has_text=False,
                has_images=False,
                confidence=0.0,
                recommendation="Use fallback multi-engine strategy",
            )

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            # Analyze first few pages (sample)
            sample_size = min(5, total_pages)
            pages_to_analyze = list(range(sample_size))

            pages_with_text = 0
            pages_with_images = 0
            total_chars = 0

            for page_num in pages_to_analyze:
                page = doc[page_num]

                # Extract text
                text = page.get_text()
                char_count = len(text.strip())
                total_chars += char_count

                if char_count >= self.min_chars_per_page:
                    pages_with_text += 1

                # Check for images
                image_list = page.get_images()
                if image_list:
                    pages_with_images += 1

            doc.close()

            # Calculate extractible ratio
            if sample_size > 0:
                extractible_ratio = pages_with_text / sample_size
            else:
                extractible_ratio = 0.0

            # Determine document type
            has_text = pages_with_text > 0
            has_images = pages_with_images > 0

            if extractible_ratio >= self.text_threshold:
                doc_type = DocumentType.TEXT_BASED
                confidence = extractible_ratio
            elif extractible_ratio <= self.scanned_threshold:
                doc_type = DocumentType.SCANNED
                confidence = 1.0 - extractible_ratio
            else:
                doc_type = DocumentType.HYBRID
                confidence = 0.7  # Medium confidence for hybrid

            analysis = DocumentAnalysis(
                doc_type=doc_type,
                extractible_ratio=extractible_ratio,
                total_pages=total_pages,
                has_text=has_text,
                has_images=has_images,
                confidence=confidence,
            )

            LOGGER.info(f"PDF analysis: {analysis}")
            return analysis

        except Exception as e:
            LOGGER.error(f"Failed to analyze PDF {pdf_path}: {e}", exc_info=True)
            return DocumentAnalysis(
                doc_type=DocumentType.UNKNOWN,
                extractible_ratio=0.0,
                total_pages=0,
                has_text=False,
                has_images=False,
                confidence=0.0,
                recommendation="Use fallback multi-engine strategy",
            )

    def detect_image_type(self, image_path: Path) -> DocumentAnalysis:
        """
        Detect image document type (always requires OCR).

        Args:
            image_path: Path to image file

        Returns:
            DocumentAnalysis indicating OCR is required
        """
        return DocumentAnalysis(
            doc_type=DocumentType.IMAGE,
            extractible_ratio=0.0,
            total_pages=1,
            has_text=False,
            has_images=True,
            confidence=1.0,
            recommendation="Use EasyOCR direct",
        )

    def detect_document_type(self, file_path: Path) -> DocumentAnalysis:
        """
        Detect document type based on file extension.

        Args:
            file_path: Path to document

        Returns:
            DocumentAnalysis with detection results
        """
        extension = file_path.suffix.lower()

        # Image files
        if extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
            return self.detect_image_type(file_path)

        # PDF files
        elif extension == ".pdf":
            return self.detect_pdf_type(file_path)

        # Other documents (DOCX, HTML, TXT) - text-based by default
        elif extension in {".docx", ".doc", ".html", ".htm", ".txt", ".md"}:
            return DocumentAnalysis(
                doc_type=DocumentType.TEXT_BASED,
                extractible_ratio=1.0,
                total_pages=1,
                has_text=True,
                has_images=False,
                confidence=1.0,
                recommendation="Use native loader (no OCR)",
            )

        # Unknown
        else:
            return DocumentAnalysis(
                doc_type=DocumentType.UNKNOWN,
                extractible_ratio=0.0,
                total_pages=0,
                has_text=False,
                has_images=False,
                confidence=0.0,
                recommendation="Unsupported file type",
            )


def quick_detect(file_path: Path) -> DocumentAnalysis:
    """
    Quick helper function to detect document type.

    Args:
        file_path: Path to document

    Returns:
        DocumentAnalysis result
    """
    detector = DocumentTypeDetector()
    return detector.detect_document_type(file_path)
