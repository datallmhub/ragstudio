"""
Utilities to clean raw text and detect language on ingested documents.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, Optional, Set

from langdetect import DetectorFactory, LangDetectException, detect

from src.workflows.io.schema import Document
from src.workflows.ingest.ocr_repair import OCRTextRepairer, OCRRepairConfig, detect_ocr_issues

DetectorFactory.seed = 42
LOGGER = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
PAGE_ARTIFACT_RE = re.compile(r"-\s*Page \d+\s*-", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _strip_artifacts(text: str) -> str:
    text = PAGE_ARTIFACT_RE.sub(" ", text)
    return text


def clean_text(text: str) -> str:
    """
    Normalize Unicode, remove OCR artefacts and collapse whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _strip_artifacts(text)
    text = _normalize_whitespace(text)
    return text


def detect_language(text: str) -> Optional[str]:
    """
    Detect the dominant language of the text. Returns None on failure.
    """
    snippet = text[:5000]
    try:
        return detect(snippet)
    except LangDetectException:
        LOGGER.debug("Could not detect language for text snippet")
        return None


def normalize_document(
    document: Document,
    *,
    allowed_languages: Optional[Iterable[str]] = None,
    enable_ocr_repair: bool = True,
    ocr_repair_config: Optional[OCRRepairConfig] = None,
) -> Optional[Document]:
    """
    Clean document text, apply OCR repair if needed, and optionally filter on language.
    Returns the updated document or None if filtered out.
    """
    cleaned = clean_text(document.text)
    
    # Apply OCR repair if enabled and text quality is poor
    if enable_ocr_repair and cleaned:
        quality_metrics = detect_ocr_issues(cleaned)
        
        if quality_metrics["needs_repair"]:
            LOGGER.info(
                "Applying OCR repair to document %s (quality_score=%.2f, issues=%s)",
                document.source_path,
                quality_metrics["quality_score"],
                quality_metrics["issues_detected"]
            )
            
            repairer = OCRTextRepairer(ocr_repair_config)
            context = _infer_document_context(document)
            repaired_text, repair_metrics = repairer.repair_text(cleaned, context)
            
            # Store repair metrics in document metadata
            document.metadata["ocr_repair"] = {
                "applied": True,
                "quality_score": quality_metrics["quality_score"],
                "issues_detected": quality_metrics["issues_detected"],
                "repair_score": repair_metrics["repair_score"],
                "corrections_applied": repair_metrics["corrections_applied"],
                "method": repair_metrics["method"]
            }
            
            cleaned = repaired_text
        else:
            document.metadata["ocr_repair"] = {
                "applied": False,
                "quality_score": quality_metrics["quality_score"],
                "reason": "text_quality_acceptable"
            }
    
    language = detect_language(cleaned) if cleaned else None

    if allowed_languages:
        allowed: Set[str] = {lang.lower() for lang in allowed_languages}
        if language is None or language.lower() not in allowed:
            LOGGER.info(
                "Skipping document %s due to language filter (detected=%s)",
                document.source_path,
                language,
            )
            return None

    document.text = cleaned
    document.language = language
    document.metadata["language"] = language
    return document


def _infer_document_context(document: Document) -> Optional[str]:
    """
    Infer document context for OCR repair based on metadata and content.
    """
    # Check file extension and metadata for context hints
    if document.source_path.suffix.lower() in ['.pdf']:
        if 'automotive' in str(document.source_path).lower() or '208' in str(document.source_path).lower():
            return "automotive_catalog"
        elif 'manual' in str(document.source_path).lower() or 'guide' in str(document.source_path).lower():
            return "technical_manual"
        else:
            return "technical_document"
    
    # Check content for context clues
    text_lower = document.text.lower()[:1000]  # First 1000 chars
    if any(keyword in text_lower for keyword in ['peugeot', 'renault', 'citroën', 'voiture', 'automobile']):
        return "automotive_catalog"
    elif any(keyword in text_lower for keyword in ['installation', 'configuration', 'manuel', 'guide']):
        return "technical_manual"
    elif any(keyword in text_lower for keyword in ['rapport', 'analyse', 'étude']):
        return "technical_report"
    
    return None
