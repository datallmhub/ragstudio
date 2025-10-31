"""
Intelligent document loading orchestrator.

Automatically selects the best extraction strategy based on document type:
- Text-based PDFs: PyMuPDF (fast, no OCR)
- Scanned PDFs: EasyOCR (high quality)
- Hybrid PDFs: PyMuPDF + selective EasyOCR
- Images: EasyOCR direct
- Tables: PDFPlumber + structure
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.workflows.ingest.document_detector import (
    DocumentTypeDetector,
    DocumentType,
    DocumentAnalysis,
)
from src.workflows.ingest.ocr.easyocr_engine import EasyOCREngine, EASYOCR_AVAILABLE
from src.workflows.ingest.ocr.paddleocr_engine import PaddleOCREngine, PADDLEOCR_AVAILABLE

try:
    from langchain_community.document_loaders import (
        PyMuPDFLoader,
        PDFPlumberLoader,
    )
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    try:
        from langchain.schema import Document as LangChainDocument
    except ImportError:
        LangChainDocument = None
    PyMuPDFLoader = None
    PDFPlumberLoader = None

LOGGER = logging.getLogger(__name__)


class IntelligentDocumentOrchestrator:
    """
    Intelligent orchestrator that chooses optimal extraction strategy.

    Workflow:
    1. Detect document type (text-based, scanned, hybrid)
    2. Select best extraction engine
    3. Execute extraction with fallback
    4. Return rich metadata
    """

    def __init__(
        self,
        ocr_languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        enable_fallback: bool = True,
    ):
        """
        Initialize orchestrator.

        Args:
            ocr_languages: Languages for OCR (default: ['en', 'fr'])
            use_gpu: Use GPU for OCR if available
            enable_fallback: Enable fallback strategies (default: True)
        """
        self.ocr_languages = ocr_languages or ['en', 'fr']
        self.use_gpu = use_gpu
        self.enable_fallback = enable_fallback

        # Lazy-load components
        self._detector = None
        self._ocr_engine = None
        self._paddleocr_engine = None

        LOGGER.info(
            f"Initialized IntelligentDocumentOrchestrator "
            f"(OCR languages: {self.ocr_languages}, GPU: {self.use_gpu}, "
            f"Fallback: {self.enable_fallback})"
        )

    @property
    def detector(self) -> DocumentTypeDetector:
        """Lazy-load document detector."""
        if self._detector is None:
            self._detector = DocumentTypeDetector()
        return self._detector

    @property
    def ocr_engine(self) -> Optional[EasyOCREngine]:
        """Lazy-load OCR engine."""
        if not EASYOCR_AVAILABLE:
            LOGGER.warning("EasyOCR not available, OCR will not be used")
            return None

        if self._ocr_engine is None:
            try:
                self._ocr_engine = EasyOCREngine(
                    languages=self.ocr_languages,
                    gpu=self.use_gpu,
                )
            except Exception as e:
                LOGGER.error(f"Failed to initialize EasyOCR: {e}")
                return None

        return self._ocr_engine

    @property
    def paddleocr_engine(self) -> Optional[PaddleOCREngine]:
        """Lazy-load PaddleOCR engine (fallback)."""
        if not PADDLEOCR_AVAILABLE:
            LOGGER.debug("PaddleOCR not available")
            return None

        if self._paddleocr_engine is None:
            try:
                self._paddleocr_engine = PaddleOCREngine(
                    languages=self.ocr_languages,
                    use_gpu=self.use_gpu,
                )
            except Exception as e:
                LOGGER.error(f"Failed to initialize PaddleOCR: {e}")
                return None

        return self._paddleocr_engine

    def load_pdf_text_based(
        self,
        pdf_path: Path,
    ) -> Tuple[List[LangChainDocument], Dict]:
        """
        Load text-based PDF with PyMuPDF (fast, no OCR).

        Args:
            pdf_path: Path to PDF

        Returns:
            Tuple of (documents, metadata)
        """
        try:
            LOGGER.info(f"Loading text-based PDF with PyMuPDF: {pdf_path.name}")

            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()

            metadata = {
                "extraction_strategy": "text_based",
                "engine": "pymupdf",
                "ocr_used": False,
                "total_pages": len(docs),
                "total_chars": sum(len(doc.page_content) for doc in docs),
            }

            LOGGER.info(
                f"✓ PyMuPDF loaded {len(docs)} pages, "
                f"{metadata['total_chars']} chars"
            )

            return docs, metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"PyMuPDF extraction failed: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Extraction PyMuPDF échouée")
            raise

    def load_pdf_scanned(
        self,
        pdf_path: Path,
        max_pages: Optional[int] = None,
    ) -> Tuple[List[LangChainDocument], Dict]:
        """
        Load scanned PDF with EasyOCR (high quality).

        Args:
            pdf_path: Path to PDF
            max_pages: Maximum pages to process (for performance)

        Returns:
            Tuple of (documents, metadata)
        """
        if not self.ocr_engine:
            raise RuntimeError("EasyOCR not available for scanned PDF extraction")

        try:
            LOGGER.info(f"Loading scanned PDF with EasyOCR: {pdf_path.name}")

            text, ocr_meta = self.ocr_engine.extract_text_from_pdf(
                pdf_path,
                max_pages=max_pages,
            )

            # Create LangChain document
            doc = LangChainDocument(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "ocr_engine": "easyocr",
                    **ocr_meta,
                },
            )

            metadata = {
                "extraction_strategy": "scanned_ocr",
                "engine": "easyocr",
                "ocr_used": True,
                "total_chars": len(text),
                **ocr_meta,
            }

            LOGGER.info(
                f"✓ EasyOCR extracted {len(text)} chars from scanned PDF"
            )

            return [doc], metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"EasyOCR extraction failed: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Extraction EasyOCR échouée pour PDF scanné")
            raise

    def load_pdf_hybrid(
        self,
        pdf_path: Path,
    ) -> Tuple[List[LangChainDocument], Dict]:
        """
        Load hybrid PDF (text + scanned regions).

        Strategy: Try PyMuPDF first, if text is insufficient, use EasyOCR as fallback.

        Args:
            pdf_path: Path to PDF

        Returns:
            Tuple of (documents, metadata)
        """
        try:
            LOGGER.info(f"Loading hybrid PDF: {pdf_path.name}")

            # Try PyMuPDF first
            try:
                docs, text_meta = self.load_pdf_text_based(pdf_path)

                # Check if we got sufficient text
                total_chars = text_meta.get('total_chars', 0)
                total_pages = text_meta.get('total_pages', 1)
                avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0

                # If avg < 200 chars/page, text extraction is insufficient
                if avg_chars_per_page < 200:
                    LOGGER.warning(
                        f"Insufficient text extracted ({avg_chars_per_page:.0f} chars/page), "
                        f"falling back to OCR"
                    )
                    raise ValueError("Insufficient text extraction")

                metadata = {
                    **text_meta,
                    "extraction_strategy": "hybrid_text_only",
                }

                return docs, metadata

            except Exception:
                # Fallback to OCR
                if self.enable_fallback and self.ocr_engine:
                    LOGGER.info("Falling back to EasyOCR for hybrid PDF")
                    docs, ocr_meta = self.load_pdf_scanned(pdf_path)

                    metadata = {
                        **ocr_meta,
                        "extraction_strategy": "hybrid_ocr_fallback",
                    }

                    return docs, metadata
                else:
                    raise

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"Hybrid PDF extraction failed: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Extraction PDF hybride échouée")
            raise

    def load_image(
        self,
        image_path: Path,
    ) -> Tuple[List[LangChainDocument], Dict]:
        """
        Load image with OCR (EasyOCR → PaddleOCR fallback).

        Cascade strategy:
        1. Try EasyOCR (best quality)
        2. If fails, try PaddleOCR (faster, good with layouts)
        3. If all fail, raise error

        Args:
            image_path: Path to image

        Returns:
            Tuple of (documents, metadata)
        """
        # Try EasyOCR first
        if self.ocr_engine:
            try:
                LOGGER.info(f"Extracting text from image with EasyOCR: {image_path.name}")

                text, ocr_meta = self.ocr_engine.extract_text_from_image(image_path, detail=1)

                doc = LangChainDocument(
                    page_content=text,
                    metadata={
                        "source": str(image_path),
                        "ocr_engine": "easyocr",
                        **ocr_meta,
                    },
                )

                metadata = {
                    "extraction_strategy": "image_ocr",
                    "engine": "easyocr",
                    "ocr_used": True,
                    "total_chars": len(text),
                    **ocr_meta,
                }

                LOGGER.info(f"✓ EasyOCR extracted {len(text)} chars from image")

                return [doc], metadata

            except Exception as e:
                # Dev mode: show full traceback, Prod mode: simple message
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.error(f"EasyOCR failed: {e}", exc_info=True)
                else:
                    LOGGER.warning(f"EasyOCR échoué pour {image_path.name}")

                # Try PaddleOCR fallback if enabled
                if self.enable_fallback and self.paddleocr_engine:
                    LOGGER.info(f"Falling back to PaddleOCR for {image_path.name}")
                    try:
                        text, ocr_meta = self.paddleocr_engine.extract_text_from_image(image_path)

                        doc = LangChainDocument(
                            page_content=text,
                            metadata={
                                "source": str(image_path),
                                "ocr_engine": "paddleocr",
                                **ocr_meta,
                            },
                        )

                        metadata = {
                            "extraction_strategy": "image_ocr_paddleocr_fallback",
                            "engine": "paddleocr",
                            "ocr_used": True,
                            "total_chars": len(text),
                            **ocr_meta,
                        }

                        LOGGER.info(f"✓ PaddleOCR extracted {len(text)} chars from image (fallback)")

                        return [doc], metadata

                    except Exception as paddle_e:
                        if LOGGER.isEnabledFor(logging.DEBUG):
                            LOGGER.error(f"PaddleOCR also failed: {paddle_e}", exc_info=True)
                        else:
                            LOGGER.warning(f"PaddleOCR aussi échoué pour {image_path.name}")
                        # Re-raise original EasyOCR error
                        raise e
                else:
                    # No fallback available, re-raise
                    raise

        # No OCR engines available
        raise RuntimeError("No OCR engine available for image extraction")

    def load_document(
        self,
        file_path: Path,
        force_ocr: bool = False,
    ) -> Tuple[List[LangChainDocument], Dict]:
        """
        Intelligently load document with optimal strategy.

        Workflow:
        1. Detect document type
        2. Choose optimal extraction strategy
        3. Execute with fallback
        4. Return documents + rich metadata

        Args:
            file_path: Path to document
            force_ocr: Force OCR regardless of document type (default: False)

        Returns:
            Tuple of (langchain_documents, extraction_metadata)
        """
        LOGGER.info(f"🎯 Orchestrating extraction for: {file_path.name}")

        # Step 1: Detect document type
        analysis = self.detector.detect_document_type(file_path)
        LOGGER.info(f"📊 Document analysis: {analysis}")

        # Step 2: Choose extraction strategy
        extension = file_path.suffix.lower()

        try:
            # Images always use OCR
            if extension in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}:
                docs, metadata = self.load_image(file_path)

            # PDFs: intelligent routing based on analysis
            elif extension == '.pdf':
                if force_ocr:
                    docs, metadata = self.load_pdf_scanned(file_path)
                elif analysis.doc_type == DocumentType.TEXT_BASED:
                    docs, metadata = self.load_pdf_text_based(file_path)
                elif analysis.doc_type == DocumentType.SCANNED:
                    docs, metadata = self.load_pdf_scanned(file_path)
                elif analysis.doc_type == DocumentType.HYBRID:
                    docs, metadata = self.load_pdf_hybrid(file_path)
                else:
                    # Unknown type, try text-based first
                    docs, metadata = self.load_pdf_text_based(file_path)

            else:
                raise ValueError(f"Unsupported file type: {extension}")

            # Enrich metadata with analysis
            metadata['document_analysis'] = analysis.to_dict()

            LOGGER.info(
                f"✅ Successfully loaded {file_path.name} using {metadata['extraction_strategy']}"
            )

            return docs, metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"Failed to load document: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Échec du chargement du document: {file_path.name}")

            # Final fallback: try basic PyMuPDF
            if self.enable_fallback and extension == '.pdf':
                try:
                    LOGGER.info("Attempting final fallback to PyMuPDF...")
                    docs, metadata = self.load_pdf_text_based(file_path)
                    metadata['extraction_strategy'] = 'fallback_pymupdf'
                    metadata['document_analysis'] = analysis.to_dict()
                    return docs, metadata
                except Exception as fallback_error:
                    LOGGER.warning(f"Final fallback to PyMuPDF failed: {fallback_error}")
                    pass

            raise
