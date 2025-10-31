"""
EasyOCR wrapper for high-quality multi-language OCR.

Provides intelligent OCR extraction with excellent multi-language support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import time

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from PIL import Image
import numpy as np

LOGGER = logging.getLogger(__name__)


class EasyOCREngine:
    """
    High-quality OCR engine using EasyOCR.

    Features:
    - 80+ language support
    - High accuracy
    - GPU acceleration (if available)
    - Batch processing
    """

    # ISO 639-3 to ISO 639-1 mapping (EasyOCR expects short codes)
    LANG_CODE_MAP = {
        'eng': 'en',
        'fra': 'fr',
        'deu': 'de',
        'spa': 'es',
        'ita': 'it',
        'por': 'pt',
        'rus': 'ru',
        'jpn': 'ja',
        'kor': 'ko',
        'chi_sim': 'ch_sim',
        'chi_tra': 'ch_tra',
        'ara': 'ar',
    }

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        model_storage_directory: Optional[str] = None,
    ):
        """
        Initialize EasyOCR engine.

        Args:
            languages: List of language codes (accepts both ISO 639-1 and ISO 639-3)
                      Examples: ['en', 'fr'] or ['eng', 'fra']
                      Also handles sets and converts them to lists
            gpu: Use GPU acceleration if available (default: False)
            model_storage_directory: Custom directory for model files
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR not installed. Install with: pip install easyocr"
            )

        # Convert languages to list if it's a set or other iterable
        if languages is None:
            languages = ['en', 'fr']
        else:
            languages = list(languages)

        # Map ISO 639-3 codes to ISO 639-1 codes (EasyOCR format)
        self.languages = []
        for lang in languages:
            if lang in self.LANG_CODE_MAP:
                self.languages.append(self.LANG_CODE_MAP[lang])
                LOGGER.debug(f"Mapped language code: {lang} → {self.LANG_CODE_MAP[lang]}")
            else:
                # Already in correct format or unknown
                self.languages.append(lang)

        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
        self._reader = None

        LOGGER.info(
            f"Initializing EasyOCR engine with languages: {self.languages}, "
            f"GPU: {self.gpu}"
        )

    @property
    def reader(self):
        """Lazy-load EasyOCR reader (models are large)."""
        if self._reader is None:
            try:
                start_time = time.time()
                LOGGER.info("Loading EasyOCR models (this may take a while on first run)...")

                kwargs = {
                    'lang_list': self.languages,
                    'gpu': self.gpu,
                }
                if self.model_storage_directory:
                    kwargs['model_storage_directory'] = self.model_storage_directory

                self._reader = easyocr.Reader(**kwargs)

                load_time = time.time() - start_time
                LOGGER.info(f"✓ EasyOCR models loaded in {load_time:.2f}s")

            except Exception as e:
                # Dev mode: show full traceback, Prod mode: simple message
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.error(f"Failed to initialize EasyOCR reader: {e}", exc_info=True)
                else:
                    LOGGER.error(f"Impossible d'initialiser EasyOCR: {str(e)[:100]}")
                raise

        return self._reader

    def extract_text_from_image(
        self,
        image_path: Path,
        detail: int = 1,
        paragraph: bool = False,
    ) -> Tuple[str, dict]:
        """
        Extract text from image using EasyOCR.

        Args:
            image_path: Path to image file
            detail: Detail level (0=text only, 1=text+bbox, 2=text+bbox+conf)
            paragraph: Group text into paragraphs (default: False)

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            start_time = time.time()
            LOGGER.info(f"Extracting text from {image_path.name} with EasyOCR...")

            # Validate image before processing
            try:
                img = Image.open(image_path)
                width, height = img.size
                if width == 0 or height == 0:
                    raise ValueError(f"Invalid image dimensions: {width}x{height}")
                if width > 10000 or height > 10000:
                    LOGGER.warning(f"Large image detected ({width}x{height}), may cause issues")
                img.verify()  # Verify image integrity
                LOGGER.debug(f"Image validated: {width}x{height}, format={img.format}, mode={img.mode}")
            except Exception as e:
                raise ValueError(f"Invalid or corrupted image: {e}")

            # Read image
            result = self.reader.readtext(
                str(image_path),
                detail=detail,
                paragraph=paragraph,
            )

            # Extract text from results
            if detail == 0:
                # Simple text list
                text = "\n".join(result)
                metadata = {
                    "num_text_blocks": len(result),
                }
            else:
                # Detailed results with bounding boxes
                text_blocks = []
                confidences = []

                for detection in result:
                    if len(detection) >= 2:
                        bbox, text_content = detection[0], detection[1]
                        text_blocks.append(text_content)

                        if len(detection) >= 3:
                            confidence = detection[2]
                            confidences.append(confidence)

                text = "\n".join(text_blocks)

                metadata = {
                    "num_text_blocks": len(text_blocks),
                    "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                    "min_confidence": min(confidences) if confidences else 0.0,
                    "max_confidence": max(confidences) if confidences else 0.0,
                }

            extract_time = time.time() - start_time

            metadata.update({
                "extraction_time": round(extract_time, 2),
                "char_count": len(text),
                "word_count": len(text.split()),
                "ocr_engine": "easyocr",
                "languages": self.languages,
            })

            LOGGER.info(
                f"✓ EasyOCR extracted {len(text)} chars in {extract_time:.2f}s "
                f"(confidence: {metadata.get('avg_confidence', 0):.2%})"
            )

            return text, metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"EasyOCR extraction failed for {image_path}: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Impossible de lire l'image {image_path.name} avec EasyOCR")
            raise ValueError(f"Image non supportée ou corrompue")

    def extract_text_from_pdf_page(
        self,
        pdf_path: Path,
        page_num: int,
        dpi: int = 300,
    ) -> Tuple[str, dict]:
        """
        Extract text from PDF page by converting to image first.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            dpi: DPI for PDF to image conversion (default: 300)

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            from pdf2image import convert_from_path

            LOGGER.info(f"Converting PDF page {page_num} to image (DPI={dpi})...")

            # Convert single page to image
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=dpi,
            )

            if not images:
                raise ValueError(f"Failed to convert page {page_num} to image")

            # Save to temporary file for EasyOCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                images[0].save(tmp_path, 'PNG')

            # Extract text
            text, metadata = self.extract_text_from_image(tmp_path, detail=1)

            # Cleanup
            tmp_path.unlink()

            metadata['page_num'] = page_num
            metadata['source_pdf'] = pdf_path.name

            return text, metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"Failed to extract text from PDF page {page_num}: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Impossible d'extraire le texte de la page {page_num}")
            raise ValueError(f"Extraction échouée pour la page {page_num}")

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        max_pages: Optional[int] = None,
        dpi: int = 300,
    ) -> Tuple[str, dict]:
        """
        Extract text from entire PDF using OCR.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (default: all)
            dpi: DPI for PDF to image conversion (default: 300)

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

            LOGGER.info(
                f"Extracting text from PDF with EasyOCR "
                f"({pages_to_process}/{total_pages} pages)..."
            )

            all_text = []
            total_time = 0
            total_chars = 0

            for page_num in range(pages_to_process):
                text, page_meta = self.extract_text_from_pdf_page(
                    pdf_path,
                    page_num,
                    dpi=dpi,
                )
                all_text.append(text)
                total_time += page_meta.get('extraction_time', 0)
                total_chars += len(text)

            combined_text = "\n\n".join(all_text)

            metadata = {
                "total_pages": total_pages,
                "pages_processed": pages_to_process,
                "total_chars": total_chars,
                "total_time": round(total_time, 2),
                "avg_time_per_page": round(total_time / pages_to_process, 2) if pages_to_process > 0 else 0,
                "ocr_engine": "easyocr",
                "languages": self.languages,
                "dpi": dpi,
            }

            LOGGER.info(
                f"✓ EasyOCR extracted {total_chars} chars from {pages_to_process} pages "
                f"in {total_time:.2f}s"
            )

            return combined_text, metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"Failed to extract PDF with EasyOCR: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Impossible d'extraire le PDF avec EasyOCR")
            raise ValueError(f"Extraction PDF échouée")


def extract_with_easyocr(
    file_path: Path,
    languages: Optional[List[str]] = None,
    gpu: bool = False,
) -> Tuple[str, dict]:
    """
    Convenience function to extract text with EasyOCR.

    Args:
        file_path: Path to file (image or PDF)
        languages: Language codes (default: ['en', 'fr'])
        gpu: Use GPU acceleration

    Returns:
        Tuple of (extracted_text, metadata)
    """
    engine = EasyOCREngine(languages=languages, gpu=gpu)

    extension = file_path.suffix.lower()

    if extension == '.pdf':
        return engine.extract_text_from_pdf(file_path)
    elif extension in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}:
        return engine.extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
