"""
PaddleOCR wrapper for fast and accurate OCR.

Provides OCR extraction with excellent performance and layout support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import time

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

from PIL import Image

LOGGER = logging.getLogger(__name__)


class PaddleOCREngine:
    """
    Fast OCR engine using PaddleOCR.

    Features:
    - Fast inference
    - Good accuracy
    - Layout detection support
    - Multi-language support
    """

    # ISO 639-3 to PaddleOCR language codes
    LANG_CODE_MAP = {
        'eng': 'en',
        'fra': 'fr',
        'deu': 'german',
        'spa': 'es',
        'ita': 'it',
        'por': 'pt',
        'rus': 'ru',
        'jpn': 'japan',
        'kor': 'korean',
        'chi_sim': 'ch',
        'chi_tra': 'chinese_cht',
        'ara': 'ar',
    }

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        use_angle_cls: bool = True,
    ):
        """
        Initialize PaddleOCR engine.

        Args:
            languages: List of language codes (accepts both ISO 639-1 and ISO 639-3)
                      Examples: ['en', 'fr'] or ['eng', 'fra']
                      Note: PaddleOCR processes one language at a time, uses first language
            use_gpu: Use GPU acceleration if available (default: False)
            use_angle_cls: Use angle classification for rotated text (default: True)
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddleocr"
            )

        # Convert languages to list if it's a set or other iterable
        if languages is None:
            languages = ['en']
        else:
            languages = list(languages)

        # Map ISO 639-3 codes to PaddleOCR format (use first language)
        lang = languages[0] if languages else 'en'
        if lang in self.LANG_CODE_MAP:
            self.language = self.LANG_CODE_MAP[lang]
            LOGGER.debug(f"Mapped language code: {lang} → {self.language}")
        else:
            self.language = lang

        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self._ocr = None

        LOGGER.info(
            f"Initializing PaddleOCR engine with language: {self.language}, "
            f"GPU: {self.use_gpu}"
        )

    @property
    def ocr(self):
        """Lazy-load PaddleOCR instance (models are large)."""
        if self._ocr is None:
            try:
                start_time = time.time()
                LOGGER.info("Loading PaddleOCR models (this may take a while on first run)...")

                self._ocr = PaddleOCR(  # Simplified - only lang supported
                    lang=self.language,
                    # show_log not available in all versions
                )

                load_time = time.time() - start_time
                LOGGER.info(f"✓ PaddleOCR models loaded in {load_time:.2f}s")

            except Exception as e:
                # Dev mode: show full traceback, Prod mode: simple message
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
                else:
                    LOGGER.error(f"Impossible d'initialiser PaddleOCR: {str(e)[:100]}")
                raise

        return self._ocr

    def extract_text_from_image(
        self,
        image_path: Path,
    ) -> Tuple[str, dict]:
        """
        Extract text from image using PaddleOCR.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            start_time = time.time()
            LOGGER.info(f"Extracting text from {image_path.name} with PaddleOCR...")

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

            # Run OCR
            result = self.ocr.ocr(str(image_path), cls=self.use_angle_cls)

            # Extract text from results
            # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
            text_blocks = []
            confidences = []

            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        text_content = line[1][0]  # Extract text
                        confidence = line[1][1]     # Extract confidence
                        text_blocks.append(text_content)
                        confidences.append(confidence)

            text = "\n".join(text_blocks)

            extract_time = time.time() - start_time

            metadata = {
                "extraction_time": round(extract_time, 2),
                "char_count": len(text),
                "word_count": len(text.split()),
                "ocr_engine": "paddleocr",
                "language": self.language,
                "num_text_blocks": len(text_blocks),
            }

            if confidences:
                metadata.update({
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences),
                })

            LOGGER.info(
                f"✓ PaddleOCR extracted {len(text)} chars in {extract_time:.2f}s "
                f"(confidence: {metadata.get('avg_confidence', 0):.2%})"
            )

            return text, metadata

        except Exception as e:
            # Dev mode: show full traceback, Prod mode: simple message
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.error(f"PaddleOCR extraction failed for {image_path}: {e}", exc_info=True)
            else:
                LOGGER.warning(f"Impossible de lire l'image {image_path.name} avec PaddleOCR")
            raise ValueError(f"Image non supportée ou corrompue")


def extract_with_paddleocr(
    file_path: Path,
    languages: Optional[List[str]] = None,
    use_gpu: bool = False,
) -> Tuple[str, dict]:
    """
    Convenience function to extract text with PaddleOCR.

    Args:
        file_path: Path to file (image)
        languages: Language codes (default: ['en'])
        use_gpu: Use GPU acceleration

    Returns:
        Tuple of (extracted_text, metadata)
    """
    engine = PaddleOCREngine(languages=languages, use_gpu=use_gpu)

    extension = file_path.suffix.lower()

    if extension in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}:
        return engine.extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
