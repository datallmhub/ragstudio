"""
OCR Quality Detector

This module performs a QUICK OCR test to detect if the extracted text quality is good.
This is different from image quality - an image can have high technical quality (resolution, sharpness)
but produce poor OCR results (garbage characters, low recognizability).

This detector helps route documents to advanced OCR engines when classic OCR fails.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Union
import unicodedata

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from unstructured.partition.image import partition_image
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    from nltk.corpus import words
    NLTK_WORDS_AVAILABLE = True
except ImportError:
    NLTK_WORDS_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# Lazy-loaded dictionaries
_ENGLISH_WORDS = None
_FRENCH_WORDS = None


def _load_french_words():
    """Load French dictionary words (lazy loading)."""
    global _FRENCH_WORDS

    if _FRENCH_WORDS is None:
        # Common French words (subset for quick validation)
        _FRENCH_WORDS = {
            # Articles
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux',
            # Pronouns
            'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'me', 'te', 'se', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
            # Prepositions
            'dans', 'sur', 'sous', 'avec', 'pour', 'par', 'en', 'vers', 'sans',
            # Common verbs
            'est', 'sont', 'être', 'avoir', 'faire', 'dire', 'aller', 'voir',
            'savoir', 'vouloir', 'pouvoir', 'devoir', 'prendre', 'donner',
            # Common nouns
            'contrat', 'entretien', 'adresse', 'email', 'téléphone', 'tél',
            'date', 'nom', 'prénom', 'société', 'entreprise', 'client',
            'document', 'page', 'numéro', 'référence', 'conditions',
            # Common adjectives
            'bon', 'grand', 'petit', 'nouveau', 'vieux', 'jeune',
            # Others
            'oui', 'non', 'et', 'ou', 'mais', 'donc', 'car', 'si', 'comme',
            'tout', 'tous', 'toute', 'toutes', 'ce', 'cet', 'cette', 'ces',
        }

    return _FRENCH_WORDS


def _load_english_words():
    """Load English dictionary words (lazy loading)."""
    global _ENGLISH_WORDS

    if _ENGLISH_WORDS is None:
        if NLTK_WORDS_AVAILABLE:
            try:
                _ENGLISH_WORDS = set(w.lower() for w in words.words())
            except LookupError:
                # NLTK data not downloaded
                LOGGER.warning("NLTK words corpus not found. Using basic English words.")
                _ENGLISH_WORDS = _get_basic_english_words()
        else:
            _ENGLISH_WORDS = _get_basic_english_words()

    return _ENGLISH_WORDS


def _get_basic_english_words():
    """Get basic English words for validation."""
    return {
        # Articles
        'the', 'a', 'an',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        # Prepositions
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about',
        # Common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may', 'might',
        'make', 'get', 'go', 'see', 'know', 'take', 'give', 'come', 'think',
        # Common nouns
        'contract', 'maintenance', 'address', 'email', 'phone', 'tel',
        'date', 'name', 'company', 'customer', 'document', 'page', 'number',
        # Common adjectives
        'good', 'new', 'old', 'big', 'small', 'long', 'short',
        # Others
        'yes', 'no', 'and', 'or', 'but', 'not', 'all', 'some', 'this', 'that',
    }


class OCRQualityMetrics:
    """OCR quality metrics for text extraction."""

    def __init__(
        self,
        sample_text: str,
        text_length: int,
        recognizable_ratio: float,
        word_ratio: float,
        dictionary_ratio: float,
        garbage_ratio: float,
        confidence_score: float,
        overall_ocr_quality: float,
        quality_category: str,
    ):
        self.sample_text = sample_text
        self.text_length = text_length
        self.recognizable_ratio = recognizable_ratio
        self.word_ratio = word_ratio
        self.dictionary_ratio = dictionary_ratio
        self.garbage_ratio = garbage_ratio
        self.confidence_score = confidence_score
        self.overall_ocr_quality = overall_ocr_quality
        self.quality_category = quality_category

    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary."""
        return {
            "sample_text_preview": self.sample_text[:200] if self.sample_text else "",
            "text_length": self.text_length,
            "recognizable_ratio": self.recognizable_ratio,
            "word_ratio": self.word_ratio,
            "dictionary_ratio": self.dictionary_ratio,
            "garbage_ratio": self.garbage_ratio,
            "confidence_score": self.confidence_score,
            "overall_ocr_quality": self.overall_ocr_quality,
            "quality_category": self.quality_category,
        }


class OCRQualityDetector:
    """
    Detect OCR text quality by performing a quick test extraction.

    This detector performs a FAST Tesseract OCR on the image and analyzes
    the extracted text quality to determine if advanced OCR is needed.

    Quality categories:
    - HIGH (score >= 0.7): Clean text, good recognition → Classic OCR is fine
    - MEDIUM (0.4 <= score < 0.7): Some issues → Try preprocessing or advanced OCR
    - LOW (score < 0.4): Poor OCR quality → Use advanced OCR (Qwen-VL)
    """

    def __init__(
        self,
        sample_size: int = 500,  # Only analyze first N characters
        min_recognizable_ratio: float = 0.7,
        min_word_ratio: float = 0.5,
        dictionary_threshold: float = 0.30,  # Dictionary ratio threshold
        dynamic_threshold: bool = True,  # Enable dynamic threshold adjustment
    ):
        """
        Initialize the OCR quality detector.

        Args:
            sample_size: Number of characters to analyze (for speed)
            min_recognizable_ratio: Minimum ratio of recognizable characters
            min_word_ratio: Minimum ratio of valid words
            dictionary_threshold: Dictionary ratio threshold (default: 0.30)
            dynamic_threshold: Enable dynamic threshold adjustment
        """
        self.sample_size = sample_size
        self.min_recognizable_ratio = min_recognizable_ratio
        self.min_word_ratio = min_word_ratio
        self.dictionary_threshold = dictionary_threshold
        self.dynamic_threshold = dynamic_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

        if not TESSERACT_AVAILABLE:
            self.logger.warning("Tesseract not available. OCR quality detection will be limited.")

    def detect_ocr_quality(
        self,
        image_path: Union[str, Path],
        languages: str = "eng+fra"
    ) -> OCRQualityMetrics:
        """
        Detect OCR quality by performing a quick test extraction.

        Args:
            image_path: Path to the image file
            languages: Tesseract language codes (e.g., 'eng', 'fra', 'eng+fra')

        Returns:
            OCRQualityMetrics object
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.logger.info(f"Testing OCR quality for: {image_path}")

        # Try Tesseract first, fall back to unstructured
        text = None
        avg_confidence = 50.0  # Default

        if TESSERACT_AVAILABLE:
            try:
                # Load image
                img = Image.open(image_path)

                # Perform quick OCR test
                # Use PSM 3 (fully automatic page segmentation) for speed
                custom_config = f'--oem 3 --psm 3 -l {languages}'

                # Get text
                text = pytesseract.image_to_string(img, config=custom_config)

                # Get detailed data for confidence
                data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 50.0

            except Exception as e:
                self.logger.warning(f"Tesseract OCR failed: {e}, falling back to unstructured")
                text = None

        # Fallback to unstructured if Tesseract failed
        if text is None and UNSTRUCTURED_AVAILABLE:
            try:
                self.logger.info("Using unstructured for OCR quality test")
                # Convert languages format: 'eng+fra' -> ['eng', 'fra']
                ocr_langs = languages.replace('+', ',').split(',')

                elements = partition_image(
                    filename=str(image_path),
                    languages=ocr_langs,  # Use languages instead of ocr_languages
                    strategy='hi_res'  # Images require hi_res strategy
                )

                # Combine text from all elements
                text = "\n".join([str(elem) for elem in elements])
                avg_confidence = 60.0  # Assume reasonable confidence for unstructured

            except Exception as e:
                self.logger.error(f"Unstructured OCR also failed: {e}")
                return self._create_fallback_metrics()

        # If still no text, return fallback
        if text is None:
            return self._create_fallback_metrics()

        try:

            # Analyze text quality
            sample_text = text[:self.sample_size]
            text_length = len(text)

            # Calculate quality metrics
            recognizable_ratio = self._calculate_recognizable_ratio(sample_text)
            word_ratio = self._calculate_word_ratio(sample_text)
            dictionary_ratio = self._calculate_dictionary_ratio(sample_text)
            garbage_ratio = self._calculate_garbage_ratio(sample_text)

            # Calculate overall OCR quality
            overall_ocr_quality = self._calculate_overall_ocr_quality(
                recognizable_ratio,
                word_ratio,
                dictionary_ratio,
                garbage_ratio,
                avg_confidence / 100.0  # Normalize to 0-1
            )

            # Extract primary language for dynamic threshold
            primary_language = languages.split("+")[0] if "+" in languages else languages
            # Convert tesseract codes to language codes (eng -> en, fra -> fr)
            lang_code = "fr" if primary_language.startswith("fra") else "en"

            # Categorize quality (with dynamic threshold based on language and text length)
            quality_category = self._categorize_quality(
                overall_ocr_quality,
                dictionary_ratio,
                language=lang_code,
                text_length=text_length
            )

            metrics = OCRQualityMetrics(
                sample_text=sample_text,
                text_length=text_length,
                recognizable_ratio=recognizable_ratio,
                word_ratio=word_ratio,
                dictionary_ratio=dictionary_ratio,
                garbage_ratio=garbage_ratio,
                confidence_score=avg_confidence,
                overall_ocr_quality=overall_ocr_quality,
                quality_category=quality_category,
            )

            self.logger.info(
                f"OCR Quality: {quality_category} "
                f"(score={overall_ocr_quality:.3f}, "
                f"recognizable={recognizable_ratio:.2%}, "
                f"words={word_ratio:.2%}, "
                f"dictionary={dictionary_ratio:.2%}, "
                f"confidence={avg_confidence:.1f})"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"OCR quality detection failed: {e}")
            return self._create_fallback_metrics()

    def _calculate_recognizable_ratio(self, text: str) -> float:
        """
        Calculate ratio of recognizable characters (letters, numbers, punctuation).

        Args:
            text: Input text

        Returns:
            Ratio of recognizable characters (0.0 to 1.0)
        """
        if not text:
            return 0.0

        # Count recognizable characters (alphanumeric + common punctuation + whitespace)
        recognizable_chars = sum(
            1 for c in text
            if c.isalnum() or c.isspace() or c in ".,;:!?'\"()[]{}«»-—–_/\\@#$%&*+=<>"
        )

        ratio = recognizable_chars / len(text)
        return float(ratio)

    def _calculate_word_ratio(self, text: str) -> float:
        """
        Calculate ratio of valid-looking words.

        A valid word contains mostly letters (at least 70% letters).

        Args:
            text: Input text

        Returns:
            Ratio of valid words (0.0 to 1.0)
        """
        if not text:
            return 0.0

        # Split into words
        words = text.split()

        if not words:
            return 0.0

        # Count valid words (mostly alphabetic)
        valid_words = sum(
            1 for word in words
            if len(word) >= 2 and sum(c.isalpha() for c in word) / len(word) >= 0.7
        )

        ratio = valid_words / len(words)
        return float(ratio)

    def _calculate_dictionary_ratio(self, text: str) -> float:
        """
        Calculate ratio of real words found in dictionaries.

        This is the CRITICAL metric for detecting bad OCR quality.
        A document can have 90%+ "valid" words (letters only) but still be garbage.

        Args:
            text: Input text

        Returns:
            Ratio of real dictionary words (0.0 to 1.0)
        """
        if not text:
            return 0.0

        # Load dictionaries
        french_words = _load_french_words()
        english_words = _load_english_words()

        # Split into words and normalize
        words_list = text.lower().split()

        if not words_list:
            return 0.0

        # Count words found in dictionaries
        # Remove punctuation for better matching
        import string
        translator = str.maketrans('', '', string.punctuation)

        dictionary_word_count = 0
        for word in words_list:
            # Clean word (remove punctuation)
            clean_word = word.translate(translator)

            # Skip very short words
            if len(clean_word) < 2:
                continue

            # Check if in any dictionary
            if clean_word in french_words or clean_word in english_words:
                dictionary_word_count += 1

        # Calculate ratio
        total_words = len([w for w in words_list if len(w.translate(translator)) >= 2])
        if total_words == 0:
            return 0.0

        ratio = dictionary_word_count / total_words

        self.logger.debug(
            f"Dictionary validation: {dictionary_word_count}/{total_words} = {ratio:.2%}"
        )

        return float(ratio)

    def _calculate_garbage_ratio(self, text: str) -> float:
        """
        Calculate ratio of "garbage" characters (unusual Unicode, control chars).

        Args:
            text: Input text

        Returns:
            Ratio of garbage characters (0.0 to 1.0, lower is better)
        """
        if not text:
            return 0.0

        # Count garbage characters
        garbage_chars = sum(
            1 for c in text
            if unicodedata.category(c) in ['Cc', 'Cf', 'Cn', 'Co', 'Cs']  # Control, format, not assigned, private use
            or (ord(c) > 127 and not c.isalpha())  # Non-ASCII non-letter
        )

        ratio = garbage_chars / len(text)
        return float(ratio)

    def _calculate_overall_ocr_quality(
        self,
        recognizable_ratio: float,
        word_ratio: float,
        dictionary_ratio: float,
        garbage_ratio: float,
        confidence_norm: float,
    ) -> float:
        """
        Calculate overall OCR quality score (0.0 to 1.0).

        Args:
            recognizable_ratio: Ratio of recognizable characters
            word_ratio: Ratio of valid words (letters only)
            dictionary_ratio: Ratio of REAL dictionary words (CRITICAL)
            garbage_ratio: Ratio of garbage characters (inverse)
            confidence_norm: Normalized Tesseract confidence (0-1)

        Returns:
            Overall OCR quality score
        """
        # Weighted average
        # CRITICAL: dictionary_ratio has the highest weight (40%)
        # because it's the best indicator of OCR quality
        weights = {
            'dictionary': 0.40,   # NEW: Most important!
            'recognizable': 0.20,
            'words': 0.15,
            'garbage': 0.15,
            'confidence': 0.10,
        }

        # Garbage is inverted (less is better)
        garbage_score = max(0.0, 1.0 - garbage_ratio)

        overall_score = (
            weights['dictionary'] * dictionary_ratio +
            weights['recognizable'] * recognizable_ratio +
            weights['words'] * word_ratio +
            weights['garbage'] * garbage_score +
            weights['confidence'] * confidence_norm
        )

        return float(overall_score)

    def _get_dynamic_threshold(self, language: str, text_length: int) -> float:
        """
        Calculate dynamic dictionary ratio threshold.

        Adjusts threshold based on:
        - Language (French: more lenient, English: stricter)
        - Text length (short text: more lenient, long text: stricter)

        Args:
            language: Detected language (fr, en, etc.)
            text_length: Length of extracted text

        Returns:
            Adjusted threshold (clamped between 0.15 and 0.45)
        """
        if not self.dynamic_threshold:
            return self.dictionary_threshold

        threshold = self.dictionary_threshold

        # Adjust by language
        if language.lower().startswith("fr"):
            threshold -= 0.05  # French: more lenient (0.25)
        elif language.lower().startswith("en"):
            threshold += 0.05  # English: stricter (0.35)

        # Adjust by text length
        if text_length < 500:
            threshold += 0.05  # Short texts: more lenient
        elif text_length > 2000:
            threshold -= 0.05  # Long texts: stricter

        # Clamp between 0.15 and 0.45
        return max(0.15, min(0.45, threshold))

    def _categorize_quality(
        self,
        score: float,
        dictionary_ratio: float = None,
        language: str = "en",
        text_length: int = 0
    ) -> str:
        """
        Categorize OCR quality based on overall score.

        CRITICAL: If dictionary_ratio < threshold, force LOW quality
        regardless of other metrics. This catches garbage text.

        Threshold is dynamically adjusted based on language and text length.

        Args:
            score: Overall quality score (0.0 to 1.0)
            dictionary_ratio: Optional ratio of real dictionary words
            language: Detected language (for dynamic threshold)
            text_length: Length of text (for dynamic threshold)

        Returns:
            Quality category: HIGH, MEDIUM, or LOW
        """
        # CRITICAL CHECK: Use dynamic threshold if enabled
        if dictionary_ratio is not None:
            threshold = self._get_dynamic_threshold(language, text_length)

            if dictionary_ratio < threshold:
                self.logger.warning(
                    f"Low dictionary ratio ({dictionary_ratio:.2%}) detected "
                    f"(threshold: {threshold:.2%} for {language}, {text_length} chars). "
                    f"Forcing LOW quality despite overall score {score:.3f}"
                )
                return "LOW"

        # Normal categorization based on overall score
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _create_fallback_metrics(self) -> OCRQualityMetrics:
        """Create fallback metrics when OCR test cannot be performed."""
        return OCRQualityMetrics(
            sample_text="",
            text_length=0,
            recognizable_ratio=0.5,
            word_ratio=0.5,
            dictionary_ratio=0.5,
            garbage_ratio=0.2,
            confidence_score=50.0,
            overall_ocr_quality=0.5,
            quality_category="MEDIUM",
        )

    def get_recommended_ocr_engine(self, ocr_metrics: OCRQualityMetrics) -> str:
        """
        Get recommended OCR engine based on OCR quality test.

        Args:
            ocr_metrics: OCR quality metrics

        Returns:
            Recommended OCR engine: 'classic_ocr', 'classic_ocr_with_preprocessing', or 'qwen_vl'
        """
        if ocr_metrics.quality_category == "HIGH":
            return "classic_ocr"
        elif ocr_metrics.quality_category == "MEDIUM":
            return "classic_ocr_with_preprocessing"
        else:  # LOW
            return "qwen_vl"


def detect_ocr_quality(image_path: Union[str, Path], languages: str = "eng+fra") -> OCRQualityMetrics:
    """
    Convenience function to detect OCR quality.

    Args:
        image_path: Path to the image file
        languages: Tesseract language codes

    Returns:
        OCRQualityMetrics object
    """
    detector = OCRQualityDetector()
    return detector.detect_ocr_quality(image_path, languages)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_quality_detector.py <image_path> [languages]")
        print("Example: python ocr_quality_detector.py document.jpg eng+fra")
        sys.exit(1)

    image_path = sys.argv[1]
    languages = sys.argv[2] if len(sys.argv) > 2 else "eng+fra"

    try:
        detector = OCRQualityDetector()
        metrics = detector.detect_ocr_quality(image_path, languages)

        print(f"OCR Quality Analysis: {image_path}")
        print(f"=" * 60)
        print(f"Text Length:           {metrics.text_length} chars")
        print(f"Recognizable Ratio:    {metrics.recognizable_ratio:.2%}")
        print(f"Word Ratio:            {metrics.word_ratio:.2%}")
        print(f"Dictionary Ratio:      {metrics.dictionary_ratio:.2%} ⭐ CRITICAL")
        print(f"Garbage Ratio:         {metrics.garbage_ratio:.2%}")
        print(f"Confidence Score:      {metrics.confidence_score:.1f}/100")
        print(f"")
        print(f"Overall OCR Quality:   {metrics.overall_ocr_quality:.3f}")
        print(f"Quality Category:      {metrics.quality_category}")
        print(f"")
        print(f"Recommended OCR:       {detector.get_recommended_ocr_engine(metrics)}")
        print(f"")
        print(f"Sample Text (first 200 chars):")
        print(f"-" * 60)
        print(metrics.sample_text[:200])
        print(f"-" * 60)

    except Exception as e:
        print(f"Error analyzing OCR quality: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
