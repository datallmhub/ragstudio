"""
Advanced OCR Text Corrector for Poor Quality Scanned PDFs

This module provides aggressive OCR correction specifically designed for
poorly scanned PDFs with heavy OCR artifacts, mirrored text, and character corruption.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.cleaners.core import (
        clean_bullets,
        clean_dashes,
        clean_non_ascii_chars,
        clean_ordered_bullets,
        clean_postfix,
        clean_prefix,
        clean_extra_whitespace,
        clean_trailing_punctuation,
        clean_ligatures,
        replace_unicode_quotes,
        remove_punctuation,
    )
    from unstructured.documents.elements import Element, Title, NarrativeText, ListItem, Table
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("Unstructured not available. OCR correction will be limited.")
    # Define dummy classes for type hints
    class Element:
        pass
    class Title:
        pass
    class NarrativeText:
        pass
    class ListItem:
        pass
    class Table:
        pass

try:
    import pytesseract
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract/pdf2image not available. Fallback OCR disabled.")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Language detection disabled.")

LOGGER = logging.getLogger(__name__)


@dataclass
class AggressiveOCRConfig:
    """Configuration for aggressive OCR correction."""
    
    # PDF processing
    strategy: str = "fast"  # Use fast strategy to avoid OCR issues
    infer_table_structure: bool = False
    extract_images_in_pdf: bool = False
    
    # Aggressive cleaning
    remove_mirrored_text: bool = True
    remove_gibberish: bool = True
    fix_word_splitting: bool = True
    normalize_spacing: bool = True
    remove_artifacts: bool = True
    
    # Text reconstruction
    reconstruct_words: bool = True
    fix_common_ocr_errors: bool = True
    preserve_structure: bool = False  # Disable for heavily corrupted text
    
    # Fallback options
    use_tesseract_fallback: bool = True
    tesseract_config: str = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"


class AggressiveOCRCorrector:
    """
    Aggressive OCR corrector for poor quality scanned PDFs.
    
    This corrector is specifically designed to handle heavily corrupted OCR text
    with mirrored sequences, gibberish, and character artifacts.
    """
    
    def __init__(self, config: Optional[AggressiveOCRConfig] = None):
        self.config = config or AggressiveOCRConfig()
        
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("unstructured library is required for AggressiveOCRCorrector")
        
        LOGGER.info("AggressiveOCRCorrector initialized with strategy: %s", self.config.strategy)
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process a PDF with aggressive OCR correction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing processed text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        LOGGER.info("Processing PDF with aggressive correction: %s", pdf_path)
        
        try:
            # Step 1: Extract text with minimal OCR processing
            elements = self._extract_text_minimal(pdf_path)
            
            # Step 2: Apply aggressive cleaning
            cleaned_text = self._apply_aggressive_cleaning(elements)
            
            # Step 3: Reconstruct text
            reconstructed_text = self._reconstruct_text(cleaned_text)
            
            # Step 4: Final cleanup
            final_text = self._final_cleanup(reconstructed_text)
            
            # Step 5: Detect language
            language = self._detect_language(final_text)
            
            # Compile results
            result = {
                "text": final_text,
                "language": language,
                "elements_count": len(elements),
                "structure_info": {"structure_preserved": False, "aggressive_correction": True},
                "correction_metrics": {
                    "original_length": sum(len(getattr(e, 'text', '')) for e in elements),
                    "corrected_length": len(final_text),
                    "length_change": len(final_text) - sum(len(getattr(e, 'text', '')) for e in elements),
                    "elements_processed": len(elements),
                    "correction_applied": True
                },
                "source_path": str(pdf_path),
                "processing_method": "aggressive_ocr_correction"
            }
            
            LOGGER.info("Aggressive OCR correction completed: %d chars", len(final_text))
            
            return result
            
        except Exception as e:
            LOGGER.error("Aggressive OCR correction failed: %s", e)
            if self.config.use_tesseract_fallback and TESSERACT_AVAILABLE:
                LOGGER.info("Attempting Tesseract fallback")
                return self._tesseract_fallback(pdf_path)
            raise
    
    def _extract_text_minimal(self, pdf_path: Path) -> List[Element]:
        """Extract text with minimal processing to avoid OCR artifacts."""
        
        try:
            # Use fast strategy to minimize OCR processing
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=self.config.strategy,
                infer_table_structure=self.config.infer_table_structure,
                extract_images_in_pdf=self.config.extract_images_in_pdf,
            )
            
            LOGGER.info("Text extracted: %d elements", len(elements))
            return elements
            
        except Exception as e:
            LOGGER.error("Text extraction failed: %s", e)
            raise
    
    def _apply_aggressive_cleaning(self, elements: List[Element]) -> str:
        """Apply aggressive cleaning to remove OCR artifacts."""
        
        # Extract all text
        all_text = []
        for element in elements:
            if hasattr(element, 'text') and element.text:
                all_text.append(element.text)
        
        raw_text = ' '.join(all_text)
        
        # Apply aggressive cleaning
        cleaned = raw_text
        
        if self.config.remove_mirrored_text:
            cleaned = self._remove_mirrored_text(cleaned)
        
        if self.config.remove_gibberish:
            cleaned = self._remove_gibberish(cleaned)
        
        if self.config.remove_artifacts:
            cleaned = self._remove_artifacts(cleaned)
        
        if self.config.normalize_spacing:
            cleaned = self._normalize_spacing(cleaned)
        
        return cleaned
    
    def _remove_mirrored_text(self, text: str) -> str:
        """Remove mirrored or reversed text sequences."""
        
        # Pattern for mirrored text (common in OCR)
        mirrored_patterns = [
            r'[a-z]+\s+[a-z]+\s+[a-z]+\s+[a-z]+\s+[a-z]+\s+[a-z]+\s+[a-z]+\s+[a-z]+',  # Long sequences of single letters
            r'\b[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\b',  # Short sequences of single letters
            r'[éèêë]\s+[m]\s+[m]\s+[a]\s+[G]',  # Specific mirrored pattern from the PDF
        ]
        
        for pattern in mirrored_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_gibberish(self, text: str) -> str:
        """Remove gibberish text patterns."""
        
        # Patterns for gibberish
        gibberish_patterns = [
            r'\b[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\b',  # Long sequences
            r'\b[a-z]{1,2}\s+[a-z]{1,2}\s+[a-z]{1,2}\s+[a-z]{1,2}\s+[a-z]{1,2}\b',  # Short repeated patterns
            r'[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]',  # Very long sequences
        ]
        
        for pattern in gibberish_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove OCR artifacts."""
        
        # Remove common OCR artifacts
        artifacts = [
            r'[^\w\s\-.,;:!?()\[\]{}"\']',  # Remove special characters except punctuation
            r'\s+',  # Multiple spaces
            r'[■□▪▫]',  # Geometric shapes
            r'\.{3,}',  # Multiple dots
            r'={2,}',  # Multiple equals
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, ' ', text)
        
        return text
    
    def _normalize_spacing(self, text: str) -> str:
        """Normalize spacing in text."""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _reconstruct_text(self, text: str) -> str:
        """Reconstruct text by fixing common OCR issues."""
        
        if not self.config.reconstruct_words:
            return text
        
        reconstructed = text
        
        # Fix word splitting
        if self.config.fix_word_splitting:
            reconstructed = self._fix_word_splitting(reconstructed)
        
        # Fix common OCR errors
        if self.config.fix_common_ocr_errors:
            reconstructed = self._fix_common_ocr_errors(reconstructed)
        
        return reconstructed
    
    def _fix_word_splitting(self, text: str) -> str:
        """Fix words that were split by OCR."""
        
        # Common word splitting patterns
        patterns = [
            (r'P\s+E\s+U\s+G\s+E\s+O\s+T', 'PEUGEOT'),
            (r'L\s+E\s+D', 'LED'),
            (r'A\s+C\s+T\s+I\s+V\s+E', 'ACTIVE'),
            (r'S\s+T\s+Y\s+L\s+E', 'STYLE'),
            (r'A\s+L\s+L\s+U\s+R\s+E', 'ALLURE'),
            (r'P\s+A\s+C\s+K', 'PACK'),
            (r'G\s+T', 'GT'),
            (r'T\s+A\s+R\s+I\s+F', 'TARIF'),
            (r'P\s+R\s+I\s+X', 'PRIX'),
            (r'C\s+O\s+2', 'CO2'),
            (r'C\s+V', 'CV'),
            (r'B\s+O\s+N\s+U\s+S', 'BONUS'),
            (r'M\s+A\s+L\s+U\s+S', 'MALUS'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR character errors."""
        
        # Common OCR character substitutions
        corrections = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'æ': 'ae',
            'œ': 'oe',
            '–': '-',
            '—': '-',
            '`': "'",
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Apply final cleanup to the text."""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the text."""
        
        if not LANGDETECT_AVAILABLE or not text:
            return None
        
        try:
            # Use first 1000 characters for language detection
            sample = text[:1000]
            language = detect(sample)
            LOGGER.debug("Detected language: %s", language)
            return language
        except LangDetectException:
            LOGGER.debug("Language detection failed")
            return None
    
    def _tesseract_fallback(self, pdf_path: Path) -> Dict[str, any]:
        """Fallback to Tesseract OCR if unstructured fails."""
        
        LOGGER.info("Using Tesseract fallback for %s", pdf_path)
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            # Extract text from each page
            text_parts = []
            for i, image in enumerate(images):
                try:
                    text = pytesseract.image_to_string(
                        image, 
                        config=self.config.tesseract_config
                    )
                    if text.strip():
                        text_parts.append(text.strip())
                except Exception as e:
                    LOGGER.warning("Tesseract failed on page %d: %s", i+1, e)
            
            final_text = '\n\n'.join(text_parts)
            language = self._detect_language(final_text)
            
            return {
                "text": final_text,
                "language": language,
                "elements_count": len(images),
                "structure_info": {"structure_preserved": False, "fallback_used": True},
                "correction_metrics": {
                    "original_length": len(final_text),
                    "corrected_length": len(final_text),
                    "length_change": 0,
                    "elements_processed": len(images),
                    "correction_applied": False
                },
                "source_path": str(pdf_path),
                "processing_method": "tesseract_fallback"
            }
            
        except Exception as e:
            LOGGER.error("Tesseract fallback failed: %s", e)
            raise


def process_pdf_aggressive(
    pdf_path: Union[str, Path], 
    config: Optional[AggressiveOCRConfig] = None
) -> Dict[str, any]:
    """
    Convenience function to process a PDF with aggressive OCR correction.
    
    Args:
        pdf_path: Path to the PDF file
        config: Optional configuration
        
    Returns:
        Dictionary containing processed text and metadata
    """
    corrector = AggressiveOCRCorrector(config)
    return corrector.process_pdf(pdf_path)


def create_aggressive_config() -> AggressiveOCRConfig:
    """Create optimized configuration for aggressive OCR correction."""
    return AggressiveOCRConfig(
        strategy="fast",
        infer_table_structure=False,
        remove_mirrored_text=True,
        remove_gibberish=True,
        fix_word_splitting=True,
        normalize_spacing=True,
        remove_artifacts=True,
        reconstruct_words=True,
        fix_common_ocr_errors=True,
        preserve_structure=False,
        use_tesseract_fallback=True
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python aggressive_ocr_corrector.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Process PDF with aggressive correction
        config = create_aggressive_config()
        result = process_pdf_aggressive(pdf_path, config)
        
        print(f"Processed PDF: {result['source_path']}")
        print(f"Language: {result['language']}")
        print(f"Elements: {result['elements_count']}")
        print(f"Text length: {len(result['text'])} characters")
        print(f"Structure: {result['structure_info']}")
        print(f"Metrics: {result['correction_metrics']}")
        
        # Save result
        output_path = Path(pdf_path).with_suffix('.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        print(f"Text saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)
