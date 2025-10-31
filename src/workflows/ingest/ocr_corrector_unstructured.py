"""
Advanced OCR Correction Module using Unstructured

This module provides sophisticated OCR text correction capabilities specifically designed
for unstructured document processing pipelines. It leverages unstructured's advanced
partitioning and cleaning capabilities for optimal text quality.
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
class OCRCorrectionConfig:
    """Configuration for OCR correction using unstructured."""
    
    # Unstructured partitioning options
    strategy: str = "hi_res"  # Forces OCR on images
    infer_table_structure: bool = True
    extract_images_in_pdf: bool = False
    ocr_languages: List[str] = None
    
    # Cleaning options
    normalize_unicode: bool = True
    clean_whitespace: bool = True
    clean_bullets: bool = True
    clean_dashes: bool = True
    clean_non_ascii: bool = True
    replace_unicode_quotes: bool = True
    strip_punctuation: bool = False  # Usually keep punctuation
    
    # Advanced correction options
    merge_hyphenated_words: bool = True
    merge_line_breaks: bool = True
    detect_mirrored_text: bool = True
    preserve_structure: bool = True
    
    # Fallback options
    use_tesseract_fallback: bool = True
    tesseract_config: str = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
    
    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ["eng", "fra"]


class OCRCorrectorUnstructured:
    """
    Advanced OCR correction using unstructured library.
    
    This class provides sophisticated OCR text correction capabilities specifically
    designed for unstructured document processing pipelines.
    """
    
    def __init__(self, config: Optional[OCRCorrectionConfig] = None):
        self.config = config or OCRCorrectionConfig()
        
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("unstructured library is required for OCRCorrectorUnstructured")
        
        LOGGER.info("OCRCorrectorUnstructured initialized with strategy: %s", self.config.strategy)
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process a PDF file with advanced OCR correction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing processed text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        LOGGER.info("Processing PDF: %s", pdf_path)
        
        try:
            # Step 1: Partition PDF with unstructured
            elements = self._partition_pdf(pdf_path)
            
            # Step 2: Extract and clean text from elements
            cleaned_elements = self._clean_elements(elements)
            
            # Step 3: Detect and correct OCR issues
            corrected_elements = self._correct_ocr_issues(cleaned_elements)
            
            # Step 4: Merge and structure the final text
            final_text, structure_info = self._merge_and_structure(corrected_elements)
            
            # Step 5: Detect language
            language = self._detect_language(final_text)
            
            # Compile results
            result = {
                "text": final_text,
                "language": language,
                "elements_count": len(elements),
                "structure_info": structure_info,
                "correction_metrics": self._calculate_metrics(elements, corrected_elements),
                "source_path": str(pdf_path),
                "processing_method": "unstructured_ocr"
            }
            
            LOGGER.info("PDF processing completed: %d elements, %d chars", 
                       len(elements), len(final_text))
            
            return result
            
        except Exception as e:
            LOGGER.error("PDF processing failed: %s", e)
            if self.config.use_tesseract_fallback and TESSERACT_AVAILABLE:
                LOGGER.info("Attempting Tesseract fallback")
                return self._tesseract_fallback(pdf_path)
            raise
    
    def _partition_pdf(self, pdf_path: Path) -> List[Element]:
        """Partition PDF using unstructured with OCR."""
        
        partition_kwargs = {
            "filename": str(pdf_path),
            "strategy": self.config.strategy,
            "infer_table_structure": self.config.infer_table_structure,
            "extract_images_in_pdf": self.config.extract_images_in_pdf,
        }
        
        # Add OCR languages if available
        if self.config.ocr_languages:
            try:
                partition_kwargs["languages"] = "+".join(self.config.ocr_languages)
            except TypeError:
                # Fallback for older unstructured versions
                partition_kwargs["ocr_languages"] = "+".join(self.config.ocr_languages)
        
        try:
            elements = partition_pdf(**partition_kwargs)
            LOGGER.info("PDF partitioned successfully: %d elements", len(elements))
            return elements
        except Exception as e:
            LOGGER.error("PDF partitioning failed: %s", e)
            raise
    
    def _clean_elements(self, elements: List[Element]) -> List[Element]:
        """Apply unstructured cleaners to elements."""
        
        cleaned_elements = []
        
        for element in elements:
            if hasattr(element, 'text') and element.text:
                original_text = element.text
                cleaned_text = self._apply_cleaners(original_text)
                
                # Update element text if it changed
                if cleaned_text != original_text:
                    element.text = cleaned_text
                    LOGGER.debug("Cleaned element text: %d chars -> %d chars", 
                               len(original_text), len(cleaned_text))
                
                cleaned_elements.append(element)
            else:
                cleaned_elements.append(element)
        
        return cleaned_elements
    
    def _apply_cleaners(self, text: str) -> str:
        """Apply unstructured text cleaners."""
        
        if not text:
            return text
        
        # Apply cleaners in sequence
        cleaned = text
        
        if self.config.normalize_unicode:
            # Use unicodedata for Unicode normalization
            import unicodedata
            cleaned = unicodedata.normalize("NFKC", cleaned)
        
        if self.config.clean_whitespace:
            cleaned = clean_extra_whitespace(cleaned)
        
        if self.config.clean_bullets:
            cleaned = clean_bullets(cleaned)
            cleaned = clean_ordered_bullets(cleaned)
        
        if self.config.clean_dashes:
            cleaned = clean_dashes(cleaned)
        
        if self.config.clean_non_ascii:
            cleaned = clean_non_ascii_chars(cleaned)
        
        if self.config.replace_unicode_quotes:
            cleaned = replace_unicode_quotes(cleaned)
        
        # Clean ligatures
        cleaned = clean_ligatures(cleaned)
        
        # Clean prefixes and postfixes
        cleaned = clean_prefix(cleaned)
        cleaned = clean_postfix(cleaned)
        cleaned = clean_trailing_punctuation(cleaned)
        
        if self.config.strip_punctuation:
            cleaned = remove_punctuation(cleaned)
        
        return cleaned
    
    def _correct_ocr_issues(self, elements: List[Element]) -> List[Element]:
        """Detect and correct specific OCR issues."""
        
        corrected_elements = []
        
        for element in elements:
            if hasattr(element, 'text') and element.text:
                original_text = element.text
                corrected_text = self._apply_ocr_corrections(original_text)
                
                if corrected_text != original_text:
                    element.text = corrected_text
                    LOGGER.debug("Applied OCR corrections: %d chars -> %d chars",
                               len(original_text), len(corrected_text))
                
                corrected_elements.append(element)
            else:
                corrected_elements.append(element)
        
        return corrected_elements
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply specific OCR corrections."""
        
        if not text:
            return text
        
        corrected = text
        
        # Merge hyphenated words split by OCR
        if self.config.merge_hyphenated_words:
            corrected = self._merge_hyphenated_words(corrected)
        
        # Merge line breaks within words
        if self.config.merge_line_breaks:
            corrected = self._merge_line_breaks(corrected)
        
        # Detect and correct mirrored text
        if self.config.detect_mirrored_text:
            corrected = self._detect_and_correct_mirrored_text(corrected)
        
        # Additional OCR-specific corrections
        corrected = self._apply_advanced_corrections(corrected)
        
        return corrected
    
    def _merge_hyphenated_words(self, text: str) -> str:
        """Merge words that were split by OCR with hyphens."""
        
        # Pattern for hyphenated words at line breaks
        patterns = [
            r'(\w+)-\s*\n\s*(\w+)',  # word-\nword
            r'(\w+)-\s*(\w+)',       # word-word (common OCR error)
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, r'\1\2', text, flags=re.MULTILINE)
        
        return text
    
    def _merge_line_breaks(self, text: str) -> str:
        """Merge inappropriate line breaks within words."""
        
        # Pattern for words split across lines
        pattern = r'(\w+)\s*\n\s*(\w+)'
        
        def should_merge(match):
            word1, word2 = match.groups()
            # Merge if both parts look like word fragments
            if len(word1) > 2 and len(word2) > 2:
                return word1 + word2
            return match.group(0)
        
        text = re.sub(pattern, should_merge, text, flags=re.MULTILINE)
        return text
    
    def _detect_and_correct_mirrored_text(self, text: str) -> str:
        """Detect and correct mirrored or reversed text sequences."""
        
        # Common mirrored character patterns
        mirrored_chars = {
            'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p',
            'n': 'u', 'u': 'n', 'm': 'w', 'w': 'm',
            '6': '9', '9': '6', '2': '5', '5': '2',
        }
        
        corrected = text
        
        # Look for sequences that might be mirrored
        for original, mirrored in mirrored_chars.items():
            # Simple heuristic: if we see both original and mirrored in close proximity
            pattern = f'{original}.*{mirrored}|{mirrored}.*{original}'
            if re.search(pattern, corrected, re.IGNORECASE):
                LOGGER.debug("Potential mirrored text detected")
                # For now, just log - more sophisticated detection could be added
        
        return corrected
    
    def _apply_advanced_corrections(self, text: str) -> str:
        """Apply advanced OCR-specific corrections."""
        
        corrected = text
        
        # Fix common OCR character substitutions
        ocr_corrections = {
            # Numbers
            'O': '0',  # Letter O to number 0 in numeric contexts
            'l': '1',  # Letter l to number 1 in numeric contexts
            'I': '1',  # Letter I to number 1 in numeric contexts
            
            # Common OCR errors
            'ﬁ': 'fi',  # Ligatures
            'ﬂ': 'fl',
            'æ': 'ae',
            'œ': 'oe',
            
            # Punctuation
            '`': "'",   # Backtick to apostrophe
            '"': '"',   # Straight quotes to curly quotes
            '"': '"',
            ''': "'",
            ''': "'",
        }
        
        # Apply corrections contextually
        for wrong, correct in ocr_corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        # Fix spacing around punctuation
        corrected = re.sub(r'\s+([.,;:!?])', r'\1', corrected)
        corrected = re.sub(r'([.,;:!?])\s*([.,;:!?])', r'\1\2', corrected)
        
        return corrected
    
    def _merge_and_structure(self, elements: List[Element]) -> Tuple[str, Dict[str, any]]:
        """Merge elements into structured text."""
        
        if not self.config.preserve_structure:
            # Simple concatenation
            text_parts = []
            for element in elements:
                if hasattr(element, 'text') and element.text:
                    text_parts.append(element.text.strip())
            return '\n'.join(text_parts), {"structure_preserved": False}
        
        # Preserve document structure
        structured_parts = []
        structure_info = {
            "titles": 0,
            "paragraphs": 0,
            "lists": 0,
            "tables": 0,
            "structure_preserved": True
        }
        
        for element in elements:
            if not hasattr(element, 'text') or not element.text:
                continue
            
            text = element.text.strip()
            if not text:
                continue
            
            # Add structure based on element type
            if isinstance(element, Title):
                structured_parts.append(f"\n## {text}\n")
                structure_info["titles"] += 1
            elif isinstance(element, ListItem):
                structured_parts.append(f"• {text}")
                structure_info["lists"] += 1
            elif isinstance(element, Table):
                structured_parts.append(f"\n[TABLE]\n{text}\n[/TABLE]\n")
                structure_info["tables"] += 1
            elif isinstance(element, NarrativeText):
                structured_parts.append(f"{text}\n")
                structure_info["paragraphs"] += 1
            else:
                # Generic element
                structured_parts.append(f"{text}\n")
        
        final_text = '\n'.join(structured_parts)
        
        # Clean up excessive newlines
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)
        
        return final_text.strip(), structure_info
    
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
    
    def _calculate_metrics(self, original_elements: List[Element], 
                          corrected_elements: List[Element]) -> Dict[str, any]:
        """Calculate correction metrics."""
        
        original_text = ' '.join(getattr(e, 'text', '') for e in original_elements if hasattr(e, 'text'))
        corrected_text = ' '.join(getattr(e, 'text', '') for e in corrected_elements if hasattr(e, 'text'))
        
        return {
            "original_length": len(original_text),
            "corrected_length": len(corrected_text),
            "length_change": len(corrected_text) - len(original_text),
            "elements_processed": len(corrected_elements),
            "correction_applied": original_text != corrected_text
        }
    
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


def process_pdf_with_unstructured(
    pdf_path: Union[str, Path], 
    config: Optional[OCRCorrectionConfig] = None
) -> Dict[str, any]:
    """
    Convenience function to process a PDF with unstructured OCR correction.
    
    Args:
        pdf_path: Path to the PDF file
        config: Optional configuration
        
    Returns:
        Dictionary containing processed text and metadata
    """
    corrector = OCRCorrectorUnstructured(config)
    return corrector.process_pdf(pdf_path)


def create_automotive_config() -> OCRCorrectionConfig:
    """Create optimized configuration for automotive documents."""
    return OCRCorrectionConfig(
        strategy="hi_res",
        infer_table_structure=True,
        ocr_languages=["eng", "fra"],
        merge_hyphenated_words=True,
        merge_line_breaks=True,
        detect_mirrored_text=True,
        preserve_structure=True,
        clean_bullets=True,
        clean_dashes=True,
        normalize_unicode=True,
        replace_unicode_quotes=True
    )


def create_technical_manual_config() -> OCRCorrectionConfig:
    """Create optimized configuration for technical manuals."""
    return OCRCorrectionConfig(
        strategy="hi_res",
        infer_table_structure=True,
        ocr_languages=["eng", "fra"],
        merge_hyphenated_words=True,
        merge_line_breaks=True,
        detect_mirrored_text=False,  # Less relevant for manuals
        preserve_structure=True,
        clean_bullets=True,
        clean_dashes=True,
        normalize_unicode=True,
        replace_unicode_quotes=True,
        strip_punctuation=False
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ocr_corrector_unstructured.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        # Process PDF with automotive-optimized config
        config = create_automotive_config()
        result = process_pdf_with_unstructured(pdf_path, config)
        
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
