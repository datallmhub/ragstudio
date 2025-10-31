"""
OCR Correction Pipeline - Orchestrates Multiple Correction Strategies

This module combines rule-based and AI-powered OCR correction for optimal results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from src.workflows.ingest.aggressive_ocr_corrector import AggressiveOCRCorrector, AggressiveOCRConfig
from src.workflows.ingest.ocr_corrector_unstructured import OCRCorrectorUnstructured, OCRCorrectionConfig
from src.workflows.ingest.ai_ocr_corrector import AIOCRCorrector, AIOCRCorrectorConfig
from src.core.config.ocr_settings import get_ocr_settings

LOGGER = logging.getLogger(__name__)


class CorrectionStrategy(str, Enum):
    """OCR correction strategies."""
    RULES_ONLY = "rules_only"  # Only rule-based correction
    AI_ONLY = "ai_only"  # Only AI correction
    HYBRID = "hybrid"  # Rules + AI (recommended)
    AUTO = "auto"  # Auto-select based on document quality


@dataclass
class CorrectionPipelineConfig:
    """Configuration for OCR correction pipeline."""

    strategy: CorrectionStrategy = CorrectionStrategy.HYBRID

    # Rule-based correction
    use_aggressive_rules: bool = True  # Use aggressive corrector for poor scans
    use_unstructured_rules: bool = True  # Use unstructured corrector

    # AI correction
    use_ai_correction: bool = True
    ai_only_if_low_confidence: bool = True  # Only use AI if confidence < threshold

    # Pipeline behavior
    cascade_corrections: bool = True  # Apply corrections in sequence
    confidence_threshold: float = 0.7  # Below this, apply AI correction


class OCRCorrectionPipeline:
    """
    Orchestrates multiple OCR correction strategies.

    Combines rule-based and AI-powered correction for optimal results.
    """

    def __init__(self, config: Optional[CorrectionPipelineConfig] = None):
        """
        Initialize correction pipeline.

        Args:
            config: Pipeline configuration (uses OCRSettings if None)
        """
        self.config = config or self._load_config_from_settings()

        # Initialize correctors based on config
        self.aggressive_corrector = None
        self.unstructured_corrector = None
        self.ai_corrector = None

        if self.config.use_aggressive_rules:
            self.aggressive_corrector = AggressiveOCRCorrector()
            LOGGER.info("Aggressive rule-based corrector initialized")

        if self.config.use_unstructured_rules:
            self.unstructured_corrector = OCRCorrectorUnstructured()
            LOGGER.info("Unstructured rule-based corrector initialized")

        if self.config.use_ai_correction:
            self.ai_corrector = AIOCRCorrector()
            LOGGER.info("AI corrector initialized (provider: %s)",
                       self.ai_corrector.config.provider.value)

        LOGGER.info("OCR Correction Pipeline initialized (strategy: %s)",
                   self.config.strategy.value)

    def _load_config_from_settings(self) -> CorrectionPipelineConfig:
        """Load configuration from OCRSettings."""
        settings = get_ocr_settings()

        return CorrectionPipelineConfig(
            strategy=CorrectionStrategy.HYBRID,  # Default to hybrid
            use_ai_correction=settings.quality.ai_correction_enabled,
            confidence_threshold=settings.quality.min_confidence_for_correction
        )

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        confidence: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Process PDF with intelligent correction pipeline.

        Args:
            pdf_path: Path to PDF file
            confidence: Optional OCR confidence score

        Returns:
            Dictionary with corrected text and metadata
        """
        pdf_path = Path(pdf_path)

        LOGGER.info("Processing PDF with correction pipeline: %s", pdf_path)
        LOGGER.info("Strategy: %s, Confidence: %.2f",
                   self.config.strategy.value, confidence or 0.0)

        # Auto-select strategy based on confidence
        if self.config.strategy == CorrectionStrategy.AUTO:
            strategy = self._auto_select_strategy(confidence)
        else:
            strategy = self.config.strategy

        LOGGER.info("Selected strategy: %s", strategy.value)

        # Execute correction based on strategy
        if strategy == CorrectionStrategy.RULES_ONLY:
            return self._process_with_rules(pdf_path)
        elif strategy == CorrectionStrategy.AI_ONLY:
            return self._process_with_ai(pdf_path, confidence)
        else:  # HYBRID
            return self._process_hybrid(pdf_path, confidence)

    def correct_text(
        self,
        text: str,
        confidence: Optional[float] = None,
        context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Correct already-extracted OCR text.

        Args:
            text: Raw OCR text
            confidence: OCR confidence score
            context: Optional context

        Returns:
            Dictionary with corrected text and metadata
        """
        LOGGER.info("Correcting text (strategy: %s, length: %d chars)",
                   self.config.strategy.value, len(text))

        result = {
            "original_text": text,
            "original_length": len(text),
            "confidence": confidence,
            "stages": []
        }

        current_text = text

        # Stage 1: Rule-based corrections
        if self.config.strategy in [CorrectionStrategy.RULES_ONLY, CorrectionStrategy.HYBRID]:
            current_text, rules_metadata = self._apply_rules_to_text(current_text)
            result["stages"].append({
                "stage": "rule_based",
                "metadata": rules_metadata
            })

        # Stage 2: AI correction (if enabled and needed)
        if self.config.strategy in [CorrectionStrategy.AI_ONLY, CorrectionStrategy.HYBRID]:
            if self._should_apply_ai(confidence):
                current_text, ai_metadata = self._apply_ai_to_text(
                    current_text, confidence, context
                )
                result["stages"].append({
                    "stage": "ai_correction",
                    "metadata": ai_metadata
                })

        # Final result
        result.update({
            "corrected_text": current_text,
            "corrected_length": len(current_text),
            "length_change": len(current_text) - len(text),
            "stages_applied": len(result["stages"]),
            "strategy_used": self.config.strategy.value
        })

        LOGGER.info("Text correction completed: %d chars -> %d chars (%d stages)",
                   len(text), len(current_text), len(result["stages"]))

        return result

    def _auto_select_strategy(self, confidence: Optional[float]) -> CorrectionStrategy:
        """Auto-select correction strategy based on document quality."""

        if confidence is None:
            # No confidence info, use hybrid
            return CorrectionStrategy.HYBRID

        if confidence < 0.5:
            # Very low confidence, use all corrections
            LOGGER.info("Low confidence (%.2f), using HYBRID strategy", confidence)
            return CorrectionStrategy.HYBRID
        elif confidence < self.config.confidence_threshold:
            # Medium confidence, AI might help
            LOGGER.info("Medium confidence (%.2f), using HYBRID strategy", confidence)
            return CorrectionStrategy.HYBRID
        else:
            # High confidence, rules only
            LOGGER.info("High confidence (%.2f), using RULES_ONLY strategy", confidence)
            return CorrectionStrategy.RULES_ONLY

    def _should_apply_ai(self, confidence: Optional[float]) -> bool:
        """Determine if AI correction should be applied."""

        if not self.ai_corrector:
            return False

        if not self.config.ai_only_if_low_confidence:
            return True

        if confidence is None:
            return True

        return confidence < self.config.confidence_threshold

    def _process_with_rules(self, pdf_path: Path) -> Dict[str, any]:
        """Process PDF with rule-based corrections only."""

        LOGGER.info("Processing with rule-based corrections only")

        # Try unstructured corrector first (better structure preservation)
        if self.unstructured_corrector:
            try:
                result = self.unstructured_corrector.process_pdf(pdf_path)
                result["correction_pipeline"] = {
                    "strategy": "rules_only",
                    "corrector": "unstructured"
                }
                return result
            except Exception as e:
                LOGGER.warning("Unstructured corrector failed: %s", e)

        # Fallback to aggressive corrector
        if self.aggressive_corrector:
            result = self.aggressive_corrector.process_pdf(pdf_path)
            result["correction_pipeline"] = {
                "strategy": "rules_only",
                "corrector": "aggressive"
            }
            return result

        raise RuntimeError("No rule-based corrector available")

    def _process_with_ai(
        self,
        pdf_path: Path,
        confidence: Optional[float]
    ) -> Dict[str, any]:
        """Process PDF with AI correction only."""

        if not self.ai_corrector:
            raise RuntimeError("AI corrector not available")

        LOGGER.info("Processing with AI correction only")

        # First, extract text (using any available method)
        if self.unstructured_corrector:
            result = self.unstructured_corrector.process_pdf(pdf_path)
        elif self.aggressive_corrector:
            result = self.aggressive_corrector.process_pdf(pdf_path)
        else:
            raise RuntimeError("No text extraction method available")

        original_text = result["text"]

        # Apply AI correction
        corrected_text, ai_metadata = self.ai_corrector.correct_text(
            original_text, confidence
        )

        result.update({
            "text": corrected_text,
            "original_text": original_text,
            "ai_correction_applied": ai_metadata.get("corrected", False),
            "ai_metadata": ai_metadata,
            "correction_pipeline": {
                "strategy": "ai_only",
                "provider": self.ai_corrector.config.provider.value
            }
        })

        return result

    def _process_hybrid(
        self,
        pdf_path: Path,
        confidence: Optional[float]
    ) -> Dict[str, any]:
        """Process PDF with hybrid approach (rules + AI)."""

        LOGGER.info("Processing with hybrid correction (rules + AI)")

        # Stage 1: Rule-based extraction and correction
        if self.unstructured_corrector:
            result = self.unstructured_corrector.process_pdf(pdf_path)
            rule_corrector = "unstructured"
        elif self.aggressive_corrector:
            result = self.aggressive_corrector.process_pdf(pdf_path)
            rule_corrector = "aggressive"
        else:
            raise RuntimeError("No rule-based corrector available")

        rules_corrected_text = result["text"]

        # Stage 2: AI correction (if needed)
        if self._should_apply_ai(confidence):
            LOGGER.info("Applying AI correction to rule-corrected text")

            corrected_text, ai_metadata = self.ai_corrector.correct_text(
                rules_corrected_text, confidence
            )

            result.update({
                "text": corrected_text,
                "rules_corrected_text": rules_corrected_text,
                "ai_correction_applied": ai_metadata.get("corrected", False),
                "ai_metadata": ai_metadata,
                "correction_pipeline": {
                    "strategy": "hybrid",
                    "stages": [rule_corrector, "ai"],
                    "ai_provider": self.ai_corrector.config.provider.value
                }
            })
        else:
            LOGGER.info("Skipping AI correction (confidence sufficient)")
            result["correction_pipeline"] = {
                "strategy": "hybrid_rules_only",
                "stages": [rule_corrector],
                "ai_skipped": True,
                "reason": "high_confidence"
            }

        return result

    def _apply_rules_to_text(self, text: str) -> tuple:
        """Apply rule-based corrections to text."""

        # For now, we apply basic cleanup
        # In a full implementation, this would use the rule-based correctors

        import re

        cleaned = text

        # Basic cleaning
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        metadata = {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "rules_applied": ["normalize_whitespace"]
        }

        return cleaned, metadata

    def _apply_ai_to_text(
        self,
        text: str,
        confidence: Optional[float],
        context: Optional[str]
    ) -> tuple:
        """Apply AI correction to text."""

        if not self.ai_corrector:
            return text, {"applied": False, "reason": "no_ai_corrector"}

        corrected_text, metadata = self.ai_corrector.correct_text(
            text, confidence, context
        )

        return corrected_text, metadata


def create_pipeline(
    strategy: str = "hybrid",
    use_ai: bool = True
) -> OCRCorrectionPipeline:
    """
    Create correction pipeline with specified strategy.

    Args:
        strategy: Correction strategy (rules_only, ai_only, hybrid, auto)
        use_ai: Whether to enable AI correction

    Returns:
        OCRCorrectionPipeline instance
    """
    config = CorrectionPipelineConfig(
        strategy=CorrectionStrategy(strategy),
        use_ai_correction=use_ai
    )

    return OCRCorrectionPipeline(config)


def process_pdf_with_correction(
    pdf_path: Union[str, Path],
    strategy: str = "hybrid",
    confidence: Optional[float] = None
) -> Dict[str, any]:
    """
    Convenience function to process PDF with correction.

    Args:
        pdf_path: Path to PDF
        strategy: Correction strategy
        confidence: Optional OCR confidence

    Returns:
        Processing results
    """
    pipeline = create_pipeline(strategy)
    return pipeline.process_pdf(pdf_path, confidence)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_correction_pipeline.py <pdf_path> [strategy]")
        print("Strategies: rules_only, ai_only, hybrid (default), auto")
        sys.exit(1)

    pdf_path = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else "hybrid"

    try:
        # Process PDF
        result = process_pdf_with_correction(pdf_path, strategy)

        print(f"PDF processed: {result['source_path']}")
        print(f"Strategy: {result.get('correction_pipeline', {}).get('strategy', 'unknown')}")
        print(f"Language: {result.get('language', 'unknown')}")
        print(f"Text length: {len(result['text'])} chars")

        if 'ai_metadata' in result:
            print(f"AI correction: {result['ai_correction_applied']}")
            print(f"AI provider: {result['correction_pipeline'].get('ai_provider', 'N/A')}")

        # Save result
        output_path = Path(pdf_path).with_suffix('.corrected.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])

        print(f"Corrected text saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)