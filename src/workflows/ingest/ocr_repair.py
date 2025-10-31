"""
OCR Text Repair & Normalization Module for ChunkForge

This module provides intelligent OCR text correction capabilities to improve
the quality of text extracted from scanned PDFs and other documents.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.info("OpenAI not available. OCR repair will use rule-based correction only.")

LOGGER = logging.getLogger(__name__)


@dataclass
class OCRRepairConfig:
    """Configuration for OCR text repair."""
    use_ai_correction: bool = True
    ai_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 2000
    correction_threshold: float = 0.2  # If >20% chars modified, trigger second pass
    enable_rule_based: bool = True
    preserve_structure: bool = True


class OCRTextRepairer:
    """
    Intelligent OCR text repair system that combines rule-based and AI-powered correction.
    """
    
    def __init__(self, config: Optional[OCRRepairConfig] = None):
        self.config = config or OCRRepairConfig()
        self.openai_client = None
        
        if self.config.use_ai_correction and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                LOGGER.info("OpenAI client initialized for OCR repair")
            except Exception as e:
                LOGGER.warning(f"Failed to initialize OpenAI client: {e}")
                self.config.use_ai_correction = False
    
    def repair_text(self, text: str, context: Optional[str] = None) -> Tuple[str, Dict[str, any]]:
        """
        Repair OCR text using intelligent correction.
        
        Args:
            text: Raw OCR text to repair
            context: Optional context about document type (e.g., "automotive_catalog")
            
        Returns:
            Tuple of (corrected_text, repair_metrics)
        """
        if not text or not text.strip():
            return text, {"repair_score": 0.0, "corrections_applied": 0, "method": "none"}
        
        original_text = text
        repair_metrics = {
            "original_length": len(text),
            "corrections_applied": 0,
            "method": "rule_based",
            "repair_score": 0.0
        }
        
        # Step 1: Rule-based correction
        if self.config.enable_rule_based:
            text = self._apply_rule_based_corrections(text)
        
        # Step 2: AI-powered correction (if available and needed)
        if self.config.use_ai_correction and self.openai_client:
            text = self._apply_ai_correction(text, context)
            repair_metrics["method"] = "ai_powered"
        
        # Calculate repair metrics
        repair_metrics["final_length"] = len(text)
        repair_metrics["corrections_applied"] = self._count_corrections(original_text, text)
        repair_metrics["repair_score"] = repair_metrics["corrections_applied"] / max(len(original_text), 1)
        
        # Trigger second pass if repair score is high
        if repair_metrics["repair_score"] > self.config.correction_threshold:
            LOGGER.info(f"High repair score ({repair_metrics['repair_score']:.2f}), applying second pass")
            text = self._apply_second_pass_correction(text, context)
            repair_metrics["second_pass_applied"] = True
        
        LOGGER.info(f"OCR repair completed: {repair_metrics['corrections_applied']} corrections, "
                   f"score: {repair_metrics['repair_score']:.3f}")
        
        return text, repair_metrics
    
    def _apply_rule_based_corrections(self, text: str) -> str:
        """Apply rule-based OCR corrections."""
        
        # Common OCR errors patterns
        corrections = [
            # Ligatures and special characters
            (r'ﬁ', 'fi'),
            (r'ﬂ', 'fl'),
            (r'æ', 'ae'),
            (r'œ', 'oe'),
            (r'–', '-'),  # En dash
            (r'—', '-'),  # Em dash
            
            # Common French accent errors
            (r'\be\b', 'é'),  # Context-dependent, be careful
            (r'([aeiou])e\b', r'\1é'),  # More specific pattern
            
            # Spacing issues
            (r'\s+', ' '),  # Multiple spaces to single
            (r'([a-zA-Z])([A-Z])', r'\1 \2'),  # Add space between camelCase
            
            # Common OCR artifacts
            (r'[■□▪▫]', ''),  # Remove geometric shapes
            (r'\.{3,}', '...'),  # Normalize ellipsis
            (r'={2,}', ''),  # Remove equal signs
            (r'[^\w\s\-.,;:!?()\[\]{}"\']', ''),  # Remove special chars except punctuation
            
            # Word boundary fixes
            (r'([a-z])([A-Z])', r'\1 \2'),  # Add space between words
        ]
        
        corrected_text = text
        for pattern, replacement in corrections:
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        return corrected_text.strip()
    
    def _apply_ai_correction(self, text: str, context: Optional[str] = None) -> str:
        """Apply AI-powered OCR correction using OpenAI."""
        
        context_hint = ""
        if context:
            context_hint = f"Contexte: {context}. "
        
        prompt = f"""Tu es un expert en post-traitement OCR pour documents techniques et commerciaux français.
Ta mission est de restaurer la lisibilité du texte issu d'un OCR dégradé, 
sans altérer le sens ni la mise en forme d'origine.

{context_hint}Corrige toutes les erreurs de lettres manquantes, accents, mots fusionnés ou tronqués.
Préserve la structure et les noms propres.

Texte original OCR :
\"\"\"
{text}
\"\"\"

Fournis le texte corrigé, fluide, complet et lisible, prêt pour indexation sémantique.
Ne traduis pas. Si un mot reste incertain, laisse-le entre crochets [mot?]."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.ai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            corrected_text = response.choices[0].message.content.strip()
            LOGGER.info("AI-powered OCR correction applied successfully")
            return corrected_text
            
        except Exception as e:
            LOGGER.error(f"AI correction failed: {e}")
            return text  # Return original text if AI fails
    
    def _apply_second_pass_correction(self, text: str, context: Optional[str] = None) -> str:
        """Apply a second pass of correction for heavily corrupted text."""
        
        if not self.openai_client:
            return text
        
        prompt = f"""Ce texte a déjà été corrigé une fois mais nécessite une correction supplémentaire.
Applique une correction plus approfondie en te concentrant sur :
- La cohérence linguistique
- La reconstruction des mots manquants
- L'amélioration de la fluidité du texte

Texte à corriger :
\"\"\"
{text}
\"\"\"

Fournis une version finale, parfaitement lisible et cohérente."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.ai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for second pass
                max_tokens=self.config.max_tokens
            )
            
            corrected_text = response.choices[0].message.content.strip()
            LOGGER.info("Second pass OCR correction applied")
            return corrected_text
            
        except Exception as e:
            LOGGER.error(f"Second pass correction failed: {e}")
            return text
    
    def _count_corrections(self, original: str, corrected: str) -> int:
        """Count the number of character-level corrections made."""
        if len(original) != len(corrected):
            return abs(len(original) - len(corrected))
        
        corrections = sum(1 for a, b in zip(original, corrected) if a != b)
        return corrections
    
    def detect_ocr_quality(self, text: str) -> Dict[str, any]:
        """
        Detect OCR quality issues in text.
        
        Returns:
            Dictionary with quality metrics and detected issues
        """
        issues = []
        quality_score = 1.0
        
        # Check for common OCR artifacts
        if re.search(r'[ﬁﬂæœ]', text):
            issues.append("ligatures_detected")
            quality_score -= 0.1
        
        if re.search(r'[■□▪▫]', text):
            issues.append("geometric_artifacts")
            quality_score -= 0.05
        
        if re.search(r'\s{2,}', text):
            issues.append("excessive_spacing")
            quality_score -= 0.05
        
        if re.search(r'[a-z][A-Z]', text):
            issues.append("missing_spaces")
            quality_score -= 0.1
        
        # Check for missing accents in French text
        french_words = re.findall(r'\b[a-zàâäéèêëïîôöùûüÿç]+\b', text.lower())
        if french_words:
            # Simple heuristic: if many words end with 'e' but few with 'é', likely missing accents
            e_endings = sum(1 for word in french_words if word.endswith('e'))
            e_accent_endings = sum(1 for word in french_words if word.endswith('é'))
            if e_endings > e_accent_endings * 2:
                issues.append("missing_accents")
                quality_score -= 0.15
        
        return {
            "quality_score": max(0.0, quality_score),
            "issues_detected": issues,
            "needs_repair": quality_score < 0.8,
            "repair_priority": "high" if quality_score < 0.6 else "medium" if quality_score < 0.8 else "low"
        }


def repair_ocr_text(text: str, context: Optional[str] = None, config: Optional[OCRRepairConfig] = None) -> Tuple[str, Dict[str, any]]:
    """
    Convenience function for OCR text repair.
    
    Args:
        text: Raw OCR text to repair
        context: Optional context about document type
        config: Optional repair configuration
        
    Returns:
        Tuple of (corrected_text, repair_metrics)
    """
    repairer = OCRTextRepairer(config)
    return repairer.repair_text(text, context)


def detect_ocr_issues(text: str) -> Dict[str, any]:
    """
    Convenience function to detect OCR quality issues.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with quality metrics
    """
    repairer = OCRTextRepairer()
    return repairer.detect_ocr_quality(text)
