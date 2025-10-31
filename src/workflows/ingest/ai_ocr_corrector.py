"""
AI-Powered OCR Text Correction

This module provides AI-powered OCR text correction using various providers
(Ollama, OpenAI, Anthropic) for advanced error detection and repair.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available. AI correction will not work.")

from src.core.config.ocr_settings import get_ocr_settings

LOGGER = logging.getLogger(__name__)


class CorrectionProvider(str, Enum):
    """Supported AI providers for OCR correction."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


@dataclass
class AIOCRCorrectorConfig:
    """Configuration for AI-powered OCR correction."""

    # Provider selection
    provider: CorrectionProvider = CorrectionProvider.OLLAMA

    # Provider-specific settings (loaded from OCRSettings)
    model_name: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60

    # Correction behavior
    min_confidence: float = 0.7  # Only correct if confidence below this
    chunk_size: int = 1000  # Max characters per correction request
    max_retries: int = 3

    # Prompt engineering
    correction_prompt: Optional[str] = None
    include_context: bool = True
    preserve_formatting: bool = True


class AIOCRCorrector:
    """
    AI-powered OCR text corrector using LLMs.

    Supports multiple providers (Ollama, OpenAI, Anthropic) for advanced
    OCR error detection and correction.
    """

    def __init__(self, config: Optional[AIOCRCorrectorConfig] = None):
        """
        Initialize AI OCR corrector.

        Args:
            config: Optional configuration (loads from OCRSettings if None)
        """
        self.config = config or self._load_config_from_settings()

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for AIOCRCorrector")

        if self.config.provider == CorrectionProvider.NONE:
            LOGGER.warning("AI correction is disabled (provider=none)")

        LOGGER.info(
            "AIOCRCorrector initialized with provider: %s, model: %s",
            self.config.provider,
            self.config.model_name
        )

    def _load_config_from_settings(self) -> AIOCRCorrectorConfig:
        """Load configuration from global OCRSettings."""
        settings = get_ocr_settings()
        quality = settings.quality

        # Determine provider
        provider = CorrectionProvider(quality.ai_correction_provider or "none")

        # Load provider-specific settings
        if provider == CorrectionProvider.OLLAMA:
            return AIOCRCorrectorConfig(
                provider=provider,
                model_name=quality.ollama_model,
                api_base=quality.ollama_api_base,
                timeout=quality.ollama_timeout,
                min_confidence=quality.min_confidence_for_correction,
            )
        elif provider == CorrectionProvider.OPENAI:
            return AIOCRCorrectorConfig(
                provider=provider,
                model_name=quality.openai_model,
                api_key=quality.openai_api_key,
                timeout=quality.openai_timeout,
                min_confidence=quality.min_confidence_for_correction,
            )
        elif provider == CorrectionProvider.ANTHROPIC:
            return AIOCRCorrectorConfig(
                provider=provider,
                model_name=quality.anthropic_model,
                api_key=quality.anthropic_api_key,
                timeout=quality.anthropic_timeout,
                min_confidence=quality.min_confidence_for_correction,
            )
        else:
            return AIOCRCorrectorConfig(provider=CorrectionProvider.NONE)

    def correct_text(
        self,
        text: str,
        confidence: Optional[float] = None,
        context: Optional[str] = None
    ) -> Tuple[str, Dict[str, any]]:
        """
        Correct OCR text using AI.

        Args:
            text: Raw OCR text to correct
            confidence: OCR confidence score (0.0-1.0)
            context: Additional context for correction

        Returns:
            Tuple of (corrected_text, metadata)
        """
        if self.config.provider == CorrectionProvider.NONE:
            LOGGER.debug("AI correction disabled, returning original text")
            return text, {"corrected": False, "reason": "disabled"}

        # Check if correction is needed based on confidence
        if confidence is not None and confidence >= self.config.min_confidence:
            LOGGER.debug(
                "Skipping AI correction (confidence %.2f >= threshold %.2f)",
                confidence,
                self.config.min_confidence
            )
            return text, {
                "corrected": False,
                "reason": "high_confidence",
                "original_confidence": confidence
            }

        LOGGER.info("Correcting text with AI (provider: %s)", self.config.provider)

        start_time = time.time()

        try:
            # Split text into chunks if too long
            if len(text) > self.config.chunk_size:
                corrected_text, metadata = self._correct_chunked(text, context)
            else:
                corrected_text, metadata = self._correct_single(text, context)

            processing_time = time.time() - start_time

            metadata.update({
                "corrected": True,
                "processing_time": processing_time,
                "original_length": len(text),
                "corrected_length": len(corrected_text),
                "length_change": len(corrected_text) - len(text),
                "provider": self.config.provider.value,
                "model": self.config.model_name
            })

            LOGGER.info(
                "AI correction completed: %d chars -> %d chars (%.2fs)",
                len(text),
                len(corrected_text),
                processing_time
            )

            return corrected_text, metadata

        except Exception as e:
            LOGGER.error("AI correction failed: %s", e)
            return text, {
                "corrected": False,
                "error": str(e),
                "reason": "correction_failed"
            }

    def _correct_single(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Tuple[str, Dict[str, any]]:
        """Correct a single text chunk."""

        # Build prompt
        prompt = self._build_correction_prompt(text, context)

        # Call AI provider
        if self.config.provider == CorrectionProvider.OLLAMA:
            return self._correct_with_ollama(prompt)
        elif self.config.provider == CorrectionProvider.OPENAI:
            return self._correct_with_openai(prompt)
        elif self.config.provider == CorrectionProvider.ANTHROPIC:
            return self._correct_with_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _correct_chunked(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Tuple[str, Dict[str, any]]:
        """Correct text in chunks to handle long documents."""

        chunks = self._split_into_chunks(text, self.config.chunk_size)
        corrected_chunks = []
        total_metadata = {
            "chunks_processed": 0,
            "chunks_total": len(chunks)
        }

        for i, chunk in enumerate(chunks):
            LOGGER.debug("Correcting chunk %d/%d", i+1, len(chunks))

            corrected_chunk, metadata = self._correct_single(chunk, context)
            corrected_chunks.append(corrected_chunk)
            total_metadata["chunks_processed"] += 1

        return " ".join(corrected_chunks), total_metadata

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""

        # Try to split at sentence boundaries
        import re
        sentences = re.split(r'([.!?]+\s+)', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            full_sentence = sentence + delimiter

            if current_length + len(full_sentence) > chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0

            current_chunk.append(full_sentence)
            current_length += len(full_sentence)

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _build_correction_prompt(
        self,
        text: str,
        context: Optional[str] = None
    ) -> str:
        """Build prompt for AI correction."""

        if self.config.correction_prompt:
            return self.config.correction_prompt.format(text=text, context=context or "")

        # Default prompt
        base_prompt = """Tu es un expert en correction de texte OCR. Ton rôle est de corriger les erreurs d'OCR dans le texte suivant tout en préservant le sens original et la structure.

Erreurs OCR courantes à corriger :
- Caractères mal reconnus (ex: 'rn' → 'm', 'l' → '1', 'O' → '0')
- Espaces manquants ou en trop
- Mots coupés incorrectement
- Ligatures mal interprétées (ﬁ, ﬂ, etc.)
- Accents manquants ou incorrects

Instructions :
1. Corrige UNIQUEMENT les erreurs d'OCR évidentes
2. Préserve la structure et la ponctuation originale
3. Ne modifie PAS le sens ou le style du texte
4. Retourne UNIQUEMENT le texte corrigé, sans explications
5. Si un passage est illisible, garde-le tel quel

"""

        if context and self.config.include_context:
            base_prompt += f"\nContexte du document : {context}\n"

        base_prompt += f"\nTexte à corriger :\n{text}\n\nTexte corrigé :"

        return base_prompt

    def _correct_with_ollama(self, prompt: str) -> Tuple[str, Dict[str, any]]:
        """Correct text using Ollama."""

        url = f"{self.config.api_base}/api/generate"

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent corrections
                "top_p": 0.9,
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            corrected_text = result.get("response", "").strip()

            metadata = {
                "tokens_generated": result.get("eval_count", 0),
                "tokens_prompt": result.get("prompt_eval_count", 0),
                "api_response_time": result.get("total_duration", 0) / 1e9  # ns to seconds
            }

            return corrected_text, metadata

        except requests.RequestException as e:
            LOGGER.error("Ollama API request failed: %s", e)
            raise

    def _correct_with_openai(self, prompt: str) -> Tuple[str, Dict[str, any]]:
        """Correct text using OpenAI."""

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            corrected_text = result["choices"][0]["message"]["content"].strip()

            metadata = {
                "tokens_generated": result.get("usage", {}).get("completion_tokens", 0),
                "tokens_prompt": result.get("usage", {}).get("prompt_tokens", 0),
                "tokens_total": result.get("usage", {}).get("total_tokens", 0)
            }

            return corrected_text, metadata

        except requests.RequestException as e:
            LOGGER.error("OpenAI API request failed: %s", e)
            raise

    def _correct_with_anthropic(self, prompt: str) -> Tuple[str, Dict[str, any]]:
        """Correct text using Anthropic Claude."""

        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            corrected_text = result["content"][0]["text"].strip()

            metadata = {
                "tokens_generated": result.get("usage", {}).get("output_tokens", 0),
                "tokens_prompt": result.get("usage", {}).get("input_tokens", 0)
            }

            return corrected_text, metadata

        except requests.RequestException as e:
            LOGGER.error("Anthropic API request failed: %s", e)
            raise


def create_ai_corrector(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> AIOCRCorrector:
    """
    Create an AI OCR corrector with optional overrides.

    Args:
        provider: Optional provider override (ollama, openai, anthropic, none)
        model: Optional model name override

    Returns:
        AIOCRCorrector instance
    """
    if provider or model:
        # Load base config from settings
        corrector = AIOCRCorrector()

        # Override if specified
        if provider:
            corrector.config.provider = CorrectionProvider(provider)
        if model:
            corrector.config.model_name = model

        return corrector

    # Use defaults from settings
    return AIOCRCorrector()


def correct_ocr_text(
    text: str,
    confidence: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Convenience function to correct OCR text.

    Args:
        text: Text to correct
        confidence: Optional OCR confidence score
        provider: Optional provider (ollama, openai, anthropic, none)
        model: Optional model name

    Returns:
        Corrected text
    """
    corrector = create_ai_corrector(provider, model)
    corrected_text, _ = corrector.correct_text(text, confidence)
    return corrected_text


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ai_ocr_corrector.py <text_or_file>")
        sys.exit(1)

    input_text = sys.argv[1]

    # Check if input is a file
    if Path(input_text).exists():
        with open(input_text, 'r', encoding='utf-8') as f:
            input_text = f.read()

    try:
        # Create corrector (uses settings from .env)
        corrector = AIOCRCorrector()

        print(f"Provider: {corrector.config.provider.value}")
        print(f"Model: {corrector.config.model_name}")
        print(f"\nOriginal text ({len(input_text)} chars):")
        print("-" * 80)
        print(input_text)
        print("-" * 80)

        # Correct text
        corrected_text, metadata = corrector.correct_text(input_text)

        print(f"\nCorrected text ({len(corrected_text)} chars):")
        print("-" * 80)
        print(corrected_text)
        print("-" * 80)

        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)