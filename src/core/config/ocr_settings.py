"""
OCR Configuration Settings for Atlas-RAG

This module provides centralized configuration for all OCR engines and routing logic.
Supports environment variables, configuration files, and runtime overrides.
"""

from pathlib import Path
from typing import Dict, Literal, Optional

try:
    from pydantic import BaseSettings, Field, validator
    PYDANTIC_V2 = False
except ImportError:
    try:
        from pydantic_settings import BaseSettings
        from pydantic import Field, validator
        PYDANTIC_V2 = True
    except ImportError:
        # Fallback to basic settings if Pydantic not available
        class BaseSettings:
            pass


class QwenVLSettings(BaseSettings):
    """Configuration for Qwen-VL OCR engine."""

    enabled: bool = Field(
        default=True,
        description="Enable Qwen-VL OCR engine"
    )

    model_name: str = Field(
        default="qwen/qwen2.5-vl-7b",
        description="Qwen-VL model to use (e.g., qwen/qwen2.5-vl-7b, minicpm-o-2_6)"
    )

    api_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )

    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )

    retry_delay: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Base delay between retries in seconds"
    )

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for generation"
    )

    class Config:
        env_prefix = "ATLAS_QWEN_"
        case_sensitive = False


class NougatSettings(BaseSettings):
    """Configuration for Nougat OCR engine."""

    enabled: bool = Field(
        default=True,
        description="Enable Nougat OCR engine"
    )

    model_name: str = Field(
        default="facebook/nougat-base",
        description="Nougat model from HuggingFace"
    )

    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)"
    )

    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Batch size for processing"
    )

    beam_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Beam search size"
    )

    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion"
    )

    timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Processing timeout in seconds"
    )

    class Config:
        env_prefix = "ATLAS_NOUGAT_"
        case_sensitive = False


class ClassicOCRSettings(BaseSettings):
    """Configuration for Classic OCR (Tesseract/docTR/unstructured)."""

    enabled: bool = Field(
        default=True,
        description="Enable classic OCR fallback"
    )

    preferred_engine: Literal["unstructured", "tesseract", "doctr", "pdfminer"] = Field(
        default="unstructured",
        description="Preferred OCR engine for classic mode"
    )

    tesseract_languages: str = Field(
        default="fra+eng",
        description="Tesseract language packs (e.g. 'fra+eng')"
    )

    tesseract_psm: int = Field(
        default=6,
        ge=0,
        le=13,
        description="Tesseract page segmentation mode"
    )

    unstructured_strategy: Literal["auto", "hi_res", "fast", "ocr_only"] = Field(
        default="hi_res",
        description="Unstructured partitioning strategy"
    )

    enable_doctr: bool = Field(
        default=False,
        description="Enable docTR as fallback option"
    )

    class Config:
        env_prefix = "ATLAS_CLASSIC_"
        case_sensitive = False


class OCRRoutingSettings(BaseSettings):
    """Configuration for intelligent OCR routing."""

    complexity_threshold_low: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Below this: use classic OCR"
    )

    complexity_threshold_high: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Above this: use advanced OCR (Qwen-VL)"
    )

    scientific_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Math density threshold for Nougat routing"
    )

    enable_scientific_detection: bool = Field(
        default=True,
        description="Enable scientific document detection"
    )

    enable_complexity_analysis: bool = Field(
        default=True,
        description="Enable document complexity analysis"
    )

    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback to classic OCR on failures"
    )

    max_retries_per_engine: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Max retries per OCR engine before fallback"
    )

    @validator("complexity_threshold_high")
    def validate_thresholds(cls, v, values):
        """Ensure high threshold is greater than low threshold."""
        if "complexity_threshold_low" in values:
            low = values["complexity_threshold_low"]
            if v <= low:
                raise ValueError(
                    f"complexity_threshold_high ({v}) must be > "
                    f"complexity_threshold_low ({low})"
                )
        return v

    class Config:
        env_prefix = "ATLAS_ROUTING_"
        case_sensitive = False


class OCRQualitySettings(BaseSettings):
    """Configuration for OCR quality assessment and validation."""

    enable_spell_check: bool = Field(
        default=False,
        description="Enable spell-checking for quality validation"
    )

    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable confidence score"
    )

    enable_post_correction: bool = Field(
        default=True,
        description="Enable post-OCR correction (AI + rules)"
    )

    ai_correction_provider: Optional[Literal["openai", "anthropic", "ollama", "none"]] = Field(
        default="ollama",
        description="AI provider for text correction (openai, anthropic, ollama, none)"
    )

    # OpenAI settings
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for correction"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (or use OPENAI_API_KEY env var)"
    )

    # Anthropic settings
    anthropic_model: str = Field(
        default="claude-3-haiku-20240307",
        description="Anthropic model for correction"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )

    # Ollama settings (for local Mistral/Llama/etc)
    ollama_model: str = Field(
        default="mistral:latest",
        description="Ollama model for correction (mistral, llama3, etc)"
    )

    ollama_api_base: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )

    ollama_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Ollama request timeout in seconds"
    )

    enable_two_pass_correction: bool = Field(
        default=False,
        description="Enable two-pass correction for heavily corrupted text"
    )

    class Config:
        env_prefix = "ATLAS_QUALITY_"
        case_sensitive = False


class OCRSettings(BaseSettings):
    """
    Master OCR configuration combining all sub-configurations.

    Usage:
        # From environment variables
        settings = OCRSettings()

        # From .env file
        settings = OCRSettings(_env_file='.env')

        # Override specific settings
        settings = OCRSettings(qwen_vl__api_base_url='http://custom-url')

    Environment Variables:
        ATLAS_QWEN_API_BASE_URL - Qwen-VL API URL
        ATLAS_NOUGAT_DEVICE - Nougat device (auto, cpu, cuda, mps)
        ATLAS_ROUTING_COMPLEXITY_THRESHOLD_LOW - Low complexity threshold
        ATLAS_ROUTING_SCIENTIFIC_THRESHOLD - Scientific detection threshold
        ... (see individual settings classes for all options)
    """

    # Sub-configurations
    qwen_vl: QwenVLSettings = Field(
        default_factory=QwenVLSettings,
        description="Qwen-VL OCR settings"
    )

    nougat: NougatSettings = Field(
        default_factory=NougatSettings,
        description="Nougat OCR settings"
    )

    classic: ClassicOCRSettings = Field(
        default_factory=ClassicOCRSettings,
        description="Classic OCR settings"
    )

    routing: OCRRoutingSettings = Field(
        default_factory=OCRRoutingSettings,
        description="OCR routing settings"
    )

    quality: OCRQualitySettings = Field(
        default_factory=OCRQualitySettings,
        description="OCR quality settings"
    )

    # Global settings
    cache_enabled: bool = Field(
        default=False,
        description="Enable OCR result caching"
    )

    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for OCR cache (None = temp)"
    )

    cache_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        ge=0,
        description="Cache time-to-live in seconds"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level for OCR operations"
    )

    class Config:
        env_prefix = "ATLAS_OCR_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"  # Use ATLAS_QWEN_VL__API_BASE_URL

    def to_dict(self) -> Dict:
        """Export settings as dictionary."""
        return {
            "qwen_vl": self.qwen_vl.dict() if hasattr(self.qwen_vl, "dict") else vars(self.qwen_vl),
            "nougat": self.nougat.dict() if hasattr(self.nougat, "dict") else vars(self.nougat),
            "classic": self.classic.dict() if hasattr(self.classic, "dict") else vars(self.classic),
            "routing": self.routing.dict() if hasattr(self.routing, "dict") else vars(self.routing),
            "quality": self.quality.dict() if hasattr(self.quality, "dict") else vars(self.quality),
            "cache_enabled": self.cache_enabled,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "log_level": self.log_level,
        }

    @classmethod
    def from_file(cls, config_path: Path) -> "OCRSettings":
        """Load settings from a JSON/YAML file."""
        import json

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix == ".json":
                config_dict = json.load(f)
            elif config_path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls(**config_dict)


# Global settings instance (lazy-loaded)
_settings: Optional[OCRSettings] = None


def get_ocr_settings(reload: bool = False) -> OCRSettings:
    """
    Get or create the global OCR settings instance.

    Args:
        reload: Force reload settings from environment

    Returns:
        OCRSettings instance

    Example:
        >>> settings = get_ocr_settings()
        >>> print(settings.qwen_vl.api_base_url)
        'http://localhost:11434'
    """
    global _settings

    if _settings is None or reload:
        _settings = OCRSettings()

    return _settings


# Convenience function for backward compatibility
def load_ocr_config() -> OCRSettings:
    """Load OCR configuration (alias for get_ocr_settings)."""
    return get_ocr_settings()