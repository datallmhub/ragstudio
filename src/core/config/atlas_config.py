"""
Atlas-RAG Unified Configuration System

Hiérarchie: CLI > ENV > YAML > Defaults

Usage:
    # Get global config
    config = get_atlas_config()

    # Override from CLI args
    config = AtlasConfig.from_cli_args(advanced_ocr=True, llm_url="http://localhost:11434")

    # Override from file
    config = AtlasConfig.from_file("~/.atlasrag/config.yml")
"""

import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM Configuration for text correction and analysis."""

    use_llm: bool = False
    """Enable LLM usage for OCR correction and analysis"""

    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    """LLM provider"""

    url: Optional[str] = None
    """LLM API URL (auto-detected if None)"""

    api_key: Optional[str] = None
    """API key for remote providers (auto-detected from env if None)"""

    model: str = "mistral:latest"
    """Model name/identifier"""

    timeout: int = 60
    """Request timeout in seconds"""

    max_tokens: int = 4096
    """Maximum tokens per request"""

    temperature: float = 0.1
    """Temperature for generation"""

    def __post_init__(self):
        """Auto-detect configuration based on provider and URL"""
        # Auto-detect URL if not provided
        if self.url is None:
            if self.provider == "ollama":
                self.url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
            elif self.provider == "openai":
                self.url = "https://api.openai.com/v1"
            elif self.provider == "anthropic":
                self.url = "https://api.anthropic.com/v1"

        # Auto-detect API key from environment if not provided
        if self.api_key is None and self.provider != "ollama":
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @property
    def is_local(self) -> bool:
        """Check if LLM endpoint is local (no auth required)"""
        if not self.url:
            return False

        local_indicators = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",  # nosec B104 - String literal for IP detection, not binding
            "192.168.",
            "10.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31."
        ]

        url_lower = self.url.lower()
        return any(indicator in url_lower for indicator in local_indicators)

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration.

        Returns:
            (is_valid, error_message)
        """
        if not self.use_llm:
            return True, None

        # Local LLM doesn't need API key
        if self.is_local:
            return True, None

        # Remote LLM requires API key
        if not self.api_key:
            return False, f"API key required for remote {self.provider} endpoint: {self.url}"

        return True, None


@dataclass
class OCRConfig:
    """OCR Configuration"""

    use_advanced_ocr: bool = False
    """Enable intelligent OCR routing"""

    dictionary_threshold: float = 0.30
    """Dictionary ratio threshold (< threshold = LOW quality → Qwen-VL)"""

    dynamic_threshold: bool = True
    """Enable dynamic threshold adjustment based on language and text length"""

    enable_fallback: bool = True
    """Enable fallback to classic OCR on advanced OCR failures"""

    qwen_vl_url: str = "http://localhost:11434"
    """Qwen-VL API URL (Ollama)"""

    qwen_vl_model: str = "qwen/qwen2.5-vl-7b"
    """Qwen-VL model name"""

    qwen_vl_max_tokens: int = 16384
    """Maximum tokens for Qwen-VL"""

    qwen_vl_timeout: int = 120
    """Qwen-VL request timeout in seconds"""

    classic_ocr_engine: Literal["unstructured", "tesseract", "pdfminer"] = "unstructured"
    """Classic OCR engine to use"""

    tesseract_languages: str = "fra+eng"
    """Tesseract language packs"""

    def get_dynamic_threshold(self, language: str, text_length: int) -> float:
        """
        Calculate dynamic dictionary ratio threshold.

        Args:
            language: Detected language (fr, en, etc.)
            text_length: Length of extracted text

        Returns:
            Adjusted threshold
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


@dataclass
class ChunkingConfig:
    """Chunking Configuration"""

    strategy: Literal["semantic", "sentence", "token"] = "semantic"
    """Chunking strategy"""

    max_tokens: int = 400
    """Maximum tokens per chunk"""

    overlap: int = 50
    """Token overlap between chunks"""

    model: str = "gpt-3.5-turbo"
    """Model for tokenization"""


@dataclass
class OutputConfig:
    """Output Configuration"""

    format: Literal["json", "jsonl", "yaml"] = "json"
    """Output format"""

    include_metadata: bool = True
    """Include metadata in output"""

    pretty_print: bool = True
    """Pretty print JSON output"""

    generate_summary: bool = False
    """Generate processing summary (JSON) - use --summary flag to enable"""


@dataclass
class AtlasConfig:
    """
    Unified Atlas-RAG Configuration

    Hierarchy: CLI > ENV > YAML > Defaults

    Example:
        # Default config
        config = AtlasConfig()

        # From environment variables
        config = AtlasConfig.from_env()

        # From YAML file
        config = AtlasConfig.from_file("~/.atlasrag/config.yml")

        # From CLI args (highest priority)
        config = AtlasConfig.from_cli_args(
            use_llm=True,
            llm_url="http://localhost:11434",
            use_advanced_ocr=True
        )

        # Merge multiple sources (CLI > ENV > FILE)
        config = AtlasConfig.from_file("config.yml")
        config.merge_from_env()
        config.merge_from_cli_args(use_llm=True)
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """Global log level"""

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate LLM config
        is_valid, error = self.llm.validate()
        if not is_valid:
            if self.llm.use_llm:
                # Only raise error if LLM is explicitly enabled
                raise ValueError(f"Invalid LLM configuration: {error}")
            else:
                # Just log warning if LLM is disabled
                logger.warning(f"LLM configuration invalid (but disabled): {error}")

    @classmethod
    def from_env(cls) -> "AtlasConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            ATLAS_USE_LLM - Enable LLM (true/false)
            ATLAS_LLM_PROVIDER - LLM provider (ollama/openai/anthropic)
            ATLAS_LLM_URL - LLM API URL
            ATLAS_LLM_API_KEY - API key
            ATLAS_LLM_MODEL - Model name
            ATLAS_LLM_TIMEOUT - Timeout in seconds

            ATLAS_USE_ADVANCED_OCR - Enable advanced OCR (true/false)
            ATLAS_OCR_DICTIONARY_THRESHOLD - Dictionary ratio threshold (0.0-1.0)
            ATLAS_OCR_DYNAMIC_THRESHOLD - Enable dynamic threshold (true/false)
            ATLAS_QWEN_VL_URL - Qwen-VL API URL
            ATLAS_QWEN_VL_MODEL - Qwen-VL model name

            ATLAS_CHUNK_STRATEGY - Chunking strategy (semantic/sentence/token)
            ATLAS_CHUNK_MAX_TOKENS - Max tokens per chunk
            ATLAS_CHUNK_OVERLAP - Token overlap

            ATLAS_LOG_LEVEL - Log level (DEBUG/INFO/WARNING/ERROR)
        """
        def getenv_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, str(default)).lower()
            return val in ("true", "1", "yes", "on")

        def getenv_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, default))
            except (ValueError, TypeError):
                return default

        def getenv_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, default))
            except (ValueError, TypeError):
                return default

        llm_config = LLMConfig(
            use_llm=getenv_bool("ATLAS_USE_LLM", False),
            provider=os.getenv("ATLAS_LLM_PROVIDER", "ollama"),
            url=os.getenv("ATLAS_LLM_URL"),
            api_key=os.getenv("ATLAS_LLM_API_KEY"),
            model=os.getenv("ATLAS_LLM_MODEL", "mistral:latest"),
            timeout=getenv_int("ATLAS_LLM_TIMEOUT", 60),
            max_tokens=getenv_int("ATLAS_LLM_MAX_TOKENS", 4096),
            temperature=getenv_float("ATLAS_LLM_TEMPERATURE", 0.1),
        )

        ocr_config = OCRConfig(
            use_advanced_ocr=getenv_bool("ATLAS_USE_ADVANCED_OCR", False),
            dictionary_threshold=getenv_float("ATLAS_OCR_DICTIONARY_THRESHOLD", 0.30),
            dynamic_threshold=getenv_bool("ATLAS_OCR_DYNAMIC_THRESHOLD", True),
            enable_fallback=getenv_bool("ATLAS_OCR_ENABLE_FALLBACK", True),
            qwen_vl_url=os.getenv("ATLAS_QWEN_VL_URL", "http://localhost:11434"),
            qwen_vl_model=os.getenv("ATLAS_QWEN_VL_MODEL", "qwen/qwen2.5-vl-7b"),
            qwen_vl_max_tokens=getenv_int("ATLAS_QWEN_VL_MAX_TOKENS", 16384),
            qwen_vl_timeout=getenv_int("ATLAS_QWEN_VL_TIMEOUT", 120),
            classic_ocr_engine=os.getenv("ATLAS_CLASSIC_OCR_ENGINE", "unstructured"),
            tesseract_languages=os.getenv("ATLAS_TESSERACT_LANGUAGES", "fra+eng"),
        )

        chunking_config = ChunkingConfig(
            strategy=os.getenv("ATLAS_CHUNK_STRATEGY", "semantic"),
            max_tokens=getenv_int("ATLAS_CHUNK_MAX_TOKENS", 400),
            overlap=getenv_int("ATLAS_CHUNK_OVERLAP", 50),
            model=os.getenv("ATLAS_CHUNK_MODEL", "gpt-3.5-turbo"),
        )

        output_config = OutputConfig(
            format=os.getenv("ATLAS_OUTPUT_FORMAT", "json"),
            include_metadata=getenv_bool("ATLAS_OUTPUT_INCLUDE_METADATA", True),
            pretty_print=getenv_bool("ATLAS_OUTPUT_PRETTY_PRINT", True),
            generate_summary=getenv_bool("ATLAS_OUTPUT_GENERATE_SUMMARY", False),
        )

        return cls(
            llm=llm_config,
            ocr=ocr_config,
            chunking=chunking_config,
            output=output_config,
            log_level=os.getenv("ATLAS_LOG_LEVEL", "INFO"),
        )

    @classmethod
    def from_file(cls, config_path: str | Path) -> "AtlasConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yml

        Returns:
            AtlasConfig instance

        Example YAML:
            llm:
              use_llm: true
              provider: ollama
              url: http://localhost:11434
              model: mistral:latest

            ocr:
              use_advanced_ocr: true
              dictionary_threshold: 0.30
              dynamic_threshold: true

            chunking:
              strategy: semantic
              max_tokens: 400
              overlap: 50
        """
        config_path = Path(config_path).expanduser()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML config. Install: pip install pyyaml")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Build nested configs
        llm_config = LLMConfig(**config_dict.get("llm", {}))
        ocr_config = OCRConfig(**config_dict.get("ocr", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))

        return cls(
            llm=llm_config,
            ocr=ocr_config,
            chunking=chunking_config,
            output=output_config,
            log_level=config_dict.get("log_level", "INFO"),
        )

    @classmethod
    def from_cli_args(cls, **kwargs) -> "AtlasConfig":
        """
        Create configuration from CLI arguments (highest priority).

        Args:
            **kwargs: CLI arguments (use_llm, llm_url, use_advanced_ocr, etc.)

        Returns:
            AtlasConfig instance

        Example:
            config = AtlasConfig.from_cli_args(
                use_llm=True,
                llm_url="http://localhost:11434",
                use_advanced_ocr=True,
                ocr_threshold=0.25,
                max_tokens=500
            )
        """
        # Start with env config
        config = cls.from_env()

        # Override with CLI args
        config.merge_from_cli_args(**kwargs)

        return config

    def merge_from_env(self):
        """Merge configuration from environment variables (in-place)"""
        env_config = self.from_env()

        # Merge LLM
        if os.getenv("ATLAS_USE_LLM"):
            self.llm = env_config.llm

        # Merge OCR
        if os.getenv("ATLAS_USE_ADVANCED_OCR"):
            self.ocr = env_config.ocr

        # Merge Chunking
        if os.getenv("ATLAS_CHUNK_STRATEGY"):
            self.chunking = env_config.chunking

        # Merge Output
        if os.getenv("ATLAS_OUTPUT_FORMAT"):
            self.output = env_config.output

        if os.getenv("ATLAS_LOG_LEVEL"):
            self.log_level = env_config.log_level

    def merge_from_cli_args(self, **kwargs):
        """
        Merge CLI arguments into config (in-place, highest priority).

        Args:
            **kwargs: CLI arguments
                LLM args: use_llm, llm_provider, llm_url, llm_api_key, llm_model, llm_timeout
                OCR args: use_advanced_ocr, ocr_threshold, ocr_dynamic_threshold, ocr_fallback
                Chunking args: chunk_strategy, max_tokens, overlap
                Output args: output_format, include_metadata
                Global: log_level
        """
        # LLM args
        if "use_llm" in kwargs:
            self.llm.use_llm = kwargs["use_llm"]
        if "llm_provider" in kwargs:
            self.llm.provider = kwargs["llm_provider"]
        if "llm_url" in kwargs:
            self.llm.url = kwargs["llm_url"]
        if "llm_api_key" in kwargs:
            self.llm.api_key = kwargs["llm_api_key"]
        if "llm_model" in kwargs:
            self.llm.model = kwargs["llm_model"]
        if "llm_timeout" in kwargs:
            self.llm.timeout = kwargs["llm_timeout"]

        # OCR args
        if "use_advanced_ocr" in kwargs or "advanced_ocr" in kwargs:
            self.ocr.use_advanced_ocr = kwargs.get("use_advanced_ocr") or kwargs.get("advanced_ocr", False)
        if "ocr_threshold" in kwargs:
            self.ocr.dictionary_threshold = kwargs["ocr_threshold"]
        if "ocr_dynamic_threshold" in kwargs:
            self.ocr.dynamic_threshold = kwargs["ocr_dynamic_threshold"]
        if "ocr_fallback" in kwargs:
            self.ocr.enable_fallback = kwargs["ocr_fallback"]

        # Chunking args
        if "chunk_strategy" in kwargs or "strategy" in kwargs:
            self.chunking.strategy = kwargs.get("chunk_strategy") or kwargs.get("strategy", "semantic")
        if "max_tokens" in kwargs:
            self.chunking.max_tokens = kwargs["max_tokens"]
        if "overlap" in kwargs:
            self.chunking.overlap = kwargs["overlap"]

        # Output args
        if "output_format" in kwargs:
            self.output.format = kwargs["output_format"]
        if "include_metadata" in kwargs:
            self.output.include_metadata = kwargs["include_metadata"]
        if "generate_summary" in kwargs:
            self.output.generate_summary = kwargs["generate_summary"]

        # Global
        if "log_level" in kwargs:
            self.log_level = kwargs["log_level"]

        # Re-validate after merge
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "llm": asdict(self.llm),
            "ocr": asdict(self.ocr),
            "chunking": asdict(self.chunking),
            "output": asdict(self.output),
            "log_level": self.log_level,
        }

    def print_config(self):
        """Print configuration in a human-readable format"""
        print("\n" + "=" * 60)
        print("📋 ATLAS-RAG CONFIGURATION")
        print("=" * 60)

        print("\n🤖 LLM:")
        print(f"  Enabled:    {self.llm.use_llm}")
        if self.llm.use_llm:
            print(f"  Provider:   {self.llm.provider}")
            print(f"  URL:        {self.llm.url}")
            print(f"  Model:      {self.llm.model}")
            print(f"  Local:      {self.llm.is_local}")
            if not self.llm.is_local:
                print(f"  API Key:    {'✓ Set' if self.llm.api_key else '✗ Missing'}")

        print("\n🔍 OCR:")
        print(f"  Advanced:   {self.ocr.use_advanced_ocr}")
        if self.ocr.use_advanced_ocr:
            print(f"  Threshold:  {self.ocr.dictionary_threshold}")
            print(f"  Dynamic:    {self.ocr.dynamic_threshold}")
            print(f"  Fallback:   {self.ocr.enable_fallback}")
            print(f"  Qwen-VL:    {self.ocr.qwen_vl_url}")

        print("\n📝 Chunking:")
        print(f"  Strategy:   {self.chunking.strategy}")
        print(f"  Max Tokens: {self.chunking.max_tokens}")
        print(f"  Overlap:    {self.chunking.overlap}")

        print("\n📤 Output:")
        print(f"  Format:     {self.output.format}")
        print(f"  Metadata:   {self.output.include_metadata}")
        print(f"  Summary:    {self.output.generate_summary}")

        print(f"\n📊 Log Level: {self.log_level}")
        print("=" * 60 + "\n")


# Global config instance (lazy-loaded)
_global_config: Optional[AtlasConfig] = None


def get_atlas_config(reload: bool = False) -> AtlasConfig:
    """
    Get or create global Atlas configuration.

    Loads from (in order):
    1. ~/.atlasrag/config.yml (if exists)
    2. Environment variables
    3. Defaults

    Args:
        reload: Force reload from sources

    Returns:
        AtlasConfig instance

    Example:
        >>> config = get_atlas_config()
        >>> print(config.llm.url)
        'http://localhost:11434'
    """
    global _global_config

    if _global_config is None or reload:
        # Try loading from default config file
        default_config_path = Path.home() / ".atlasrag" / "config.yml"

        if default_config_path.exists():
            logger.info(f"Loading config from: {default_config_path}")
            _global_config = AtlasConfig.from_file(default_config_path)
            # Merge environment variables (higher priority)
            _global_config.merge_from_env()
        else:
            # Load from environment only
            _global_config = AtlasConfig.from_env()

    return _global_config


def reset_global_config():
    """Reset global configuration (for testing)"""
    global _global_config
    _global_config = None
