"""
Tests unitaires pour le module atlas_config.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.core.config.atlas_config import (
    LLMConfig,
    OCRConfig,
    ChunkingConfig,
    OutputConfig,
    AtlasConfig,
    get_atlas_config,
    reset_global_config,
)


class TestLLMConfig:
    """Tests pour LLMConfig."""

    def test_default_values(self):
        """Test valeurs par défaut."""
        config = LLMConfig()
        assert config.use_llm is False
        assert config.provider == "ollama"
        assert config.model == "mistral:latest"
        assert config.timeout == 60
        assert config.max_tokens == 4096
        assert config.temperature == 0.1

    def test_post_init_ollama_url(self):
        """Test auto-détection URL Ollama."""
        config = LLMConfig(provider="ollama")
        assert config.url == "http://localhost:11434"

    def test_post_init_openai_url(self):
        """Test auto-détection URL OpenAI."""
        config = LLMConfig(provider="openai")
        assert config.url == "https://api.openai.com/v1"

    def test_post_init_anthropic_url(self):
        """Test auto-détection URL Anthropic."""
        config = LLMConfig(provider="anthropic")
        assert config.url == "https://api.anthropic.com/v1"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"})
    def test_post_init_openai_api_key_from_env(self):
        """Test auto-détection API key OpenAI depuis ENV."""
        config = LLMConfig(provider="openai")
        assert config.api_key == "test-key-123"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_post_init_anthropic_api_key_from_env(self):
        """Test auto-détection API key Anthropic depuis ENV."""
        config = LLMConfig(provider="anthropic")
        assert config.api_key == "test-anthropic-key"

    def test_is_local_localhost(self):
        """Test détection endpoint local (localhost)."""
        config = LLMConfig(url="http://localhost:11434")
        assert config.is_local is True

    def test_is_local_127_0_0_1(self):
        """Test détection endpoint local (127.0.0.1)."""
        config = LLMConfig(url="http://127.0.0.1:8080")
        assert config.is_local is True

    def test_is_local_192_168(self):
        """Test détection endpoint local (192.168.x.x)."""
        config = LLMConfig(url="http://192.168.1.100:8080")
        assert config.is_local is True

    def test_is_local_remote(self):
        """Test détection endpoint remote."""
        config = LLMConfig(url="https://api.openai.com/v1")
        assert config.is_local is False

    def test_validate_llm_disabled(self):
        """Test validation quand LLM désactivé."""
        config = LLMConfig(use_llm=False)
        is_valid, error = config.validate()
        assert is_valid is True
        assert error is None

    def test_validate_local_llm_no_api_key(self):
        """Test validation LLM local sans API key (OK)."""
        config = LLMConfig(
            use_llm=True,
            provider="ollama",
            url="http://localhost:11434"
        )
        is_valid, error = config.validate()
        assert is_valid is True
        assert error is None

    def test_validate_remote_llm_no_api_key(self):
        """Test validation LLM remote sans API key (FAIL)."""
        config = LLMConfig(
            use_llm=True,
            provider="openai",
            url="https://api.openai.com/v1",
            api_key=None
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert "API key required" in error

    def test_validate_remote_llm_with_api_key(self):
        """Test validation LLM remote avec API key (OK)."""
        config = LLMConfig(
            use_llm=True,
            provider="openai",
            url="https://api.openai.com/v1",
            api_key="test-key"
        )
        is_valid, error = config.validate()
        assert is_valid is True
        assert error is None


class TestOCRConfig:
    """Tests pour OCRConfig."""

    def test_default_values(self):
        """Test valeurs par défaut."""
        config = OCRConfig()
        assert config.use_advanced_ocr is False
        assert config.dictionary_threshold == 0.30
        assert config.dynamic_threshold is True
        assert config.enable_fallback is True
        assert config.qwen_vl_url == "http://localhost:11434"
        assert "qwen" in config.qwen_vl_model.lower()

    def test_get_dynamic_threshold_english_short(self):
        """Test seuil dynamique pour texte anglais court."""
        config = OCRConfig(dictionary_threshold=0.30)
        threshold = config.get_dynamic_threshold("en", 100)
        # Court texte anglais → seuil ajusté vers ~0.40
        assert 0.39 <= threshold <= 0.41

    def test_get_dynamic_threshold_english_long(self):
        """Test seuil dynamique pour texte anglais long."""
        config = OCRConfig(dictionary_threshold=0.30)
        threshold = config.get_dynamic_threshold("en", 2000)
        # Long texte anglais → seuil ajusté vers ~0.35
        assert 0.34 <= threshold <= 0.36

    def test_get_dynamic_threshold_non_english(self):
        """Test seuil dynamique pour texte non-anglais."""
        config = OCRConfig(dictionary_threshold=0.30)
        threshold = config.get_dynamic_threshold("fr", 1000)
        # Non-anglais → seuil ajusté vers ~0.25
        assert 0.24 <= threshold <= 0.26

    def test_get_dynamic_threshold_disabled(self):
        """Test seuil dynamique désactivé."""
        config = OCRConfig(dictionary_threshold=0.30, dynamic_threshold=False)
        threshold = config.get_dynamic_threshold("en", 100)
        # Threshold dynamique désactivé → toujours la valeur de base
        assert threshold == 0.30


class TestChunkingConfig:
    """Tests pour ChunkingConfig."""

    def test_default_values(self):
        """Test valeurs par défaut."""
        config = ChunkingConfig()
        assert config.strategy == "semantic"
        assert config.max_tokens == 400
        assert config.overlap == 50
        assert config.model == "gpt-3.5-turbo"


class TestOutputConfig:
    """Tests pour OutputConfig."""

    def test_default_values(self):
        """Test valeurs par défaut."""
        config = OutputConfig()
        assert config.format == "json"
        assert config.include_metadata is True
        assert config.pretty_print is True
        assert config.generate_summary is False


class TestAtlasConfig:
    """Tests pour AtlasConfig (classe principale)."""

    def test_default_values(self):
        """Test valeurs par défaut."""
        config = AtlasConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.ocr, OCRConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.output, OutputConfig)
        assert config.log_level == "INFO"

    @patch.dict(os.environ, {
        "ATLAS_USE_LLM": "true",
        "ATLAS_LLM_PROVIDER": "ollama",
        "ATLAS_LLM_URL": "http://localhost:11434",
        "ATLAS_LOG_LEVEL": "DEBUG",
    })
    def test_from_env(self):
        """Test chargement depuis variables d'environnement."""
        config = AtlasConfig.from_env()
        assert config.llm.use_llm is True
        assert config.llm.provider == "ollama"
        assert config.llm.url == "http://localhost:11434"
        assert config.log_level == "DEBUG"

    def test_from_cli_args(self):
        """Test chargement depuis arguments CLI."""
        config = AtlasConfig.from_cli_args(
            use_llm=True,
            llm_url="http://localhost:8080",
            use_advanced_ocr=True,
            log_level="DEBUG"
        )
        assert config.llm.use_llm is True
        assert config.llm.url == "http://localhost:8080"
        assert config.ocr.use_advanced_ocr is True
        assert config.log_level == "DEBUG"

    def test_merge_from_cli_args(self):
        """Test merge avec arguments CLI."""
        config = AtlasConfig()
        assert config.llm.use_llm is False

        config.merge_from_cli_args(use_llm=True, llm_provider="openai")
        assert config.llm.use_llm is True
        assert config.llm.provider == "openai"

    def test_to_dict(self):
        """Test conversion en dictionnaire."""
        config = AtlasConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "llm" in config_dict
        assert "ocr" in config_dict
        assert "chunking" in config_dict
        assert "output" in config_dict
        assert isinstance(config_dict["llm"], dict)


class TestGlobalConfig:
    """Tests pour les fonctions globales."""

    def teardown_method(self):
        """Nettoyer après chaque test."""
        reset_global_config()

    def test_get_atlas_config_singleton(self):
        """Test que get_atlas_config retourne un singleton."""
        config1 = get_atlas_config()
        config2 = get_atlas_config()
        assert config1 is config2

    def test_get_atlas_config_reload(self):
        """Test reload de la config globale."""
        config1 = get_atlas_config()
        config2 = get_atlas_config(reload=True)
        assert config1 is not config2

    def test_reset_global_config(self):
        """Test reset de la config globale."""
        config1 = get_atlas_config()
        reset_global_config()
        config2 = get_atlas_config()
        assert config1 is not config2
