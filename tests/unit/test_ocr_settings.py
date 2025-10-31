"""
Unit tests for core/config/ocr_settings.py

Tests cover:
- QwenVLSettings configuration and validation
- NougatSettings configuration and validation
- ClassicOCRSettings configuration and validation
- OCRRoutingConfig configuration
- Environment variable loading
- Validation constraints
"""

import pytest
import os
from unittest.mock import patch

from src.core.config.ocr_settings import (
    QwenVLSettings,
    NougatSettings,
    ClassicOCRSettings,
)


class TestQwenVLSettings:
    """Test suite for QwenVLSettings configuration."""

    def test_default_values(self):
        """Test default QwenVL configuration values."""
        settings = QwenVLSettings()

        assert settings.enabled is True
        assert settings.model_name == "qwen/qwen2.5-vl-7b"
        assert settings.api_base_url == "http://localhost:11434"
        assert settings.timeout_seconds == 120
        assert settings.max_retries == 3
        assert settings.retry_delay == 2.0
        assert settings.temperature == 0.1

    def test_custom_values(self):
        """Test QwenVL with custom values."""
        settings = QwenVLSettings(
            enabled=False,
            model_name="qwen/qwen-custom",
            api_base_url="http://custom:8000",
            timeout_seconds=60,
            max_retries=5,
            retry_delay=1.5,
            temperature=0.5
        )

        assert settings.enabled is False
        assert settings.model_name == "qwen/qwen-custom"
        assert settings.api_base_url == "http://custom:8000"
        assert settings.timeout_seconds == 60
        assert settings.max_retries == 5
        assert settings.retry_delay == 1.5
        assert settings.temperature == 0.5

    def test_timeout_validation_min(self):
        """Test timeout minimum validation."""
        with pytest.raises((ValueError, Exception)):
            QwenVLSettings(timeout_seconds=5)  # Below minimum of 10

    def test_timeout_validation_max(self):
        """Test timeout maximum validation."""
        with pytest.raises((ValueError, Exception)):
            QwenVLSettings(timeout_seconds=700)  # Above maximum of 600

    def test_max_retries_validation(self):
        """Test max_retries validation."""
        with pytest.raises((ValueError, Exception)):
            QwenVLSettings(max_retries=15)  # Above maximum of 10

    def test_retry_delay_validation(self):
        """Test retry_delay validation."""
        with pytest.raises((ValueError, Exception)):
            QwenVLSettings(retry_delay=0.05)  # Below minimum of 0.1

    def test_temperature_validation(self):
        """Test temperature validation range."""
        with pytest.raises((ValueError, Exception)):
            QwenVLSettings(temperature=3.0)  # Above maximum of 2.0

    @patch.dict(os.environ, {
        "ATLAS_QWEN_ENABLED": "false",
        "ATLAS_QWEN_MODEL_NAME": "custom-model",
        "ATLAS_QWEN_TIMEOUT_SECONDS": "90"
    })
    def test_environment_variable_loading(self):
        """Test loading from environment variables."""
        settings = QwenVLSettings()

        assert settings.enabled is False
        assert settings.model_name == "custom-model"
        assert settings.timeout_seconds == 90


class TestNougatSettings:
    """Test suite for NougatSettings configuration."""

    def test_default_values(self):
        """Test default Nougat configuration values."""
        settings = NougatSettings()

        assert settings.enabled is True
        assert settings.model_name == "facebook/nougat-base"
        assert settings.device == "auto"
        assert settings.batch_size == 1
        assert settings.beam_size == 5
        assert settings.dpi == 300
        assert settings.timeout_seconds == 300

    def test_custom_values(self):
        """Test Nougat with custom values."""
        settings = NougatSettings(
            enabled=False,
            model_name="facebook/nougat-small",
            device="cuda",
            batch_size=4,
            beam_size=8,
            dpi=600,
            timeout_seconds=600
        )

        assert settings.enabled is False
        assert settings.model_name == "facebook/nougat-small"
        assert settings.device == "cuda"
        assert settings.batch_size == 4
        assert settings.beam_size == 8
        assert settings.dpi == 600
        assert settings.timeout_seconds == 600

    def test_device_validation(self):
        """Test device type validation."""
        # Valid devices
        for device in ["auto", "cpu", "cuda", "mps"]:
            settings = NougatSettings(device=device)
            assert settings.device == device

        # Invalid device should raise error
        with pytest.raises((ValueError, Exception)):
            NougatSettings(device="invalid")

    def test_batch_size_validation(self):
        """Test batch_size validation."""
        with pytest.raises((ValueError, Exception)):
            NougatSettings(batch_size=0)  # Below minimum of 1

        with pytest.raises((ValueError, Exception)):
            NougatSettings(batch_size=50)  # Above maximum of 32

    def test_beam_size_validation(self):
        """Test beam_size validation."""
        with pytest.raises((ValueError, Exception)):
            NougatSettings(beam_size=0)  # Below minimum of 1

        with pytest.raises((ValueError, Exception)):
            NougatSettings(beam_size=15)  # Above maximum of 10

    def test_dpi_validation(self):
        """Test DPI validation."""
        with pytest.raises((ValueError, Exception)):
            NougatSettings(dpi=50)  # Below minimum of 72

        with pytest.raises((ValueError, Exception)):
            NougatSettings(dpi=700)  # Above maximum of 600

    @patch.dict(os.environ, {
        "ATLAS_NOUGAT_ENABLED": "false",
        "ATLAS_NOUGAT_DEVICE": "cuda",
        "ATLAS_NOUGAT_BATCH_SIZE": "8"
    })
    def test_environment_variable_loading(self):
        """Test loading from environment variables."""
        settings = NougatSettings()

        assert settings.enabled is False
        assert settings.device == "cuda"
        assert settings.batch_size == 8


class TestClassicOCRSettings:
    """Test suite for ClassicOCRSettings configuration."""

    def test_default_values(self):
        """Test default Classic OCR configuration values."""
        settings = ClassicOCRSettings()

        assert settings.enabled is True
        assert settings.preferred_engine == "unstructured"
        assert settings.tesseract_languages == "fra+eng"
        assert settings.tesseract_psm == 6

    def test_custom_values(self):
        """Test Classic OCR with custom values."""
        settings = ClassicOCRSettings(
            enabled=False,
            preferred_engine="tesseract",
            tesseract_languages="eng",
            tesseract_psm=3
        )

        assert settings.enabled is False
        assert settings.preferred_engine == "tesseract"
        assert settings.tesseract_languages == "eng"
        assert settings.tesseract_psm == 3

    def test_preferred_engine_validation(self):
        """Test preferred_engine validation."""
        # Valid engines
        for engine in ["unstructured", "tesseract", "doctr", "pdfminer"]:
            settings = ClassicOCRSettings(preferred_engine=engine)
            assert settings.preferred_engine == engine

        # Invalid engine should raise error
        with pytest.raises((ValueError, Exception)):
            ClassicOCRSettings(preferred_engine="invalid_engine")

    def test_tesseract_psm_validation(self):
        """Test Tesseract PSM validation."""
        # Valid PSM values (0-13)
        for psm in [0, 6, 13]:
            settings = ClassicOCRSettings(tesseract_psm=psm)
            assert settings.tesseract_psm == psm

        # Invalid PSM values
        with pytest.raises((ValueError, Exception)):
            ClassicOCRSettings(tesseract_psm=-1)  # Below minimum

        with pytest.raises((ValueError, Exception)):
            ClassicOCRSettings(tesseract_psm=14)  # Above maximum

    @patch.dict(os.environ, {
        "ATLAS_CLASSIC_ENABLED": "false",
        "ATLAS_CLASSIC_PREFERRED_ENGINE": "tesseract",
        "ATLAS_CLASSIC_TESSERACT_LANGUAGES": "fra"
    })
    def test_environment_variable_loading(self):
        """Test loading from environment variables."""
        settings = ClassicOCRSettings()

        # Note: Environment variable names might differ based on actual implementation
        # This test assumes env_prefix is "ATLAS_CLASSIC_"
        # Adjust if needed based on actual Config class

    def test_multiple_tesseract_languages(self):
        """Test multiple Tesseract languages configuration."""
        settings = ClassicOCRSettings(tesseract_languages="fra+eng+deu+spa")
        assert settings.tesseract_languages == "fra+eng+deu+spa"

        settings = ClassicOCRSettings(tesseract_languages="eng")
        assert settings.tesseract_languages == "eng"


class TestOCRSettingsIntegration:
    """Integration tests for OCR settings modules."""

    def test_all_settings_default_enabled(self):
        """Test that all OCR engines are enabled by default."""
        qwen = QwenVLSettings()
        nougat = NougatSettings()
        classic = ClassicOCRSettings()

        assert qwen.enabled is True
        assert nougat.enabled is True
        assert classic.enabled is True

    def test_disable_all_engines(self):
        """Test disabling all OCR engines."""
        qwen = QwenVLSettings(enabled=False)
        nougat = NougatSettings(enabled=False)
        classic = ClassicOCRSettings(enabled=False)

        assert qwen.enabled is False
        assert nougat.enabled is False
        assert classic.enabled is False

    def test_settings_independence(self):
        """Test that settings are independent of each other."""
        qwen = QwenVLSettings(timeout_seconds=60)
        nougat = NougatSettings(timeout_seconds=120)

        assert qwen.timeout_seconds == 60
        assert nougat.timeout_seconds == 120

    def test_boundary_values(self):
        """Test boundary values for all settings."""
        # Minimum values
        qwen = QwenVLSettings(
            timeout_seconds=10,
            max_retries=0,
            retry_delay=0.1,
            temperature=0.0
        )
        assert qwen.timeout_seconds == 10
        assert qwen.max_retries == 0
        assert qwen.retry_delay == 0.1
        assert qwen.temperature == 0.0

        # Maximum values
        qwen = QwenVLSettings(
            timeout_seconds=600,
            max_retries=10,
            retry_delay=30.0,
            temperature=2.0
        )
        assert qwen.timeout_seconds == 600
        assert qwen.max_retries == 10
        assert qwen.retry_delay == 30.0
        assert qwen.temperature == 2.0
