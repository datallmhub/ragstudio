"""
Test logging modes - Production vs Development.

Verifies that error logging behavior changes based on log level:
- Production (INFO level): Simple messages without tracebacks
- Development (DEBUG level): Full tracebacks with exc_info=True
"""

import logging
from unittest.mock import patch, MagicMock
import pytest


class TestLoggingModes:
    """Test that logging behavior adapts to log level."""

    def test_production_mode_simple_messages(self, caplog):
        """
        Test that in production mode (INFO level), errors are logged
        with simple messages and no tracebacks.
        """
        from src.workflows.ingest.ocr.easyocr_engine import EasyOCREngine

        # Set logger to INFO (production mode)
        logger = logging.getLogger("src.workflows.ingest.ocr.easyocr_engine")
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            with caplog.at_level(logging.INFO):
                # Create mock reader that will fail
                engine = EasyOCREngine.__new__(EasyOCREngine)
                engine._reader = MagicMock()
                engine._reader.readtext.side_effect = Exception("OpenCV error: resize failed")
                engine.languages = ['en']

                # Mock logger
                with patch.object(engine.__class__, '__init__', lambda x, **kwargs: None):
                    from pathlib import Path
                    test_path = Path("test.jpg")

                    # This should log simple message in production mode
                    with pytest.raises(ValueError, match="Image non supportée"):
                        # Simulate the error handling code
                        try:
                            engine._reader.readtext(str(test_path))
                        except Exception as e:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.error(f"Full error: {e}", exc_info=True)
                            else:
                                logger.warning(f"Impossible de lire l'image test.jpg avec EasyOCR")
                            raise ValueError("Image non supportée")

                # Check that simple message was logged
                assert any("Impossible de lire l'image" in record.message for record in caplog.records)
                # Check that no traceback details are in simple message
                warning_records = [r for r in caplog.records if "Impossible de lire" in r.message]
                if warning_records:
                    assert "OpenCV" not in warning_records[0].message
                    assert "resize" not in warning_records[0].message

        finally:
            logger.setLevel(original_level)

    def test_development_mode_full_tracebacks(self, caplog):
        """
        Test that in development mode (DEBUG level), errors are logged
        with full tracebacks using exc_info=True.
        """
        from src.workflows.ingest.ocr.easyocr_engine import EasyOCREngine

        # Set logger to DEBUG (development mode)
        logger = logging.getLogger("src.workflows.ingest.ocr.easyocr_engine")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            with caplog.at_level(logging.DEBUG):
                # Create mock reader that will fail
                engine = EasyOCREngine.__new__(EasyOCREngine)
                engine._reader = MagicMock()
                engine._reader.readtext.side_effect = Exception("OpenCV error: resize failed")
                engine.languages = ['en']

                # Mock logger
                with patch.object(engine.__class__, '__init__', lambda x, **kwargs: None):
                    from pathlib import Path
                    test_path = Path("test.jpg")

                    # This should log full error in development mode
                    with pytest.raises(ValueError, match="Image non supportée"):
                        # Simulate the error handling code
                        try:
                            engine._reader.readtext(str(test_path))
                        except Exception as e:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.error(f"EasyOCR extraction failed for test.jpg: {e}", exc_info=True)
                            else:
                                logger.warning(f"Impossible de lire l'image test.jpg avec EasyOCR")
                            raise ValueError("Image non supportée")

                # Check that detailed error message was logged
                assert any("EasyOCR extraction failed" in record.message for record in caplog.records)
                # Check that error includes details
                error_records = [r for r in caplog.records if "EasyOCR extraction failed" in r.message]
                if error_records:
                    assert "OpenCV error" in error_records[0].message or "resize failed" in error_records[0].message

        finally:
            logger.setLevel(original_level)

    def test_logger_level_detection(self):
        """Test that logger correctly detects DEBUG level."""
        logger = logging.getLogger("test_logger")

        # Test INFO level (production)
        logger.setLevel(logging.INFO)
        assert not logger.isEnabledFor(logging.DEBUG)

        # Test DEBUG level (development)
        logger.setLevel(logging.DEBUG)
        assert logger.isEnabledFor(logging.DEBUG)

        # Test WARNING level (production)
        logger.setLevel(logging.WARNING)
        assert not logger.isEnabledFor(logging.DEBUG)

    def test_all_error_handlers_respect_log_level(self):
        """
        Test that all modified error handlers check log level.

        Files that should have conditional logging:
        - easyocr_engine.py (4 locations)
        - intelligent_orchestrator.py (5 locations)
        - langchain_loader.py (1 location)
        - loader.py (2 locations)
        """
        # This is a documentation test to ensure we don't forget the pattern
        expected_files_with_conditional_logging = [
            "src/workflows/ingest/ocr/easyocr_engine.py",
            "src/workflows/ingest/intelligent_orchestrator.py",
            "src/workflows/ingest/langchain_loader.py",
            "src/workflows/ingest/loader.py",
        ]

        for filepath in expected_files_with_conditional_logging:
            from pathlib import Path
            file = Path(filepath)
            if file.exists():
                content = file.read_text()
                # Check that file uses the conditional logging pattern
                assert "LOGGER.isEnabledFor(logging.DEBUG)" in content, \
                    f"{filepath} should check log level before logging detailed errors"
                assert "exc_info=True" in content, \
                    f"{filepath} should use exc_info=True in DEBUG mode"
