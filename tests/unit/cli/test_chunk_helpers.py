"""Unit tests for chunk_helpers - SOLID refactored functions."""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.core.cli.commands.chunk_helpers import (
    generate_processing_summary,
    display_routing_decisions,
    load_document_universal
)


class TestGenerateProcessingSummary:
    """Tests for generate_processing_summary function (SRP: summary generation)."""

    def test_generate_summary_basic(self, tmp_path):
        """Test basic summary generation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Mock config
        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        # Mock chunks
        chunk1 = Mock()
        chunk1.text = "A" * 100
        chunks = [chunk1]

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data={},
            chunks=chunks
        )

        # Verify structure
        assert "metadata" in summary
        assert "document" in summary
        assert "configuration" in summary
        assert "processing" in summary
        assert "results" in summary

        # Verify metadata
        assert summary["metadata"]["success"] is True
        assert summary["metadata"]["atlas_rag_version"] == "1.0.0"

        # Verify document
        assert summary["document"]["filename"] == "test.txt"
        assert summary["document"]["format"] == ".txt"

        # Verify chunks
        assert summary["results"]["chunks"]["total_count"] == 1

    def test_generate_summary_with_errors(self, tmp_path):
        """Test summary generation with errors."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        errors = ["Error 1", "Error 2"]
        summary = generate_processing_summary(
            test_file,
            config,
            processing_data={},
            chunks=[],
            success=False,
            errors=errors
        )

        assert summary["metadata"]["success"] is False
        assert summary["metadata"]["errors"] == errors

    def test_generate_summary_with_llm_config(self, tmp_path):
        """Test summary with LLM configuration."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        config = Mock()
        config.llm.use_llm = True
        config.llm.provider = "openai"
        config.llm.model = "gpt-4"
        config.llm.is_local = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data={},
            chunks=[]
        )

        assert summary["configuration"]["llm"]["enabled"] is True
        assert summary["configuration"]["llm"]["provider"] == "openai"
        assert summary["configuration"]["llm"]["model"] == "gpt-4"

    def test_generate_summary_with_ocr_data(self, tmp_path):
        """Test summary with OCR processing data."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("pdf content")

        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = True
        config.ocr.dictionary_threshold = 0.7
        config.ocr.dynamic_threshold = True
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        ocr_result = {
            "metadata": {
                "ocr_engine": "Qwen-VL",
                "success": True,
                "quality_metrics": {"confidence": 0.95}
            },
            "routing_decisions": ["decision1", "decision2"]
        }

        processing_data = {
            "ocr_result": ocr_result,
            "ocr_time": 2.5
        }

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data=processing_data,
            chunks=[]
        )

        assert "ocr" in summary["processing"]["stages"]
        assert summary["processing"]["stages"]["ocr"]["engine"] == "Qwen-VL"
        assert summary["processing"]["stages"]["ocr"]["success"] is True
        assert summary["processing"]["stages"]["ocr"]["time_seconds"] == 2.5

    def test_generate_summary_chunk_statistics(self, tmp_path):
        """Test chunk statistics calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        # Create chunks with different sizes
        chunk1 = Mock()
        chunk1.text = "A" * 50

        chunk2 = Mock()
        chunk2.text = "B" * 100

        chunk3 = Mock()
        chunk3.text = "C" * 200

        chunks = [chunk1, chunk2, chunk3]

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data={},
            chunks=chunks
        )

        chunk_stats = summary["results"]["chunks"]
        assert chunk_stats["total_count"] == 3
        assert chunk_stats["min_size_chars"] == 50
        assert chunk_stats["max_size_chars"] == 200
        assert chunk_stats["total_text_length"] == 350
        assert chunk_stats["average_size_chars"] == 116  # 350 // 3

    def test_generate_summary_empty_chunks(self, tmp_path):
        """Test summary with no chunks."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data={},
            chunks=[]
        )

        chunk_stats = summary["results"]["chunks"]
        assert chunk_stats["total_count"] == 0
        assert chunk_stats["average_size_chars"] == 0
        assert chunk_stats["min_size_chars"] == 0
        assert chunk_stats["max_size_chars"] == 0

    def test_generate_summary_with_strategy_selection(self, tmp_path):
        """Test summary with strategy selection data."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        processing_data = {
            "strategy_selection": {
                "selected": "semantic",
                "reason": "Best for this document"
            }
        }

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data=processing_data,
            chunks=[]
        )

        assert "strategy_selection" in summary["processing"]["stages"]
        assert summary["processing"]["stages"]["strategy_selection"]["selected"] == "semantic"

    def test_generate_summary_with_chunking_time(self, tmp_path):
        """Test summary with chunking timing data."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        config = Mock()
        config.llm.use_llm = False
        config.ocr.use_advanced_ocr = False
        config.ocr.dictionary_threshold = 0.5
        config.ocr.dynamic_threshold = False
        config.ocr.enable_fallback = True
        config.chunking.strategy = "semantic"
        config.chunking.max_tokens = 400
        config.chunking.overlap = 50

        processing_data = {
            "chunking_time": 1.23
        }

        summary = generate_processing_summary(
            test_file,
            config,
            processing_data=processing_data,
            chunks=[]
        )

        assert "chunking" in summary["processing"]["stages"]
        assert summary["processing"]["stages"]["chunking"]["time_seconds"] == 1.23


class TestDisplayRoutingDecisions:
    """Tests for display_routing_decisions function (SRP: display OCR routing)."""

    def test_display_no_routing_decisions(self):
        """Test with no routing decisions."""
        console_mock = MagicMock()
        result = {}

        display_routing_decisions(result, console_mock)

        # Should not print anything
        console_mock.print.assert_not_called()

    def test_display_ocr_quality_detection_high(self):
        """Test OCR quality detection with HIGH quality."""
        console_mock = MagicMock()
        result = {
            "routing_decisions": [
                {
                    "step": "ocr_quality_detection",
                    "ocr_quality_category": "HIGH",
                    "ocr_quality_score": 0.95,
                    "recommended_engine": "classic"
                }
            ]
        }

        display_routing_decisions(result, console_mock)

        # Should print header and decision
        assert console_mock.print.call_count >= 2

    def test_display_scientific_detection_true(self):
        """Test scientific content detection with high math density."""
        console_mock = MagicMock()
        result = {
            "routing_decisions": [
                {
                    "step": "scientific_detection",
                    "is_scientific": True,
                    "math_density": 0.75,
                    "recommended_engine": "Nougat"
                }
            ]
        }

        display_routing_decisions(result, console_mock)

        assert console_mock.print.call_count >= 2

    def test_display_complexity_analysis_high(self):
        """Test complexity analysis with HIGH complexity."""
        console_mock = MagicMock()
        result = {
            "routing_decisions": [
                {
                    "step": "complexity_analysis",
                    "complexity_score": 0.8,
                    "recommended_strategy": "Qwen-VL"
                }
            ]
        }

        display_routing_decisions(result, console_mock)

        assert console_mock.print.call_count >= 2

    def test_display_ocr_routing_qwen(self):
        """Test OCR routing with Qwen engine."""
        console_mock = MagicMock()
        result = {
            "routing_decisions": [
                {
                    "step": "ocr_routing",
                    "engine_used": "Qwen-VL",
                    "routing_reason": "High complexity document"
                }
            ]
        }

        display_routing_decisions(result, console_mock)

        assert console_mock.print.call_count >= 2

    def test_display_fallback(self):
        """Test fallback display."""
        console_mock = MagicMock()
        result = {
            "routing_decisions": [
                {
                    "step": "fallback",
                    "engine_used": "classic",
                    "reason": "Primary engine unavailable"
                }
            ]
        }

        display_routing_decisions(result, console_mock)

        assert console_mock.print.call_count >= 2

    def test_display_multiple_decisions(self):
        """Test multiple routing decisions."""
        console_mock = MagicMock()
        result = {
            "routing_decisions": [
                {
                    "step": "ocr_quality_detection",
                    "ocr_quality_category": "LOW",
                    "ocr_quality_score": 0.2,
                    "recommended_engine": "Qwen-VL"
                },
                {
                    "step": "ocr_routing",
                    "engine_used": "Qwen-VL",
                    "routing_reason": "Low quality requires advanced OCR"
                }
            ]
        }

        display_routing_decisions(result, console_mock)

        # Should print header + decisions
        assert console_mock.print.call_count >= 3


class TestLoadDocumentUniversal:
    """Tests for load_document_universal function (SRP: document loading)."""

    def test_load_text_file_without_status(self, tmp_path):
        """Test loading simple text file without status."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world", encoding='utf-8')

        print_info_mock = Mock()

        text = load_document_universal(test_file, print_info_mock, use_status=False)

        assert text == "Hello world"

    def test_load_markdown_file_with_status(self, tmp_path):
        """Test loading markdown file with status."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\nContent", encoding='utf-8')

        print_info_mock = Mock()
        console_mock = MagicMock()

        text = load_document_universal(test_file, print_info_mock, use_status=True, console=console_mock)

        assert text == "# Header\nContent"
        console_mock.status.assert_called_once()

    @patch('src.workflows.ingest.loader.ingest_file')
    def test_load_pdf_file_success(self, mock_ingest, tmp_path):
        """Test loading PDF file successfully."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("pdf content")

        # Mock document with text
        mock_doc = Mock()
        mock_doc.text = "Extracted PDF text"
        mock_ingest.return_value = mock_doc

        print_info_mock = Mock()
        console_mock = MagicMock()

        text = load_document_universal(test_file, print_info_mock, use_status=True, console=console_mock)

        assert text == "Extracted PDF text"
        mock_ingest.assert_called_once()

    @patch('src.workflows.ingest.loader.ingest_file')
    def test_load_unsupported_format(self, mock_ingest, tmp_path):
        """Test loading unsupported format."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("xyz content")

        mock_doc = Mock()
        mock_doc.text = "Some text"
        mock_ingest.return_value = mock_doc

        print_info_mock = Mock()

        text = load_document_universal(test_file, print_info_mock, use_status=False)

        # Should call print_info about unsupported format
        print_info_mock.assert_called()

    @patch('src.workflows.ingest.loader.ingest_file')
    def test_load_pdf_no_text_extracted(self, mock_ingest, tmp_path):
        """Test loading PDF with no text extracted."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("pdf content")

        mock_doc = Mock()
        mock_doc.text = ""
        mock_ingest.return_value = mock_doc

        print_info_mock = Mock()

        # ValueError is caught and re-raised as RuntimeError
        with pytest.raises(RuntimeError, match="Failed to load document"):
            load_document_universal(test_file, print_info_mock, use_status=False)

    @patch('src.workflows.ingest.loader.ingest_file')
    def test_load_pdf_import_error_fallback(self, mock_ingest, tmp_path):
        """Test fallback to text reading when import fails."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("fallback content", encoding='utf-8')

        mock_ingest.side_effect = ImportError("Module not found")

        print_info_mock = Mock()

        text = load_document_universal(test_file, print_info_mock, use_status=False)

        assert text == "fallback content"

    @patch('src.workflows.ingest.loader.ingest_file')
    def test_load_pdf_generic_error(self, mock_ingest, tmp_path):
        """Test generic error during PDF loading."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("pdf content")

        mock_ingest.side_effect = Exception("Processing error")

        print_info_mock = Mock()

        with pytest.raises(RuntimeError, match="Failed to load document"):
            load_document_universal(test_file, print_info_mock, use_status=False)
