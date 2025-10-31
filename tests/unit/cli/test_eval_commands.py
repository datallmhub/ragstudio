"""Unit tests for eval command."""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from click.exceptions import Exit as ClickExit
import typer

from src.core.cli.commands.eval import (
    eval_command,
    calculate_metrics,
    display_single_evaluation,
    display_comparison,
    provide_recommendations,
    provide_comparison_recommendations
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_normal_chunks(self):
        """Test metrics calculation with normal chunks."""
        chunks = [
            {"text": "A" * 100},
            {"text": "B" * 200},
            {"text": "C" * 300},
        ]

        metrics = calculate_metrics(chunks)

        assert metrics["total_chunks"] == 3
        assert metrics["avg_size"] == 200.0
        assert metrics["min_size"] == 100
        assert metrics["max_size"] == 300
        assert metrics["std_dev"] > 0
        assert 0 <= metrics["consistency_score"] <= 1
        assert metrics["distribution"]["empty"] == 0

    def test_calculate_metrics_with_empty_chunks(self):
        """Test metrics calculation with empty chunks."""
        chunks = [
            {"text": "A" * 100},
            {"text": ""},
            {"text": "C" * 300},
        ]

        metrics = calculate_metrics(chunks)

        assert metrics["total_chunks"] == 3
        assert metrics["distribution"]["empty"] == 1

    def test_calculate_metrics_size_distribution(self):
        """Test size distribution calculation."""
        chunks = [
            {"text": "A" * 50},   # small
            {"text": "B" * 150},  # medium
            {"text": "C" * 600},  # large
        ]

        metrics = calculate_metrics(chunks)

        assert metrics["distribution"]["small"] == 1
        assert metrics["distribution"]["medium"] == 1
        assert metrics["distribution"]["large"] == 1

    def test_calculate_metrics_single_chunk(self):
        """Test metrics with single chunk."""
        chunks = [{"text": "A" * 100}]

        metrics = calculate_metrics(chunks)

        assert metrics["total_chunks"] == 1
        assert metrics["avg_size"] == 100.0
        assert metrics["min_size"] == 100
        assert metrics["max_size"] == 100
        assert metrics["std_dev"] == 0.0

    def test_calculate_metrics_all_same_size(self):
        """Test metrics with uniform chunk sizes."""
        chunks = [{"text": "A" * 100} for _ in range(5)]

        metrics = calculate_metrics(chunks)

        # All same size = perfect consistency
        assert metrics["consistency_score"] == 1.0
        assert metrics["std_dev"] == 0.0


class TestEvalCommand:
    """Tests for eval_command function."""

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_single_file_valid(self, mock_validate, tmp_path):
        """Test evaluation of single valid file."""
        # Create test file
        test_file = tmp_path / "chunks.json"
        chunks = [{"text": "A" * 100}, {"text": "B" * 200}]
        test_file.write_text(json.dumps(chunks))

        # Execute
        eval_command(chunks_files=[test_file])

        # File should be validated
        mock_validate.assert_called_once()

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_invalid_format(self, mock_validate, tmp_path):
        """Test evaluation with invalid JSON format (not an array)."""
        # Create test file with dict instead of array
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps({"data": "not an array"}))

        # Should handle gracefully and exit
        with pytest.raises((ClickExit, typer.Exit)):
            eval_command(chunks_files=[test_file])

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_empty_chunks_array(self, mock_validate, tmp_path):
        """Test evaluation with empty chunks array."""
        # Create test file with empty array
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([]))

        # Should handle gracefully and exit
        with pytest.raises((ClickExit, typer.Exit)):
            eval_command(chunks_files=[test_file])

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_file_load_error(self, mock_validate, tmp_path):
        """Test evaluation when file loading fails."""
        # Create invalid JSON file
        test_file = tmp_path / "chunks.json"
        test_file.write_text("invalid json content {]")

        # Should handle exception and exit
        with pytest.raises((ClickExit, typer.Exit)):
            eval_command(chunks_files=[test_file])

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_compare_multiple_files(self, mock_validate, tmp_path):
        """Test comparison of multiple chunking files."""
        # Create multiple test files
        file1 = tmp_path / "semantic.json"
        file1.write_text(json.dumps([{"text": "A" * 100}, {"text": "B" * 150}]))

        file2 = tmp_path / "sentence.json"
        file2.write_text(json.dumps([{"text": "C" * 200}, {"text": "D" * 250}]))

        # Execute with compare flag
        eval_command(chunks_files=[file1, file2], compare=True)

        # Should validate both files
        assert mock_validate.call_count == 2

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_save_report(self, mock_validate, tmp_path):
        """Test saving evaluation report."""
        # Create test chunks file
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(json.dumps([{"text": "A" * 100}]))

        # Report file
        report_file = tmp_path / "report.json"

        # Execute with report flag
        eval_command(chunks_files=[chunks_file], report=report_file)

        # Report should be created
        assert report_file.exists()
        report_data = json.loads(report_file.read_text())
        assert "files" in report_data
        assert "metrics" in report_data
        assert "comparison" in report_data

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_with_details(self, mock_validate, tmp_path):
        """Test evaluation with details flag."""
        # Create test file
        test_file = tmp_path / "chunks.json"
        chunks = [{"text": "A" * 100}, {"text": "B" * 200}]
        test_file.write_text(json.dumps(chunks))

        # Execute with details flag
        eval_command(chunks_files=[test_file], show_details=True)

        # Should complete without error
        mock_validate.assert_called_once()

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_partial_file_failures(self, mock_validate, tmp_path):
        """Test evaluation when some files fail to load."""
        # Create one valid and one invalid file
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(json.dumps([{"text": "A" * 100}]))

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json {]")

        # Should process valid file and skip invalid
        eval_command(chunks_files=[valid_file, invalid_file])

        # Should validate both files
        assert mock_validate.call_count == 2

    @patch('src.core.cli.commands.eval.validate_file_exists')
    def test_eval_report_save_error(self, mock_validate, tmp_path):
        """Test handling of report save error."""
        # Create test chunks file
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(json.dumps([{"text": "A" * 100}]))

        # Invalid report path (directory that doesn't exist)
        report_file = tmp_path / "nonexistent_dir" / "report.json"

        # Should handle error gracefully (no exception)
        eval_command(chunks_files=[chunks_file], report=report_file)


class TestDisplayFunctions:
    """Tests for display and recommendation functions."""

    def test_display_single_evaluation(self):
        """Test display of single file evaluation."""
        metrics = {
            "total_chunks": 10,
            "avg_size": 200.5,
            "min_size": 100,
            "max_size": 300,
            "std_dev": 50.2,
            "consistency_score": 0.75,
            "distribution": {
                "small": 2,
                "medium": 6,
                "large": 2,
                "empty": 0
            }
        }

        # Should not raise any exception
        display_single_evaluation("test.json", metrics, show_details=False)

    def test_display_single_evaluation_with_empty_chunks(self):
        """Test display when empty chunks are present."""
        metrics = {
            "total_chunks": 10,
            "avg_size": 180.0,
            "min_size": 0,
            "max_size": 300,
            "std_dev": 60.0,
            "consistency_score": 0.65,
            "distribution": {
                "small": 2,
                "medium": 5,
                "large": 1,
                "empty": 2  # Empty chunks present
            }
        }

        # Should handle and display warning about empty chunks
        display_single_evaluation("test.json", metrics, show_details=True)

    def test_display_comparison(self):
        """Test comparison display for multiple files."""
        results = {
            "semantic.json": {
                "total_chunks": 10,
                "avg_size": 200.0,
                "std_dev": 30.0,
                "consistency_score": 0.85  # 3 stars
            },
            "sentence.json": {
                "total_chunks": 15,
                "avg_size": 150.0,
                "std_dev": 50.0,
                "consistency_score": 0.65  # 2 stars
            },
            "token.json": {
                "total_chunks": 20,
                "avg_size": 100.0,
                "std_dev": 80.0,
                "consistency_score": 0.45  # 1 star
            }
        }

        # Should display comparison table without error
        display_comparison(results)

    def test_provide_recommendations_good_metrics(self):
        """Test recommendations with good metrics."""
        metrics = {
            "consistency_score": 0.8,  # Good
            "avg_size": 300,  # Good (100-1000)
            "total_chunks": 10,
            "distribution": {
                "small": 1,
                "medium": 8,
                "large": 1,
                "empty": 0
            }
        }

        # Should provide positive recommendations
        provide_recommendations(metrics)

    def test_provide_recommendations_poor_consistency(self):
        """Test recommendations with poor consistency."""
        metrics = {
            "consistency_score": 0.4,  # Low
            "avg_size": 300,
            "total_chunks": 10,
            "distribution": {
                "small": 2,
                "medium": 5,
                "large": 3,
                "empty": 0
            }
        }

        # Should recommend improving consistency
        provide_recommendations(metrics)

    def test_provide_recommendations_small_avg_size(self):
        """Test recommendations with small average size."""
        metrics = {
            "consistency_score": 0.7,
            "avg_size": 50,  # Too small
            "total_chunks": 10,
            "distribution": {
                "small": 8,
                "medium": 2,
                "large": 0,
                "empty": 0
            }
        }

        # Should recommend increasing chunk size
        provide_recommendations(metrics)

    def test_provide_recommendations_large_avg_size(self):
        """Test recommendations with large average size."""
        metrics = {
            "consistency_score": 0.7,
            "avg_size": 1500,  # Too large
            "total_chunks": 10,
            "distribution": {
                "small": 0,
                "medium": 2,
                "large": 8,
                "empty": 0
            }
        }

        # Should recommend decreasing chunk size
        provide_recommendations(metrics)

    def test_provide_recommendations_with_empty_chunks(self):
        """Test recommendations when empty chunks detected."""
        metrics = {
            "consistency_score": 0.7,
            "avg_size": 300,
            "total_chunks": 10,
            "distribution": {
                "small": 2,
                "medium": 5,
                "large": 1,
                "empty": 2  # Empty chunks
            }
        }

        # Should warn about empty chunks
        provide_recommendations(metrics)

    def test_provide_recommendations_many_small_chunks(self):
        """Test recommendations when many small chunks."""
        metrics = {
            "consistency_score": 0.7,
            "avg_size": 300,
            "total_chunks": 10,
            "distribution": {
                "small": 5,  # 50% small chunks
                "medium": 3,
                "large": 2,
                "empty": 0
            }
        }

        # Should warn about many small chunks
        provide_recommendations(metrics)

    def test_provide_comparison_recommendations(self):
        """Test comparison recommendations."""
        results = {
            "semantic.json": {
                "consistency_score": 0.85,
                "avg_size": 400.0  # Perfect
            },
            "sentence.json": {
                "consistency_score": 0.65,
                "avg_size": 200.0
            },
            "token.json": {
                "consistency_score": 0.75,
                "avg_size": 600.0
            }
        }

        # Should identify best strategies
        provide_comparison_recommendations(results)
