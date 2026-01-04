"""Tests for the batch command."""
import json
import pytest
from pathlib import Path

from src.core.cli.app import app


class TestBatchCommand:
    """Test cases for ragctl batch command."""

    def test_batch_basic(self, cli_runner, sample_batch_files):
        """Test basic batch processing."""
        batch_dir, files = sample_batch_files

        result = cli_runner.invoke(app, ["batch", str(batch_dir)])

        assert result.exit_code == 0
        assert "All files processed successfully" in result.stdout or "Batch Summary" in result.stdout
        assert "3" in result.stdout  # Should show 3 files processed

    def test_batch_with_output(self, cli_runner, sample_batch_files, tmp_path):
        """Test batch processing with output file."""
        batch_dir, files = sample_batch_files
        output_file = tmp_path / "batch_output.json"

        result = cli_runner.invoke(app, [
            "batch", str(batch_dir),
            "-o", str(output_file),
            "--single-file",
            "--auto-continue"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) >= 3  # At least one chunk per file

    def test_batch_with_pattern(self, cli_runner, tmp_path):
        """Test batch processing with file pattern."""
        # Create files with different extensions
        batch_dir = tmp_path / "mixed"
        batch_dir.mkdir()

        # Create files with sufficient content
        (batch_dir / "doc1.txt").write_text("Text file 1 with enough content to pass quality validation checks. This text is longer than 50 characters.")
        (batch_dir / "doc2.txt").write_text("Text file 2 with enough content to pass quality validation checks. This text is longer than 50 characters.")
        (batch_dir / "doc3.md").write_text("Markdown file with enough content to pass quality validation checks. This text is longer than 50 characters.")

        # Process only .txt files
        result = cli_runner.invoke(app, [
            "batch", str(batch_dir),
            "--pattern", "*.txt",
            "--auto-continue"
        ])

        assert result.exit_code == 0
        assert "2" in result.stdout  # Should process 2 files

    def test_batch_recursive(self, cli_runner, tmp_path):
        """Test batch processing with recursive option."""
        # Create nested directory structure
        root_dir = tmp_path / "root"
        root_dir.mkdir()
        (root_dir / "doc1.txt").write_text("Root document with enough content to pass quality validation checks. This text is longer than 50 characters.")

        sub_dir = root_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "doc2.txt").write_text("Sub document with enough content to pass quality validation checks. This text is longer than 50 characters.")

        # Non-recursive (should find 1 file)
        result = cli_runner.invoke(app, [
            "batch", str(root_dir),
            "--pattern", "*.txt",
            "--auto-continue"
        ])

        assert result.exit_code == 0
        assert "1" in result.stdout

        # Recursive (should find 2 files)
        result = cli_runner.invoke(app, [
            "batch", str(root_dir),
            "--pattern", "*.txt",
            "--recursive",
            "--auto-continue"
        ])

        assert result.exit_code == 0
        assert "2" in result.stdout

    def test_batch_with_strategy(self, cli_runner, sample_batch_files):
        """Test batch processing with different strategies."""
        batch_dir, files = sample_batch_files

        strategies = ["semantic", "sentence", "token"]

        for strategy in strategies:
            result = cli_runner.invoke(app, [
                "batch", str(batch_dir),
                "--strategy", strategy
            ])

            assert result.exit_code == 0
            assert strategy in result.stdout.lower()

    def test_batch_no_files_found(self, cli_runner, tmp_path):
        """Test batch processing when no files match pattern."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = cli_runner.invoke(app, [
            "batch", str(empty_dir),
            "--pattern", "*.txt"
        ])

        # Should exit with success (0) but show warning
        assert result.exit_code == 0
        assert "No files found" in result.stdout

    def test_batch_invalid_directory(self, cli_runner):
        """Test batch processing with non-existent directory."""
        result = cli_runner.invoke(app, ["batch", "/nonexistent/dir"])

        assert result.exit_code != 0

    def test_batch_with_max_tokens(self, cli_runner, sample_batch_files):
        """Test batch processing with custom max tokens."""
        batch_dir, files = sample_batch_files

        result = cli_runner.invoke(app, [
            "batch", str(batch_dir),
            "--max-tokens", "300"
        ])

        assert result.exit_code == 0
        assert "300" in result.stdout

    def test_batch_dry_run(self, cli_runner, sample_batch_files):
        """Test batch processing dry-run mode."""
        batch_dir, files = sample_batch_files

        result = cli_runner.invoke(app, [
            "batch", str(batch_dir),
            "--dry-run"
        ])

        assert result.exit_code == 0
        assert "Would process" in result.stdout
        assert "Total:" in result.stdout

    def test_batch_mutually_exclusive_flags(self, cli_runner, tmp_path):
        """Test that --auto-continue and --auto-stop cannot be used together."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        result = cli_runner.invoke(app, [
            "batch", 
            str(input_dir), 
            "--auto-continue", 
            "--auto-stop"
        ])
        
        assert result.exit_code != 0
        assert "Cannot use both --auto-continue and --auto-stop" in result.stdout
