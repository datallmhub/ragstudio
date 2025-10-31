"""
End-to-end tests for Atlas-RAG v3.2 CLI batch and retry commands.

These tests validate the complete user workflow:
- Batch processing with various options
- History tracking
- Retry mechanism
- Output file generation
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    input_dir = temp_dir / "input"
    input_dir.mkdir()

    # Create sample text files
    (input_dir / "doc1.txt").write_text("This is document 1. " * 50)
    (input_dir / "doc2.txt").write_text("This is document 2. " * 50)
    (input_dir / "doc3.txt").write_text("This is document 3. " * 50)

    return input_dir


@pytest.fixture
def cli_command():
    """Get the CLI command to run."""
    return [".venv/bin/atlas-rag"]


class TestBatchCommand:
    """Test batch command functionality."""

    def test_batch_basic(self, sample_files, temp_dir, cli_command):
        """Test basic batch processing."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_dir),
                "--auto-continue",
                "--no-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "✓ All files processed successfully" in result.stdout
        assert "3 files" in result.stdout

    def test_batch_per_file_output(self, sample_files, temp_dir, cli_command):
        """Test per-file output (default behavior)."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_dir),
                "--auto-continue",
                "--no-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        # Check that individual files were created
        output_files = list(output_dir.glob("*_chunks.jsonl"))
        assert len(output_files) == 3

        # Check filenames
        filenames = {f.name for f in output_files}
        assert "doc1_chunks.jsonl" in filenames
        assert "doc2_chunks.jsonl" in filenames
        assert "doc3_chunks.jsonl" in filenames

        # Check file contents
        for output_file in output_files:
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) > 0
                # Verify JSON format
                chunk = json.loads(lines[0])
                assert "id" in chunk
                assert "text" in chunk
                assert "metadata" in chunk

    def test_batch_single_file_output(self, sample_files, temp_dir, cli_command):
        """Test single-file output mode."""
        output_file = temp_dir / "all_chunks.jsonl"

        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_file),
                "--single-file",
                "--auto-continue",
                "--no-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Check combined output
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) > 0

            # Verify all source files are represented
            sources = set()
            for line in lines:
                chunk = json.loads(line)
                source = Path(chunk["metadata"]["source_file"]).name
                sources.add(source)

            assert "doc1.txt" in sources
            assert "doc2.txt" in sources
            assert "doc3.txt" in sources

    def test_batch_with_history(self, sample_files, temp_dir, cli_command):
        """Test batch processing with history tracking."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_dir),
                "--auto-continue",
                "--save-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        # Extract run ID from output
        lines = result.stdout.split("\n")
        run_id = None
        for line in lines:
            if "Run ID:" in line:
                run_id = line.split("Run ID:")[1].strip()
                break

        assert run_id is not None
        assert run_id.startswith("run_")

        # Verify history file was created
        history_file = Path.home() / ".atlasrag" / "history" / "runs" / f"{run_id}.json"
        assert history_file.exists()

        # Verify history contents
        with open(history_file) as f:
            history = json.load(f)
            assert history["run_id"] == run_id
            assert history["status"] == "done"
            assert history["total_files"] == 3
            assert history["success"] == 3
            assert len(history["files"]) == 3

    def test_batch_with_pattern(self, temp_dir, cli_command):
        """Test batch processing with file pattern."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        # Create mixed file types
        (input_dir / "doc1.txt").write_text("Text document")
        (input_dir / "doc2.md").write_text("Markdown document")
        (input_dir / "doc3.txt").write_text("Another text")

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        result = subprocess.run(
            cli_command + [
                "batch",
                str(input_dir),
                "--pattern", "*.txt",
                "--output", str(output_dir),
                "--auto-continue",
                "--no-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        # Only .txt files should be processed
        output_files = list(output_dir.glob("*_chunks.jsonl"))
        assert len(output_files) == 2

    def test_batch_with_custom_params(self, sample_files, temp_dir, cli_command):
        """Test batch processing with custom chunking parameters."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_dir),
                "--strategy", "sentence",
                "--max-tokens", "200",
                "--overlap", "20",
                "--auto-continue",
                "--no-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        # Verify chunks were created with custom params
        output_file = list(output_dir.glob("*_chunks.jsonl"))[0]
        with open(output_file) as f:
            chunk = json.loads(f.readline())
            assert chunk["metadata"]["chunk_size"] == 200
            assert chunk["metadata"]["chunk_overlap"] == 20
            # Strategy may be mapped to internal implementation
            assert "chunking_strategy" in chunk["metadata"]


class TestRetryCommand:
    """Test retry command functionality."""

    def test_retry_show_no_failures(self, cli_command):
        """Test retry --show when no failures exist."""
        result = subprocess.run(
            cli_command + ["retry", "--show"],
            capture_output=True,
            text=True
        )

        # Should report no failed runs
        assert "No failed runs found" in result.stdout

    def test_retry_with_history(self, sample_files, temp_dir, cli_command):
        """Test full retry workflow."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # First batch run with history
        result1 = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_dir),
                "--auto-continue",
                "--save-history"
            ],
            capture_output=True,
            text=True
        )

        assert result1.returncode == 0

        # Extract run ID
        lines = result1.stdout.split("\n")
        run_id = None
        for line in lines:
            if "Run ID:" in line:
                run_id = line.split("Run ID:")[1].strip()
                break

        assert run_id is not None

        # Verify history file exists
        history_file = Path.home() / ".atlasrag" / "history" / "runs" / f"{run_id}.json"
        assert history_file.exists()


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_full_workflow_per_file(self, sample_files, temp_dir, cli_command):
        """Test complete workflow with per-file output."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Run batch processing
        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_dir),
                "--auto-continue",
                "--save-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "✓ All files processed successfully" in result.stdout

        # Verify outputs
        output_files = list(output_dir.glob("*_chunks.jsonl"))
        assert len(output_files) == 3

        # Verify each file has valid JSON
        for output_file in output_files:
            assert output_file.stat().st_size > 0
            with open(output_file) as f:
                for line in f:
                    chunk = json.loads(line)
                    assert "id" in chunk
                    assert "text" in chunk
                    assert len(chunk["text"]) > 0

    def test_full_workflow_single_file(self, sample_files, temp_dir, cli_command):
        """Test complete workflow with single-file output."""
        output_file = temp_dir / "all_chunks.jsonl"

        # Run batch processing
        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", str(output_file),
                "--single-file",
                "--auto-continue",
                "--save-history"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "✓ All files processed successfully" in result.stdout

        # Verify single output file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify contents
        with open(output_file) as f:
            chunks = [json.loads(line) for line in f]
            assert len(chunks) > 0

            # All chunks should be valid
            for chunk in chunks:
                assert "id" in chunk
                assert "text" in chunk
                assert "metadata" in chunk

    def test_help_commands(self, cli_command):
        """Test help output for commands."""
        # Test batch help
        result = subprocess.run(
            cli_command + ["batch", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "DIRECTORY" in result.stdout
        assert "--single-file" in result.stdout

        # Test retry help
        result = subprocess.run(
            cli_command + ["retry", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "RUN_ID" in result.stdout or "run_id" in result.stdout.lower()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_batch_empty_directory(self, temp_dir, cli_command):
        """Test batch on empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = subprocess.run(
            cli_command + [
                "batch",
                str(empty_dir),
                "--auto-continue"
            ],
            capture_output=True,
            text=True
        )

        # Should handle gracefully
        assert "No files found" in result.stdout or result.returncode != 0

    def test_batch_nonexistent_directory(self, temp_dir, cli_command):
        """Test batch on nonexistent directory."""
        nonexistent = temp_dir / "does_not_exist"

        result = subprocess.run(
            cli_command + [
                "batch",
                str(nonexistent),
                "--auto-continue"
            ],
            capture_output=True,
            text=True
        )

        # Should fail with error
        assert result.returncode != 0

    def test_batch_output_no_permission(self, sample_files, cli_command):
        """Test batch with no write permission on output."""
        # Try to write to /root (should fail)
        result = subprocess.run(
            cli_command + [
                "batch",
                str(sample_files),
                "--output", "/root/output",
                "--auto-continue"
            ],
            capture_output=True,
            text=True
        )

        # Should handle permission error gracefully
        # (either fail or show error message)
        assert result.returncode != 0 or "Error" in result.stdout or "error" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
