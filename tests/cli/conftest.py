"""Pytest configuration and fixtures for CLI tests."""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest


class RealCliResult:
    """Result from a real CLI invocation."""

    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.output = stdout  # Alias for compatibility


class RealCliRunner:
    """
    Real CLI Runner that executes commands as subprocess.

    This avoids the Typer CliRunner issues with closed file streams.
    """

    def invoke(self, app, args: List[str], **kwargs) -> RealCliResult:
        """
        Invoke CLI command as subprocess.

        Args:
            app: Typer app (ignored, we use atlas-rag command directly)
            args: Command arguments (e.g., ["chunk", "file.txt"])
            **kwargs: Additional options (ignored)

        Returns:
            RealCliResult with stdout, stderr, exit_code
        """
        # Build command: python -m src.core.cli.app <args>
        cmd = [sys.executable, "-m", "src.core.cli.app"] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent.parent  # Project root
            )

            return RealCliResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode
            )

        except subprocess.TimeoutExpired:
            return RealCliResult(
                stdout="",
                stderr="Command timeout",
                exit_code=124
            )
        except Exception as e:
            return RealCliResult(
                stdout="",
                stderr=str(e),
                exit_code=1
            )


@pytest.fixture
def cli_runner():
    """Provide a real CLI test runner using subprocess."""
    return RealCliRunner()


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    file_path = tmp_path / "sample.txt"
    content = """This is a test document for Atlas-RAG.

It contains multiple paragraphs to test the chunking functionality.

The chunking engine should split this into meaningful segments.
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_batch_files(tmp_path):
    """Create multiple sample files for batch testing."""
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()

    # Create 3 sample files with sufficient content for quality checks
    files = []
    for i in range(1, 4):
        file_path = batch_dir / f"doc{i}.txt"
        # Content must be > 50 chars to pass quality check
        content = f"""Document {i} - Test Content for Atlas-RAG

This is the first paragraph of document {i}. It contains enough text to pass the quality validation checks that require chunks to be at least 50 characters long.

This is the second paragraph providing additional context and information for proper chunking and testing purposes.

The document includes multiple paragraphs to ensure proper semantic chunking and quality validation."""
        file_path.write_text(content)
        files.append(file_path)

    return batch_dir, files


@pytest.fixture
def empty_file(tmp_path):
    """Create an empty file for testing edge cases."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    return file_path
