"""Unit tests for batch_helpers - SOLID refactored functions."""
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.core.cli.commands.batch_helpers import (
    discover_files,
    filter_supported_files,
    validate_files_for_batch,
    SUPPORTED_EXTENSIONS
)


class TestDiscoverFiles:
    """Tests for discover_files function (SRP: file discovery)."""

    def test_discover_files_non_recursive(self, tmp_path):
        """Test non-recursive file discovery."""
        # Create test files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.md").touch()

        # Create subdirectory with file (should not be discovered)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file4.txt").touch()

        files = discover_files(tmp_path, "*.txt", recursive=False)

        assert len(files) == 2
        assert all(f.suffix == ".txt" for f in files)
        assert all(f.parent == tmp_path for f in files)

    def test_discover_files_recursive(self, tmp_path):
        """Test recursive file discovery."""
        # Create test files in root
        (tmp_path / "file1.txt").touch()

        # Create files in subdirectories
        sub1 = tmp_path / "sub1"
        sub1.mkdir()
        (sub1 / "file2.txt").touch()

        sub2 = sub1 / "sub2"
        sub2.mkdir()
        (sub2 / "file3.txt").touch()

        files = discover_files(tmp_path, "*.txt", recursive=True)

        assert len(files) == 3
        assert all(f.suffix == ".txt" for f in files)

    def test_discover_files_with_pattern(self, tmp_path):
        """Test file discovery with specific pattern."""
        (tmp_path / "doc.pdf").touch()
        (tmp_path / "image.jpg").touch()
        (tmp_path / "text.txt").touch()

        pdf_files = discover_files(tmp_path, "*.pdf", recursive=False)
        assert len(pdf_files) == 1
        assert pdf_files[0].suffix == ".pdf"

    def test_discover_files_no_matches(self, tmp_path):
        """Test file discovery with no matches."""
        (tmp_path / "file.txt").touch()

        files = discover_files(tmp_path, "*.pdf", recursive=False)
        assert len(files) == 0


class TestFilterSupportedFiles:
    """Tests for filter_supported_files function (SRP: extension filtering)."""

    def test_filter_all_supported_files(self, tmp_path):
        """Test filtering when all files are supported."""
        files = [
            tmp_path / "doc.pdf",
            tmp_path / "text.txt",
            tmp_path / "image.jpg",
        ]
        # Create actual files
        for f in files:
            f.touch()

        supported, unsupported = filter_supported_files(files)

        assert len(supported) == 3
        assert len(unsupported) == 0

    def test_filter_mixed_files(self, tmp_path):
        """Test filtering with mix of supported and unsupported."""
        files = [
            tmp_path / "doc.pdf",      # supported
            tmp_path / "image.gif",    # unsupported
            tmp_path / "text.txt",     # supported
            tmp_path / "video.mp4",    # unsupported
        ]
        # Create actual files
        for f in files:
            f.touch()

        supported, unsupported = filter_supported_files(files)

        assert len(supported) == 2
        assert len(unsupported) == 2

        # Check unsupported format
        unsupported_names = [name for name, ext in unsupported]
        assert "image.gif" in unsupported_names
        assert "video.mp4" in unsupported_names

    def test_filter_all_unsupported_files(self, tmp_path):
        """Test filtering when all files are unsupported."""
        files = [
            tmp_path / "video.mp4",
            tmp_path / "audio.wav",
        ]
        # Create actual files
        for f in files:
            f.touch()

        supported, unsupported = filter_supported_files(files)

        assert len(supported) == 0
        assert len(unsupported) == 2

    def test_filter_skips_directories(self, tmp_path):
        """Test that directories are skipped."""
        # Create a file and a directory
        file1 = tmp_path / "doc.pdf"
        file1.touch()

        dir1 = tmp_path / "folder"
        dir1.mkdir()

        files = [file1, dir1]

        supported, unsupported = filter_supported_files(files)

        # Directory should not be in either list
        assert len(supported) == 1
        assert len(unsupported) == 0

    def test_filter_case_insensitive_extensions(self, tmp_path):
        """Test that extension matching is case-insensitive."""
        files = [
            tmp_path / "doc.PDF",
            tmp_path / "text.TXT",
            tmp_path / "image.JPG",
        ]
        # Create actual files
        for f in files:
            f.touch()

        supported, unsupported = filter_supported_files(files)

        assert len(supported) == 3
        assert len(unsupported) == 0


class TestValidateFilesForBatch:
    """Tests for validate_files_for_batch function (SRP: security validation)."""

    def test_validate_all_files_pass(self, tmp_path):
        """Test when all files pass validation."""
        files = [tmp_path / f"file{i}.txt" for i in range(3)]
        for f in files:
            f.touch()

        # Mock security config and validation function
        security_config = Mock()
        validate_fn = Mock()  # Does not raise exception

        validated = validate_files_for_batch(files, security_config, validate_fn)

        assert len(validated) == 3
        assert validate_fn.call_count == 3

    def test_validate_some_files_fail(self, tmp_path):
        """Test when some files fail validation."""
        files = [tmp_path / f"file{i}.txt" for i in range(3)]
        for f in files:
            f.touch()

        security_config = Mock()

        # Mock validation function that fails for second file
        def validate_fn_with_failure(file_path, config):
            if "file1" in str(file_path):
                raise Exception("Symlink detected")

        validated = validate_files_for_batch(
            files, security_config, validate_fn_with_failure
        )

        # Only 2 files should pass (file0 and file2)
        assert len(validated) == 2
        assert all("file1" not in str(f) for f in validated)

    def test_validate_all_files_fail(self, tmp_path):
        """Test when all files fail validation."""
        files = [tmp_path / f"file{i}.txt" for i in range(3)]
        for f in files:
            f.touch()

        security_config = Mock()

        # Mock validation function that always fails
        def validate_fn_always_fails(file_path, config):
            raise Exception("Validation failed")

        validated = validate_files_for_batch(
            files, security_config, validate_fn_always_fails
        )

        assert len(validated) == 0

    def test_validate_empty_file_list(self, tmp_path):
        """Test validation with empty file list."""
        security_config = Mock()
        validate_fn = Mock()

        validated = validate_files_for_batch([], security_config, validate_fn)

        assert len(validated) == 0
        assert validate_fn.call_count == 0


class TestSupportedExtensions:
    """Tests for SUPPORTED_EXTENSIONS constant."""

    def test_supported_extensions_contains_common_formats(self):
        """Test that supported extensions include common document formats."""
        assert '.pdf' in SUPPORTED_EXTENSIONS
        assert '.txt' in SUPPORTED_EXTENSIONS
        assert '.docx' in SUPPORTED_EXTENSIONS
        assert '.md' in SUPPORTED_EXTENSIONS

    def test_supported_extensions_contains_image_formats(self):
        """Test that supported extensions include image formats."""
        assert '.jpg' in SUPPORTED_EXTENSIONS
        assert '.jpeg' in SUPPORTED_EXTENSIONS
        assert '.png' in SUPPORTED_EXTENSIONS

    def test_supported_extensions_is_set(self):
        """Test that SUPPORTED_EXTENSIONS is a set (fast lookup)."""
        assert isinstance(SUPPORTED_EXTENSIONS, set)
