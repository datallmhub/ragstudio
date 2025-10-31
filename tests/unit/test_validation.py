"""
Tests for CLI validation utilities.

Tests all validation functions in src/core/cli/utils/validation.py.
"""

import pytest
from pathlib import Path
import typer

from src.core.cli.utils.validation import (
    validate_file_exists,
    validate_directory_exists,
    validate_output_path,
    validate_token_range,
)


class TestValidateFileExists:
    """Test validate_file_exists function."""

    def test_validate_existing_file(self, tmp_path):
        """Test validation passes for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = validate_file_exists(test_file)

        assert result == test_file

    def test_validate_nonexistent_file(self, tmp_path):
        """Test validation fails for nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_file_exists(test_file)

        assert "File does not exist" in str(exc_info.value)
        assert str(test_file) in str(exc_info.value)

    def test_validate_directory_instead_of_file(self, tmp_path):
        """Test validation fails when path is a directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_file_exists(test_dir)

        assert "Path is not a file" in str(exc_info.value)
        assert str(test_dir) in str(exc_info.value)


class TestValidateDirectoryExists:
    """Test validate_directory_exists function."""

    def test_validate_existing_directory(self, tmp_path):
        """Test validation passes for existing directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = validate_directory_exists(test_dir)

        assert result == test_dir

    def test_validate_nonexistent_directory(self, tmp_path):
        """Test validation fails for nonexistent directory."""
        test_dir = tmp_path / "nonexistent_dir"

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_directory_exists(test_dir)

        assert "Directory does not exist" in str(exc_info.value)
        assert str(test_dir) in str(exc_info.value)

    def test_validate_file_instead_of_directory(self, tmp_path):
        """Test validation fails when path is a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_directory_exists(test_file)

        assert "Path is not a directory" in str(exc_info.value)
        assert str(test_file) in str(exc_info.value)


class TestValidateOutputPath:
    """Test validate_output_path function."""

    def test_validate_none_output_path(self):
        """Test validation passes for None."""
        result = validate_output_path(None)

        assert result is None

    def test_validate_output_path_with_existing_parent(self, tmp_path):
        """Test validation passes when parent directory exists."""
        output_file = tmp_path / "output.json"

        result = validate_output_path(output_file)

        assert result == output_file

    def test_validate_output_path_with_nonexistent_parent(self, tmp_path):
        """Test validation fails when parent directory doesn't exist."""
        output_file = tmp_path / "nonexistent_dir" / "output.json"

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_output_path(output_file)

        assert "Parent directory does not exist" in str(exc_info.value)
        assert str(output_file.parent) in str(exc_info.value)
        assert "Please create it first" in str(exc_info.value)

    def test_validate_output_path_nested_existing_parent(self, tmp_path):
        """Test validation passes for nested path with existing parent."""
        nested_dir = tmp_path / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        output_file = nested_dir / "output.json"

        result = validate_output_path(output_file)

        assert result == output_file

    def test_validate_output_path_file_already_exists(self, tmp_path):
        """Test validation passes even if output file already exists."""
        output_file = tmp_path / "existing.json"
        output_file.write_text("existing content")

        result = validate_output_path(output_file)

        assert result == output_file


class TestValidateTokenRange:
    """Test validate_token_range function."""

    def test_validate_token_in_default_range(self):
        """Test validation passes for value in default range."""
        assert validate_token_range(100) == 100
        assert validate_token_range(50) == 50
        assert validate_token_range(2000) == 2000
        assert validate_token_range(500) == 500

    def test_validate_token_below_minimum(self):
        """Test validation fails for value below minimum."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_token_range(49)

        assert "must be between 50 and 2000" in str(exc_info.value)

    def test_validate_token_above_maximum(self):
        """Test validation fails for value above maximum."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_token_range(2001)

        assert "must be between 50 and 2000" in str(exc_info.value)

    def test_validate_token_custom_range(self):
        """Test validation with custom min/max values."""
        assert validate_token_range(100, min_val=100, max_val=500) == 100
        assert validate_token_range(500, min_val=100, max_val=500) == 500
        assert validate_token_range(300, min_val=100, max_val=500) == 300

    def test_validate_token_custom_range_below_min(self):
        """Test validation fails below custom minimum."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_token_range(99, min_val=100, max_val=500)

        assert "must be between 100 and 500" in str(exc_info.value)

    def test_validate_token_custom_range_above_max(self):
        """Test validation fails above custom maximum."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_token_range(501, min_val=100, max_val=500)

        assert "must be between 100 and 500" in str(exc_info.value)

    def test_validate_token_edge_cases(self):
        """Test edge cases for token validation."""
        # At exact boundaries
        assert validate_token_range(50, min_val=50, max_val=2000) == 50
        assert validate_token_range(2000, min_val=50, max_val=2000) == 2000

        # Just outside boundaries
        with pytest.raises(typer.BadParameter):
            validate_token_range(49, min_val=50, max_val=2000)

        with pytest.raises(typer.BadParameter):
            validate_token_range(2001, min_val=50, max_val=2000)

    def test_validate_token_zero_value(self):
        """Test validation with zero value."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_token_range(0)

        assert "must be between" in str(exc_info.value)

    def test_validate_token_negative_value(self):
        """Test validation with negative value."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_token_range(-10)

        assert "must be between" in str(exc_info.value)
