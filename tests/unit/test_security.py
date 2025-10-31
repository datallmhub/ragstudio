"""
Tests for CLI security utilities.

Comprehensive tests for all security functions in src/core/cli/utils/security.py.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import typer

from src.core.cli.utils.security import (
    SecurityConfig,
    get_security_config,
    set_security_config,
    validate_path_safe,
    validate_no_symlinks,
    validate_pattern_safe,
    validate_file_size,
    validate_total_size,
    validate_batch_size,
    validate_disk_space,
    validate_mime_type,
    sanitize_metadata,
    validate_file_comprehensive,
    validate_batch_comprehensive,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Sample content for testing")
    return file_path


@pytest.fixture
def large_file(temp_dir):
    """Create a large file (60MB) for testing."""
    file_path = temp_dir / "large_file.bin"
    # Create 60MB file
    with open(file_path, 'wb') as f:
        f.write(b'0' * (60 * 1024 * 1024))
    return file_path


@pytest.fixture
def default_config():
    """Default security configuration."""
    return SecurityConfig()


@pytest.fixture
def permissive_config():
    """Permissive security configuration for testing."""
    return SecurityConfig(
        max_file_size_mb=1000,
        warn_file_size_mb=500,
        max_batch_files=100000,
        warn_batch_files=10000,
        allow_symlinks=True,
        allow_absolute_patterns=True,
        allow_parent_traversal=True,
        require_disk_space_mb=1,
        validate_mime_types=False,
        sanitize_metadata=False,
    )


# ============================================================================
# TEST SECURITY CONFIG
# ============================================================================

class TestSecurityConfig:
    """Tests for SecurityConfig class."""

    def test_default_config(self):
        """Test default security configuration values."""
        config = SecurityConfig()

        assert config.max_file_size_mb == 100
        assert config.warn_file_size_mb == 50
        assert config.max_batch_files == 10000
        assert config.warn_batch_files == 1000
        assert config.allow_symlinks is False
        assert config.allow_absolute_patterns is False
        assert config.allow_parent_traversal is False
        assert config.require_disk_space_mb == 100
        assert config.validate_mime_types is False
        assert config.max_batch_timeout_seconds == 3600
        assert config.max_metadata_string_length == 1000
        assert config.sanitize_metadata is True

    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            max_file_size_mb=200,
            allow_symlinks=True,
            validate_mime_types=True,
        )

        assert config.max_file_size_mb == 200
        assert config.allow_symlinks is True
        assert config.validate_mime_types is True

    @patch.dict(os.environ, {
        'ATLAS_MAX_FILE_SIZE_MB': '250',
        'ATLAS_ALLOW_SYMLINKS': 'true',
        'ATLAS_VALIDATE_MIME_TYPES': 'TRUE',
        'ATLAS_SANITIZE_METADATA': 'false',
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = SecurityConfig.load_from_env()

        assert config.max_file_size_mb == 250
        assert config.allow_symlinks is True
        assert config.validate_mime_types is True
        assert config.sanitize_metadata is False

    @patch.dict(os.environ, {
        'ATLAS_MAX_FILE_SIZE_MB': '150',
        'ATLAS_WARN_FILE_SIZE_MB': '75',
    })
    def test_load_from_env_partial(self):
        """Test loading with only some env vars set."""
        config = SecurityConfig.load_from_env()

        assert config.max_file_size_mb == 150
        assert config.warn_file_size_mb == 75
        # Others should use defaults
        assert config.max_batch_files == 10000

    def test_get_set_security_config(self):
        """Test getting and setting global security config."""
        custom_config = SecurityConfig(max_file_size_mb=500)

        set_security_config(custom_config)
        retrieved_config = get_security_config()

        assert retrieved_config.max_file_size_mb == 500


# ============================================================================
# TEST PATH SECURITY
# ============================================================================

class TestValidatePathSafe:
    """Tests for validate_path_safe function."""

    def test_valid_path_within_base(self, temp_dir, sample_file, default_config):
        """Test validation passes for path within base directory."""
        result = validate_path_safe(temp_dir, sample_file, default_config)

        assert result == sample_file

    def test_path_outside_base(self, temp_dir, default_config):
        """Test validation fails for path outside base directory."""
        outside_path = Path("/etc/passwd")

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_path_safe(temp_dir, outside_path, default_config)

        assert "outside allowed directory" in str(exc_info.value)

    def test_path_traversal_attack(self, temp_dir, default_config):
        """Test prevention of path traversal attack."""
        # Create a path that tries to escape
        malicious_path = temp_dir / ".." / ".." / "etc" / "passwd"

        with pytest.raises(typer.BadParameter):
            validate_path_safe(temp_dir, malicious_path, default_config)

    def test_valid_nested_path(self, temp_dir, default_config):
        """Test validation for nested paths within base."""
        nested_dir = temp_dir / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        nested_file = nested_dir / "file.txt"
        nested_file.write_text("content")

        result = validate_path_safe(temp_dir, nested_file, default_config)

        assert result == nested_file

    def test_uses_global_config_when_none(self, temp_dir, sample_file):
        """Test that global config is used when config parameter is None."""
        # Should not raise - uses global config
        result = validate_path_safe(temp_dir, sample_file, config=None)
        assert result == sample_file


class TestValidateNoSymlinks:
    """Tests for validate_no_symlinks function."""

    def test_regular_file_allowed(self, sample_file, default_config):
        """Test validation passes for regular file."""
        result = validate_no_symlinks(sample_file, default_config)

        assert result == sample_file

    def test_symlink_blocked_by_default(self, temp_dir, sample_file, default_config):
        """Test symlinks are blocked with default config."""
        symlink_path = temp_dir / "symlink.txt"
        symlink_path.symlink_to(sample_file)

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_no_symlinks(symlink_path, default_config)

        assert "Symbolic links are not allowed" in str(exc_info.value)

    def test_symlink_allowed_with_permissive_config(self, temp_dir, sample_file, permissive_config):
        """Test symlinks are allowed with permissive config."""
        symlink_path = temp_dir / "symlink.txt"
        symlink_path.symlink_to(sample_file)

        # Should not raise
        result = validate_no_symlinks(symlink_path, permissive_config)
        assert result == symlink_path

    def test_string_path_converted_to_path(self, sample_file, default_config):
        """Test that string paths are converted to Path objects."""
        result = validate_no_symlinks(str(sample_file), default_config)

        assert isinstance(result, Path)
        assert result == sample_file


class TestValidatePatternSafe:
    """Tests for validate_pattern_safe function."""

    def test_safe_pattern_allowed(self, default_config):
        """Test safe patterns are allowed."""
        safe_patterns = ["*.txt", "**/*.pdf", "docs/*.md", "data/**/*.json"]

        for pattern in safe_patterns:
            result = validate_pattern_safe(pattern, default_config)
            assert result == pattern

    def test_parent_traversal_blocked(self, default_config):
        """Test patterns with parent traversal are blocked."""
        dangerous_patterns = ["../config.txt", "../../etc/passwd", "data/../secrets.txt"]

        for pattern in dangerous_patterns:
            with pytest.raises(typer.BadParameter) as exc_info:
                validate_pattern_safe(pattern, default_config)

            assert "parent directory traversal" in str(exc_info.value)

    def test_absolute_path_blocked(self, default_config):
        """Test absolute paths in patterns are blocked."""
        absolute_patterns = ["/etc/passwd", "/home/user/file.txt"]

        for pattern in absolute_patterns:
            with pytest.raises(typer.BadParameter) as exc_info:
                validate_pattern_safe(pattern, default_config)

            assert "Absolute paths in patterns are not allowed" in str(exc_info.value)

    def test_windows_absolute_path_blocked(self, default_config):
        """Test Windows-style absolute paths are blocked."""
        with pytest.raises(typer.BadParameter):
            validate_pattern_safe("C:\\Windows\\System32\\*", default_config)

    def test_home_directory_expansion_blocked(self, default_config):
        """Test home directory expansion is blocked."""
        with pytest.raises(typer.BadParameter) as exc_info:
            validate_pattern_safe("~/.ssh/id_rsa", default_config)

        assert "Home directory expansion (~) is not allowed" in str(exc_info.value)

    def test_dangerous_patterns_allowed_with_permissive_config(self, permissive_config):
        """Test dangerous patterns allowed with permissive config."""
        # Parent traversal allowed
        result1 = validate_pattern_safe("../file.txt", permissive_config)
        assert result1 == "../file.txt"

        # Absolute paths allowed
        result2 = validate_pattern_safe("/etc/config.txt", permissive_config)
        assert result2 == "/etc/config.txt"

        # But home directory expansion is ALWAYS blocked
        with pytest.raises(typer.BadParameter):
            validate_pattern_safe("~/file.txt", permissive_config)


# ============================================================================
# TEST FILE SIZE VALIDATION
# ============================================================================

class TestValidateFileSize:
    """Tests for validate_file_size function."""

    def test_small_file_allowed(self, sample_file, default_config):
        """Test small files pass validation."""
        result = validate_file_size(sample_file, default_config)

        assert result == sample_file

    @patch('src.core.cli.utils.display.print_warning')
    def test_large_file_warning(self, mock_warning, temp_dir, default_config):
        """Test warning for large files (but within limit)."""
        # Create 55MB file (above warn threshold of 50MB, below max of 100MB)
        large_file = temp_dir / "large.bin"
        with open(large_file, 'wb') as f:
            f.write(b'0' * (55 * 1024 * 1024))

        result = validate_file_size(large_file, default_config)

        assert result == large_file
        mock_warning.assert_called_once()
        assert "Large file detected" in mock_warning.call_args[0][0]

    def test_oversized_file_blocked(self, temp_dir, default_config):
        """Test files exceeding max size are blocked."""
        # Create 110MB file (above max of 100MB)
        huge_file = temp_dir / "huge.bin"
        with open(huge_file, 'wb') as f:
            f.write(b'0' * (110 * 1024 * 1024))

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_file_size(huge_file, default_config)

        assert "File too large" in str(exc_info.value)
        assert "110" in str(exc_info.value)  # Size in MB
        assert "100MB" in str(exc_info.value)  # Max limit

    @patch('src.core.cli.utils.display.print_warning')
    def test_oversized_file_warn_only(self, mock_warning, temp_dir, default_config):
        """Test warn_only mode for oversized files."""
        huge_file = temp_dir / "huge.bin"
        with open(huge_file, 'wb') as f:
            f.write(b'0' * (110 * 1024 * 1024))

        # Should not raise with warn_only=True
        result = validate_file_size(huge_file, default_config, warn_only=True)

        assert result == huge_file
        mock_warning.assert_called_once()

    def test_nonexistent_file_error(self, temp_dir, default_config):
        """Test error for nonexistent file."""
        nonexistent = temp_dir / "doesnt_exist.txt"

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_file_size(nonexistent, default_config)

        assert "Cannot access file" in str(exc_info.value)


class TestValidateTotalSize:
    """Tests for validate_total_size function."""

    @patch('src.core.cli.utils.display.print_warning')
    def test_small_batch_no_warning(self, mock_warning, temp_dir, default_config):
        """Test small batch doesn't trigger warnings."""
        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text("small content")
            files.append(f)

        result = validate_total_size(files, default_config)

        assert result == files
        mock_warning.assert_not_called()

    @patch('src.core.cli.utils.display.print_warning')
    def test_large_batch_warning(self, mock_warning, temp_dir, default_config):
        """Test large batch triggers warning."""
        files = []
        # Create files that total > max_file_size_mb * num_files
        for i in range(2):
            f = temp_dir / f"large{i}.bin"
            with open(f, 'wb') as fp:
                fp.write(b'0' * (150 * 1024 * 1024))  # 150MB each
            files.append(f)

        result = validate_total_size(files, default_config)

        assert result == files
        mock_warning.assert_called_once()
        assert "Total batch size" in mock_warning.call_args[0][0]

    def test_empty_batch(self, default_config):
        """Test empty batch."""
        result = validate_total_size([], default_config)

        assert result == []


# ============================================================================
# TEST BATCH VALIDATION
# ============================================================================

class TestValidateBatchSize:
    """Tests for validate_batch_size function."""

    def test_small_batch_allowed(self, temp_dir, default_config):
        """Test small batches pass validation."""
        files = [temp_dir / f"file{i}.txt" for i in range(10)]

        result = validate_batch_size(files, default_config)

        assert result == files

    @patch('src.core.cli.utils.display.print_warning')
    def test_large_batch_warning(self, mock_warning, temp_dir, default_config):
        """Test large batches trigger warning."""
        # Create 1500 files (above warn threshold of 1000)
        files = [temp_dir / f"file{i}.txt" for i in range(1500)]

        result = validate_batch_size(files, default_config)

        assert result == files
        mock_warning.assert_called_once()
        assert "1500" in mock_warning.call_args[0][0]

    def test_oversized_batch_blocked(self, temp_dir, default_config):
        """Test batches exceeding max size are blocked."""
        # Create 15000 files (above max of 10000)
        files = [temp_dir / f"file{i}.txt" for i in range(15000)]

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_batch_size(files, default_config)

        assert "Too many files in batch" in str(exc_info.value)
        assert "15000" in str(exc_info.value)
        assert "10000" in str(exc_info.value)

    def test_empty_batch(self, default_config):
        """Test empty batch validation."""
        result = validate_batch_size([], default_config)

        assert result == []


# ============================================================================
# TEST DISK SPACE VALIDATION
# ============================================================================

class TestValidateDiskSpace:
    """Tests for validate_disk_space function."""

    @patch('shutil.disk_usage')
    def test_sufficient_disk_space(self, mock_disk_usage, temp_dir, default_config):
        """Test validation passes with sufficient disk space."""
        # Mock 500MB available (well above 100MB requirement)
        mock_disk_usage.return_value = MagicMock(
            free=500 * 1024 * 1024
        )

        # Should not raise
        validate_disk_space(temp_dir, default_config)

    @patch('shutil.disk_usage')
    @patch('src.core.cli.utils.display.print_warning')
    def test_low_disk_space_warning(self, mock_warning, mock_disk_usage, temp_dir, default_config):
        """Test warning for low disk space."""
        # Mock 150MB available (above 100MB requirement, but below 200MB warning threshold)
        mock_disk_usage.return_value = MagicMock(
            free=150 * 1024 * 1024
        )

        validate_disk_space(temp_dir, default_config)

        mock_warning.assert_called_once()
        assert "Low disk space" in mock_warning.call_args[0][0]

    @patch('shutil.disk_usage')
    def test_insufficient_disk_space(self, mock_disk_usage, temp_dir, default_config):
        """Test error for insufficient disk space."""
        # Mock 50MB available (below 100MB requirement)
        mock_disk_usage.return_value = MagicMock(
            free=50 * 1024 * 1024
        )

        with pytest.raises(typer.BadParameter) as exc_info:
            validate_disk_space(temp_dir, default_config)

        assert "Insufficient disk space" in str(exc_info.value)

    @patch('shutil.disk_usage')
    @patch('src.core.cli.utils.display.print_warning')
    def test_disk_check_error_warning(self, mock_warning, mock_disk_usage, temp_dir, default_config):
        """Test warning when disk check fails."""
        mock_disk_usage.side_effect = OSError("Disk check failed")

        # Should not raise, just warn
        validate_disk_space(temp_dir, default_config)

        mock_warning.assert_called_once()
        assert "Could not check disk space" in mock_warning.call_args[0][0]

    @patch('shutil.disk_usage')
    def test_validate_file_path_uses_parent(self, mock_disk_usage, temp_dir, default_config):
        """Test validation uses parent directory for file paths."""
        mock_disk_usage.return_value = MagicMock(free=500 * 1024 * 1024)

        file_path = temp_dir / "output.json"
        validate_disk_space(file_path, default_config)

        # Should check parent directory
        mock_disk_usage.assert_called_once_with(temp_dir)


# ============================================================================
# TEST MIME TYPE VALIDATION
# ============================================================================

class TestValidateMimeType:
    """Tests for validate_mime_type function."""

    def test_validation_disabled_by_default(self, sample_file, default_config):
        """Test MIME validation is disabled by default."""
        # Should pass without checking (validate_mime_types=False by default)
        result = validate_mime_type(sample_file, config=default_config)

        assert result == sample_file

    def test_validation_without_magic_library(self, sample_file):
        """Test graceful handling when python-magic is not installed."""
        config = SecurityConfig(validate_mime_types=True)

        # Mock the import to fail
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'magic':
                    raise ImportError("No module named 'magic'")
                # Call the real import for everything else
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            result = validate_mime_type(sample_file, config=config)

        assert result == sample_file

    def test_mime_type_match(self, temp_dir):
        """Test validation passes when MIME type matches extension."""
        config = SecurityConfig(validate_mime_types=True)

        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_bytes(b'%PDF-1.4 fake pdf')

        # Mock magic module at import time
        with patch.dict('sys.modules', {'magic': MagicMock()}):
            import sys
            mock_magic_instance = MagicMock()
            mock_magic_instance.from_file.return_value = "application/pdf"
            sys.modules['magic'].Magic.return_value = mock_magic_instance

            result = validate_mime_type(pdf_file, config=config)

        assert result == pdf_file

    @patch('src.core.cli.utils.display.print_warning')
    def test_mime_type_mismatch_warning(self, mock_warning, temp_dir):
        """Test warning when MIME type doesn't match extension."""
        config = SecurityConfig(validate_mime_types=True)

        # Create file with misleading extension
        fake_pdf = temp_dir / "malicious.pdf"
        fake_pdf.write_text("This is actually a text file")

        # Mock magic module
        with patch.dict('sys.modules', {'magic': MagicMock()}):
            import sys
            mock_magic_instance = MagicMock()
            mock_magic_instance.from_file.return_value = "text/plain"
            sys.modules['magic'].Magic.return_value = mock_magic_instance

            result = validate_mime_type(fake_pdf, config=config)

        # Should still return file but with warning
        assert result == fake_pdf
        mock_warning.assert_called_once()
        assert "File type mismatch" in mock_warning.call_args[0][0]

    def test_mime_validation_error_handled(self, sample_file):
        """Test that MIME validation errors don't cause failures."""
        config = SecurityConfig(validate_mime_types=True)

        # Mock magic module to raise error
        with patch.dict('sys.modules', {'magic': MagicMock()}):
            import sys
            mock_magic_instance = MagicMock()
            mock_magic_instance.from_file.side_effect = Exception("Magic error")
            sys.modules['magic'].Magic.return_value = mock_magic_instance

            # Should not raise, just return file
            result = validate_mime_type(sample_file, config=config)

        assert result == sample_file


# ============================================================================
# TEST METADATA SANITIZATION
# ============================================================================

class TestSanitizeMetadata:
    """Tests for sanitize_metadata function."""

    def test_sanitization_disabled(self):
        """Test sanitization can be disabled."""
        config = SecurityConfig(sanitize_metadata=False)
        metadata = {"key": "<script>alert('xss')</script>"}

        result = sanitize_metadata(metadata, config)

        # Should return unchanged
        assert result == metadata

    def test_sanitize_html_characters(self, default_config):
        """Test HTML/XML characters are escaped."""
        metadata = {
            "title": "<script>alert('xss')</script>",
            "description": "Safe & sound",
            "tag": 'Quote: "test"',
        }

        result = sanitize_metadata(metadata, default_config)

        assert "&lt;script&gt;" in result["title"]
        assert "&amp;" in result["description"]
        assert "&quot;" in result["tag"]

    def test_sanitize_long_strings(self, default_config):
        """Test long strings are truncated."""
        long_string = "a" * 1500  # Exceeds max_metadata_string_length of 1000
        metadata = {"field": long_string}

        result = sanitize_metadata(metadata, default_config)

        assert len(result["field"]) < len(long_string)
        assert "[truncated]" in result["field"]

    def test_preserve_numeric_types(self, default_config):
        """Test numeric types are preserved."""
        metadata = {
            "count": 42,
            "score": 3.14,
            "active": True,
        }

        result = sanitize_metadata(metadata, default_config)

        assert result["count"] == 42
        assert result["score"] == 3.14
        assert result["active"] is True

    def test_sanitize_nested_dict(self, default_config):
        """Test nested dictionaries are recursively sanitized."""
        metadata = {
            "user": {
                "name": "<script>bad</script>",
                "age": 25,
            }
        }

        result = sanitize_metadata(metadata, default_config)

        assert "&lt;script&gt;" in result["user"]["name"]
        assert result["user"]["age"] == 25

    def test_sanitize_lists(self, default_config):
        """Test lists are sanitized."""
        metadata = {
            "tags": ["safe", "<script>bad</script>", "normal"],
            "numbers": [1, 2, 3],
        }

        result = sanitize_metadata(metadata, default_config)

        assert result["tags"][0] == "safe"
        assert "&lt;script&gt;" in result["tags"][1]
        assert result["numbers"] == [1, 2, 3]

    def test_sanitize_key_names(self, default_config):
        """Test that keys are also sanitized."""
        metadata = {
            "<script>key</script>": "value"
        }

        result = sanitize_metadata(metadata, default_config)

        # Key should be sanitized
        assert list(result.keys())[0] != "<script>key</script>"
        assert "&lt;script&gt;" in list(result.keys())[0]

    def test_sanitize_complex_nested_structure(self, default_config):
        """Test complex nested structures."""
        metadata = {
            "data": {
                "items": [
                    {"name": "<bad>", "value": 1},
                    {"name": "good", "value": 2}
                ]
            }
        }

        result = sanitize_metadata(metadata, default_config)

        assert "&lt;bad&gt;" in result["data"]["items"][0]["name"]
        assert result["data"]["items"][1]["name"] == "good"


# ============================================================================
# TEST COMPREHENSIVE VALIDATION
# ============================================================================

class TestValidateFileComprehensive:
    """Tests for validate_file_comprehensive function."""

    def test_all_validations_pass(self, temp_dir, sample_file, default_config):
        """Test all validations pass for valid file."""
        result = validate_file_comprehensive(
            sample_file,
            base_dir=temp_dir,
            config=default_config
        )

        assert result == sample_file

    def test_path_traversal_detected(self, temp_dir, default_config):
        """Test path traversal is detected in comprehensive validation."""
        malicious_path = Path("/etc/passwd")

        with pytest.raises(typer.BadParameter):
            validate_file_comprehensive(
                malicious_path,
                base_dir=temp_dir,
                config=default_config
            )

    def test_symlink_detected(self, temp_dir, sample_file, default_config):
        """Test symlinks are detected in comprehensive validation."""
        symlink = temp_dir / "link.txt"
        symlink.symlink_to(sample_file)

        with pytest.raises(typer.BadParameter):
            validate_file_comprehensive(
                symlink,
                base_dir=temp_dir,
                config=default_config
            )

    def test_oversized_file_detected(self, temp_dir, default_config):
        """Test oversized files are detected."""
        huge_file = temp_dir / "huge.bin"
        with open(huge_file, 'wb') as f:
            f.write(b'0' * (110 * 1024 * 1024))

        with pytest.raises(typer.BadParameter):
            validate_file_comprehensive(
                huge_file,
                base_dir=temp_dir,
                config=default_config
            )

    def test_without_base_dir(self, sample_file, default_config):
        """Test validation without base directory check."""
        # Should still check symlinks and file size
        result = validate_file_comprehensive(
            sample_file,
            base_dir=None,
            config=default_config
        )

        assert result == sample_file


class TestValidateBatchComprehensive:
    """Tests for validate_batch_comprehensive function."""

    def test_valid_batch(self, temp_dir, default_config):
        """Test validation passes for valid batch."""
        files = []
        for i in range(5):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        result = validate_batch_comprehensive(
            files,
            base_dir=temp_dir,
            config=default_config,
            validate_files=True
        )

        assert len(result) == 5

    def test_batch_too_large(self, temp_dir, default_config):
        """Test batch size limit is enforced."""
        files = [temp_dir / f"file{i}.txt" for i in range(15000)]

        with pytest.raises(typer.BadParameter):
            validate_batch_comprehensive(
                files,
                base_dir=temp_dir,
                config=default_config,
                validate_files=False  # Skip individual validation
            )

    @patch('src.core.cli.utils.display.print_warning')
    def test_invalid_files_skipped(self, mock_warning, temp_dir, default_config):
        """Test that invalid files are skipped instead of failing entire batch."""
        files = []
        # Create valid file
        valid_file = temp_dir / "valid.txt"
        valid_file.write_text("content")
        files.append(valid_file)

        # Create oversized file
        huge_file = temp_dir / "huge.bin"
        with open(huge_file, 'wb') as f:
            f.write(b'0' * (110 * 1024 * 1024))
        files.append(huge_file)

        result = validate_batch_comprehensive(
            files,
            base_dir=temp_dir,
            config=default_config,
            validate_files=True
        )

        # Should only include valid file
        assert len(result) == 1
        assert result[0] == valid_file

        # Should have warned about skipping invalid file
        mock_warning.assert_called()

    def test_skip_individual_validation(self, temp_dir, default_config):
        """Test skipping individual file validation for performance."""
        files = []
        for i in range(10):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        result = validate_batch_comprehensive(
            files,
            base_dir=temp_dir,
            config=default_config,
            validate_files=False  # Skip individual checks
        )

        # All files returned without individual validation
        assert len(result) == 10

    @patch('src.core.cli.utils.display.print_warning')
    def test_batch_total_size_warning(self, mock_warning, temp_dir, default_config):
        """Test warning for large total batch size."""
        files = []
        for i in range(2):
            f = temp_dir / f"large{i}.bin"
            with open(f, 'wb') as fp:
                fp.write(b'0' * (150 * 1024 * 1024))
            files.append(f)

        validate_batch_comprehensive(
            files,
            base_dir=temp_dir,
            config=default_config,
            validate_files=False
        )

        # Should warn about large batch
        mock_warning.assert_called()
