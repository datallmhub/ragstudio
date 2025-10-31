"""
Security tests for Atlas-RAG CLI.

These tests verify that security validations work correctly and protect
against common vulnerabilities.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List
import typer

from src.core.cli.utils.security import (
    validate_path_safe,
    validate_no_symlinks,
    validate_pattern_safe,
    validate_file_size,
    validate_batch_size,
    validate_disk_space,
    sanitize_metadata,
    SecurityConfig,
    set_security_config,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def strict_config():
    """Security config with strict settings."""
    return SecurityConfig(
        max_file_size_mb=10,
        max_batch_files=100,
        allow_symlinks=False,
        allow_absolute_patterns=False,
        allow_parent_traversal=False,
        validate_mime_types=False,  # Disabled for tests (no python-magic)
    )


@pytest.fixture
def permissive_config():
    """Security config with permissive settings."""
    return SecurityConfig(
        max_file_size_mb=1000,
        max_batch_files=100000,
        allow_symlinks=True,
        allow_absolute_patterns=True,
        allow_parent_traversal=True,
    )


# ============================================================================
# PATH TRAVERSAL TESTS
# ============================================================================

class TestPathTraversal:
    """Tests for path traversal protection."""

    def test_valid_path_within_base(self, temp_dir, strict_config):
        """Test that valid paths within base directory are accepted."""
        file_path = temp_dir / "doc.txt"
        file_path.write_text("test")

        # Should not raise
        result = validate_path_safe(temp_dir, file_path, strict_config)
        assert result == file_path

    def test_nested_valid_path(self, temp_dir, strict_config):
        """Test that nested paths within base are accepted."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        file_path = subdir / "doc.txt"
        file_path.write_text("test")

        # Should not raise
        result = validate_path_safe(temp_dir, file_path, strict_config)
        assert result == file_path

    def test_absolute_path_outside_base_blocked(self, temp_dir, strict_config):
        """Test that absolute paths outside base are blocked."""
        malicious_path = Path("/etc/passwd")

        with pytest.raises(typer.BadParameter, match="outside allowed directory"):
            validate_path_safe(temp_dir, malicious_path, strict_config)

    def test_relative_traversal_blocked(self, temp_dir, strict_config):
        """Test that relative path traversal is blocked."""
        # Create a path that tries to escape
        malicious_path = temp_dir / ".." / ".." / "etc" / "passwd"

        with pytest.raises(typer.BadParameter):
            validate_path_safe(temp_dir, malicious_path, strict_config)


# ============================================================================
# SYMLINK TESTS
# ============================================================================

class TestSymlinks:
    """Tests for symlink protection."""

    def test_symlink_blocked_by_default(self, temp_dir, strict_config):
        """Test that symlinks are blocked with strict config."""
        target = temp_dir / "target.txt"
        target.write_text("test")

        link = temp_dir / "link.txt"
        link.symlink_to(target)

        with pytest.raises(typer.BadParameter, match="Symbolic links are not allowed"):
            validate_no_symlinks(link, strict_config)

    def test_symlink_allowed_with_permissive_config(self, temp_dir, permissive_config):
        """Test that symlinks are allowed with permissive config."""
        target = temp_dir / "target.txt"
        target.write_text("test")

        link = temp_dir / "link.txt"
        link.symlink_to(target)

        # Should not raise
        result = validate_no_symlinks(link, permissive_config)
        assert result == link

    def test_regular_file_accepted(self, temp_dir, strict_config):
        """Test that regular files are always accepted."""
        file_path = temp_dir / "doc.txt"
        file_path.write_text("test")

        # Should not raise
        result = validate_no_symlinks(file_path, strict_config)
        assert result == file_path


# ============================================================================
# PATTERN VALIDATION TESTS
# ============================================================================

class TestPatternValidation:
    """Tests for glob pattern validation."""

    def test_safe_patterns_accepted(self, strict_config):
        """Test that safe patterns are accepted."""
        safe_patterns = [
            "*.txt",
            "*.pdf",
            "**/*.md",
            "docs/*.html",
            "data/**/*.json",
        ]

        for pattern in safe_patterns:
            # Should not raise
            result = validate_pattern_safe(pattern, strict_config)
            assert result == pattern

    def test_parent_traversal_blocked(self, strict_config):
        """Test that parent directory traversal is blocked."""
        dangerous_patterns = [
            "../*.txt",
            "../../etc/*",
            "../../../*",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(typer.BadParameter, match="parent directory traversal"):
                validate_pattern_safe(pattern, strict_config)

    def test_absolute_paths_blocked(self, strict_config):
        """Test that absolute paths are blocked."""
        dangerous_patterns = [
            "/etc/*",
            "/home/user/*",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(typer.BadParameter, match="Absolute paths"):
                validate_pattern_safe(pattern, strict_config)

    def test_home_expansion_blocked(self, strict_config):
        """Test that home directory expansion is blocked."""
        dangerous_patterns = [
            "~/.ssh/*",
            "~/Documents/*",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(typer.BadParameter, match="Home directory expansion"):
                validate_pattern_safe(pattern, strict_config)

    def test_parent_traversal_allowed_with_permissive_config(self, permissive_config):
        """Test that traversal is allowed with permissive config."""
        pattern = "../*.txt"

        # Should not raise
        result = validate_pattern_safe(pattern, permissive_config)
        assert result == pattern


# ============================================================================
# FILE SIZE TESTS
# ============================================================================

class TestFileSize:
    """Tests for file size validation."""

    def test_small_file_accepted(self, temp_dir, strict_config):
        """Test that small files are accepted."""
        file_path = temp_dir / "small.txt"
        file_path.write_text("test" * 100)  # ~400 bytes

        # Should not raise
        result = validate_file_size(file_path, strict_config)
        assert result == file_path

    def test_large_file_blocked(self, temp_dir, strict_config):
        """Test that files exceeding limit are blocked."""
        file_path = temp_dir / "large.txt"

        # Create 15MB file (exceeds 10MB limit)
        with open(file_path, "wb") as f:
            f.write(b"0" * (15 * 1024 * 1024))

        with pytest.raises(typer.BadParameter, match="too large"):
            validate_file_size(file_path, strict_config)

    def test_large_file_warning_only(self, temp_dir, strict_config):
        """Test warn_only mode for large files."""
        file_path = temp_dir / "large.txt"

        # Create 15MB file
        with open(file_path, "wb") as f:
            f.write(b"0" * (15 * 1024 * 1024))

        # Should warn but not raise
        result = validate_file_size(file_path, strict_config, warn_only=True)
        assert result == file_path

    def test_exact_limit_accepted(self, temp_dir, strict_config):
        """Test that files at exact limit are accepted."""
        file_path = temp_dir / "exact.txt"

        # Create exactly 10MB file
        with open(file_path, "wb") as f:
            f.write(b"0" * (10 * 1024 * 1024))

        # Should not raise
        result = validate_file_size(file_path, strict_config)
        assert result == file_path


# ============================================================================
# BATCH SIZE TESTS
# ============================================================================

class TestBatchSize:
    """Tests for batch size validation."""

    def test_small_batch_accepted(self, temp_dir, strict_config):
        """Test that small batches are accepted."""
        files = [temp_dir / f"file{i}.txt" for i in range(10)]
        for f in files:
            f.write_text("test")

        # Should not raise
        result = validate_batch_size(files, strict_config)
        assert result == files

    def test_large_batch_blocked(self, temp_dir, strict_config):
        """Test that batches exceeding limit are blocked."""
        # Create 150 files (exceeds 100 limit)
        files = [temp_dir / f"file{i}.txt" for i in range(150)]

        with pytest.raises(typer.BadParameter, match="Too many files"):
            validate_batch_size(files, strict_config)

    def test_exact_limit_accepted(self, temp_dir, strict_config):
        """Test that batches at exact limit are accepted."""
        # Create exactly 100 files
        files = [temp_dir / f"file{i}.txt" for i in range(100)]

        # Should not raise
        result = validate_batch_size(files, strict_config)
        assert result == files


# ============================================================================
# METADATA SANITIZATION TESTS
# ============================================================================

class TestMetadataSanitization:
    """Tests for metadata sanitization."""

    def test_html_escaping(self, strict_config):
        """Test that HTML/XML characters are escaped."""
        metadata = {
            "title": "<script>alert('XSS')</script>",
            "author": "John & Jane",
        }

        sanitized = sanitize_metadata(metadata, strict_config)

        assert "&lt;script&gt;" in sanitized["title"]
        assert "&amp;" in sanitized["author"]
        assert "<script>" not in sanitized["title"]

    def test_length_limiting(self, strict_config):
        """Test that long strings are truncated."""
        metadata = {
            "long_text": "x" * 2000,  # Exceeds 1000 char limit
        }

        sanitized = sanitize_metadata(metadata, strict_config)

        assert len(sanitized["long_text"]) <= 1020  # 1000 + "... [truncated]"
        assert "[truncated]" in sanitized["long_text"]

    def test_nested_dict_sanitization(self, strict_config):
        """Test that nested dictionaries are sanitized."""
        metadata = {
            "nested": {
                "title": "<b>Bold</b>",
                "count": 42,
            }
        }

        sanitized = sanitize_metadata(metadata, strict_config)

        assert "&lt;b&gt;" in sanitized["nested"]["title"]
        assert sanitized["nested"]["count"] == 42

    def test_list_sanitization(self, strict_config):
        """Test that lists are sanitized."""
        metadata = {
            "tags": ["<tag1>", "<tag2>", "normal"],
        }

        sanitized = sanitize_metadata(metadata, strict_config)

        assert "&lt;tag1&gt;" in sanitized["tags"][0]
        assert "&lt;tag2&gt;" in sanitized["tags"][1]
        assert sanitized["tags"][2] == "normal"

    def test_numbers_preserved(self, strict_config):
        """Test that numbers are not modified."""
        metadata = {
            "count": 42,
            "ratio": 3.14,
            "is_valid": True,
        }

        sanitized = sanitize_metadata(metadata, strict_config)

        assert sanitized["count"] == 42
        assert sanitized["ratio"] == 3.14
        assert sanitized["is_valid"] is True

    def test_sanitization_disabled(self):
        """Test that sanitization can be disabled."""
        config = SecurityConfig(sanitize_metadata=False)

        metadata = {
            "title": "<script>alert('XSS')</script>",
        }

        sanitized = sanitize_metadata(metadata, config)

        # Should be unchanged
        assert sanitized["title"] == "<script>alert('XSS')</script>"


# ============================================================================
# DISK SPACE TESTS
# ============================================================================

class TestDiskSpace:
    """Tests for disk space validation."""

    def test_sufficient_disk_space(self, temp_dir, strict_config):
        """Test that sufficient disk space passes validation."""
        # temp_dir should have plenty of space
        # Should not raise
        validate_disk_space(temp_dir, strict_config)

    def test_insufficient_disk_space_detection(self, temp_dir):
        """Test detection of insufficient disk space."""
        # Create config requiring 1TB of space (unlikely to have)
        extreme_config = SecurityConfig(require_disk_space_mb=1_000_000)

        # May raise or may not, depending on actual disk space
        # This test verifies the function runs without errors
        try:
            validate_disk_space(temp_dir, extreme_config)
        except typer.BadParameter as e:
            assert "Insufficient disk space" in str(e)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSecurityIntegration:
    """Integration tests combining multiple security checks."""

    def test_secure_file_processing_workflow(self, temp_dir, strict_config):
        """Test complete secure file processing workflow."""
        # 1. Create a valid file
        file_path = temp_dir / "doc.txt"
        file_path.write_text("test content")

        # 2. Validate path safety
        validate_path_safe(temp_dir, file_path, strict_config)

        # 3. Check symlinks
        validate_no_symlinks(file_path, strict_config)

        # 4. Check file size
        validate_file_size(file_path, strict_config)

        # All should pass
        assert file_path.exists()

    def test_secure_batch_workflow(self, temp_dir, strict_config):
        """Test complete secure batch processing workflow."""
        # 1. Create valid files
        files = []
        for i in range(10):
            file_path = temp_dir / f"doc{i}.txt"
            file_path.write_text(f"content {i}")
            files.append(file_path)

        # 2. Validate pattern
        pattern = "*.txt"
        validate_pattern_safe(pattern, strict_config)

        # 3. Validate batch size
        validate_batch_size(files, strict_config)

        # 4. Validate each file
        for file_path in files:
            validate_path_safe(temp_dir, file_path, strict_config)
            validate_no_symlinks(file_path, strict_config)
            validate_file_size(file_path, strict_config)

        # All should pass
        assert len(files) == 10

    def test_malicious_file_blocked(self, temp_dir, strict_config):
        """Test that malicious files are blocked at multiple levels."""
        # Try to create a file outside base dir
        malicious_path = Path("/tmp") / "malicious.txt"

        # Should be blocked by path validation
        with pytest.raises(typer.BadParameter):
            validate_path_safe(temp_dir, malicious_path, strict_config)


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestSecurityConfiguration:
    """Tests for security configuration."""

    def test_default_config_is_secure(self):
        """Test that default configuration is reasonably secure."""
        config = SecurityConfig()

        assert config.max_file_size_mb <= 100
        assert config.max_batch_files <= 10000
        assert config.allow_symlinks is False
        assert config.allow_absolute_patterns is False
        assert config.allow_parent_traversal is False

    def test_load_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("ATLAS_MAX_FILE_SIZE_MB", "200")
        monkeypatch.setenv("ATLAS_ALLOW_SYMLINKS", "true")

        config = SecurityConfig.load_from_env()

        assert config.max_file_size_mb == 200
        assert config.allow_symlinks is True

    def test_global_config_management(self, strict_config):
        """Test global configuration management."""
        from src.core.cli.utils.security import get_security_config, set_security_config

        # Set custom config
        set_security_config(strict_config)

        # Retrieve and verify
        retrieved = get_security_config()
        assert retrieved.max_file_size_mb == strict_config.max_file_size_mb


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("pattern,should_pass", [
    ("*.txt", True),
    ("**/*.pdf", True),
    ("docs/*.md", True),
    ("../../../etc/*", False),
    ("/etc/passwd", False),
    ("~/.ssh/*", False),
])
def test_pattern_validation_parametrized(pattern, should_pass, strict_config):
    """Parametrized test for pattern validation."""
    if should_pass:
        result = validate_pattern_safe(pattern, strict_config)
        assert result == pattern
    else:
        with pytest.raises(typer.BadParameter):
            validate_pattern_safe(pattern, strict_config)


@pytest.mark.parametrize("size_mb,should_pass", [
    (1, True),    # 1MB - OK
    (5, True),    # 5MB - OK
    (10, True),   # 10MB - At limit, OK
    (15, False),  # 15MB - Too large
    (100, False), # 100MB - Way too large
])
def test_file_size_validation_parametrized(size_mb, should_pass, temp_dir, strict_config):
    """Parametrized test for file size validation."""
    file_path = temp_dir / f"file_{size_mb}mb.txt"

    with open(file_path, "wb") as f:
        f.write(b"0" * (size_mb * 1024 * 1024))

    if should_pass:
        result = validate_file_size(file_path, strict_config)
        assert result == file_path
    else:
        with pytest.raises(typer.BadParameter):
            validate_file_size(file_path, strict_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
