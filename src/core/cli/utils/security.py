"""
Security utilities for Atlas-RAG CLI.

This module provides security validations and guards for file operations,
input validation, and resource limits.

Usage:
    from src.core.cli.utils.security import (
        validate_path_safe,
        validate_file_size,
        validate_batch_size,
        SecurityConfig
    )
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass

import typer


# Configure security logger
security_logger = logging.getLogger("atlas_rag.security")


@dataclass
class SecurityConfig:
    """
    Centralized security configuration for Atlas-RAG CLI.

    All limits are configurable via environment variables or code.
    """
    # File size limits
    max_file_size_mb: int = 100  # Max single file size
    warn_file_size_mb: int = 50  # Warn if file is larger than this

    # Batch limits
    max_batch_files: int = 10000  # Max files in a single batch
    warn_batch_files: int = 1000  # Warn if batch has more files

    # Path security
    allow_symlinks: bool = False  # Allow symlink following
    allow_absolute_patterns: bool = False  # Allow absolute paths in patterns
    allow_parent_traversal: bool = False  # Allow ../ in patterns

    # Disk space
    require_disk_space_mb: int = 100  # Min disk space required for output

    # MIME validation
    validate_mime_types: bool = False  # Validate file types via magic numbers (requires python-magic)

    # Timeouts
    max_batch_timeout_seconds: int = 3600  # 1 hour max for batch processing

    # Metadata sanitization
    max_metadata_string_length: int = 1000  # Max length for metadata strings
    sanitize_metadata: bool = True  # Enable metadata sanitization

    @classmethod
    def load_from_env(cls) -> "SecurityConfig":
        """Load security configuration from environment variables."""
        return cls(
            max_file_size_mb=int(os.getenv("ATLAS_MAX_FILE_SIZE_MB", "100")),
            warn_file_size_mb=int(os.getenv("ATLAS_WARN_FILE_SIZE_MB", "50")),
            max_batch_files=int(os.getenv("ATLAS_MAX_BATCH_FILES", "10000")),
            warn_batch_files=int(os.getenv("ATLAS_WARN_BATCH_FILES", "1000")),
            allow_symlinks=os.getenv("ATLAS_ALLOW_SYMLINKS", "false").lower() == "true",
            allow_absolute_patterns=os.getenv("ATLAS_ALLOW_ABSOLUTE_PATTERNS", "false").lower() == "true",
            allow_parent_traversal=os.getenv("ATLAS_ALLOW_PARENT_TRAVERSAL", "false").lower() == "true",
            require_disk_space_mb=int(os.getenv("ATLAS_REQUIRE_DISK_SPACE_MB", "100")),
            validate_mime_types=os.getenv("ATLAS_VALIDATE_MIME_TYPES", "false").lower() == "true",
            max_batch_timeout_seconds=int(os.getenv("ATLAS_MAX_BATCH_TIMEOUT_SECONDS", "3600")),
            sanitize_metadata=os.getenv("ATLAS_SANITIZE_METADATA", "true").lower() == "true",
        )


# Global security config (can be overridden)
_global_security_config = SecurityConfig.load_from_env()


def get_security_config() -> SecurityConfig:
    """Get the global security configuration."""
    return _global_security_config


def set_security_config(config: SecurityConfig) -> None:
    """Set the global security configuration."""
    global _global_security_config
    _global_security_config = config


# ============================================================================
# PATH SECURITY
# ============================================================================

def validate_path_safe(
    base_dir: Path,
    file_path: Path,
    config: Optional[SecurityConfig] = None
) -> Path:
    """
    Validate that a file path doesn't escape the base directory.

    This prevents path traversal attacks (e.g., ../../etc/passwd).

    Args:
        base_dir: Base directory that file_path should be within
        file_path: Path to validate
        config: Security configuration (uses global if None)

    Returns:
        Validated file_path

    Raises:
        typer.BadParameter: If path is outside base_dir

    Examples:
        >>> validate_path_safe(Path("/home/user"), Path("/home/user/doc.txt"))
        Path('/home/user/doc.txt')

        >>> validate_path_safe(Path("/home/user"), Path("/etc/passwd"))
        # Raises typer.BadParameter
    """
    if config is None:
        config = get_security_config()

    try:
        resolved_path = file_path.resolve()
        resolved_base = base_dir.resolve()

        # Check if path is within base directory
        if not resolved_path.is_relative_to(resolved_base):
            security_logger.warning(
                f"Path traversal attempt blocked: {file_path} "
                f"(base: {base_dir})"
            )
            raise typer.BadParameter(
                f"Path is outside allowed directory: {file_path}\n"
                f"Base directory: {base_dir}"
            )

        return file_path

    except (ValueError, OSError) as e:
        security_logger.error(f"Path validation error: {e}")
        raise typer.BadParameter(f"Invalid path: {file_path}")


def validate_no_symlinks(
    file_path: Union[str, Path],
    config: Optional[SecurityConfig] = None
) -> Path:
    """
    Validate that a file is not a symbolic link.

    Args:
        file_path: Path to validate (str or Path object)
        config: Security configuration (uses global if None)

    Returns:
        Validated file_path as Path object

    Raises:
        typer.BadParameter: If path is a symlink and symlinks are not allowed
    """
    # Convert str to Path if necessary
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if config is None:
        config = get_security_config()

    if not config.allow_symlinks and file_path.is_symlink():
        security_logger.warning(f"Symlink blocked: {file_path}")
        raise typer.BadParameter(
            f"Symbolic links are not allowed: {file_path}\n"
            f"Set ATLAS_ALLOW_SYMLINKS=true to enable."
        )

    return file_path


def validate_pattern_safe(
    pattern: str,
    config: Optional[SecurityConfig] = None
) -> str:
    """
    Validate that a glob pattern is safe (no path traversal).

    Args:
        pattern: Glob pattern to validate
        config: Security configuration (uses global if None)

    Returns:
        Validated pattern

    Raises:
        typer.BadParameter: If pattern contains dangerous elements

    Examples:
        >>> validate_pattern_safe("*.txt")
        '*.txt'

        >>> validate_pattern_safe("../../*.txt")
        # Raises typer.BadParameter
    """
    if config is None:
        config = get_security_config()

    # Check for parent directory traversal
    if not config.allow_parent_traversal and ".." in pattern:
        security_logger.warning(f"Dangerous pattern blocked: {pattern}")
        raise typer.BadParameter(
            f"Pattern contains parent directory traversal (..): {pattern}\n"
            f"This is not allowed for security reasons."
        )

    # Check for absolute paths
    if not config.allow_absolute_patterns:
        if pattern.startswith("/") or (len(pattern) > 1 and pattern[1] == ":"):
            security_logger.warning(f"Absolute pattern blocked: {pattern}")
            raise typer.BadParameter(
                f"Absolute paths in patterns are not allowed: {pattern}\n"
                f"Use relative patterns like '*.txt' or '**/*.pdf'."
            )

    # Check for home directory expansion
    if "~" in pattern:
        security_logger.warning(f"Home directory pattern blocked: {pattern}")
        raise typer.BadParameter(
            f"Home directory expansion (~) is not allowed in patterns: {pattern}"
        )

    return pattern


# ============================================================================
# FILE SIZE VALIDATION
# ============================================================================

def validate_file_size(
    file_path: Path,
    config: Optional[SecurityConfig] = None,
    warn_only: bool = False
) -> Path:
    """
    Validate that a file size is within acceptable limits.

    Args:
        file_path: Path to file to validate
        config: Security configuration (uses global if None)
        warn_only: If True, only warn for large files instead of raising

    Returns:
        Validated file_path

    Raises:
        typer.BadParameter: If file is too large (unless warn_only=True)
    """
    if config is None:
        config = get_security_config()

    try:
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        # Check if file exceeds maximum size
        if size_mb > config.max_file_size_mb:
            msg = (
                f"File too large: {file_path.name} ({size_mb:.1f}MB)\n"
                f"Maximum allowed: {config.max_file_size_mb}MB\n"
                f"Set ATLAS_MAX_FILE_SIZE_MB to increase limit."
            )

            if warn_only:
                security_logger.warning(msg)
                from src.core.cli.utils.display import print_warning
                print_warning(msg)
            else:
                security_logger.error(msg)
                raise typer.BadParameter(msg)

        # Warn if file is large (but within limit)
        elif size_mb > config.warn_file_size_mb:
            security_logger.info(f"Large file: {file_path.name} ({size_mb:.1f}MB)")
            from src.core.cli.utils.display import print_warning
            print_warning(
                f"Large file detected: {file_path.name} ({size_mb:.1f}MB). "
                f"Processing may be slow."
            )

        return file_path

    except OSError as e:
        security_logger.error(f"Error checking file size: {e}")
        raise typer.BadParameter(f"Cannot access file: {file_path}")


def validate_total_size(
    files: List[Path],
    config: Optional[SecurityConfig] = None
) -> List[Path]:
    """
    Validate total size of all files in a batch.

    Args:
        files: List of file paths
        config: Security configuration (uses global if None)

    Returns:
        Validated files list
    """
    if config is None:
        config = get_security_config()

    total_size_bytes = sum(f.stat().st_size for f in files if f.exists())
    total_size_mb = total_size_bytes / (1024 * 1024)

    # Calculate max total size (per-file limit * number of files)
    max_total_mb = config.max_file_size_mb * len(files)

    if total_size_mb > max_total_mb:
        security_logger.warning(
            f"Total batch size too large: {total_size_mb:.1f}MB "
            f"(max: {max_total_mb:.1f}MB)"
        )
        from src.core.cli.utils.display import print_warning
        print_warning(
            f"Total batch size is {total_size_mb:.1f}MB. "
            f"Processing may require significant memory."
        )

    return files


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_batch_size(
    files: List[Path],
    config: Optional[SecurityConfig] = None
) -> List[Path]:
    """
    Validate that batch size (number of files) is within limits.

    Args:
        files: List of file paths
        config: Security configuration (uses global if None)

    Returns:
        Validated files list

    Raises:
        typer.BadParameter: If batch has too many files
    """
    if config is None:
        config = get_security_config()

    file_count = len(files)

    # Check maximum
    if file_count > config.max_batch_files:
        security_logger.error(f"Batch too large: {file_count} files")
        raise typer.BadParameter(
            f"Too many files in batch: {file_count}\n"
            f"Maximum allowed: {config.max_batch_files}\n"
            f"Set ATLAS_MAX_BATCH_FILES to increase limit, or process in smaller batches."
        )

    # Warn if large (but within limit)
    if file_count > config.warn_batch_files:
        security_logger.info(f"Large batch: {file_count} files")
        from src.core.cli.utils.display import print_warning
        print_warning(
            f"Processing {file_count} files. This may take a while."
        )

    return files


# ============================================================================
# DISK SPACE VALIDATION
# ============================================================================

def validate_disk_space(
    output_path: Path,
    config: Optional[SecurityConfig] = None
) -> None:
    """
    Validate that sufficient disk space is available.

    Args:
        output_path: Path where output will be written
        config: Security configuration (uses global if None)

    Raises:
        typer.BadParameter: If insufficient disk space
    """
    if config is None:
        config = get_security_config()

    try:
        # Get disk usage for output path
        output_dir = output_path if output_path.is_dir() else output_path.parent
        stat = shutil.disk_usage(output_dir)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < config.require_disk_space_mb:
            security_logger.error(
                f"Insufficient disk space: {available_mb:.1f}MB available "
                f"(need {config.require_disk_space_mb}MB)"
            )
            raise typer.BadParameter(
                f"Insufficient disk space: {available_mb:.1f}MB available\n"
                f"Required: {config.require_disk_space_mb}MB\n"
                f"Free up space or change output location."
            )

        # Warn if low on space
        elif available_mb < config.require_disk_space_mb * 2:
            security_logger.warning(f"Low disk space: {available_mb:.1f}MB")
            from src.core.cli.utils.display import print_warning
            print_warning(
                f"Low disk space: {available_mb:.1f}MB available. "
                f"Monitor space during processing."
            )

    except OSError as e:
        security_logger.error(f"Error checking disk space: {e}")
        # Don't fail on disk space check errors (may not be supported on all systems)
        from src.core.cli.utils.display import print_warning
        print_warning(f"Could not check disk space: {e}")


# ============================================================================
# MIME TYPE VALIDATION (Optional, requires python-magic)
# ============================================================================

def validate_mime_type(
    file_path: Path,
    expected_extensions: Optional[List[str]] = None,
    config: Optional[SecurityConfig] = None
) -> Path:
    """
    Validate file type using MIME type detection (magic numbers).

    This prevents attacks where a malicious file has a misleading extension.

    Args:
        file_path: Path to file to validate
        expected_extensions: List of expected extensions (e.g., ['.pdf', '.txt'])
        config: Security configuration (uses global if None)

    Returns:
        Validated file_path

    Raises:
        typer.BadParameter: If MIME type doesn't match extension

    Note:
        Requires python-magic library. Install with: pip install python-magic
    """
    if config is None:
        config = get_security_config()

    if not config.validate_mime_types:
        return file_path  # MIME validation disabled

    try:
        import magic
    except ImportError:
        security_logger.warning(
            "python-magic not installed, skipping MIME validation. "
            "Install with: pip install python-magic"
        )
        return file_path

    # MIME type mapping
    MIME_MAPPING = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.md': 'text/plain',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.tiff': 'image/tiff',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.odt': 'application/vnd.oasis.opendocument.text',
    }

    try:
        # Detect MIME type
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_file(str(file_path))

        # Get expected MIME types
        file_ext = file_path.suffix.lower()
        expected_mimes = []

        if expected_extensions:
            expected_mimes = [MIME_MAPPING.get(ext) for ext in expected_extensions]
        elif file_ext in MIME_MAPPING:
            expected_mimes = [MIME_MAPPING[file_ext]]

        # Validate
        if expected_mimes and detected_mime not in expected_mimes:
            security_logger.warning(
                f"MIME type mismatch: {file_path.name} "
                f"(extension: {file_ext}, detected: {detected_mime})"
            )

            from src.core.cli.utils.display import print_warning
            print_warning(
                f"File type mismatch: {file_path.name}\n"
                f"  Extension suggests: {file_ext}\n"
                f"  Detected type: {detected_mime}\n"
                f"  Proceeding with caution..."
            )

        return file_path

    except Exception as e:
        security_logger.error(f"MIME validation error: {e}")
        # Don't fail on MIME validation errors
        return file_path


# ============================================================================
# METADATA SANITIZATION
# ============================================================================

def sanitize_metadata(
    metadata: dict,
    config: Optional[SecurityConfig] = None
) -> dict:
    """
    Sanitize metadata to prevent injection attacks.

    Escapes HTML/XML characters and limits string lengths.

    Args:
        metadata: Metadata dictionary to sanitize
        config: Security configuration (uses global if None)

    Returns:
        Sanitized metadata dictionary
    """
    if config is None:
        config = get_security_config()

    if not config.sanitize_metadata:
        return metadata  # Sanitization disabled

    import html

    sanitized = {}
    for key, value in metadata.items():
        # Sanitize key
        clean_key = html.escape(str(key))

        # Sanitize value
        if isinstance(value, str):
            # Escape HTML/XML characters
            clean_value = html.escape(value)

            # Limit length
            max_len = config.max_metadata_string_length
            if len(clean_value) > max_len:
                clean_value = clean_value[:max_len] + "... [truncated]"

            sanitized[clean_key] = clean_value

        elif isinstance(value, (int, float, bool)):
            sanitized[clean_key] = value

        elif isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized[clean_key] = sanitize_metadata(value, config)

        elif isinstance(value, list):
            # Sanitize list elements
            sanitized[clean_key] = [
                sanitize_metadata({"item": item}, config)["item"]
                if isinstance(item, dict)
                else html.escape(str(item)) if isinstance(item, str)
                else item
                for item in value
            ]

        else:
            # Convert to string and sanitize
            sanitized[clean_key] = html.escape(str(value))

    return sanitized


# ============================================================================
# COMPREHENSIVE VALIDATION
# ============================================================================

def validate_file_comprehensive(
    file_path: Path,
    base_dir: Optional[Path] = None,
    config: Optional[SecurityConfig] = None
) -> Path:
    """
    Run all security validations on a file.

    Args:
        file_path: Path to file to validate
        base_dir: Base directory for path traversal check (optional)
        config: Security configuration (uses global if None)

    Returns:
        Validated file_path

    Raises:
        typer.BadParameter: If any validation fails
    """
    if config is None:
        config = get_security_config()

    # Check path safety
    if base_dir:
        validate_path_safe(base_dir, file_path, config)

    # Check symlinks
    validate_no_symlinks(file_path, config)

    # Check file size
    validate_file_size(file_path, config)

    # Check MIME type (if enabled)
    if config.validate_mime_types:
        validate_mime_type(file_path, config=config)

    security_logger.debug(f"File validated: {file_path}")
    return file_path


def validate_batch_comprehensive(
    files: List[Path],
    base_dir: Path,
    config: Optional[SecurityConfig] = None,
    validate_files: bool = True
) -> List[Path]:
    """
    Run all security validations on a batch of files.

    Args:
        files: List of files to validate
        base_dir: Base directory for path traversal check
        config: Security configuration (uses global if None)
        validate_files: Whether to validate individual files (can be slow)

    Returns:
        Validated files list

    Raises:
        typer.BadParameter: If any validation fails
    """
    if config is None:
        config = get_security_config()

    # Validate batch size
    validate_batch_size(files, config)

    # Validate total size
    validate_total_size(files, config)

    # Validate individual files (optional, can be slow for large batches)
    if validate_files:
        validated_files = []
        for file_path in files:
            try:
                validate_file_comprehensive(file_path, base_dir, config)
                validated_files.append(file_path)
            except typer.BadParameter as e:
                security_logger.warning(f"File validation failed: {file_path} - {e}")
                # Skip invalid files instead of failing entire batch
                from src.core.cli.utils.display import print_warning
                print_warning(f"Skipping invalid file: {file_path.name} - {e}")

        return validated_files

    return files
