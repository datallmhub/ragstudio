"""Helper functions for batch command - following SOLID principles.

Each function has a single responsibility (SRP).
"""
from pathlib import Path
from typing import List, Tuple, Set


# Supported file extensions for batch processing
SUPPORTED_EXTENSIONS: Set[str] = {
    '.pdf', '.txt', '.jpeg', '.jpg', '.png',
    '.docx', '.md', '.html', '.htm'
}


def discover_files(directory: Path, pattern: str, recursive: bool) -> List[Path]:
    """
    Discover files matching pattern in directory.

    Single Responsibility: File discovery only.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.txt", "*.*")
        recursive: Whether to search recursively

    Returns:
        List of Path objects matching pattern

    Examples:
        >>> discover_files(Path("/docs"), "*.txt", False)
        [Path("/docs/file1.txt"), Path("/docs/file2.txt")]
    """
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def filter_supported_files(
    files: List[Path]
) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """
    Filter files to only supported extensions.

    Single Responsibility: Extension filtering only.

    Args:
        files: List of file paths to filter

    Returns:
        Tuple of (supported_files, unsupported_files)
        where unsupported_files = [(filename, extension), ...]

    Examples:
        >>> files = [Path("doc.pdf"), Path("img.gif"), Path("text.txt")]
        >>> supported, unsupported = filter_supported_files(files)
        >>> len(supported)
        2
        >>> len(unsupported)
        1
    """
    supported_files = []
    unsupported_files = []

    for file in files:
        if file.is_file():  # Skip directories
            ext = file.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                supported_files.append(file)
            else:
                unsupported_files.append((file.name, ext))

    return supported_files, unsupported_files


def validate_files_for_batch(
    files: List[Path],
    security_config,
    validate_no_symlinks_fn,
) -> List[Path]:
    """
    Validate files for batch processing (security checks).

    Single Responsibility: Security validation only.

    Args:
        files: List of files to validate
        security_config: Security configuration object
        validate_no_symlinks_fn: Function to validate symlinks

    Returns:
        List of validated file paths (invalid ones removed)

    Note:
        This function silently filters out invalid files.
        In production, you'd want to log/report skipped files.
    """
    validated_files = []

    for file_path in files:
        try:
            # Check symlinks
            validate_no_symlinks_fn(file_path, security_config)
            validated_files.append(file_path)
        except Exception:
            # Silently skip invalid files
            # In production, you'd log this
            pass

    return validated_files
