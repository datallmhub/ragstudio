"""Validation utilities for CLI."""
from pathlib import Path
from typing import Optional
import typer


def validate_file_exists(file_path: Path) -> Path:
    """Validate that a file exists."""
    if not file_path.exists():
        raise typer.BadParameter(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise typer.BadParameter(f"Path is not a file: {file_path}")
    return file_path


def validate_directory_exists(dir_path: Path) -> Path:
    """Validate that a directory exists."""
    if not dir_path.exists():
        raise typer.BadParameter(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise typer.BadParameter(f"Path is not a directory: {dir_path}")
    return dir_path


def validate_output_path(output_path: Optional[Path]) -> Optional[Path]:
    """Validate output path (parent directory must exist)."""
    if output_path is None:
        return None

    parent = output_path.parent
    if not parent.exists():
        raise typer.BadParameter(
            f"Parent directory does not exist: {parent}. "
            f"Please create it first or use an existing directory."
        )
    return output_path


def validate_token_range(value: int, min_val: int = 50, max_val: int = 2000) -> int:
    """Validate token value is within range."""
    if value < min_val or value > max_val:
        raise typer.BadParameter(f"Token value must be between {min_val} and {max_val}")
    return value
