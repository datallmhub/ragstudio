"""Helper functions for retry command - following SOLID principles.

Each function has a single responsibility (SRP).
"""
from typing import Optional, List, Dict, Any, Tuple

from src.core.pipeline import HistoryManager, FileStatus, PipelineStatus
from src.core.cli.utils.display import console, print_error, print_info, print_warning


def get_run_to_retry(run_id: Optional[str]) -> Optional[Any]:
    """
    Retrieve run to retry from history. SRP: Run retrieval only.

    Args:
        run_id: Specific run ID or None to get last failed run

    Returns:
        Run object or None if not found

    Raises:
        ValueError: If run not found
    """
    history = HistoryManager()

    if run_id is None:
        console.print("[yellow]ℹ️[/yellow] No run_id provided, finding last failed run...")
        run = history.get_last_failed_run()

        if not run:
            raise ValueError("No failed runs found in history")

        return run

    try:
        run = history.get_run(run_id)
        if not run:
            raise ValueError(f"Run '{run_id}' not found in history")
        return run
    except Exception as e:
        raise ValueError(f"Error retrieving run: {e}")


def extract_failed_files(run: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract failed and skipped files from run. SRP: File extraction only.

    Args:
        run: Run object from history

    Returns:
        Tuple of (failed_files, skipped_files) with file details

    Examples:
        >>> failed, skipped = extract_failed_files(mock_run)
        >>> isinstance(failed, list) and isinstance(skipped, list)
        True
    """
    failed_files = []
    skipped_files = []

    try:
        # Get files from run history
        files = run.files if hasattr(run, 'files') else {}

        for filename, file_info in files.items():
            file_status = file_info.get('status') if isinstance(file_info, dict) else None

            if file_status == FileStatus.FAILED.value:
                failed_files.append({
                    'filename': filename,
                    'status': FileStatus.FAILED.value,
                    'error': file_info.get('error', 'Unknown error') if isinstance(file_info, dict) else None
                })

            elif file_status == FileStatus.SKIPPED.value:
                skipped_files.append({
                    'filename': filename,
                    'status': FileStatus.SKIPPED.value,
                    'reason': file_info.get('reason', 'Unknown reason') if isinstance(file_info, dict) else None
                })

    except Exception as e:
        print_warning(f"Error extracting files from run: {e}")

    return failed_files, skipped_files


def display_retry_files(failed_files: List[Dict[str, Any]], skipped_files: List[Dict[str, Any]]) -> None:
    """
    Display files that will be retried. SRP: Display formatting only.

    Args:
        failed_files: List of failed file dictionaries
        skipped_files: List of skipped file dictionaries
    """
    console.print()

    if not failed_files and not skipped_files:
        print_info("No failed or skipped files found - nothing to retry")
        console.print()
        return

    if failed_files:
        console.print(f"[bold red]Failed files ({len(failed_files)}):[/bold red]")
        for i, file_info in enumerate(failed_files[:10], 1):
            error = file_info.get('error', 'Unknown error')
            console.print(f"  {i}. {file_info['filename']}")
            console.print(f"     Error: {error}")

        if len(failed_files) > 10:
            console.print(f"  ... and {len(failed_files) - 10} more")
        console.print()

    if skipped_files:
        console.print(f"[bold yellow]Skipped files ({len(skipped_files)}):[/bold yellow]")
        for i, file_info in enumerate(skipped_files[:10], 1):
            reason = file_info.get('reason', 'Unknown reason')
            console.print(f"  {i}. {file_info['filename']}")
            console.print(f"     Reason: {reason}")

        if len(skipped_files) > 10:
            console.print(f"  ... and {len(skipped_files) - 10} more")
        console.print()


def validate_retry_mode(mode: str) -> bool:
    """
    Validate retry execution mode. SRP: Mode validation only.

    Args:
        mode: Execution mode string

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_retry_mode("interactive")
        True
        >>> validate_retry_mode("invalid_mode")
        False
    """
    valid_modes = ["interactive", "auto-continue", "auto-stop", "auto-skip"]
    return mode.lower() in valid_modes


def get_run_summary(run: Any) -> Dict[str, Any]:
    """
    Get summary statistics from run. SRP: Summary extraction only.

    Args:
        run: Run object from history

    Returns:
        Dictionary with run statistics

    Examples:
        >>> summary = get_run_summary(mock_run)
        >>> "total_files" in summary
        True
    """
    try:
        return {
            "run_id": run.run_id if hasattr(run, 'run_id') else "unknown",
            "status": run.status if hasattr(run, 'status') else "unknown",
            "created_at": run.created_at if hasattr(run, 'created_at') else "unknown",
            "total_files": len(run.files) if hasattr(run, 'files') else 0,
            "mode": run.mode if hasattr(run, 'mode') else "unknown",
        }
    except Exception as e:
        print_warning(f"Error extracting run summary: {e}")
        return {}
