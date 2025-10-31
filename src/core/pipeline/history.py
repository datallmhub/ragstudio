"""History tracking for Atlas-RAG pipeline runs.

This module manages the history of pipeline executions, storing detailed
information about each run in ~/.atlasrag/history/.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

from src.core.pipeline.status import FileStatus, PipelineStatus

LOGGER = logging.getLogger(__name__)

# Default history directory
DEFAULT_HISTORY_DIR = Path.home() / ".atlasrag" / "history" / "runs"


@dataclass
class FileResult:
    """Result for a single file processing."""

    filename: str
    filepath: str  # Full path for retry
    status: FileStatus
    chunks_created: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    reason: Optional[str] = None
    duration: Optional[float] = None  # seconds
    retries: int = 0
    user_decision: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert FileStatus enum to string
        data['status'] = self.status.value if isinstance(self.status, FileStatus) else self.status
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileResult':
        """Create from dictionary."""
        # Convert status string to FileStatus enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = FileStatus(data['status'])
        return cls(**data)


@dataclass
class PipelineRun:
    """Complete pipeline run information."""

    run_id: str
    timestamp: str
    status: PipelineStatus
    total_files: int
    success: int = 0
    failed: int = 0
    skipped: int = 0
    aborted: int = 0
    duration: Optional[float] = None  # seconds
    mode: str = "interactive"
    config: Dict[str, Any] = field(default_factory=dict)
    files: List[FileResult] = field(default_factory=list)

    @property
    def processed(self) -> int:
        """Total processed files."""
        return self.success + self.failed + self.skipped + self.aborted

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        if self.processed == 0:
            return 0.0
        return self.success / self.processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert status enum to string
        data['status'] = self.status.value if isinstance(self.status, PipelineStatus) else self.status
        # Convert file results
        data['files'] = [f.to_dict() if isinstance(f, FileResult) else f for f in self.files]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineRun':
        """Create from dictionary."""
        # Convert status string to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = PipelineStatus(data['status'])

        # Convert file results
        if 'files' in data:
            data['files'] = [
                FileResult.from_dict(f) if isinstance(f, dict) else f
                for f in data['files']
            ]

        return cls(**data)


class HistoryManager:
    """Manages pipeline run history."""

    def __init__(self, history_dir: Optional[Path] = None):
        """
        Initialize history manager.

        Args:
            history_dir: Directory for storing run history
                        (defaults to ~/.atlasrag/history/runs)
        """
        self.history_dir = history_dir or DEFAULT_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.history_dir.parent / "index.json"

    def create_run(
        self,
        total_files: int,
        mode: str = "interactive",
        config: Optional[Dict[str, Any]] = None,
    ) -> PipelineRun:
        """
        Create a new pipeline run.

        Args:
            total_files: Total number of files to process
            mode: Execution mode (interactive, auto-continue, etc.)
            config: Optional configuration dict

        Returns:
            New PipelineRun instance
        """
        now = datetime.now()
        run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond:06d}"
        timestamp = now.isoformat()

        run = PipelineRun(
            run_id=run_id,
            timestamp=timestamp,
            status=PipelineStatus.INITIALIZING,
            total_files=total_files,
            mode=mode,
            config=config or {},
        )

        self._save_run(run)
        self._update_index(run)

        LOGGER.info(f"Created run: {run_id} ({total_files} files)")
        return run

    def start_run(self, run: PipelineRun) -> None:
        """Mark run as started."""
        run.status = PipelineStatus.RUNNING
        self._save_run(run)
        self._update_index(run)
        LOGGER.info(f"Started run: {run.run_id}")

    def update_run(
        self,
        run: PipelineRun,
        file_result: Optional[FileResult] = None,
    ) -> None:
        """
        Update run with new file result.

        Args:
            run: Pipeline run to update
            file_result: Optional file result to add
        """
        if file_result:
            run.files.append(file_result)

            # Update counters
            if file_result.status == FileStatus.SUCCESS:
                run.success += 1
            elif file_result.status == FileStatus.FAILED:
                run.failed += 1
            elif file_result.status == FileStatus.SKIPPED:
                run.skipped += 1
            elif file_result.status == FileStatus.ABORTED:
                run.aborted += 1

        self._save_run(run)
        # Don't update index for every file (performance)

    def finalize_run(self, run: PipelineRun, duration: float) -> None:
        """
        Finalize a run.

        Args:
            run: Pipeline run to finalize
            duration: Total duration in seconds
        """
        run.status = PipelineStatus.DONE
        run.duration = duration
        self._save_run(run)
        self._update_index(run)
        LOGGER.info(
            f"Finalized run: {run.run_id} "
            f"({run.success}/{run.total_files} success, {duration:.1f}s)"
        )

    def abort_run(self, run: PipelineRun, reason: Optional[str] = None) -> None:
        """
        Mark run as aborted.

        Args:
            run: Pipeline run to abort
            reason: Optional reason for abort
        """
        run.status = PipelineStatus.ABORTED
        if reason:
            run.config['abort_reason'] = reason
        self._save_run(run)
        self._update_index(run)
        LOGGER.warning(f"Aborted run: {run.run_id} - {reason or 'User requested'}")

    def fail_run(self, run: PipelineRun, reason: str) -> None:
        """
        Mark run as failed.

        Args:
            run: Pipeline run to fail
            reason: Reason for failure
        """
        run.status = PipelineStatus.FAILED
        run.config['failure_reason'] = reason
        self._save_run(run)
        self._update_index(run)
        LOGGER.error(f"Failed run: {run.run_id} - {reason}")

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """
        Get a specific run.

        Args:
            run_id: Run ID to retrieve

        Returns:
            PipelineRun if found, None otherwise
        """
        run_file = self.history_dir / f"{run_id}.json"
        if not run_file.exists():
            LOGGER.warning(f"Run not found: {run_id}")
            return None

        try:
            data = json.loads(run_file.read_text(encoding='utf-8'))
            return PipelineRun.from_dict(data)
        except Exception as e:
            LOGGER.error(f"Error loading run {run_id}: {e}")
            return None

    def list_runs(
        self,
        limit: Optional[int] = None,
        status_filter: Optional[PipelineStatus] = None,
    ) -> List[PipelineRun]:
        """
        List all runs (most recent first).

        Args:
            limit: Maximum number of runs to return
            status_filter: Optional status to filter by

        Returns:
            List of PipelineRun instances
        """
        if not self.index_file.exists():
            return []

        try:
            index = json.loads(self.index_file.read_text(encoding='utf-8'))
        except Exception as e:
            LOGGER.error(f"Error reading index: {e}")
            return []

        runs = []
        for entry in reversed(index):  # most recent first
            run = self.get_run(entry['run_id'])
            if run:
                if status_filter is None or run.status == status_filter:
                    runs.append(run)
                    if limit and len(runs) >= limit:
                        break

        return runs

    def get_last_run(self) -> Optional[PipelineRun]:
        """Get the most recent run."""
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def get_last_failed_run(self) -> Optional[PipelineRun]:
        """Get the most recent run with failures."""
        runs = self.list_runs(limit=20)  # Check last 20 runs
        for run in runs:
            if run.failed > 0 or run.skipped > 0:
                return run
        return None

    def get_failed_files(self, run_id: str) -> List[FileResult]:
        """
        Get all failed/skipped files from a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of FileResult with failed or skipped status
        """
        run = self.get_run(run_id)
        if not run:
            return []

        return [
            f for f in run.files
            if f.status in (FileStatus.FAILED, FileStatus.SKIPPED)
        ]

    def cleanup_old_runs(self, days: int = 30) -> int:
        """
        Delete runs older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of runs deleted
        """
        if not self.index_file.exists():
            return 0

        try:
            index = json.loads(self.index_file.read_text(encoding='utf-8'))
        except Exception as e:
            LOGGER.error(f"Error reading index: {e}")
            return 0

        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted = 0

        new_index = []
        for entry in index:
            try:
                # Parse timestamp
                ts = datetime.fromisoformat(entry['timestamp']).timestamp()
                if ts < cutoff:
                    # Delete run file
                    run_file = self.history_dir / f"{entry['run_id']}.json"
                    if run_file.exists():
                        run_file.unlink()
                        deleted += 1
                else:
                    new_index.append(entry)
            except Exception as e:
                LOGGER.warning(f"Error processing entry {entry.get('run_id')}: {e}")
                new_index.append(entry)  # Keep on error

        # Save updated index
        if deleted > 0:
            self.index_file.write_text(
                json.dumps(new_index, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            LOGGER.info(f"Cleaned up {deleted} old runs")

        return deleted

    def _save_run(self, run: PipelineRun) -> None:
        """Save run to disk."""
        run_file = self.history_dir / f"{run.run_id}.json"
        try:
            run_file.write_text(
                json.dumps(run.to_dict(), indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            LOGGER.error(f"Error saving run {run.run_id}: {e}")

    def _update_index(self, run: PipelineRun) -> None:
        """Update the index file."""
        try:
            if self.index_file.exists():
                index = json.loads(self.index_file.read_text(encoding='utf-8'))
            else:
                index = []

            # Remove existing entry if any
            index = [e for e in index if e['run_id'] != run.run_id]

            # Add new entry
            index.append({
                "run_id": run.run_id,
                "timestamp": run.timestamp,
                "status": run.status.value if isinstance(run.status, PipelineStatus) else run.status,
                "total_files": run.total_files,
                "success": run.success,
                "failed": run.failed,
                "skipped": run.skipped,
                "aborted": run.aborted,
                "mode": run.mode,
            })

            self.index_file.write_text(
                json.dumps(index, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            LOGGER.error(f"Error updating index: {e}")
