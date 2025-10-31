"""Status management for pipeline execution.

This module provides enums and utilities for tracking the status of files
and pipeline runs during Atlas-RAG execution.
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass, asdict


class FileStatus(str, Enum):
    """Status of a file in the pipeline."""

    PENDING = "pending"          # Not yet processed
    PROCESSING = "processing"    # Currently processing
    SUCCESS = "success"          # Successfully processed
    FAILED = "failed"            # Failed after retries
    SKIPPED = "skipped"          # Skipped by user or system
    ABORTED = "aborted"          # Pipeline aborted

    def is_final(self) -> bool:
        """Check if this is a final status."""
        return self in (FileStatus.SUCCESS, FileStatus.FAILED,
                       FileStatus.SKIPPED, FileStatus.ABORTED)

    def is_error(self) -> bool:
        """Check if this represents an error."""
        return self in (FileStatus.FAILED, FileStatus.SKIPPED, FileStatus.ABORTED)


class PipelineStatus(str, Enum):
    """Status of the entire pipeline."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    DONE = "done"
    ABORTED = "aborted"
    FAILED = "failed"

    def is_final(self) -> bool:
        """Check if this is a final status."""
        return self in (PipelineStatus.DONE, PipelineStatus.ABORTED,
                       PipelineStatus.FAILED)


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    aborted: int = 0

    @property
    def processed(self) -> int:
        """Total processed files (excluding pending)."""
        return self.success + self.failed + self.skipped + self.aborted

    @property
    def remaining(self) -> int:
        """Remaining files to process."""
        return self.total - self.processed

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        if self.processed == 0:
            return 0.0
        return self.success / self.processed

    @property
    def failure_rate(self) -> float:
        """Failure rate (0-1)."""
        if self.processed == 0:
            return 0.0
        return self.failed / self.processed

    @property
    def error_rate(self) -> float:
        """Overall error rate including skipped and aborted (0-1)."""
        if self.processed == 0:
            return 0.0
        return (self.failed + self.skipped + self.aborted) / self.processed

    def increment(self, status: FileStatus) -> None:
        """Increment counter for given status."""
        if status == FileStatus.SUCCESS:
            self.success += 1
        elif status == FileStatus.FAILED:
            self.failed += 1
        elif status == FileStatus.SKIPPED:
            self.skipped += 1
        elif status == FileStatus.ABORTED:
            self.aborted += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "aborted": self.aborted,
            "processed": self.processed,
            "remaining": self.remaining,
            "success_rate": round(self.success_rate, 3),
            "failure_rate": round(self.failure_rate, 3),
            "error_rate": round(self.error_rate, 3),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"Total: {self.total}",
            f"✅ Success: {self.success} ({self.success_rate:.1%})",
            f"❌ Failed: {self.failed}",
            f"⚠️  Skipped: {self.skipped}",
        ]
        if self.aborted > 0:
            lines.append(f"🛑 Aborted: {self.aborted}")
        lines.append(f"Remaining: {self.remaining}")
        return "\n".join(lines)


def format_status(status: FileStatus) -> str:
    """Format status with color emoji."""
    emoji_map = {
        FileStatus.PENDING: "⏳",
        FileStatus.PROCESSING: "⚙️",
        FileStatus.SUCCESS: "✅",
        FileStatus.FAILED: "❌",
        FileStatus.SKIPPED: "⏭",
        FileStatus.ABORTED: "🛑",
    }
    return f"{emoji_map.get(status, '❓')} {status.value}"


def format_pipeline_status(status: PipelineStatus) -> str:
    """Format pipeline status with color emoji."""
    emoji_map = {
        PipelineStatus.INITIALIZING: "🔄",
        PipelineStatus.RUNNING: "▶️",
        PipelineStatus.DONE: "✅",
        PipelineStatus.ABORTED: "🛑",
        PipelineStatus.FAILED: "❌",
    }
    return f"{emoji_map.get(status, '❓')} {status.value}"
