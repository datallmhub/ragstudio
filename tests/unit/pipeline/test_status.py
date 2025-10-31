"""Unit tests for src.core.pipeline.status module."""

import pytest
from src.core.pipeline.status import (
    FileStatus,
    PipelineStatus,
    PipelineStats,
    format_status,
    format_pipeline_status,
)


class TestFileStatus:
    """Tests for FileStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses are defined."""
        assert FileStatus.PENDING == "pending"
        assert FileStatus.PROCESSING == "processing"
        assert FileStatus.SUCCESS == "success"
        assert FileStatus.FAILED == "failed"
        assert FileStatus.SKIPPED == "skipped"
        assert FileStatus.ABORTED == "aborted"

    def test_is_final(self):
        """Test is_final() method."""
        assert FileStatus.SUCCESS.is_final()
        assert FileStatus.FAILED.is_final()
        assert FileStatus.SKIPPED.is_final()
        assert FileStatus.ABORTED.is_final()
        assert not FileStatus.PENDING.is_final()
        assert not FileStatus.PROCESSING.is_final()

    def test_is_error(self):
        """Test is_error() method."""
        assert FileStatus.FAILED.is_error()
        assert FileStatus.SKIPPED.is_error()
        assert FileStatus.ABORTED.is_error()
        assert not FileStatus.SUCCESS.is_error()
        assert not FileStatus.PENDING.is_error()
        assert not FileStatus.PROCESSING.is_error()


class TestPipelineStatus:
    """Tests for PipelineStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses are defined."""
        assert PipelineStatus.INITIALIZING == "initializing"
        assert PipelineStatus.RUNNING == "running"
        assert PipelineStatus.DONE == "done"
        assert PipelineStatus.ABORTED == "aborted"
        assert PipelineStatus.FAILED == "failed"

    def test_is_final(self):
        """Test is_final() method."""
        assert PipelineStatus.DONE.is_final()
        assert PipelineStatus.ABORTED.is_final()
        assert PipelineStatus.FAILED.is_final()
        assert not PipelineStatus.INITIALIZING.is_final()
        assert not PipelineStatus.RUNNING.is_final()


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        stats = PipelineStats(total=10)
        assert stats.total == 10
        assert stats.success == 0
        assert stats.failed == 0
        assert stats.skipped == 0
        assert stats.aborted == 0

    def test_processed_count(self):
        """Test processed property."""
        stats = PipelineStats(total=10, success=5, failed=2, skipped=1, aborted=1)
        assert stats.processed == 9  # 5 + 2 + 1 + 1

    def test_remaining_count(self):
        """Test remaining property."""
        stats = PipelineStats(total=10, success=5, failed=2)
        assert stats.remaining == 3  # 10 - 7

    def test_success_rate_with_data(self):
        """Test success_rate calculation."""
        stats = PipelineStats(total=10, success=8, failed=2)
        assert stats.success_rate == 0.8  # 8/10

    def test_success_rate_no_data(self):
        """Test success_rate when no files processed."""
        stats = PipelineStats(total=10)
        assert stats.success_rate == 0.0

    def test_failure_rate_with_data(self):
        """Test failure_rate calculation."""
        stats = PipelineStats(total=10, success=7, failed=3)
        assert stats.failure_rate == 0.3  # 3/10

    def test_failure_rate_no_data(self):
        """Test failure_rate when no files processed."""
        stats = PipelineStats(total=10)
        assert stats.failure_rate == 0.0

    def test_error_rate_with_mixed_errors(self):
        """Test error_rate includes failed, skipped, and aborted."""
        stats = PipelineStats(total=10, success=6, failed=2, skipped=1, aborted=1)
        assert stats.error_rate == 0.4  # (2 + 1 + 1) / 10

    def test_increment_success(self):
        """Test increment() with success status."""
        stats = PipelineStats(total=10)
        stats.increment(FileStatus.SUCCESS)
        assert stats.success == 1
        assert stats.processed == 1

    def test_increment_failed(self):
        """Test increment() with failed status."""
        stats = PipelineStats(total=10)
        stats.increment(FileStatus.FAILED)
        assert stats.failed == 1
        assert stats.processed == 1

    def test_increment_skipped(self):
        """Test increment() with skipped status."""
        stats = PipelineStats(total=10)
        stats.increment(FileStatus.SKIPPED)
        assert stats.skipped == 1
        assert stats.processed == 1

    def test_increment_aborted(self):
        """Test increment() with aborted status."""
        stats = PipelineStats(total=10)
        stats.increment(FileStatus.ABORTED)
        assert stats.aborted == 1
        assert stats.processed == 1

    def test_increment_multiple(self):
        """Test multiple increments."""
        stats = PipelineStats(total=10)
        stats.increment(FileStatus.SUCCESS)
        stats.increment(FileStatus.SUCCESS)
        stats.increment(FileStatus.FAILED)
        stats.increment(FileStatus.SKIPPED)

        assert stats.success == 2
        assert stats.failed == 1
        assert stats.skipped == 1
        assert stats.processed == 4

    def test_to_dict(self):
        """Test to_dict() conversion."""
        stats = PipelineStats(total=10, success=8, failed=2)
        data = stats.to_dict()

        assert data["total"] == 10
        assert data["success"] == 8
        assert data["failed"] == 2
        assert data["skipped"] == 0
        assert data["aborted"] == 0
        assert data["processed"] == 10
        assert data["remaining"] == 0
        assert data["success_rate"] == 0.8
        assert data["failure_rate"] == 0.2
        assert "error_rate" in data

    def test_str_representation(self):
        """Test string representation."""
        stats = PipelineStats(total=10, success=8, failed=1, skipped=1)
        str_repr = str(stats)

        assert "Total: 10" in str_repr
        assert "Success: 8" in str_repr
        assert "Failed: 1" in str_repr
        assert "Skipped: 1" in str_repr


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_status(self):
        """Test format_status() function."""
        # Test each status
        assert "⏳" in format_status(FileStatus.PENDING)
        assert "pending" in format_status(FileStatus.PENDING)

        assert "⚙️" in format_status(FileStatus.PROCESSING)
        assert "processing" in format_status(FileStatus.PROCESSING)

        assert "✅" in format_status(FileStatus.SUCCESS)
        assert "success" in format_status(FileStatus.SUCCESS)

        assert "❌" in format_status(FileStatus.FAILED)
        assert "failed" in format_status(FileStatus.FAILED)

        assert "⏭" in format_status(FileStatus.SKIPPED)
        assert "skipped" in format_status(FileStatus.SKIPPED)

        assert "🛑" in format_status(FileStatus.ABORTED)
        assert "aborted" in format_status(FileStatus.ABORTED)

    def test_format_pipeline_status(self):
        """Test format_pipeline_status() function."""
        assert "🔄" in format_pipeline_status(PipelineStatus.INITIALIZING)
        assert "initializing" in format_pipeline_status(PipelineStatus.INITIALIZING)

        assert "▶️" in format_pipeline_status(PipelineStatus.RUNNING)
        assert "running" in format_pipeline_status(PipelineStatus.RUNNING)

        assert "✅" in format_pipeline_status(PipelineStatus.DONE)
        assert "done" in format_pipeline_status(PipelineStatus.DONE)

        assert "🛑" in format_pipeline_status(PipelineStatus.ABORTED)
        assert "aborted" in format_pipeline_status(PipelineStatus.ABORTED)

        assert "❌" in format_pipeline_status(PipelineStatus.FAILED)
        assert "failed" in format_pipeline_status(PipelineStatus.FAILED)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_stats_with_zero_total(self):
        """Test stats with zero total."""
        stats = PipelineStats(total=0)
        assert stats.processed == 0
        assert stats.remaining == 0
        assert stats.success_rate == 0.0

    def test_stats_all_success(self):
        """Test stats with all files successful."""
        stats = PipelineStats(total=10, success=10)
        assert stats.success_rate == 1.0
        assert stats.failure_rate == 0.0
        assert stats.error_rate == 0.0
        assert stats.remaining == 0

    def test_stats_all_failed(self):
        """Test stats with all files failed."""
        stats = PipelineStats(total=10, failed=10)
        assert stats.success_rate == 0.0
        assert stats.failure_rate == 1.0
        assert stats.error_rate == 1.0
        assert stats.remaining == 0

    def test_stats_mixed_errors(self):
        """Test stats with mixed error types."""
        stats = PipelineStats(
            total=10,
            success=5,
            failed=2,
            skipped=2,
            aborted=1
        )
        assert stats.processed == 10
        assert stats.success_rate == 0.5
        assert stats.error_rate == 0.5  # (2 + 2 + 1) / 10
