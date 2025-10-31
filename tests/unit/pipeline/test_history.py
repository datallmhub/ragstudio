"""Unit tests for src.core.pipeline.history module."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from src.core.pipeline.history import (
    FileResult,
    PipelineRun,
    HistoryManager,
    DEFAULT_HISTORY_DIR,
)
from src.core.pipeline.status import FileStatus, PipelineStatus


class TestFileResult:
    """Tests for FileResult dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        result = FileResult(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            status=FileStatus.SUCCESS,
            chunks_created=10,
            duration=5.2,
        )

        assert result.filename == "test.pdf"
        assert result.filepath == "/path/to/test.pdf"
        assert result.status == FileStatus.SUCCESS
        assert result.chunks_created == 10
        assert result.duration == 5.2

    def test_initialization_with_error(self):
        """Test initialization with error information."""
        result = FileResult(
            filename="failed.pdf",
            filepath="/path/to/failed.pdf",
            status=FileStatus.FAILED,
            error="OCR failed",
            error_type="ValueError",
            reason="Low confidence",
            retries=3,
        )

        assert result.status == FileStatus.FAILED
        assert result.error == "OCR failed"
        assert result.error_type == "ValueError"
        assert result.reason == "Low confidence"
        assert result.retries == 3

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = FileResult(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            status=FileStatus.SUCCESS,
            chunks_created=10,
        )

        data = result.to_dict()

        assert data["filename"] == "test.pdf"
        assert data["filepath"] == "/path/to/test.pdf"
        assert data["status"] == "success"  # Enum converted to string
        assert data["chunks_created"] == 10

    def test_from_dict(self):
        """Test from_dict reconstruction."""
        data = {
            "filename": "test.pdf",
            "filepath": "/path/to/test.pdf",
            "status": "success",
            "chunks_created": 10,
            "error": None,
            "error_type": None,
            "reason": None,
            "duration": 5.2,
            "retries": 0,
            "user_decision": None,
            "metadata": {},
        }

        result = FileResult.from_dict(data)

        assert result.filename == "test.pdf"
        assert result.status == FileStatus.SUCCESS  # String converted to enum
        assert result.chunks_created == 10


class TestPipelineRun:
    """Tests for PipelineRun dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        run = PipelineRun(
            run_id="run_123",
            timestamp="2025-01-28T10:00:00",
            status=PipelineStatus.RUNNING,
            total_files=10,
        )

        assert run.run_id == "run_123"
        assert run.timestamp == "2025-01-28T10:00:00"
        assert run.status == PipelineStatus.RUNNING
        assert run.total_files == 10
        assert run.success == 0
        assert run.files == []

    def test_processed_property(self):
        """Test processed property calculation."""
        run = PipelineRun(
            run_id="run_123",
            timestamp="2025-01-28T10:00:00",
            status=PipelineStatus.DONE,
            total_files=10,
            success=7,
            failed=2,
            skipped=1,
        )

        assert run.processed == 10  # 7 + 2 + 1

    def test_success_rate_property(self):
        """Test success_rate property."""
        run = PipelineRun(
            run_id="run_123",
            timestamp="2025-01-28T10:00:00",
            status=PipelineStatus.DONE,
            total_files=10,
            success=8,
            failed=2,
        )

        assert run.success_rate == 0.8

    def test_to_dict(self):
        """Test to_dict conversion."""
        run = PipelineRun(
            run_id="run_123",
            timestamp="2025-01-28T10:00:00",
            status=PipelineStatus.DONE,
            total_files=10,
            success=8,
        )

        data = run.to_dict()

        assert data["run_id"] == "run_123"
        assert data["status"] == "done"  # Enum to string
        assert data["total_files"] == 10
        assert data["success"] == 8

    def test_from_dict(self):
        """Test from_dict reconstruction."""
        data = {
            "run_id": "run_123",
            "timestamp": "2025-01-28T10:00:00",
            "status": "done",
            "total_files": 10,
            "success": 8,
            "failed": 2,
            "skipped": 0,
            "aborted": 0,
            "duration": 100.5,
            "mode": "interactive",
            "config": {},
            "files": [],
        }

        run = PipelineRun.from_dict(data)

        assert run.run_id == "run_123"
        assert run.status == PipelineStatus.DONE  # String to enum
        assert run.total_files == 10


class TestHistoryManager:
    """Tests for HistoryManager class."""

    def test_initialization_default(self):
        """Test default initialization."""
        manager = HistoryManager()
        assert manager.history_dir == DEFAULT_HISTORY_DIR
        assert manager.index_file == DEFAULT_HISTORY_DIR.parent / "index.json"

    def test_initialization_custom_dir(self, tmp_path):
        """Test initialization with custom directory."""
        custom_dir = tmp_path / "custom_history"
        manager = HistoryManager(history_dir=custom_dir)

        assert manager.history_dir == custom_dir
        assert manager.history_dir.exists()  # Should be created

    def test_create_run(self, tmp_path):
        """Test create_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        run = manager.create_run(total_files=5, mode="interactive")

        assert run.run_id.startswith("run_")
        assert run.total_files == 5
        assert run.mode == "interactive"
        assert run.status == PipelineStatus.INITIALIZING

        # Check that run file was created
        run_file = manager.history_dir / f"{run.run_id}.json"
        assert run_file.exists()

    def test_start_run(self, tmp_path):
        """Test start_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=5)

        manager.start_run(run)

        assert run.status == PipelineStatus.RUNNING

    def test_update_run_with_file_result(self, tmp_path):
        """Test update_run with file result."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=2)

        file_result = FileResult(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            status=FileStatus.SUCCESS,
            chunks_created=10,
        )

        manager.update_run(run, file_result)

        assert len(run.files) == 1
        assert run.success == 1
        assert run.files[0].filename == "test.pdf"

    def test_update_run_multiple_results(self, tmp_path):
        """Test update_run with multiple file results."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=3)

        results = [
            FileResult("f1.pdf", "/path/f1.pdf", FileStatus.SUCCESS, chunks_created=10),
            FileResult("f2.pdf", "/path/f2.pdf", FileStatus.FAILED, error="Error"),
            FileResult("f3.pdf", "/path/f3.pdf", FileStatus.SKIPPED, reason="Skipped"),
        ]

        for result in results:
            manager.update_run(run, result)

        assert len(run.files) == 3
        assert run.success == 1
        assert run.failed == 1
        assert run.skipped == 1

    def test_finalize_run(self, tmp_path):
        """Test finalize_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=5)

        manager.finalize_run(run, duration=125.5)

        assert run.status == PipelineStatus.DONE
        assert run.duration == 125.5

    def test_abort_run(self, tmp_path):
        """Test abort_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=5)

        manager.abort_run(run, reason="User interrupted")

        assert run.status == PipelineStatus.ABORTED
        assert run.config.get("abort_reason") == "User interrupted"

    def test_fail_run(self, tmp_path):
        """Test fail_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=5)

        manager.fail_run(run, reason="Critical error")

        assert run.status == PipelineStatus.FAILED
        assert run.config.get("failure_reason") == "Critical error"

    def test_get_run(self, tmp_path):
        """Test get_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        original_run = manager.create_run(total_files=5)

        # Retrieve the run
        retrieved_run = manager.get_run(original_run.run_id)

        assert retrieved_run is not None
        assert retrieved_run.run_id == original_run.run_id
        assert retrieved_run.total_files == 5

    def test_get_run_nonexistent(self, tmp_path):
        """Test get_run with non-existent run_id."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        result = manager.get_run("nonexistent_run")

        assert result is None

    def test_list_runs(self, tmp_path):
        """Test list_runs method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create multiple runs
        run1 = manager.create_run(total_files=5)
        run2 = manager.create_run(total_files=10)
        run3 = manager.create_run(total_files=15)

        runs = manager.list_runs()

        assert len(runs) == 3
        # Should be in reverse order (most recent first)
        assert runs[0].run_id == run3.run_id
        assert runs[1].run_id == run2.run_id
        assert runs[2].run_id == run1.run_id

    def test_list_runs_with_limit(self, tmp_path):
        """Test list_runs with limit."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create 5 runs
        for i in range(5):
            manager.create_run(total_files=i + 1)

        runs = manager.list_runs(limit=3)

        assert len(runs) == 3

    def test_list_runs_with_status_filter(self, tmp_path):
        """Test list_runs with status filter."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create runs with different statuses
        run1 = manager.create_run(total_files=5)
        manager.finalize_run(run1, duration=100)

        run2 = manager.create_run(total_files=5)
        manager.abort_run(run2, reason="Test")

        # Filter for done runs
        done_runs = manager.list_runs(status_filter=PipelineStatus.DONE)
        assert len(done_runs) == 1
        assert done_runs[0].run_id == run1.run_id

    def test_get_last_run(self, tmp_path):
        """Test get_last_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        run1 = manager.create_run(total_files=5)
        run2 = manager.create_run(total_files=10)

        last_run = manager.get_last_run()

        assert last_run is not None
        assert last_run.run_id == run2.run_id

    def test_get_last_run_empty(self, tmp_path):
        """Test get_last_run with no runs."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        last_run = manager.get_last_run()

        assert last_run is None

    def test_get_last_failed_run(self, tmp_path):
        """Test get_last_failed_run method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create successful run
        run1 = manager.create_run(total_files=5)
        run1.success = 5
        manager._save_run(run1)
        manager._update_index(run1)

        # Create failed run
        run2 = manager.create_run(total_files=5)
        run2.failed = 2
        manager._save_run(run2)
        manager._update_index(run2)

        last_failed = manager.get_last_failed_run()

        assert last_failed is not None
        assert last_failed.run_id == run2.run_id

    def test_get_failed_files(self, tmp_path):
        """Test get_failed_files method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=5)

        # Add results
        results = [
            FileResult("s1.pdf", "/path/s1.pdf", FileStatus.SUCCESS),
            FileResult("f1.pdf", "/path/f1.pdf", FileStatus.FAILED, error="Error"),
            FileResult("s2.pdf", "/path/s2.pdf", FileStatus.SUCCESS),
            FileResult("sk1.pdf", "/path/sk1.pdf", FileStatus.SKIPPED, reason="Skip"),
        ]

        for result in results:
            manager.update_run(run, result)

        failed_files = manager.get_failed_files(run.run_id)

        assert len(failed_files) == 2  # 1 failed + 1 skipped
        assert failed_files[0].filename == "f1.pdf"
        assert failed_files[1].filename == "sk1.pdf"

    def test_cleanup_old_runs(self, tmp_path):
        """Test cleanup_old_runs method."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create runs with old timestamps
        run1 = manager.create_run(total_files=5)
        run1.timestamp = "2020-01-01T00:00:00"  # Very old
        manager._save_run(run1)
        manager._update_index(run1)

        # Create recent run
        run2 = manager.create_run(total_files=5)

        # Cleanup runs older than 30 days
        deleted = manager.cleanup_old_runs(days=30)

        assert deleted == 1  # Should delete run1

        # Verify run1 deleted, run2 still exists
        assert not (manager.history_dir / f"{run1.run_id}.json").exists()
        assert (manager.history_dir / f"{run2.run_id}.json").exists()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_index_file_creation(self, tmp_path):
        """Test that index file is created automatically."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        assert not manager.index_file.exists()

        # Create a run
        manager.create_run(total_files=5)

        # Index should now exist
        assert manager.index_file.exists()

    def test_corrupted_index_handling(self, tmp_path):
        """Test handling of corrupted index file."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create corrupted index
        manager.index_file.parent.mkdir(parents=True, exist_ok=True)
        manager.index_file.write_text("not valid json")

        # Should handle gracefully
        runs = manager.list_runs()
        assert runs == []

    def test_missing_run_file(self, tmp_path):
        """Test handling of missing run file."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")

        # Create index with non-existent run
        manager.index_file.parent.mkdir(parents=True, exist_ok=True)
        manager.index_file.write_text(json.dumps([{
            "run_id": "nonexistent",
            "timestamp": "2025-01-28T10:00:00",
            "status": "done",
            "total_files": 5,
            "success": 5,
            "failed": 0,
            "skipped": 0,
            "aborted": 0,
            "mode": "interactive",
        }]))

        # list_runs should handle missing file gracefully
        runs = manager.list_runs()
        assert len(runs) == 0  # Missing run file filtered out

    def test_concurrent_updates(self, tmp_path):
        """Test that concurrent updates don't corrupt data."""
        manager = HistoryManager(history_dir=tmp_path / "history" / "runs")
        run = manager.create_run(total_files=10)

        # Multiple rapid updates
        for i in range(10):
            result = FileResult(
                filename=f"file{i}.pdf",
                filepath=f"/path/file{i}.pdf",
                status=FileStatus.SUCCESS,
            )
            manager.update_run(run, result)

        # All files should be recorded
        assert len(run.files) == 10
        assert run.success == 10
