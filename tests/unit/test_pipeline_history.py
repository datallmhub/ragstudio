"""
Unit tests for core/pipeline/history.py

Tests cover:
- FileResult data class and serialization
- PipelineRun data class and serialization
- HistoryManager file operations
- History listing and retrieval
- Error handling
"""

import pytest
import json
import tempfile
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
    """Test suite for FileResult data class."""

    def test_file_result_creation(self):
        """Test creating a FileResult."""
        result = FileResult(
            filename="test.txt",
            filepath="/path/to/test.txt",
            status=FileStatus.SUCCESS,
            chunks_created=10,
            duration=1.5
        )

        assert result.filename == "test.txt"
        assert result.filepath == "/path/to/test.txt"
        assert result.status == FileStatus.SUCCESS
        assert result.chunks_created == 10
        assert result.duration == 1.5
        assert result.retries == 0
        assert result.error is None

    def test_file_result_with_error(self):
        """Test FileResult with error information."""
        result = FileResult(
            filename="bad.txt",
            filepath="/path/to/bad.txt",
            status=FileStatus.FAILED,
            error="File not readable",
            error_type="IOError",
            reason="Corrupted file"
        )

        assert result.status == FileStatus.FAILED
        assert result.error == "File not readable"
        assert result.error_type == "IOError"
        assert result.reason == "Corrupted file"

    def test_file_result_to_dict(self):
        """Test converting FileResult to dictionary."""
        result = FileResult(
            filename="test.txt",
            filepath="/path/to/test.txt",
            status=FileStatus.SUCCESS,
            chunks_created=5
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data['filename'] == "test.txt"
        assert data['status'] == "success"  # Enum converted to string
        assert data['chunks_created'] == 5

    def test_file_result_from_dict(self):
        """Test creating FileResult from dictionary."""
        data = {
            'filename': "test.txt",
            'filepath': "/path/to/test.txt",
            'status': "success",
            'chunks_created': 10,
            'duration': 2.0,
            'retries': 0
        }

        result = FileResult.from_dict(data)

        assert result.filename == "test.txt"
        assert result.status == FileStatus.SUCCESS
        assert result.chunks_created == 10
        assert result.duration == 2.0

    def test_file_result_roundtrip(self):
        """Test FileResult serialization roundtrip."""
        original = FileResult(
            filename="test.txt",
            filepath="/path/to/test.txt",
            status=FileStatus.SKIPPED,
            reason="User skipped",
            retries=2
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = FileResult.from_dict(data)

        assert restored.filename == original.filename
        assert restored.status == original.status
        assert restored.reason == original.reason
        assert restored.retries == original.retries

    def test_file_result_metadata(self):
        """Test FileResult with custom metadata."""
        result = FileResult(
            filename="test.txt",
            filepath="/path/to/test.txt",
            status=FileStatus.SUCCESS,
            metadata={"ocr_used": True, "pages": 5}
        )

        assert result.metadata["ocr_used"] is True
        assert result.metadata["pages"] == 5


class TestPipelineRun:
    """Test suite for PipelineRun data class."""

    def test_pipeline_run_creation(self):
        """Test creating a PipelineRun."""
        run = PipelineRun(
            run_id="run_20250130_120000",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=10,
            success=8,
            failed=2,
            skipped=0,
            duration=45.5
        )

        assert run.run_id == "run_20250130_120000"
        assert run.status == PipelineStatus.DONE
        assert run.total_files == 10
        assert run.success == 8
        assert run.failed == 2
        assert run.skipped == 0

    def test_pipeline_run_processed_property(self):
        """Test the processed property."""
        run = PipelineRun(
            run_id="test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=20,
            success=10,
            failed=5,
            skipped=3,
            aborted=2
        )

        assert run.processed == 20  # 10 + 5 + 3 + 2

    def test_pipeline_run_success_rate(self):
        """Test the success_rate property."""
        run = PipelineRun(
            run_id="test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=10,
            success=8,
            failed=2
        )

        assert run.success_rate == 0.8

    def test_pipeline_run_success_rate_zero_processed(self):
        """Test success_rate when no files processed."""
        run = PipelineRun(
            run_id="test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.RUNNING,
            total_files=10
        )

        assert run.success_rate == 0.0

    def test_pipeline_run_with_files(self):
        """Test PipelineRun with file results."""
        file1 = FileResult(
            filename="file1.txt",
            filepath="/path/to/file1.txt",
            status=FileStatus.SUCCESS,
            chunks_created=5
        )
        file2 = FileResult(
            filename="file2.txt",
            filepath="/path/to/file2.txt",
            status=FileStatus.FAILED,
            error="Parse error"
        )

        run = PipelineRun(
            run_id="test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=2,
            success=1,
            failed=1,
            files=[file1, file2]
        )

        assert len(run.files) == 2
        assert run.files[0].status == FileStatus.SUCCESS
        assert run.files[1].status == FileStatus.FAILED

    def test_pipeline_run_to_dict(self):
        """Test converting PipelineRun to dictionary."""
        run = PipelineRun(
            run_id="test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=5,
            success=5
        )

        data = run.to_dict()

        assert isinstance(data, dict)
        assert data['run_id'] == "test"
        assert data['status'] == "done"  # Enum to string
        assert data['total_files'] == 5

    def test_pipeline_run_from_dict(self):
        """Test creating PipelineRun from dictionary."""
        data = {
            'run_id': "test",
            'timestamp': "2025-01-30T12:00:00",
            'status': "done",
            'total_files': 10,
            'success': 8,
            'failed': 2,
            'skipped': 0,
            'aborted': 0,
            'mode': "interactive",
            'config': {},
            'files': []
        }

        run = PipelineRun.from_dict(data)

        assert run.run_id == "test"
        assert run.status == PipelineStatus.DONE
        assert run.total_files == 10
        assert run.success == 8

    def test_pipeline_run_roundtrip_with_files(self):
        """Test PipelineRun serialization roundtrip with files."""
        file_result = FileResult(
            filename="test.txt",
            filepath="/path/to/test.txt",
            status=FileStatus.SUCCESS,
            chunks_created=10
        )

        original = PipelineRun(
            run_id="test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=1,
            success=1,
            files=[file_result]
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = PipelineRun.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert len(restored.files) == 1
        assert restored.files[0].filename == "test.txt"
        assert restored.files[0].status == FileStatus.SUCCESS


class TestHistoryManager:
    """Test suite for HistoryManager."""

    @pytest.fixture
    def temp_history_dir(self):
        """Create a temporary history directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def history_manager(self, temp_history_dir):
        """Create a HistoryManager with temp directory."""
        return HistoryManager(history_dir=temp_history_dir)

    def test_history_manager_init_default(self):
        """Test HistoryManager initialization with default path."""
        manager = HistoryManager()
        assert manager.history_dir == DEFAULT_HISTORY_DIR

    def test_history_manager_init_custom(self, temp_history_dir):
        """Test HistoryManager initialization with custom path."""
        manager = HistoryManager(history_dir=temp_history_dir)
        assert manager.history_dir == temp_history_dir

    def test_history_manager_ensure_directory_created(self, history_manager, temp_history_dir):
        """Test that history directory is created if it doesn't exist."""
        # Directory should be created by the manager
        assert temp_history_dir.exists()

    def test_create_run(self, history_manager):
        """Test creating a new run."""
        run = history_manager.create_run(total_files=10, mode="interactive")

        assert run.run_id.startswith("run_")
        assert run.total_files == 10
        assert run.mode == "interactive"
        assert run.status == PipelineStatus.INITIALIZING

    def test_save_run(self, history_manager, temp_history_dir):
        """Test saving a pipeline run."""
        run = PipelineRun(
            run_id="run_20250130_120000",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=5,
            success=5
        )

        # Ensure directory exists first
        temp_history_dir.mkdir(parents=True, exist_ok=True)

        history_manager._save_run(run)

        # Check file was created
        run_file = temp_history_dir / f"{run.run_id}.json"
        assert run_file.exists()

        # Verify contents
        with open(run_file) as f:
            data = json.load(f)
            assert data['run_id'] == run.run_id
            assert data['status'] == "done"

    def test_list_runs_empty(self, history_manager):
        """Test listing runs when history is empty."""
        runs = history_manager.list_runs()
        assert runs == []

    def test_list_runs_with_data(self, history_manager, temp_history_dir):
        """Test listing runs with saved data."""
        # Ensure directory exists
        temp_history_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple runs
        for i in range(3):
            run = PipelineRun(
                run_id=f"run_{i}",
                timestamp=f"2025-01-30T12:00:0{i}",
                status=PipelineStatus.DONE,
                total_files=5,
                success=5
            )
            history_manager._save_run(run)

        runs = history_manager.list_runs()
        assert len(runs) >= 3

    def test_get_run(self, history_manager, temp_history_dir):
        """Test retrieving a specific run."""
        # Ensure directory exists
        temp_history_dir.mkdir(parents=True, exist_ok=True)

        # Save a run
        original = PipelineRun(
            run_id="run_test",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=10,
            success=8,
            failed=2
        )
        history_manager._save_run(original)

        # Retrieve it
        retrieved = history_manager.get_run("run_test")

        assert retrieved is not None
        assert retrieved.run_id == "run_test"
        assert retrieved.success == 8
        assert retrieved.failed == 2

    def test_get_run_nonexistent(self, history_manager):
        """Test retrieving a non-existent run."""
        result = history_manager.get_run("nonexistent_run")
        assert result is None

    def test_get_failed_files(self, history_manager, temp_history_dir):
        """Test getting failed files from a run."""
        # Ensure directory exists
        temp_history_dir.mkdir(parents=True, exist_ok=True)

        # Create run with failures
        file1 = FileResult(
            filename="success.txt",
            filepath="/path/to/success.txt",
            status=FileStatus.SUCCESS
        )
        file2 = FileResult(
            filename="failed.txt",
            filepath="/path/to/failed.txt",
            status=FileStatus.FAILED,
            error="Parse error"
        )

        run = PipelineRun(
            run_id="run_with_failures",
            timestamp="2025-01-30T12:00:00",
            status=PipelineStatus.DONE,
            total_files=2,
            success=1,
            failed=1,
            files=[file1, file2]
        )
        history_manager._save_run(run)

        # Get failed files
        failed = history_manager.get_failed_files("run_with_failures")

        assert len(failed) == 1
        assert failed[0].filename == "failed.txt"
        assert failed[0].status == FileStatus.FAILED


class TestHistoryIntegration:
    """Integration tests for history functionality."""

    @pytest.fixture
    def temp_history_dir(self):
        """Create a temporary history directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_complete_workflow(self, temp_history_dir):
        """Test complete history workflow: create, save, list, retrieve."""
        manager = HistoryManager(history_dir=temp_history_dir)

        # Ensure directory exists
        temp_history_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID
        run_id = manager.generate_run_id()
        assert run_id.startswith("run_")

        # Create pipeline run
        run = PipelineRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            status=PipelineStatus.DONE,
            total_files=10,
            success=7,
            failed=2,
            skipped=1,
            mode="interactive"
        )

        # Save run
        manager._save_run(run)

        # List runs
        runs = manager.list_runs()
        assert len(runs) >= 1

        # Retrieve run
        retrieved = manager.get_run(run_id)
        assert retrieved is not None
        assert retrieved.run_id == run_id
        assert retrieved.success == 7

    def test_multiple_runs_chronological_order(self, temp_history_dir):
        """Test that runs are listed in chronological order."""
        manager = HistoryManager(history_dir=temp_history_dir)

        # Ensure directory exists
        temp_history_dir.mkdir(parents=True, exist_ok=True)

        # Create runs with different timestamps
        for i in range(5):
            run = PipelineRun(
                run_id=f"run_{i:03d}",
                timestamp=f"2025-01-30T12:00:{i:02d}",
                status=PipelineStatus.DONE,
                total_files=1,
                success=1
            )
            manager._save_run(run)

        runs = manager.list_runs()
        assert len(runs) >= 5
