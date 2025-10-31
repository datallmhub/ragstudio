"""
Unit tests for core/cli/commands/retry.py

Tests cover:
- Finding last failed run (automatic retry)
- Retrying specific run by ID
- Handling cases with no failed runs
- Handling cases with no failed files
- Show mode (dry run)
- Missing/invalid file paths
- Confirmation prompts
- Error handling

Note: retry_command has complex CLI interactions, so we focus on
testing the key logic paths with mocks.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import typer
from click.exceptions import Exit as ClickExit

from src.core.cli.commands.retry import retry_command
from src.core.pipeline import FileStatus, PipelineStatus


class TestRetryFindRun:
    """Test suite for finding runs to retry."""

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.typer.confirm')
    def test_retry_last_failed_run_found(self, mock_confirm, mock_history_class):
        """Test automatic retry finds last failed run."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        # Mock a failed run
        mock_run = Mock()
        mock_run.run_id = "run_20251030_123456"
        mock_run.total_files = 5
        mock_run.success = 2
        mock_run.failed = 3
        mock_run.skipped = 0
        mock_run.timestamp = "2025-10-30 12:34:56"
        mock_run.mode = "interactive"
        mock_run.status = PipelineStatus.FAILED
        mock_run.config = {}

        mock_history.get_last_failed_run.return_value = mock_run
        mock_history.get_run.return_value = mock_run

        # Mock failed files
        mock_file = Mock()
        mock_file.filename = "test.txt"
        mock_file.filepath = "/tmp/test.txt"
        mock_file.status = FileStatus.FAILED
        mock_file.error = "Test error"
        mock_file.reason = None
        mock_file.retries = 0
        mock_history.get_failed_files.return_value = [mock_file]

        # User cancels at confirmation
        mock_confirm.return_value = False

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id=None)

        # Verify get_last_failed_run was called
        mock_history.get_last_failed_run.assert_called_once()

    @patch('src.core.cli.commands.retry.HistoryManager')
    def test_retry_no_failed_runs_show_mode(self, mock_history_class):
        """Test show mode with no failed runs exits gracefully."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history
        mock_history.get_last_failed_run.return_value = None

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id=None, show=True)

        mock_history.get_last_failed_run.assert_called_once()

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.print_error')
    def test_retry_no_failed_runs_error(self, mock_print_error, mock_history_class):
        """Test error when no failed runs found (non-show mode)."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history
        mock_history.get_last_failed_run.return_value = None

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id=None, show=False)

        assert mock_print_error.called

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.print_error')
    def test_retry_specific_run_not_found(self, mock_print_error, mock_history_class):
        """Test error when specified run_id doesn't exist."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history
        mock_history.get_run.return_value = None

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="nonexistent_run")

        mock_history.get_run.assert_called_once_with("nonexistent_run")
        assert mock_print_error.called


class TestRetryFailedFiles:
    """Test suite for handling failed files."""

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.print_info')
    def test_retry_no_failed_files_in_run(self, mock_print_info, mock_history_class):
        """Test handling run with no failed files."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.total_files = 5
        mock_run.success = 5
        mock_run.failed = 0
        mock_run.skipped = 0

        mock_history.get_run.return_value = mock_run
        mock_history.get_failed_files.return_value = []

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="run_test")

        assert mock_print_info.called

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.typer.confirm')
    def test_retry_show_mode_displays_files(self, mock_confirm, mock_history_class):
        """Test show mode displays failed files without processing."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.timestamp = "2025-10-30"
        mock_run.mode = "interactive"
        mock_run.status = PipelineStatus.FAILED
        mock_run.config = {}

        mock_history.get_run.return_value = mock_run

        # Create failed files
        mock_files = [
            Mock(
                filename=f"file{i}.txt",
                filepath=f"/tmp/file{i}.txt",
                status=FileStatus.FAILED,
                error="Test error",
                reason=None,
                retries=0
            )
            for i in range(3)
        ]
        mock_history.get_failed_files.return_value = mock_files

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="run_test", show=True)

        # Verify files were retrieved but confirm was NOT called (show mode)
        assert not mock_confirm.called

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.print_error')
    def test_retry_missing_file_paths(self, mock_print_error, mock_history_class, tmp_path):
        """Test error handling for files with missing/invalid paths."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.timestamp = "2025-10-30"
        mock_run.mode = "interactive"
        mock_run.status = PipelineStatus.FAILED
        mock_run.config = {}

        mock_history.get_run.return_value = mock_run

        # File with missing path
        mock_file = Mock()
        mock_file.filename = "missing.txt"
        mock_file.filepath = None  # Missing path
        mock_file.status = FileStatus.FAILED
        mock_file.error = "Error"
        mock_file.reason = None
        mock_file.retries = 0

        mock_history.get_failed_files.return_value = [mock_file]

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="run_test", show=False)

        assert mock_print_error.called


class TestRetryConfirmation:
    """Test suite for retry confirmation and cancellation."""

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.typer.confirm')
    @patch('src.core.cli.commands.retry.print_info')
    def test_retry_user_cancels(self, mock_print_info, mock_confirm, mock_history_class, tmp_path):
        """Test user cancels retry at confirmation."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.timestamp = "2025-10-30"
        mock_run.mode = "interactive"
        mock_run.status = PipelineStatus.FAILED
        mock_run.config = {}

        mock_history.get_run.return_value = mock_run

        # Valid file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_file = Mock()
        mock_file.filename = "test.txt"
        mock_file.filepath = str(test_file)
        mock_file.status = FileStatus.FAILED
        mock_file.error = "Error"
        mock_file.reason = None
        mock_file.retries = 0

        mock_history.get_failed_files.return_value = [mock_file]

        # User cancels
        mock_confirm.return_value = False

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="run_test")

        # Should show cancellation message
        assert mock_print_info.called


class TestRetryModes:
    """Test suite for different retry modes."""

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.typer.confirm')
    @patch('src.core.cli.commands.retry.create_pipeline_manager')
    @patch('src.core.cli.commands.retry.chunk_document')
    def test_retry_auto_continue_mode(
        self, mock_chunk, mock_create_pipeline, mock_confirm,
        mock_history_class, tmp_path
    ):
        """Test retry in auto-continue mode."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.timestamp = "2025-10-30"
        mock_run.mode = "auto-continue"
        mock_run.status = PipelineStatus.FAILED
        mock_run.config = {"strategy": "semantic", "max_tokens": 400, "overlap": 50}

        mock_history.get_run.return_value = mock_run

        # Create valid file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        mock_file = Mock()
        mock_file.filename = "test.txt"
        mock_file.filepath = str(test_file)
        mock_file.status = FileStatus.FAILED
        mock_file.error = "Error"
        mock_file.reason = None
        mock_file.retries = 0

        mock_history.get_failed_files.return_value = [mock_file]

        # Mock retry run creation
        mock_retry_run = Mock()
        mock_retry_run.run_id = "run_retry"
        mock_retry_run.success = 1
        mock_retry_run.failed = 0
        mock_retry_run.skipped = 0
        mock_retry_run.aborted = 0
        mock_history.create_run.return_value = mock_retry_run

        # Mock pipeline manager
        mock_pipeline = Mock()
        mock_pipeline.should_continue.return_value = True
        mock_create_pipeline.return_value = mock_pipeline

        # Mock chunk_document success
        mock_chunk.return_value = {"chunks": [{"text": "chunk1"}]}

        # User confirms
        mock_confirm.return_value = True

        try:
            retry_command(run_id="run_test", mode="auto-continue", output=tmp_path)
        except (ClickExit, typer.Exit):
            pass

        # Verify pipeline was created with auto_continue=True
        mock_create_pipeline.assert_called_once()
        call_kwargs = mock_create_pipeline.call_args[1]
        assert call_kwargs.get('auto_continue') is True


class TestRetryEdgeCases:
    """Test suite for edge cases."""

    @patch('src.core.cli.commands.retry.HistoryManager')
    def test_retry_with_skipped_files(self, mock_history_class):
        """Test retry handles skipped files (not just failed)."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.timestamp = "2025-10-30"
        mock_run.mode = "interactive"
        mock_run.status = PipelineStatus.DONE
        mock_run.config = {}

        mock_history.get_run.return_value = mock_run

        # Mix of failed and skipped
        mock_files = [
            Mock(
                filename="failed.txt",
                filepath="/tmp/failed.txt",
                status=FileStatus.FAILED,
                error="Error",
                reason=None,
                retries=0
            ),
            Mock(
                filename="skipped.txt",
                filepath="/tmp/skipped.txt",
                status=FileStatus.SKIPPED,
                error=None,
                reason="Skipped due to...",
                retries=0
            ),
        ]
        mock_history.get_failed_files.return_value = mock_files

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="run_test", show=True)

        # Should retrieve both failed and skipped files
        mock_history.get_failed_files.assert_called_once_with("run_test")

    @patch('src.core.cli.commands.retry.HistoryManager')
    @patch('src.core.cli.commands.retry.typer.confirm')
    def test_retry_custom_output_directory(
        self, mock_confirm, mock_history_class, tmp_path
    ):
        """Test retry with custom output directory."""
        mock_history = Mock()
        mock_history_class.return_value = mock_history

        mock_run = Mock()
        mock_run.run_id = "run_test"
        mock_run.timestamp = "2025-10-30"
        mock_run.mode = "interactive"
        mock_run.status = PipelineStatus.FAILED
        mock_run.config = {}

        mock_history.get_run.return_value = mock_run

        # Valid file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_file = Mock()
        mock_file.filename = "test.txt"
        mock_file.filepath = str(test_file)
        mock_file.status = FileStatus.FAILED
        mock_file.error = "Error"
        mock_file.reason = None
        mock_file.retries = 0

        mock_history.get_failed_files.return_value = [mock_file]

        # User cancels (to avoid full execution)
        mock_confirm.return_value = False

        custom_output = tmp_path / "custom_retry_output"

        with pytest.raises((ClickExit, typer.Exit)):
            retry_command(run_id="run_test", output=custom_output)

        # Output directory path should be passed through
        # (we can't verify creation since we cancel before that)
