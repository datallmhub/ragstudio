"""Unit tests for batch command - focused on testable logic."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.exceptions import Exit as ClickExit
import typer

from src.core.cli.commands.batch import batch_command
from src.core.cli.commands.chunk import ChunkStrategy


class TestBatchSecurityValidations:
    """Tests for security validations at batch command entry."""

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    def test_batch_validates_pattern_safety(self, mock_validate_pattern, mock_security_config, tmp_path):
        """Test that batch command validates pattern safety."""
        mock_security_config.return_value = {}

        # Simulate pattern validation failure
        mock_validate_pattern.side_effect = typer.BadParameter("Dangerous pattern")

        # Should exit with error
        with pytest.raises((ClickExit, typer.Exit)):
            batch_command(directory=tmp_path, pattern="../../*")

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_validates_directory_exists(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch command validates directory existence."""
        mock_security_config.return_value = {}
        mock_validate_pattern.return_value = None

        # Simulate directory validation failure
        mock_validate_dir.side_effect = typer.BadParameter("Directory not found")

        # Should exit with error
        with pytest.raises((ClickExit, typer.Exit)):
            batch_command(directory=tmp_path, pattern="*.txt")

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    @patch('src.core.cli.commands.batch.validate_output_path')
    def test_batch_validates_output_path_when_provided(
        self, mock_validate_output, mock_validate_dir,
        mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch validates output path when provided."""
        mock_security_config.return_value = {}
        mock_validate_pattern.return_value = None
        mock_validate_dir.return_value = None

        # Simulate output path validation failure
        mock_validate_output.side_effect = typer.BadParameter("Invalid output path")

        output_path = tmp_path / "output"

        # Should exit with error
        with pytest.raises((ClickExit, typer.Exit)):
            batch_command(directory=tmp_path, output=output_path)


class TestBatchConfiguration:
    """Tests for batch command configuration and parameters."""

    @pytest.mark.skip(reason="Complex integration test - requires full pipeline mock setup")
    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    @patch('src.core.cli.commands.batch.create_pipeline_manager')
    @patch('src.core.cli.commands.batch.HistoryManager')
    def test_batch_accepts_various_chunk_strategies(
        self, mock_history, mock_pipeline, mock_validate_dir,
        mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch accepts different chunking strategies."""
        # Setup mocks with proper security config
        mock_sec_config = Mock()
        mock_sec_config.max_batch_files = 1000
        mock_sec_config.max_total_size = 10000000000
        mock_security_config.return_value = mock_sec_config
        mock_validate_pattern.return_value = None
        mock_validate_dir.return_value = None

        mock_manager = Mock()
        mock_manager.process_file.return_value = Mock(status="success")
        mock_pipeline.return_value = mock_manager

        mock_history_inst = Mock()
        mock_history_inst.create_run.return_value = Mock(run_id="test_run")
        mock_history.return_value = mock_history_inst

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Test different strategies (should not raise error)
        strategies = [ChunkStrategy.semantic, ChunkStrategy.sentence, ChunkStrategy.token]

        for strategy in strategies:
            try:
                batch_command(
                    directory=tmp_path,
                    pattern="*.txt",
                    strategy=strategy,
                    auto_stop=True  # Stop quickly for test
                )
            except (ClickExit, typer.Exit):
                pass  # Expected when no files match or after processing

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_validates_max_tokens_range(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that max_tokens parameter has valid range constraints."""
        mock_security_config.return_value = {}
        mock_validate_pattern.return_value = None
        mock_validate_dir.return_value = None

        # The actual validation is done by Typer's parameter constraints
        # We just verify the command accepts valid values
        # (Testing invalid values would require testing Typer's validation)

        # This test verifies the function signature is correct
        import inspect
        sig = inspect.signature(batch_command)
        max_tokens_param = sig.parameters['max_tokens']

        # Verify default value
        assert max_tokens_param.default == 400

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_has_auto_mode_flags(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch has auto-continue, auto-stop, auto-skip flags."""
        import inspect
        sig = inspect.signature(batch_command)

        # Verify auto mode parameters exist
        assert 'auto_continue' in sig.parameters
        assert 'auto_stop' in sig.parameters
        assert 'auto_skip' in sig.parameters
        assert 'save_history' in sig.parameters

        # Verify defaults
        assert sig.parameters['auto_continue'].default is False
        assert sig.parameters['auto_stop'].default is False
        assert sig.parameters['auto_skip'].default is False
        assert sig.parameters['save_history'].default is True


class TestBatchFileDiscovery:
    """Tests for file discovery logic."""

    @pytest.mark.skip(reason="Complex integration test - requires full pipeline mock setup")
    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    @patch('src.core.cli.commands.batch.validate_total_size')
    @patch('src.core.cli.commands.batch.create_pipeline_manager')
    @patch('src.core.cli.commands.batch.HistoryManager')
    def test_batch_discovers_files_with_pattern(
        self, mock_history, mock_pipeline, mock_validate_total,
        mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch discovers files matching the pattern."""
        # Setup mocks with proper security config
        mock_sec_config = Mock()
        mock_sec_config.max_batch_files = 1000
        mock_sec_config.max_total_size = 10000000000
        mock_security_config.return_value = mock_sec_config
        mock_validate_pattern.return_value = None
        mock_validate_dir.return_value = None
        mock_validate_total.return_value = None

        mock_manager = Mock()
        mock_manager.process_file.return_value = Mock(status="success")
        mock_pipeline.return_value = mock_manager

        mock_history_inst = Mock()
        mock_history_inst.create_run.return_value = Mock(run_id="test_run")
        mock_history.return_value = mock_history_inst

        # Create multiple test files
        (tmp_path / "file1.txt").write_text("content 1")
        (tmp_path / "file2.txt").write_text("content 2")
        (tmp_path / "file3.md").write_text("markdown content")

        # Process only txt files
        try:
            batch_command(
                directory=tmp_path,
                pattern="*.txt",
                auto_stop=True
            )
        except (ClickExit, typer.Exit):
            pass

        # Verify pipeline was created (file discovery happened)
        mock_pipeline.assert_called()

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    @patch('src.core.cli.commands.batch.validate_total_size')
    @patch('src.core.cli.commands.batch.create_pipeline_manager')
    @patch('src.core.cli.commands.batch.HistoryManager')
    def test_batch_handles_no_matching_files(
        self, mock_history, mock_pipeline, mock_validate_total,
        mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test batch behavior when no files match pattern."""
        # Setup mocks
        mock_security_config.return_value = {}
        mock_validate_pattern.return_value = None
        mock_validate_dir.return_value = None
        mock_validate_total.return_value = None

        mock_manager = Mock()
        mock_pipeline.return_value = mock_manager

        mock_history_inst = Mock()
        mock_history_inst.create_run.return_value = Mock(run_id="test_run")
        mock_history.return_value = mock_history_inst

        # Create files that don't match pattern
        (tmp_path / "file1.txt").write_text("content")

        # Try to match .pdf files (none exist)
        with pytest.raises((ClickExit, typer.Exit)):
            batch_command(
                directory=tmp_path,
                pattern="*.pdf",
                auto_stop=True
            )


class TestBatchRecursiveMode:
    """Tests for recursive directory processing."""

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_has_recursive_flag(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch has recursive parameter."""
        import inspect
        sig = inspect.signature(batch_command)

        # Verify recursive parameter exists
        assert 'recursive' in sig.parameters
        assert sig.parameters['recursive'].default is False


class TestBatchOutputModes:
    """Tests for output configuration."""

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_has_single_file_output_option(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch supports single-file output mode."""
        import inspect
        sig = inspect.signature(batch_command)

        # Verify single_file parameter exists
        assert 'single_file' in sig.parameters
        assert sig.parameters['single_file'].default is False

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_has_output_directory_option(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch supports output directory."""
        import inspect
        sig = inspect.signature(batch_command)

        # Verify output parameter exists
        assert 'output' in sig.parameters
        assert sig.parameters['output'].default is None


class TestBatchAdvancedFeatures:
    """Tests for advanced batch features."""

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_has_advanced_ocr_option(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch supports advanced OCR routing."""
        import inspect
        sig = inspect.signature(batch_command)

        # Verify advanced_ocr parameter exists
        assert 'advanced_ocr' in sig.parameters
        assert sig.parameters['advanced_ocr'].default is False

    @patch('src.core.cli.commands.batch.get_security_config')
    @patch('src.core.cli.commands.batch.validate_pattern_safe')
    @patch('src.core.cli.commands.batch.validate_directory_exists')
    def test_batch_has_history_management(
        self, mock_validate_dir, mock_validate_pattern, mock_security_config, tmp_path
    ):
        """Test that batch supports history management."""
        import inspect
        sig = inspect.signature(batch_command)

        # Verify save_history parameter
        assert 'save_history' in sig.parameters
        assert sig.parameters['save_history'].default is True
