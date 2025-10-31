"""
Tests for CLI display utilities.

Tests all display functions in src/core/cli/utils/display.py.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from src.core.cli.utils.display import (
    print_success,
    print_error,
    print_warning,
    print_info,
    create_chunks_table,
    create_batch_progress,
    display_stats,
)


class TestPrintFunctions:
    """Test print utility functions."""

    @patch('src.core.cli.utils.display.console')
    def test_print_success(self, mock_console):
        """Test printing success message."""
        print_success("Operation completed")

        mock_console.print.assert_called_once_with("[green]✓[/green] Operation completed")

    @patch('src.core.cli.utils.display.console')
    def test_print_error(self, mock_console):
        """Test printing error message."""
        print_error("Operation failed")

        mock_console.print.assert_called_once_with("[red]✗[/red] Operation failed")

    @patch('src.core.cli.utils.display.console')
    def test_print_warning(self, mock_console):
        """Test printing warning message."""
        print_warning("Potential issue detected")

        mock_console.print.assert_called_once_with("[yellow]⚠[/yellow] Potential issue detected")

    @patch('src.core.cli.utils.display.console')
    def test_print_info(self, mock_console):
        """Test printing info message."""
        print_info("Processing started")

        mock_console.print.assert_called_once_with("[cyan]ℹ[/cyan] Processing started")

    @patch('src.core.cli.utils.display.console')
    def test_print_functions_with_special_characters(self, mock_console):
        """Test print functions with special characters."""
        test_cases = [
            (print_success, "[green]✓[/green] Test with émojis 🎉"),
            (print_error, "[red]✗[/red] Error: file not found!"),
            (print_warning, "[yellow]⚠[/yellow] Warning: 100% complete"),
            (print_info, "[cyan]ℹ[/cyan] Info: ${PATH}/file.txt"),
        ]

        for func, expected_msg in test_cases:
            mock_console.reset_mock()
            message = expected_msg.split("] ", 1)[1]
            func(message)
            assert mock_console.print.called


class TestCreateChunksTable:
    """Test create_chunks_table function."""

    def test_create_chunks_table_with_dict_chunks(self):
        """Test table creation with dictionary chunks."""
        chunks = [
            {"id": "chunk001", "text": "This is the first chunk of text"},
            {"id": "chunk002", "text": "This is the second chunk"},
            {"id": "chunk003", "text": "Short text"},
        ]

        table = create_chunks_table(chunks, title="Test Chunks", limit=10)

        assert isinstance(table, Table)
        assert table.title == "Test Chunks"
        assert len(table.columns) == 3

    def test_create_chunks_table_with_object_chunks(self):
        """Test table creation with object chunks."""
        chunk1 = Mock()
        chunk1.id = "obj-chunk-001"
        chunk1.text = "Object chunk text content"

        chunk2 = Mock()
        chunk2.id = "obj-chunk-002"
        chunk2.text = "Another object chunk"

        chunks = [chunk1, chunk2]

        table = create_chunks_table(chunks)

        assert isinstance(table, Table)
        assert table.title == "Chunks"

    def test_create_chunks_table_with_long_text(self):
        """Test table creation with text longer than 80 chars."""
        long_text = "A" * 100
        chunks = [{"id": "long-chunk", "text": long_text}]

        table = create_chunks_table(chunks)

        assert isinstance(table, Table)
        # Text should be truncated to 80 chars + "..."

    def test_create_chunks_table_with_limit(self):
        """Test table creation with limit parameter."""
        chunks = [
            {"id": f"chunk{i:03d}", "text": f"Text {i}"}
            for i in range(20)
        ]

        table = create_chunks_table(chunks, limit=5)

        assert isinstance(table, Table)
        # Should only show 5 chunks, with caption indicating more exist
        assert table.caption == "Showing 5 of 20 chunks"

    def test_create_chunks_table_no_limit_needed(self):
        """Test table when chunks count is below limit."""
        chunks = [
            {"id": "chunk001", "text": "Text 1"},
            {"id": "chunk002", "text": "Text 2"},
        ]

        table = create_chunks_table(chunks, limit=10)

        assert isinstance(table, Table)
        assert table.caption is None  # No caption when under limit

    def test_create_chunks_table_empty_list(self):
        """Test table creation with empty chunk list."""
        chunks = []

        table = create_chunks_table(chunks)

        assert isinstance(table, Table)
        assert len(table.columns) == 3

    def test_create_chunks_table_missing_id(self):
        """Test table with chunks missing ID field."""
        chunks = [{"text": "Text without ID"}]

        table = create_chunks_table(chunks)

        assert isinstance(table, Table)
        # Should handle missing ID gracefully

    def test_create_chunks_table_missing_text(self):
        """Test table with chunks missing text field."""
        chunks = [{"id": "chunk001"}]

        table = create_chunks_table(chunks)

        assert isinstance(table, Table)
        # Should handle missing text gracefully

    def test_create_chunks_table_mixed_chunk_types(self):
        """Test table with mixed dict and object chunks."""
        chunk_dict = {"id": "dict-chunk", "text": "Dictionary chunk"}

        chunk_obj = Mock()
        chunk_obj.id = "obj-chunk"
        chunk_obj.text = "Object chunk"

        chunks = [chunk_dict, chunk_obj]

        table = create_chunks_table(chunks)

        assert isinstance(table, Table)


class TestCreateBatchProgress:
    """Test create_batch_progress function."""

    def test_create_batch_progress(self):
        """Test progress bar creation."""
        progress = create_batch_progress()

        assert isinstance(progress, Progress)
        # Progress should have spinner, text, bar, and task progress columns
        assert len(progress.columns) == 4

    def test_create_batch_progress_has_console(self):
        """Test that progress bar uses console."""
        progress = create_batch_progress()

        assert progress.console is not None


class TestDisplayStats:
    """Test display_stats function."""

    @patch('src.core.cli.utils.display.console')
    def test_display_stats_basic(self, mock_console):
        """Test displaying basic statistics."""
        stats = {
            "Total Files": 100,
            "Processed": 85,
            "Failed": 5,
            "Skipped": 10,
        }

        display_stats(stats)

        mock_console.print.assert_called_once()
        # Should print a Table
        call_args = mock_console.print.call_args[0]
        assert isinstance(call_args[0], Table)

    @patch('src.core.cli.utils.display.console')
    def test_display_stats_empty(self, mock_console):
        """Test displaying empty statistics."""
        stats = {}

        display_stats(stats)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0]
        assert isinstance(call_args[0], Table)

    @patch('src.core.cli.utils.display.console')
    def test_display_stats_various_types(self, mock_console):
        """Test displaying stats with various value types."""
        stats = {
            "String Value": "test",
            "Integer": 42,
            "Float": 3.14,
            "Boolean": True,
            "None Value": None,
        }

        display_stats(stats)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0]
        assert isinstance(call_args[0], Table)

    @patch('src.core.cli.utils.display.console')
    def test_display_stats_special_characters(self, mock_console):
        """Test stats with special characters in keys/values."""
        stats = {
            "Files (%)": "85.5%",
            "Path": "/tmp/test/file.txt",
            "Status": "✓ Complete",
        }

        display_stats(stats)

        mock_console.print.assert_called_once()
