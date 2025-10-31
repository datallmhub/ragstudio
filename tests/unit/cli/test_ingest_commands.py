"""
Unit tests for core/cli/commands/ingest.py

Tests cover:
- File validation (existence, symlinks, size)
- JSON loading (array format, dict format, invalid JSON)
- Qdrant connection and configuration
- Collection management (create, recreate, exists)
- Chunk ingestion with batching
- Error handling and edge cases

Note: QdrantVectorStore and VectorStoreConfig are imported dynamically in ingest_command,
so we need to patch at the import location.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import typer
from click.exceptions import Exit as ClickExit

# Import the ingest command
from src.core.cli.commands.ingest import ingest_command


class TestIngestFileValidation:
    """Test suite for input file validation."""

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_valid_file_passes_all_validations(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test that valid file passes all security validations."""
        # Create test file
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([
            {"id": "1", "text": "test", "metadata": {}}
        ]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        # Mock Qdrant
        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 1
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(chunks_file=test_file, yes=True)
        except (ClickExit, typer.Exit):
            pass

        # Verify validations were called
        assert mock_validate_symlinks.called
        assert mock_validate_size.called
        assert mock_validate_exists.called

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.print_error')
    def test_symlink_validation_fails(
        self, mock_print_error, mock_validate_symlinks,
        mock_security_config, tmp_path
    ):
        """Test that symlink validation failure exits with error."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.side_effect = typer.BadParameter("Symlinks not allowed")

        with pytest.raises((ClickExit, typer.Exit)):
            ingest_command(chunks_file=test_file, yes=True)

        assert mock_print_error.called

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.print_error')
    def test_file_size_validation_fails(
        self, mock_print_error, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test that file size validation failure exits with error."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.side_effect = typer.BadParameter("File too large")

        with pytest.raises((ClickExit, typer.Exit)):
            ingest_command(chunks_file=test_file, yes=True)

        assert mock_print_error.called


class TestIngestJSONLoading:
    """Test suite for JSON file loading."""

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_load_json_array_format(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test loading chunks in array format [...]."""
        test_file = tmp_path / "chunks.json"
        chunks = [
            {"id": "1", "text": "chunk1", "metadata": {}},
            {"id": "2", "text": "chunk2", "metadata": {}}
        ]
        test_file.write_text(json.dumps(chunks))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 2
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(chunks_file=test_file, yes=True)
        except (ClickExit, typer.Exit):
            pass

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_load_json_dict_format(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test loading chunks in dict format {chunks: [...]}."""
        test_file = tmp_path / "chunks.json"
        data = {
            "chunks": [
                {"id": "1", "text": "chunk1", "metadata": {}},
                {"id": "2", "text": "chunk2", "metadata": {}}
            ]
        }
        test_file.write_text(json.dumps(data))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 2
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(chunks_file=test_file, yes=True)
        except (ClickExit, typer.Exit):
            pass

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.cli.commands.ingest.print_error')
    def test_invalid_json_format(
        self, mock_print_error, mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test that invalid JSON fails with error."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text("{invalid json")

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        with pytest.raises((ClickExit, typer.Exit)):
            ingest_command(chunks_file=test_file, yes=True)

        assert mock_print_error.called

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.cli.commands.ingest.print_warning')
    def test_empty_chunks_array(
        self, mock_print_warning, mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test handling of empty chunks array."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        with pytest.raises((ClickExit, typer.Exit)):
            ingest_command(chunks_file=test_file, yes=True)

        assert mock_print_warning.called


class TestIngestQdrantConnection:
    """Test suite for Qdrant connection management."""

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_qdrant_connection_successful(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test successful Qdrant connection."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([
            {"id": "1", "text": "test", "metadata": {}}
        ]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 1
        mock_store.get_collection_info.return_value = {"vectors_count": 1, "status": "ready"}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(
                chunks_file=test_file,
                collection="test_collection",
                qdrant_url="http://localhost:6333",
                yes=True
            )
        except (ClickExit, typer.Exit):
            pass

        # Verify connection was attempted
        mock_store.connect.assert_called_once()

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.VectorStoreConfig')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.cli.commands.ingest.print_error')
    def test_qdrant_connection_failure(
        self, mock_print_error, mock_vector_store_class, mock_config_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test Qdrant connection failure handling."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([{"id": "1", "text": "test", "metadata": {}}]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.connect.side_effect = ConnectionError("Cannot connect")
        mock_vector_store_class.return_value = mock_store

        with pytest.raises((ClickExit, typer.Exit)):
            ingest_command(chunks_file=test_file, yes=True)

        assert mock_print_error.called


class TestIngestCollectionManagement:
    """Test suite for collection creation and management."""

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_create_new_collection(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test creating new collection when it doesn't exist."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([{"id": "1", "text": "test", "metadata": {}}]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = False  # Collection doesn't exist
        mock_store.store_chunks.return_value = 1
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(chunks_file=test_file, yes=True)
        except (ClickExit, typer.Exit):
            pass

        # Verify collection was created
        mock_store.create_collection.assert_called_once()

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_use_existing_collection(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test using existing collection."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([{"id": "1", "text": "test", "metadata": {}}]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True  # Collection exists
        mock_store.store_chunks.return_value = 1
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(chunks_file=test_file, recreate=False, yes=True)
        except (ClickExit, typer.Exit):
            pass

        # Verify collection was NOT created (using existing)
        assert not mock_store.create_collection.called

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_recreate_collection_with_yes_flag(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test recreating collection with --yes flag (no confirmation)."""
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps([{"id": "1", "text": "test", "metadata": {}}]))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 1
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(
                chunks_file=test_file,
                recreate=True,
                yes=True  # Auto-confirm
            )
        except (ClickExit, typer.Exit):
            pass

        # Verify recreate was called
        mock_store.create_collection.assert_called_once_with(recreate=True)


class TestIngestBatchProcessing:
    """Test suite for batch processing of chunks."""

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_batch_processing_multiple_batches(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test processing chunks in multiple batches."""
        # Create file with 5 chunks, batch size = 2 → 3 batches
        chunks = [{"id": str(i), "text": f"chunk{i}", "metadata": {}} for i in range(5)]
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps(chunks))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 2  # Each batch stores 2 chunks
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(
                chunks_file=test_file,
                batch_size=2,
                yes=True
            )
        except (ClickExit, typer.Exit):
            pass

        # Verify store_chunks was called 3 times (for 3 batches)
        assert mock_store.store_chunks.call_count == 3

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_batch_error_handling(
        self, mock_config_class, mock_vector_store_class,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test error handling when a batch fails."""
        chunks = [{"id": str(i), "text": f"chunk{i}", "metadata": {}} for i in range(3)]
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps(chunks))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        # First batch succeeds, second fails
        mock_store.store_chunks.side_effect = [1, Exception("Batch failed"), 1]
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(
                chunks_file=test_file,
                batch_size=1,
                yes=True
            )
        except (ClickExit, typer.Exit):
            pass

        # Should continue processing despite batch failure
        assert mock_store.store_chunks.call_count == 3


class TestIngestMetadataSanitization:
    """Test suite for metadata sanitization."""

    @patch('src.core.cli.commands.ingest.get_security_config')
    @patch('src.core.cli.commands.ingest.validate_no_symlinks')
    @patch('src.core.cli.commands.ingest.validate_file_size')
    @patch('src.core.cli.commands.ingest.validate_file_exists')
    @patch('src.core.cli.commands.ingest.sanitize_metadata')
    @patch('src.core.vector.QdrantVectorStore')
    @patch('src.core.vector.VectorStoreConfig')
    def test_metadata_sanitization_called(
        self, mock_config_class, mock_vector_store_class, mock_sanitize,
        mock_validate_exists, mock_validate_size,
        mock_validate_symlinks, mock_security_config, tmp_path
    ):
        """Test that metadata sanitization is called for each chunk."""
        chunks = [
            {"id": "1", "text": "chunk1", "metadata": {"key": "value1"}},
            {"id": "2", "text": "chunk2", "metadata": {"key": "value2"}}
        ]
        test_file = tmp_path / "chunks.json"
        test_file.write_text(json.dumps(chunks))

        mock_security_config.return_value = {}
        mock_validate_symlinks.return_value = None
        mock_validate_size.return_value = None
        mock_validate_exists.return_value = None
        mock_sanitize.return_value = {"key": "sanitized"}

        mock_store = Mock()
        mock_store.index_exists.return_value = True
        mock_store.store_chunks.return_value = 2
        mock_store.get_collection_info.return_value = {}
        mock_vector_store_class.return_value = mock_store

        try:
            ingest_command(chunks_file=test_file, yes=True)
        except (ClickExit, typer.Exit):
            pass

        # Verify sanitize_metadata was called for each chunk
        assert mock_sanitize.call_count == 2
