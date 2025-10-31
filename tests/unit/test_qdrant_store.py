"""
Unit tests for core/vector/qdrant_store.py

Tests cover:
- QdrantStore initialization with/without qdrant-client
- Connection establishment (server, in-memory, with API key)
- Collection management (create, exists, delete)
- Vector operations (upsert, search, delete)
- Error handling and edge cases

All tests use mocks for QdrantClient since qdrant-client may not be installed.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Mock qdrant_client before importing qdrant_store
mock_qdrant_client = MagicMock()
mock_distance = MagicMock()
mock_distance.COSINE = "COSINE"
mock_distance.EUCLID = "EUCLID"
mock_distance.DOT = "DOT"

sys.modules['qdrant_client'] = mock_qdrant_client
sys.modules['qdrant_client.models'] = MagicMock()

# Now import after mocking
from src.core.vector.base import VectorStoreConfig
from src.workflows.io.schema import Chunk


class TestQdrantStoreInit:
    """Test suite for QdrantStore initialization."""

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    def test_init_with_qdrant_available(self):
        """Test initialization when qdrant-client is available."""
        from src.core.vector.qdrant_store import QdrantStore

        config = VectorStoreConfig(
            url="http://localhost:6333",
            api_key="test_key",
            index_name="test_collection"
        )

        store = QdrantStore(config)

        assert store.config == config
        assert store.client is None  # Not connected yet

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', False)
    def test_init_without_qdrant_client(self):
        """Test initialization fails when qdrant-client is not installed."""
        from src.core.vector.qdrant_store import QdrantStore

        config = VectorStoreConfig(index_name="test")

        with pytest.raises(ImportError, match="qdrant-client is not installed"):
            QdrantStore(config)

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    def test_init_validates_config(self):
        """Test that config is properly stored."""
        from src.core.vector.qdrant_store import QdrantStore

        config = VectorStoreConfig(
            index_name="my_collection",
            embedding_dimension=768,
            batch_size=50
        )

        store = QdrantStore(config)

        assert store.config.index_name == "my_collection"
        assert store.config.embedding_dimension == 768
        assert store.config.batch_size == 50


class TestQdrantConnection:
    """Test suite for Qdrant connection management."""

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_connect_to_server(self, mock_client_class):
        """Test connecting to Qdrant server."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(
            url="http://localhost:6333",
            api_key="test_key",
            index_name="test"
        )

        store = QdrantStore(config)
        store.connect()

        # Verify QdrantClient was called with correct parameters
        mock_client_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key="test_key"
        )
        assert store.client == mock_client

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_connect_in_memory(self, mock_client_class):
        """Test connecting to in-memory Qdrant."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")  # No URL = in-memory

        store = QdrantStore(config)
        store.connect()

        # Verify in-memory connection
        mock_client_class.assert_called_once_with(":memory:")
        assert store.client == mock_client

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_connect_with_api_key(self, mock_client_class):
        """Test connection with API key authentication."""
        from src.core.vector.qdrant_store import QdrantStore

        config = VectorStoreConfig(
            url="https://xyz.qdrant.io",
            api_key="secret_key_123",
            index_name="test"
        )

        store = QdrantStore(config)
        store.connect()

        mock_client_class.assert_called_once_with(
            url="https://xyz.qdrant.io",
            api_key="secret_key_123"
        )

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_connection_error_handling(self, mock_client_class):
        """Test connection error handling."""
        from src.core.vector.qdrant_store import QdrantStore

        # Simulate connection error
        mock_client_class.side_effect = ConnectionError("Cannot connect to Qdrant")

        config = VectorStoreConfig(
            url="http://invalid:6333",
            index_name="test"
        )

        store = QdrantStore(config)

        with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
            store.connect()


class TestQdrantCollections:
    """Test suite for Qdrant collection management."""

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.Distance')
    @patch('src.core.vector.qdrant_store.VectorParams')
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_create_collection(self, mock_client_class, mock_vector_params, mock_distance):
        """Test creating a new collection."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_distance.COSINE = "COSINE"

        config = VectorStoreConfig(
            index_name="test_collection",
            embedding_dimension=384
        )

        store = QdrantStore(config)
        store.client = mock_client

        store.create_index()

        # Verify create_collection was called
        assert mock_client.create_collection.called
        call_args = mock_client.create_collection.call_args
        assert call_args[1]['collection_name'] == "test_collection"

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_create_collection_custom_dimension(self, mock_client_class):
        """Test creating collection with custom dimension."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(
            index_name="custom_collection",
            embedding_dimension=768
        )

        store = QdrantStore(config)
        store.client = mock_client

        # Create with custom dimension
        store.create_index(dimension=1536)

        assert mock_client.create_collection.called

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_index_exists_true(self, mock_client_class):
        """Test checking if collection exists (exists)."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()  # Collection exists
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="existing_collection")

        store = QdrantStore(config)
        store.client = mock_client

        exists = store.index_exists()

        assert exists is True
        mock_client.get_collection.assert_called_once_with("existing_collection")

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_index_exists_false(self, mock_client_class):
        """Test checking if collection exists (doesn't exist)."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="nonexistent")

        store = QdrantStore(config)
        store.client = mock_client

        exists = store.index_exists()

        assert exists is False

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_delete_collection(self, mock_client_class):
        """Test deleting a collection."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()  # Exists
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="to_delete")

        store = QdrantStore(config)
        store.client = mock_client

        store.create_collection(recreate=True)

        # Should delete then create
        mock_client.delete_collection.assert_called_once_with(collection_name="to_delete")


class TestQdrantOperations:
    """Test suite for Qdrant vector operations."""

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.PointStruct')
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_upsert_vectors(self, mock_client_class, mock_point_struct):
        """Test inserting vectors."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()  # Collection exists
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(
            index_name="test_collection",
            batch_size=100
        )

        store = QdrantStore(config)
        store.client = mock_client

        # Create test chunks
        chunks = [
            Chunk(id="chunk1", text="text1", document_id="doc1", metadata={}),
            Chunk(id="chunk2", text="text2", document_id="doc1", metadata={})
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        result = store.upsert(chunks, embeddings)

        assert result['count'] == 2
        assert result['collection'] == "test_collection"
        assert mock_client.upsert.called

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_upsert_batch_vectors(self, mock_client_class):
        """Test inserting vectors in batches."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(
            index_name="test",
            batch_size=2  # Small batch size to test batching
        )

        store = QdrantStore(config)
        store.client = mock_client

        # Create 5 chunks (should be 3 batches: 2, 2, 1)
        chunks = [
            Chunk(id=f"chunk{i}", text=f"text{i}", document_id="doc1", metadata={})
            for i in range(5)
        ]
        embeddings = [[0.1, 0.2, 0.3] for _ in range(5)]

        result = store.upsert(chunks, embeddings)

        assert result['count'] == 5
        # Verify multiple upsert calls (batching)
        assert mock_client.upsert.call_count == 3

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_upsert_mismatched_lengths(self, mock_client_class):
        """Test upsert with mismatched chunks and embeddings."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        chunks = [Chunk(id="1", text="text", document_id="doc", metadata={})]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 chunk

        with pytest.raises(ValueError, match="Number of chunks must match"):
            store.upsert(chunks, embeddings)

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_search_vectors(self, mock_client_class):
        """Test searching for similar vectors."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock search results
        mock_result = Mock()
        mock_result.id = "uuid-123"
        mock_result.score = 0.95
        mock_result.payload = {
            "chunk_id": "chunk1",
            "text": "result text",
            "metadata": {"key": "value"},
            "document_id": "doc1"
        }
        mock_client.search.return_value = [mock_result]

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        query_embedding = [0.1, 0.2, 0.3]
        results = store.search(query_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]['id'] == "chunk1"
        assert results[0]['score'] == 0.95
        assert results[0]['text'] == "result text"
        assert results[0]['document_id'] == "doc1"

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.Filter')
    @patch('src.core.vector.qdrant_store.FieldCondition')
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_search_with_filter(self, mock_client_class, mock_field_condition, mock_filter):
        """Test search with metadata filters."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.search.return_value = []
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        query_embedding = [0.1, 0.2, 0.3]
        filter_dict = {"document_id": "doc123"}

        results = store.search(
            query_embedding,
            top_k=10,
            filter_dict=filter_dict
        )

        # Verify search was called with filter
        assert mock_client.search.called
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs['query_vector'] == query_embedding
        assert call_kwargs['limit'] == 10


class TestQdrantDelete:
    """Test suite for Qdrant delete operations."""

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_delete_by_ids(self, mock_client_class):
        """Test deleting vectors by IDs."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        ids = ["chunk1", "chunk2", "chunk3"]
        result = store.delete(ids=ids)

        assert result['count'] == 3
        assert result['collection'] == "test"
        assert mock_client.delete.called

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_delete_by_filter(self, mock_client_class):
        """Test deleting vectors by metadata filter."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        filter_dict = {"document_id": "doc_to_delete"}
        result = store.delete(filter_dict=filter_dict)

        assert result['count'] == "unknown"  # Filter deletes don't return count
        assert result['collection'] == "test"
        assert mock_client.delete.called

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_delete_without_params(self, mock_client_class):
        """Test delete fails without IDs or filter."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        with pytest.raises(ValueError, match="Either ids or filter_dict must be provided"):
            store.delete()


class TestQdrantUtilities:
    """Test suite for Qdrant utility methods."""

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_close_connection(self, mock_client_class):
        """Test closing Qdrant connection."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")
        store = QdrantStore(config)
        store.client = mock_client

        store.close()

        mock_client.close.assert_called_once()
        assert store.client is None

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_get_stats(self, mock_client_class):
        """Test getting collection statistics."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_collection_info = Mock()
        mock_collection_info.vectors_count = 1000
        mock_collection_info.indexed_vectors_count = 1000
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(
            index_name="test_collection",
            embedding_dimension=384
        )
        store = QdrantStore(config)
        store.client = mock_client

        stats = store.get_stats()

        assert stats['index_name'] == "test_collection"
        assert stats['embedding_dimension'] == 384
        assert stats['connected'] is True
        assert stats['vectors_count'] == 1000

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    def test_collection_name_property(self):
        """Test collection_name property."""
        from src.core.vector.qdrant_store import QdrantStore

        config = VectorStoreConfig(index_name="my_collection")
        store = QdrantStore(config)

        assert store.collection_name == "my_collection"

    @patch('src.core.vector.qdrant_store.QDRANT_AVAILABLE', True)
    @patch('src.core.vector.qdrant_store.QdrantClient')
    def test_context_manager(self, mock_client_class):
        """Test using QdrantStore as context manager."""
        from src.core.vector.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = VectorStoreConfig(index_name="test")

        with QdrantStore(config) as store:
            # Should auto-connect
            assert store.client == mock_client

        # Should auto-close
        mock_client.close.assert_called_once()
