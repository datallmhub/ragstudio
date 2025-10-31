"""
Base classes and interfaces for vector database connectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.workflows.io.schema import Chunk


@dataclass
class VectorStoreConfig:
    """Configuration for vector store connection."""

    # Connection settings
    api_key: Optional[str] = None
    url: Optional[str] = None
    index_name: str = "atlas-rag"

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Batch settings
    batch_size: int = 100

    # Metadata settings
    metadata_fields: Optional[List[str]] = None  # Fields to include in metadata

    # Additional provider-specific config
    extra_config: Optional[Dict[str, Any]] = None


class VectorStore(ABC):
    """Abstract base class for vector database connectors."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store with configuration.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self.client = None

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the vector database."""
        pass

    @abstractmethod
    def create_index(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Create a new index/collection in the vector database.

        Args:
            index_name: Name of the index (defaults to config.index_name)
            dimension: Vector dimension (defaults to config.embedding_dimension)
            **kwargs: Provider-specific index creation parameters
        """
        pass

    @abstractmethod
    def index_exists(self, index_name: Optional[str] = None) -> bool:
        """
        Check if an index/collection exists.

        Args:
            index_name: Name of the index (defaults to config.index_name)

        Returns:
            True if index exists, False otherwise
        """
        pass

    @abstractmethod
    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert or update chunks with their embeddings in the vector database.

        Args:
            chunks: List of chunks to upsert
            embeddings: List of embedding vectors (same length as chunks)
            **kwargs: Provider-specific upsert parameters

        Returns:
            Dictionary with upsert results (e.g., count, errors)
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filters
            **kwargs: Provider-specific search parameters

        Returns:
            List of search results with scores and metadata
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete vectors from the database.

        Args:
            ids: List of chunk IDs to delete
            filter_dict: Metadata filters for deletion
            **kwargs: Provider-specific delete parameters

        Returns:
            Dictionary with deletion results
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection to the vector database."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.

        Returns:
            Dictionary with database statistics
        """
        return {
            "index_name": self.config.index_name,
            "embedding_dimension": self.config.embedding_dimension,
            "connected": self.client is not None,
        }
