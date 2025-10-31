"""
Qdrant vector database connector.

Qdrant is a high-performance vector search engine with a focus on simplicity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.workflows.io.schema import Chunk
from src.core.vector.base import VectorStore, VectorStoreConfig

LOGGER = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None


class QdrantStore(VectorStore):
    """Qdrant vector database connector."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize Qdrant store.

        Args:
            config: Vector store configuration
                - url: Qdrant server URL (default: None for in-memory)
                - api_key: Qdrant API key (optional)
                - index_name: Collection name
        """
        super().__init__(config)

        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install with: pip install qdrant-client"
            )

    def connect(self) -> None:
        """Establish connection to Qdrant."""
        if self.config.url:
            # Connect to Qdrant server
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
            )
            LOGGER.info("Connected to Qdrant at %s", self.config.url)
        else:
            # Use in-memory Qdrant
            self.client = QdrantClient(":memory:")
            LOGGER.info("Using in-memory Qdrant")

    def create_index(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance: str = "Cosine",
        **kwargs
    ) -> None:
        """
        Create a new collection in Qdrant.

        Args:
            index_name: Collection name (defaults to config.index_name)
            dimension: Vector dimension (defaults to config.embedding_dimension)
            distance: Distance metric (Cosine, Euclidean, Dot)
            **kwargs: Additional Qdrant-specific parameters
        """
        if self.client is None:
            self.connect()

        collection_name = index_name or self.config.index_name
        vector_dimension = dimension or self.config.embedding_dimension

        # Map distance string to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance_metric = distance_map.get(distance, Distance.COSINE)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dimension,
                distance=distance_metric,
            ),
        )

        LOGGER.info(
            "Created Qdrant collection '%s' with dimension %d and %s distance",
            collection_name,
            vector_dimension,
            distance,
        )

    def index_exists(self, index_name: Optional[str] = None) -> bool:
        """Check if a collection exists."""
        if self.client is None:
            self.connect()

        collection_name = index_name or self.config.index_name

        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert or update chunks in Qdrant.

        Args:
            chunks: List of chunks to upsert
            embeddings: List of embedding vectors
            **kwargs: Additional Qdrant-specific parameters

        Returns:
            Dictionary with upsert results
        """
        if self.client is None:
            self.connect()

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        collection_name = self.config.index_name

        # Ensure collection exists
        if not self.index_exists():
            self.create_index()

        # Convert chunks to Qdrant points
        import uuid
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Prepare metadata
            payload = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,  # Store original chunk ID in metadata
                "metadata": chunk.metadata,
            }

            # Add specific metadata fields if configured
            if self.config.metadata_fields:
                for field in self.config.metadata_fields:
                    if field in chunk.metadata:
                        payload[field] = chunk.metadata[field]

            # Convert chunk ID to UUID-compatible format
            # Use UUID5 to generate deterministic UUID from chunk ID
            chunk_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id)

            point = PointStruct(
                id=str(chunk_uuid),
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        # Upsert in batches
        batch_size = self.config.batch_size
        total_upserted = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            total_upserted += len(batch)

        LOGGER.info(
            "Upserted %d chunks to Qdrant collection '%s'",
            total_upserted,
            collection_name,
        )

        return {
            "count": total_upserted,
            "collection": collection_name,
        }

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"document_id": "doc123"})
            score_threshold: Minimum similarity score
            **kwargs: Additional Qdrant-specific search parameters

        Returns:
            List of search results with scores and metadata
        """
        if self.client is None:
            self.connect()

        collection_name = self.config.index_name

        # Build Qdrant filter
        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            **kwargs
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.payload.get("chunk_id", result.id),  # Use original chunk_id
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "document_id": result.payload.get("document_id", ""),
            })

        LOGGER.debug("Found %d results for query", len(formatted_results))

        return formatted_results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete vectors from Qdrant.

        Args:
            ids: List of chunk IDs to delete
            filter_dict: Metadata filters for deletion
            **kwargs: Additional Qdrant-specific delete parameters

        Returns:
            Dictionary with deletion results
        """
        if self.client is None:
            self.connect()

        collection_name = self.config.index_name

        if ids:
            # Delete by IDs - convert chunk IDs to UUIDs
            import uuid
            uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)) for chunk_id in ids]
            self.client.delete(
                collection_name=collection_name,
                points_selector=uuid_ids,
            )
            count = len(ids)
        elif filter_dict:
            # Delete by filter
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            query_filter = Filter(must=conditions) if conditions else None

            self.client.delete(
                collection_name=collection_name,
                points_selector=query_filter,
            )
            count = "unknown"  # Qdrant doesn't return delete count with filters
        else:
            raise ValueError("Either ids or filter_dict must be provided")

        LOGGER.info("Deleted %s vectors from Qdrant", count)

        return {
            "count": count,
            "collection": collection_name,
        }

    def close(self) -> None:
        """Close connection to Qdrant."""
        if self.client is not None:
            self.client.close()
            self.client = None
            LOGGER.info("Closed Qdrant connection")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection."""
        stats = super().get_stats()

        if self.client and self.index_exists():
            collection_info = self.client.get_collection(self.config.index_name)
            stats.update({
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
            })

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics (alias for get_stats)."""
        return self.get_stats()

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self.config.index_name

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create collection (alias for create_index).

        Args:
            recreate: If True, delete existing collection before creating
        """
        if recreate and self.index_exists():
            # Delete existing collection
            if self.client is None:
                self.connect()
            self.client.delete_collection(collection_name=self.config.index_name)
            LOGGER.info("Deleted existing collection '%s'", self.config.index_name)

        # Create new collection
        self.create_index()

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store chunks with automatic embedding generation.

        Args:
            chunks: List of chunk dictionaries with 'text', 'id', 'metadata'

        Returns:
            Number of chunks stored
        """
        from src.workflows.io.schema import Chunk
        from src.core.vector.embeddings import embed_chunks

        # Convert dict chunks to Chunk objects
        chunk_objects = []
        for chunk_dict in chunks:
            chunk_obj = Chunk(
                id=chunk_dict.get("id", ""),
                text=chunk_dict.get("text", ""),
                metadata=chunk_dict.get("metadata", {}),
                document_id=chunk_dict.get("metadata", {}).get("source_file", "unknown")
            )
            chunk_objects.append(chunk_obj)

        # Generate embeddings
        LOGGER.info("Generating embeddings for %d chunks...", len(chunk_objects))
        chunk_embeddings = embed_chunks(chunk_objects)

        # Extract embeddings
        embeddings = [ce.embedding for ce in chunk_embeddings]

        # Store using upsert
        result = self.upsert(chunk_objects, embeddings)

        return result.get("count", 0)

    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (generates embedding automatically).

        Args:
            query: Text query
            top_k: Number of results
            score_threshold: Minimum score
            metadata_filter: Metadata filters

        Returns:
            List of search results
        """
        from src.workflows.ml.embeddings import compute_batch_embeddings

        # Generate embedding for query (batch of 1)
        query_embeddings = compute_batch_embeddings([query])
        query_embedding = query_embeddings[0]

        # Search using embedding
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=metadata_filter,
            score_threshold=score_threshold
        )
