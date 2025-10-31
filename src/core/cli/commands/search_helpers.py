"""Helper functions for search command - following SOLID principles.

Each function has a single responsibility (SRP).
"""
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.core.cli.utils.display import console, display_stats


def configure_search_client(qdrant_url: str, collection: str) -> Any:
    """
    Initialize and configure Qdrant search client. SRP: Configuration only.

    Args:
        qdrant_url: Qdrant server URL
        collection: Collection name to search in

    Returns:
        Configured QdrantVectorStore instance

    Raises:
        ImportError: If vector store dependencies not available
        Exception: If connection fails
    """
    try:
        from src.core.vector import QdrantVectorStore, VectorStoreConfig
    except ImportError as e:
        raise ImportError(f"Failed to import vector store dependencies: {e}")

    config = VectorStoreConfig()
    config.url = qdrant_url
    config.index_name = collection

    try:
        vector_store = QdrantVectorStore(config)
        vector_store.connect()
        return vector_store
    except Exception as e:
        raise Exception(f"Failed to connect to Qdrant at {qdrant_url}: {e}")


def perform_similarity_search(
    vector_store: Any,
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic similarity search. SRP: Search execution only.

    Args:
        vector_store: QdrantVectorStore instance
        query: Search query text
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0.0-1.0)

    Returns:
        List of search results with scores and metadata

    Raises:
        Exception: If search fails
    """
    try:
        results = vector_store.search(
            query_text=query,
            top_k=limit,
            score_threshold=score_threshold
        )
        return results
    except Exception as e:
        raise Exception(f"Search failed: {e}")


def format_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format search results for display. SRP: Result formatting only.

    Args:
        results: Raw search results from vector store

    Returns:
        Formatted results with relevance scores and clean display text

    Examples:
        >>> results = [{"text": "sample", "score": 0.95}]
        >>> formatted = format_search_results(results)
        >>> len(formatted) >= 0
        True
    """
    formatted = []
    for i, result in enumerate(results, 1):
        formatted_result = {
            "rank": i,
            "text": result.get("text", ""),
            "score": result.get("score", 0.0),
            "relevance": f"{result.get('score', 0.0) * 100:.1f}%",
            "metadata": result.get("metadata", {})
        }
        formatted.append(formatted_result)
    return formatted


def display_search_stats(
    query: str,
    results_count: int,
    collection: str,
    qdrant_url: str
) -> None:
    """
    Display search statistics. SRP: Statistics display only.

    Args:
        query: Original search query
        results_count: Number of results found
        collection: Collection searched
        qdrant_url: Qdrant server URL
    """
    stats = {
        "Query": query,
        "Results found": results_count,
        "Collection": collection,
        "Qdrant URL": qdrant_url
    }
    display_stats(stats)
