"""Helper functions for ingest command - following SOLID principles.

Each function has a single responsibility (SRP).
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.core.cli.utils.display import console, print_error, print_success, print_info
from src.core.cli.utils.security import sanitize_metadata


def load_chunks_from_file(
    chunks_file: Path,
    security_config: Any
) -> List[Dict[str, Any]]:
    """
    Load and parse chunks from JSON file. SRP: File loading and parsing only.

    Args:
        chunks_file: Path to JSON file containing chunks
        security_config: Security configuration for metadata sanitization

    Returns:
        List of chunk dictionaries

    Raises:
        ValueError: If JSON format is invalid or empty
    """
    console.print(f"\n[bold]Loading chunks from {chunks_file.name}...[/bold]")

    try:
        with console.status("[bold green]Reading JSON..."):
            chunks_data = json.loads(chunks_file.read_text())

        # Support both formats: [...] and {"chunks": [...]}
        if isinstance(chunks_data, dict) and "chunks" in chunks_data:
            chunks_data = chunks_data["chunks"]
        elif not isinstance(chunks_data, list):
            raise ValueError("Invalid JSON format: expected array of chunks or {\"chunks\": [...]}")

        if len(chunks_data) == 0:
            raise ValueError("No chunks found in file")

        # Sanitize metadata in chunks
        sanitized_chunks = []
        for chunk in chunks_data:
            if isinstance(chunk, dict) and "metadata" in chunk:
                chunk["metadata"] = sanitize_metadata(chunk["metadata"], security_config)
            sanitized_chunks.append(chunk)

        print_success(f"Loaded {len(sanitized_chunks)} chunks")
        return sanitized_chunks

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def configure_qdrant(
    qdrant_url: str,
    collection: str,
    embedding_dim: int,
    batch_size: int
) -> Tuple[Any, Any]:
    """
    Configure Qdrant connection and vector store. SRP: Configuration only.

    Args:
        qdrant_url: URL of Qdrant server
        collection: Collection name
        embedding_dim: Embedding vector dimension
        batch_size: Batch processing size

    Returns:
        Tuple of (vector_store, config)

    Raises:
        ImportError: If vector store dependencies not available
        Exception: If connection fails
    """
    # Import Qdrant here (lazy import to avoid NumPy issues on other commands)
    try:
        from src.core.vector import QdrantVectorStore, VectorStoreConfig
    except ImportError as e:
        raise ImportError(f"Failed to import vector store dependencies: {e}")

    # Configure Qdrant
    console.print(f"\n[bold]Configuring Qdrant connection...[/bold]")
    config = VectorStoreConfig()
    config.url = qdrant_url
    config.index_name = collection
    config.embedding_dimension = embedding_dim

    from src.core.cli.utils.display import display_stats
    display_stats({
        "Qdrant URL": qdrant_url,
        "Collection": collection,
        "Embedding dimension": embedding_dim,
        "Batch size": batch_size
    })

    # Initialize vector store and connect
    try:
        with console.status("[bold green]Connecting to Qdrant..."):
            vector_store = QdrantVectorStore(config)
            vector_store.connect()

        print_success("Connected to Qdrant")
        return vector_store, config

    except Exception as e:
        raise Exception(f"Failed to connect to Qdrant: {e}")


def prepare_collection(
    vector_store: Any,
    collection: str,
    recreate: bool,
    yes: bool
) -> None:
    """
    Create or verify Qdrant collection. SRP: Collection management only.

    Args:
        vector_store: QdrantVectorStore instance
        collection: Collection name
        recreate: Whether to recreate collection (delete existing data)
        yes: Whether to skip confirmation prompts

    Raises:
        Exception: If collection preparation fails
    """
    import typer

    console.print(f"\n[bold]Preparing collection...[/bold]")
    try:
        if recreate:
            from src.core.cli.utils.display import print_warning
            print_warning(f"Recreating collection '{collection}' (existing data will be deleted)")

            # Ask for confirmation in interactive mode (unless --yes flag is used)
            if not yes and not typer.confirm("⚠️  Are you sure you want to delete existing data?"):
                print_info("Operation cancelled")
                raise typer.Exit(code=0)

            with console.status(f"[bold yellow]Recreating collection..."):
                vector_store.create_collection(recreate=True)
            print_success(f"Collection '{collection}' recreated")

        elif not vector_store.index_exists():
            with console.status(f"[bold green]Creating collection '{collection}'..."):
                vector_store.create_collection()
            print_success(f"Collection '{collection}' created")

        else:
            print_info(f"Using existing collection '{collection}'")

    except typer.Exit:
        raise
    except Exception as e:
        raise Exception(f"Failed to prepare collection: {e}")


def ingest_chunks_batch(
    vector_store: Any,
    chunks_data: List[Dict[str, Any]],
    batch_size: int,
    collection: str,
    qdrant_url: str
) -> Dict[str, int]:
    """
    Ingest chunks into vector store in batches. SRP: Batch ingestion only.

    Args:
        vector_store: QdrantVectorStore instance
        chunks_data: List of chunks to ingest
        batch_size: Number of chunks per batch
        collection: Collection name (for display)
        qdrant_url: Qdrant URL (for display)

    Returns:
        Dictionary with ingestion statistics

    Raises:
        KeyboardInterrupt: If user interrupts
        Exception: If ingestion fails
    """
    import typer
    from src.core.cli.utils.display import display_stats, print_warning

    console.print(f"\n[bold]Ingesting chunks...[/bold]")
    console.print("[dim]This may take a while for large datasets (embedding generation)[/dim]\n")

    try:
        # Process in batches
        total_stored = 0
        failed_count = 0

        with console.status("[bold green]Generating embeddings and storing...") as status:
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunks_data) + batch_size - 1) // batch_size

                status.update(
                    f"[bold green]Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} chunks)..."
                )

                try:
                    stored = vector_store.store_chunks(batch)
                    total_stored += stored
                except Exception as e:
                    console.print(f"[red]Warning: Batch {batch_num} failed: {e}[/red]")
                    failed_count += len(batch)

        console.print()
        print_success("Ingestion complete!")

        stats = {
            "Total chunks": len(chunks_data),
            "Successfully stored": total_stored,
            "Failed": failed_count,
            "Collection": collection,
            "Qdrant URL": qdrant_url
        }
        display_stats(stats)

        # Show collection info
        try:
            info = vector_store.get_collection_info()
            console.print(f"\n[bold]Collection status:[/bold]")
            console.print(f"  Vectors in collection: {info.get('vectors_count', 'N/A'):,}")
            console.print(f"  Collection status: [green]{info.get('status', 'unknown')}[/green]")
        except Exception:  # nosec B110 - Collection info is optional, safe to skip
            # Collection info is optional, silently skip if not available
            pass

        return {
            "total_chunks": len(chunks_data),
            "successfully_stored": total_stored,
            "failed": failed_count
        }

    except KeyboardInterrupt:
        console.print("\n")
        print_warning("Ingestion interrupted by user")
        print_info(f"Partial data may have been stored in collection '{collection}'")
        raise

    except Exception as e:
        raise Exception(f"Ingestion failed: {e}")
