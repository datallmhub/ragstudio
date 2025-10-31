"""Ingest command for Atlas-RAG CLI."""
import json
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer

from src.core.cli.utils.display import (
    console,
    print_success,
    print_error,
    print_warning,
    print_info,
    display_stats
)
from src.core.cli.utils.validation import validate_file_exists
from src.core.cli.utils.security import (
    validate_file_size,
    validate_no_symlinks,
    sanitize_metadata,
    get_security_config
)


def ingest_command(
    chunks_file: Annotated[
        Path,
        typer.Argument(
            help="JSON file containing chunks to ingest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        )
    ],
    collection: Annotated[
        str,
        typer.Option(
            "--collection", "-c",
            help="Qdrant collection name"
        )
    ] = "atlas_chunks",
    qdrant_url: Annotated[
        str,
        typer.Option(
            "--qdrant-url",
            help="Qdrant server URL"
        )
    ] = "http://localhost:6333",
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate",
            help="Recreate collection if it exists (⚠️ deletes existing data)"
        )
    ] = False,
    embedding_dim: Annotated[
        int,
        typer.Option(
            "--embedding-dim",
            help="Embedding vector dimension",
            min=128,
            max=4096
        )
    ] = 384,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Number of chunks to process at once",
            min=1,
            max=1000
        )
    ] = 32,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y",
            help="Skip confirmation prompts (auto-confirm)"
        )
    ] = False,
) -> None:
    """
    Ingest chunks into Qdrant vector store.

    This command takes a JSON file containing document chunks and stores them
    in Qdrant with automatic embedding generation.

    \b
    Workflow:
      1. Load chunks from JSON file
      2. Connect to Qdrant
      3. Create/verify collection
      4. Generate embeddings (automatic)
      5. Store chunks with vectors

    \b
    Examples:
        # Basic ingestion
        ragctl ingest chunks.json

        # Custom collection
        ragctl ingest chunks.json --collection my_docs

        # Recreate collection (⚠️ deletes existing data)
        ragctl ingest chunks.json --collection my_docs --recreate

        # Custom Qdrant URL
        ragctl ingest chunks.json --qdrant-url http://192.168.1.100:6333

        # Pipeline: chunk → ingest
        ragctl chunk doc.txt -o /tmp/chunks.json && \\
        ragctl ingest /tmp/chunks.json
    """
    # === SECURITY VALIDATIONS ===
    security_config = get_security_config()

    # Validate file is not a symlink
    try:
        validate_no_symlinks(chunks_file, security_config)
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # Validate file size
    try:
        validate_file_size(chunks_file, security_config, warn_only=False)
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # Validate file
    try:
        validate_file_exists(chunks_file)
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # Load chunks
    console.print(f"\n[bold]Loading chunks from {chunks_file.name}...[/bold]")
    try:
        with console.status("[bold green]Reading JSON..."):
            chunks_data = json.loads(chunks_file.read_text())

        # Support both formats: [...] and {"chunks": [...]}
        if isinstance(chunks_data, dict) and "chunks" in chunks_data:
            chunks_data = chunks_data["chunks"]
        elif not isinstance(chunks_data, list):
            print_error("Invalid JSON format: expected array of chunks or {\"chunks\": [...]}")
            raise typer.Exit(code=1)

        if len(chunks_data) == 0:
            print_warning("No chunks found in file")
            raise typer.Exit(code=0)

        # Sanitize metadata in chunks
        sanitized_chunks = []
        for chunk in chunks_data:
            if isinstance(chunk, dict) and "metadata" in chunk:
                chunk["metadata"] = sanitize_metadata(chunk["metadata"], security_config)
            sanitized_chunks.append(chunk)
        chunks_data = sanitized_chunks

        print_success(f"Loaded {len(chunks_data)} chunks")

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Error reading file: {e}")
        raise typer.Exit(code=1)

    # Import Qdrant here (lazy import to avoid NumPy issues on other commands)
    try:
        from src.core.vector import QdrantVectorStore, VectorStoreConfig
    except ImportError as e:
        print_error(f"Failed to import vector store dependencies: {e}")
        print_warning("This may be due to NumPy/scipy compatibility issues.")
        raise typer.Exit(code=1)

    # Configure Qdrant
    console.print(f"\n[bold]Configuring Qdrant connection...[/bold]")
    config = VectorStoreConfig()
    config.url = qdrant_url
    config.index_name = collection
    config.embedding_dimension = embedding_dim

    display_stats({
        "Qdrant URL": qdrant_url,
        "Collection": collection,
        "Embedding dimension": embedding_dim,
        "Batch size": batch_size
    })

    # Initialize vector store
    try:
        with console.status("[bold green]Connecting to Qdrant..."):
            vector_store = QdrantVectorStore(config)
            vector_store.connect()

        print_success("Connected to Qdrant")

    except Exception as e:
        print_error(f"Failed to connect to Qdrant: {e}")
        print_warning("Make sure Qdrant is running:")
        console.print("  [cyan]→[/cyan] docker-compose up -d")
        console.print("  [cyan]→[/cyan] or check the Qdrant URL")
        raise typer.Exit(code=1)

    # Create or verify collection
    console.print(f"\n[bold]Preparing collection...[/bold]")
    try:
        if recreate:
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

    except Exception as e:
        print_error(f"Failed to prepare collection: {e}")
        raise typer.Exit(code=1)

    # Store chunks
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

    except KeyboardInterrupt:
        console.print("\n")
        print_warning("Ingestion interrupted by user")
        print_info(f"Partial data may have been stored in collection '{collection}'")
        raise typer.Exit(code=130)

    except Exception as e:
        print_error(f"Ingestion failed: {e}")
        raise typer.Exit(code=1)

    # Success message
    console.print(f"\n[bold green]✓ Ready for search![/bold green]")
    console.print(f"  Try: [cyan]ragctl search \"your query\" --collection {collection}[/cyan]\n")
