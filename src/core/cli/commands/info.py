"""Info command for Atlas-RAG CLI."""
from typing_extensions import Annotated

import typer
import requests

from src.core.cli.utils.display import (
    console,
    print_success,
    print_error,
    print_warning
)


def info_command(
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            help="API server URL"
        )
    ] = "http://localhost:8000",
) -> None:
    """
    Display system information and status.

    Shows the status of the API server, vector store, and local capabilities.

    \b
    Examples:
        # Display system info
        ragctl info

        # Check custom API URL
        ragctl info --api-url http://192.168.1.100:8000
    """
    console.print("\n[bold]RAG Studio System Information[/bold]\n")

    # Check API health
    api_available = False
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        health = response.json()

        print_success(f"API Status: {health.get('status', 'unknown')}")
        console.print(f"  Version: {health.get('version', 'N/A')}")
        console.print(f"  URL: {api_url}")
        api_available = True

    except requests.exceptions.ConnectionError:
        print_error(f"API: Not available at {api_url}")
        print_warning("Start the API with: docker-compose up -d")
        api_available = False

    except Exception as e:
        print_error(f"API: Error - {e}")
        api_available = False

    # Check Vector Store info
    if api_available:
        console.print()
        try:
            response = requests.get(f"{api_url}/api/v1/vector/info", timeout=5)
            info = response.json()

            print_success("Vector Store: Connected")
            console.print(f"  Collection: {info.get('name', 'N/A')}")
            console.print(f"  Vectors: {info.get('vectors_count', 0):,}")
            console.print(f"  Dimension: {info.get('config', {}).get('dimension', 'N/A')}")

        except Exception as e:
            print_warning(f"Vector Store: Not initialized or error - {e}")

    # Display local capabilities
    console.print(f"\n[bold]Local Capabilities:[/bold]")
    console.print("  [green]✓[/green] Document chunking (semantic, sentence, token)")
    console.print("  [green]✓[/green] Multiple file formats (TXT, MD, PDF, DOCX)")
    console.print("  [green]✓[/green] Batch processing")
    console.print("  [green]✓[/green] JSON/JSONL export")

    # Display API capabilities if available
    if api_available:
        console.print(f"\n[bold]API Capabilities:[/bold]")
        console.print("  [green]✓[/green] Vector storage (Qdrant)")
        console.print("  [green]✓[/green] Semantic search")
        console.print("  [green]✓[/green] Embeddings (SentenceTransformers)")
        console.print("  [green]✓[/green] REST API endpoints")

    # CLI commands available
    console.print(f"\n[bold]Available Commands:[/bold]")
    console.print("  [cyan]ragctl chunk[/cyan]   - Chunk a single document")
    console.print("  [cyan]ragctl batch[/cyan]   - Process multiple files")
    console.print("  [cyan]ragctl search[/cyan]  - Semantic search (requires API)")
    console.print("  [cyan]ragctl info[/cyan]    - Display this information")

    console.print(f"\n[dim]Run 'ragctl --help' for more information[/dim]\n")
