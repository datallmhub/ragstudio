"""Search command for Atlas-RAG CLI."""
from typing import Optional
from typing_extensions import Annotated

import typer
import requests

from src.core.cli.utils.display import (
    console,
    print_success,
    print_error,
    print_warning
)


def search_command(
    query: Annotated[
        str,
        typer.Argument(help="Search query text")
    ],
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k", "-k",
            help="Number of results to return",
            min=1,
            max=100
        )
    ] = 5,
    threshold: Annotated[
        Optional[float],
        typer.Option(
            "--threshold", "-t",
            help="Minimum similarity score threshold (0.0-1.0)",
            min=0.0,
            max=1.0
        )
    ] = None,
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            help="API server URL"
        )
    ] = "http://localhost:8000",
) -> None:
    """
    Search in vector store using semantic search.

    This command requires the Atlas-RAG API to be running.
    Start it with: docker-compose up -d

    \b
    Examples:
        # Basic search
        ragctl search "machine learning applications"

        # Search with more results
        ragctl search "deep learning" --top-k 10

        # Search with score threshold
        ragctl search "neural networks" --threshold 0.7

        # Search with custom API URL
        ragctl search "transformers" --api-url http://192.168.1.100:8000
    """
    # Check API health
    try:
        with console.status(f"[bold cyan]Connecting to API at {api_url}..."):
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to API at {api_url}")
        print_warning("Make sure the API is running:")
        console.print("  [cyan]→[/cyan] docker-compose up -d")
        console.print("  [cyan]→[/cyan] or check the API URL")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"API health check failed: {e}")
        raise typer.Exit(code=1)

    # Perform search
    with console.status("[bold green]Searching..."):
        try:
            payload = {
                "query": query,
                "top_k": top_k,
            }
            if threshold is not None:
                payload["score_threshold"] = threshold

            response = requests.post(
                f"{api_url}/api/v1/vector/search",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.HTTPError as e:
            print_error(f"Search request failed: {e}")
            if e.response is not None:
                console.print(f"  Response: {e.response.text}")
            raise typer.Exit(code=1)
        except Exception as e:
            print_error(f"Search error: {e}")
            raise typer.Exit(code=1)

    # Display results
    results = result.get('results', [])
    search_time = result.get('search_time_seconds', 0)

    console.print("\n")
    print_success(f"Search completed in {search_time:.3f}s")
    console.print(f"  Query: [bold cyan]{query}[/bold cyan]")
    console.print(f"  Results: {len(results)}")
    if threshold:
        console.print(f"  Min score: {threshold}")
    console.print()

    if not results:
        print_warning("No results found")
        console.print("  Try:")
        console.print("    • Lowering the threshold")
        console.print("    • Using different keywords")
        console.print("    • Checking if documents are indexed")
        return

    # Display each result
    for i, res in enumerate(results, 1):
        score = res.get('score', 0.0)

        # Color-code scores
        if score > 0.7:
            score_color = "green"
        elif score > 0.5:
            score_color = "yellow"
        else:
            score_color = "white"

        console.print(f"[bold cyan]{i}.[/bold cyan] Score: [{score_color}]{score:.3f}[/{score_color}]")
        console.print(f"   ID: [dim]{res.get('chunk_id', 'N/A')}[/dim]")

        # Display text with truncation
        text = res.get('text', '')
        if len(text) > 300:
            text_display = text[:300] + "..."
        else:
            text_display = text

        console.print(f"   [dim]{text_display}[/dim]")

        # Display metadata if available
        metadata = res.get('metadata', {})
        if metadata:
            source = metadata.get('source_file', metadata.get('source', 'Unknown'))
            console.print(f"   Source: [cyan]{source}[/cyan]")

        console.print()
