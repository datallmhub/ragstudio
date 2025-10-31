"""Display utilities for CLI using Rich."""
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import List, Dict, Any

console = Console()


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[cyan]ℹ[/cyan] {message}")


def create_chunks_table(chunks: List[Dict[str, Any]], title: str = "Chunks", limit: int = 10) -> Table:
    """Create a Rich table for displaying chunks."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True, width=12)
    table.add_column("Length", justify="right", style="magenta", width=8)
    table.add_column("Preview", style="white")

    for chunk in chunks[:limit]:
        chunk_id = chunk.get("id", "N/A")
        if hasattr(chunk, 'id'):
            chunk_id = chunk.id

        text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, 'text', "")
        preview = text[:80] + "..." if len(text) > 80 else text

        table.add_row(
            chunk_id[:12],
            str(len(text)),
            preview
        )

    if len(chunks) > limit:
        table.caption = f"Showing {limit} of {len(chunks)} chunks"

    return table


def create_batch_progress() -> Progress:
    """Create a progress bar for batch operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )


def display_stats(stats: Dict[str, Any]) -> None:
    """Display statistics in a formatted way."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in stats.items():
        table.add_row(f"  {key}:", str(value))

    console.print(table)
