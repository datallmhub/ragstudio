"""Eval command for Atlas-RAG CLI."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated

import typer
from rich.table import Table

from src.core.cli.utils.display import (
    console,
    print_success,
    print_error,
    print_info,
    display_stats
)
from src.core.cli.utils.validation import validate_file_exists


def eval_command(
    chunks_files: Annotated[
        List[Path],
        typer.Argument(
            help="One or more JSON files containing chunks to evaluate",
            exists=True,
        )
    ],
    compare: Annotated[
        bool,
        typer.Option(
            "--compare",
            help="Compare multiple chunking results"
        )
    ] = False,
    report: Annotated[
        Optional[Path],
        typer.Option(
            "--report", "-r",
            help="Save evaluation report to JSON file"
        )
    ] = None,
    show_details: Annotated[
        bool,
        typer.Option(
            "--details",
            help="Show detailed metrics per chunk"
        )
    ] = False,
) -> None:
    """
    Evaluate chunking quality and compare strategies.

    This command analyzes document chunks and provides quality metrics
    to help optimize your chunking strategy.

    \b
    Metrics calculated:
      • Chunk count and distribution
      • Average, min, max chunk sizes
      • Size variance and consistency
      • Token efficiency
      • Empty/small chunk detection

    \b
    Examples:
        # Evaluate single chunking result
        ragctl eval chunks.json

        # Compare multiple strategies
        ragctl eval semantic.json sentence.json token.json --compare

        # Save detailed report
        ragctl eval chunks.json --report eval_report.json --details

        # Pipeline: chunk with different strategies → evaluate
        ragctl chunk doc.txt --strategy semantic -o semantic.json
        ragctl chunk doc.txt --strategy sentence -o sentence.json
        ragctl eval semantic.json sentence.json --compare
    """
    console.print(f"\n[bold]Evaluating chunking quality...[/bold]\n")

    # Validate and load files
    results = {}

    for file_path in chunks_files:
        try:
            validate_file_exists(file_path)

            with console.status(f"[bold green]Loading {file_path.name}..."):
                chunks = json.loads(file_path.read_text())

            if not isinstance(chunks, list):
                print_error(f"Invalid format in {file_path.name}: expected array")
                continue

            if len(chunks) == 0:
                print_error(f"No chunks found in {file_path.name}")
                continue

            # Calculate metrics
            metrics = calculate_metrics(chunks)
            results[file_path.name] = metrics

            print_success(f"Loaded {len(chunks)} chunks from {file_path.name}")

        except Exception as e:
            print_error(f"Error loading {file_path.name}: {e}")
            continue

    if not results:
        print_error("No valid chunk files loaded")
        raise typer.Exit(code=1)

    # Display results
    if compare and len(results) > 1:
        display_comparison(results)
    else:
        for filename, metrics in results.items():
            display_single_evaluation(filename, metrics, show_details)

    # Save report if requested
    if report:
        try:
            report_data = {
                "files": list(results.keys()),
                "metrics": results,
                "comparison": compare
            }
            report.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
            console.print()
            print_success(f"Report saved to [bold]{report}[/bold]")
        except Exception as e:
            print_error(f"Failed to save report: {e}")

    # Recommendations
    console.print(f"\n[bold]Recommendations:[/bold]")
    if len(results) == 1:
        metrics = list(results.values())[0]
        provide_recommendations(metrics)
    elif compare:
        provide_comparison_recommendations(results)


def calculate_metrics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate quality metrics for chunks."""
    sizes = [len(chunk.get("text", "")) for chunk in chunks]

    # Basic stats
    total_chunks = len(chunks)
    avg_size = sum(sizes) / total_chunks if total_chunks > 0 else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    # Variance
    variance = sum((s - avg_size) ** 2 for s in sizes) / total_chunks if total_chunks > 0 else 0
    std_dev = variance ** 0.5

    # Distribution
    small_chunks = sum(1 for s in sizes if s < 100)  # < 100 chars
    medium_chunks = sum(1 for s in sizes if 100 <= s < 500)
    large_chunks = sum(1 for s in sizes if s >= 500)
    empty_chunks = sum(1 for s in sizes if s == 0)

    # Consistency score (lower std_dev relative to mean = more consistent)
    consistency_score = 1 - min(std_dev / avg_size if avg_size > 0 else 0, 1)

    return {
        "total_chunks": total_chunks,
        "avg_size": round(avg_size, 1),
        "min_size": min_size,
        "max_size": max_size,
        "std_dev": round(std_dev, 1),
        "consistency_score": round(consistency_score, 3),
        "distribution": {
            "small": small_chunks,
            "medium": medium_chunks,
            "large": large_chunks,
            "empty": empty_chunks
        },
        "sizes": sizes
    }


def display_single_evaluation(filename: str, metrics: Dict[str, Any], show_details: bool = False):
    """Display evaluation for a single file."""
    console.print(f"\n[bold cyan]📊 {filename}[/bold cyan]")

    stats = {
        "Total chunks": metrics["total_chunks"],
        "Average size": f"{metrics['avg_size']:.0f} chars",
        "Min size": f"{metrics['min_size']} chars",
        "Max size": f"{metrics['max_size']} chars",
        "Std deviation": f"{metrics['std_dev']:.1f}",
        "Consistency": f"{metrics['consistency_score']:.1%}"
    }
    display_stats(stats)

    # Distribution
    dist = metrics["distribution"]
    console.print(f"\n[bold]Size distribution:[/bold]")
    console.print(f"  Small (<100):    {dist['small']:4d} chunks ({dist['small']/metrics['total_chunks']*100:.1f}%)")
    console.print(f"  Medium (100-500): {dist['medium']:4d} chunks ({dist['medium']/metrics['total_chunks']*100:.1f}%)")
    console.print(f"  Large (>500):     {dist['large']:4d} chunks ({dist['large']/metrics['total_chunks']*100:.1f}%)")

    if dist['empty'] > 0:
        console.print(f"  [yellow]⚠ Empty chunks:  {dist['empty']:4d}[/yellow]")


def display_comparison(results: Dict[str, Dict[str, Any]]):
    """Display comparison table for multiple files."""
    console.print(f"\n[bold cyan]📊 Chunking Strategy Comparison[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Chunks", justify="right", style="white")
    table.add_column("Avg Size", justify="right", style="white")
    table.add_column("Std Dev", justify="right", style="white")
    table.add_column("Consistency", justify="right", style="green")
    table.add_column("Rating", justify="center", style="yellow")

    for filename, metrics in results.items():
        # Simple rating based on consistency
        consistency = metrics["consistency_score"]
        if consistency > 0.8:
            rating = "⭐⭐⭐"
        elif consistency > 0.6:
            rating = "⭐⭐"
        else:
            rating = "⭐"

        table.add_row(
            filename[:30],
            str(metrics["total_chunks"]),
            f"{metrics['avg_size']:.0f}",
            f"{metrics['std_dev']:.1f}",
            f"{consistency:.1%}",
            rating
        )

    console.print(table)


def provide_recommendations(metrics: Dict[str, Any]):
    """Provide recommendations based on metrics."""
    consistency = metrics["consistency_score"]
    avg_size = metrics["avg_size"]
    dist = metrics["distribution"]

    recommendations = []

    # Consistency
    if consistency < 0.5:
        recommendations.append(
            "❌ [red]Low consistency[/red] - Chunks vary significantly in size. "
            "Consider using a different strategy (e.g., token-based)."
        )
    elif consistency < 0.7:
        recommendations.append(
            "⚠️  [yellow]Moderate consistency[/yellow] - Some variation in chunk sizes. "
            "This may be acceptable depending on your use case."
        )
    else:
        recommendations.append(
            "✅ [green]Good consistency[/green] - Chunks are relatively uniform in size."
        )

    # Average size
    if avg_size < 100:
        recommendations.append(
            "⚠️  [yellow]Small average chunk size[/yellow] - May lose context. "
            "Consider increasing max_tokens."
        )
    elif avg_size > 1000:
        recommendations.append(
            "⚠️  [yellow]Large average chunk size[/yellow] - May exceed embedding limits. "
            "Consider decreasing max_tokens."
        )
    else:
        recommendations.append(
            "✅ [green]Good average size[/green] - Suitable for most embedding models."
        )

    # Empty chunks
    if dist["empty"] > 0:
        recommendations.append(
            f"❌ [red]{dist['empty']} empty chunks detected[/red] - "
            "Review your chunking logic or input data."
        )

    # Small chunks
    small_ratio = dist["small"] / metrics["total_chunks"]
    if small_ratio > 0.3:
        recommendations.append(
            f"⚠️  [yellow]{small_ratio:.1%} of chunks are very small[/yellow] - "
            "May not contain enough context for embeddings."
        )

    for rec in recommendations:
        console.print(f"  {rec}")


def provide_comparison_recommendations(results: Dict[str, Dict[str, Any]]):
    """Provide recommendations when comparing multiple strategies."""
    # Find best by consistency
    best_consistency = max(results.items(), key=lambda x: x[1]["consistency_score"])
    console.print(
        f"  ✅ [green]Best consistency:[/green] {best_consistency[0]} "
        f"({best_consistency[1]['consistency_score']:.1%})"
    )

    # Find best by avg size (closest to 300-500 chars ideal range)
    best_size = min(
        results.items(),
        key=lambda x: abs(x[1]["avg_size"] - 400)  # 400 = middle of ideal range
    )
    console.print(
        f"  ✅ [green]Best average size:[/green] {best_size[0]} "
        f"({best_size[1]['avg_size']:.0f} chars)"
    )

    console.print(f"\n  💡 [cyan]Consider using the strategy with best consistency and suitable average size.[/cyan]")
