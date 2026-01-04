"""ragctl CLI Application (Typer-based)."""
import warnings
import typer
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from src.core.cli.commands.chunk import ChunkStrategy

# Suppress common warnings for better UX
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

# Create main Typer app
app = typer.Typer(
    name="ragctl",
    help="RAG Studio - Production-ready RAG toolkit with intelligent document processing",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Display version and exit."""
    if value:
        from importlib.metadata import version, PackageNotFoundError
        try:
            app_version = version("ragctl")
        except PackageNotFoundError:
            try:
                # Fallback to old name for development
                app_version = version("ragctl")
            except PackageNotFoundError:
                app_version = "0.1.0 (dev)"

        typer.echo(f"RAG Studio (ragctl) version {app_version}")
        raise typer.Exit()


# Lazy import - commands are imported only when actually used
# This makes --version and --help instant instead of loading all heavy dependencies


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True
        )
    ] = False,
):
    """
    RAG Studio - Production-ready RAG toolkit with intelligent document processing.

    A comprehensive CLI for building RAG (Retrieval-Augmented Generation) systems
    with advanced OCR, semantic chunking, and vector store integration.

    \b
    Quick Start:
        1. Chunk a document:
           $ ragctl chunk document.txt --show

        2. Process multiple files:
           $ ragctl batch ./documents -o chunks.json

        3. Ingest to vector store:
           $ ragctl ingest chunks.json

        4. Evaluate chunking quality:
           $ ragctl eval chunks.json

        5. System info:
           $ ragctl info

    \b
    Documentation:
        https://github.com/datallmhub/ragctl
        Documentation: https://datallmhub.github.io/ragctl

    \b
    Support:
        Report issues at: https://github.com/datallmhub/ragctl/issues
    """
    pass


# Register commands - imports happen inside each command function (lazy loading)
@app.command(name="chunk", help="Chunk a single document")
def chunk(
    input_file: Path = typer.Argument(..., help="Input file to chunk"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    strategy: str = typer.Option("semantic", "--strategy", "-s", help="Chunking strategy"),
    max_tokens: int = typer.Option(400, "--max-tokens", help="Maximum tokens per chunk"),
    overlap: int = typer.Option(50, "--overlap", help="Overlap between chunks"),
    show: bool = typer.Option(False, "--show", help="Display chunks in terminal"),
    advanced_ocr: bool = typer.Option(False, "--advanced-ocr", help="Use advanced OCR"),
):
    """Chunk a single document."""
    from src.core.cli.commands.chunk import chunk_command
    # Pass all required parameters with defaults for missing ones
    return chunk_command(
        file=input_file,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap=overlap,
        output=output,
        show=show,
        limit=10,
        advanced_ocr=advanced_ocr,
        use_llm=False,
        llm_url=None,
        llm_model=None,
        llm_provider=None,
        ocr_threshold=None,
        ocr_dynamic_threshold=None,
        ocr_fallback=None,
        config_file=None,
        generate_summary=False
    )


@app.command(name="batch", help="Process multiple files in batch mode")
def batch(
    directory: Path = typer.Argument(..., help="Input directory with files"),
    pattern: str = typer.Option("*", "--pattern", "-p", help="File pattern to match"),
    strategy: ChunkStrategy = typer.Option(ChunkStrategy.semantic, "--strategy", "-s", help="Chunking strategy to use"),
    max_tokens: int = typer.Option(400, "--max-tokens", "-m", help="Maximum tokens per chunk", min=50, max=2000),
    overlap: int = typer.Option(50, "--overlap", "-ol", help="Token overlap between chunks", min=0, max=500),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory or file"),
    single_file: bool = typer.Option(False, "--single-file", help="Combine all chunks into a single output file"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process recursively"),
    advanced_ocr: bool = typer.Option(False, "--advanced-ocr", help="Use intelligent OCR routing for scanned PDFs"),
    auto_continue: bool = typer.Option(False, "--auto-continue", help="Continue automatically on errors"),
    auto_stop: bool = typer.Option(False, "--auto-stop", help="Stop on first error"),
    auto_skip: bool = typer.Option(False, "--auto-skip", help="Skip failed files automatically"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="List files without processing"),
    save_history: bool = typer.Option(True, "--save-history/--no-history", help="Save run to history for retry capability"),
):
    """Process multiple files in batch mode."""
    if auto_continue and auto_stop:
        from src.core.cli.utils.display import print_error
        print_error("Cannot use both --auto-continue and --auto-stop")
        raise typer.Exit(code=1)

    from src.core.cli.commands.batch import batch_command
    
    return batch_command(
        directory=directory,
        pattern=pattern,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap=overlap,
        output=output,
        single_file=single_file,
        recursive=recursive,
        advanced_ocr=advanced_ocr,
        auto_continue=auto_continue,
        auto_stop=auto_stop,
        auto_skip=auto_skip,
        dry_run=dry_run,
        save_history=save_history
    )


@app.command(name="ingest", help="Ingest chunks into Qdrant vector store")
def ingest(
    input_file: Path = typer.Argument(..., help="Input JSON/JSONL file with chunks"),
    collection: str = typer.Option("atlas_chunks", "--collection", "-c", help="Collection name"),
    url: str = typer.Option("http://localhost:6333", "--qdrant-url", help="Qdrant URL"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate collection if exists"),
    embedding_dim: int = typer.Option(384, "--embedding-dim", help="Embedding dimension"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """Ingest chunks into Qdrant vector store."""
    from src.core.cli.commands.ingest import ingest_command
    return ingest_command(
        chunks_file=input_file,
        collection=collection,
        qdrant_url=url,
        recreate=recreate,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        yes=yes
    )


@app.command(name="search", help="Search in vector store using semantic search", hidden=True)
def search(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results"),
):
    """Search in vector store using semantic search."""
    from src.core.cli.commands.search import search_command
    return search_command(query, collection, limit)


@app.command(name="eval", help="Evaluate chunking quality and compare strategies")
def eval(
    input_file: Path = typer.Argument(..., help="Input file to evaluate"),
    strategies: str = typer.Option("semantic,sentence,token", "--strategies", help="Strategies to compare"),
):
    """Evaluate chunking quality and compare strategies."""
    import tempfile
    from src.core.cli.commands.chunk import chunk_command
    from src.core.cli.commands.eval import eval_command

    # Parse strategies
    strategy_list = [s.strip() for s in strategies.split(",")]

    # Chunk the file with each strategy
    temp_files = []
    for strategy in strategy_list:
        # Create temp file for this strategy's chunks (secure temp file creation)
        temp_fd = tempfile.NamedTemporaryFile(mode='w', suffix=f"_{strategy}.json", delete=False)
        temp_file = Path(temp_fd.name)
        temp_fd.close()

        # Chunk with this strategy
        result = chunk_command(
            file=input_file,
            strategy=strategy,
            max_tokens=400,
            overlap=50,
            output=temp_file,
            show=False,
            limit=None,
            advanced_ocr=False
        )

        # Only add to list if chunking succeeded and file was created
        if result != 0 and temp_file.exists():
            temp_files.append(temp_file)

    # Handle case where no chunks were created (empty file)
    if not temp_files:
        from src.core.cli.utils.display import print_warning
        print_warning(f"Cannot evaluate empty file: {input_file.name}")
        return 0

    # Evaluate and compare the results
    compare = len(strategy_list) > 1
    result = eval_command(temp_files, compare, None, False)

    # Cleanup temp files
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()

    return result


@app.command(name="info", help="Display system information and status")
def info(
    api_url: str = typer.Option("http://localhost:8000", "--api-url", help="API server URL"),
):
    """Display system information and status."""
    from src.core.cli.commands.info import info_command
    return info_command(api_url)


@app.command(name="retry", help="Retry failed files from a previous run")
def retry(
    run_id: str = typer.Argument(None, help="Run ID to retry (optional)"),
    show: bool = typer.Option(False, "--show", help="Show failed runs"),
):
    """Retry failed files from a previous run."""
    from src.core.cli.commands.retry import retry_command
    # Match signature: (run_id, output, mode, show)
    if show:
        return retry_command(None, None, "list", show=True)
    return retry_command(run_id, None, "retry", show=False)


if __name__ == "__main__":
    app()
