"""Retry command for Atlas-RAG CLI.

Allows retrying failed/skipped files from a previous pipeline run.
"""

from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import time

import typer

from src.core.pipeline import (
    HistoryManager,
    FileStatus,
    PipelineStatus,
)
from src.core.cli.utils.display import (
    console,
    print_error,
    print_success,
    print_info,
    print_warning,
)


def retry_command(
    run_id: Annotated[
        Optional[str],
        typer.Argument(
            help="Run ID to retry (e.g., run_20251028_223532). If not provided, retries last failed run."
        )
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output directory for retry results"
        )
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode", "-m",
            help="Execution mode: interactive / auto-continue / auto-stop / auto-skip"
        )
    ] = "interactive",
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            help="Show files that will be retried (dry run)"
        )
    ] = False,
) -> None:
    """
    Retry failed/skipped files from a previous run.

    This command reads the history, identifies failed/skipped files,
    and reprocesses them with the same or different configuration.

    \b
    Examples:
        # Retry last failed run
        ragctl retry

        # Retry specific run
        ragctl retry run_20251028_223532

        # Show what would be retried (dry run)
        ragctl retry --show

        # Retry with auto-continue mode
        ragctl retry --mode auto-continue

        # Retry to specific output
        ragctl retry -o ./output_retry/
    """
    console.print("[bold cyan]🔄 ragctl Retry Command[/bold cyan]")
    console.print()

    # Get history manager
    history = HistoryManager()

    # Find run to retry
    if run_id is None:
        console.print("[yellow]ℹ️[/yellow] No run_id provided, finding last failed run...")
        run = history.get_last_failed_run()

        if not run:
            if show:
                # In show mode, no failed runs is success, not an error
                print_info("No failed runs found in history - all runs succeeded!")
                console.print()
                console.print("[dim]✓ No files need to be retried[/dim]")
                raise typer.Exit(0)
            else:
                print_error("No failed runs found in history")
                console.print()
                console.print("[dim]Tip: Use 'ragctl history list' to see all runs[/dim]")
                raise typer.Exit(1)

        run_id = run.run_id
        console.print(f"[green]✓[/green] Found run: {run_id}")

    else:
        run = history.get_run(run_id)
        if not run:
            print_error(f"Run not found: {run_id}")
            console.print()
            console.print("[dim]Tip: Use 'ragctl history list' to see available runs[/dim]")
            raise typer.Exit(1)

    # Get failed files
    failed_files = history.get_failed_files(run_id)

    if not failed_files:
        print_info(f"No failed/skipped files in run {run_id}")
        console.print()
        console.print(f"[dim]Run stats:[/dim]")
        console.print(f"  • Total: {run.total_files}")
        console.print(f"  • Success: {run.success}")
        console.print(f"  • Failed: {run.failed}")
        console.print(f"  • Skipped: {run.skipped}")
        raise typer.Exit(0)

    # Display summary
    console.print()
    console.print(f"[bold]Run Information:[/bold]")
    console.print(f"  • Run ID: {run.run_id}")
    console.print(f"  • Date: {run.timestamp}")
    console.print(f"  • Mode: {run.mode}")
    console.print(f"  • Status: {run.status.value}")
    console.print()

    console.print(f"[bold]Found {len(failed_files)} file(s) to retry:[/bold]")
    console.print()

    for i, f in enumerate(failed_files, 1):
        status_color = "red" if f.status == FileStatus.FAILED else "yellow"
        console.print(f"  {i}. [{status_color}]{f.status.value}[/{status_color}] {f.filename}")

        if f.error:
            console.print(f"     [dim]Error: {f.error}[/dim]")
        if f.reason:
            console.print(f"     [dim]Reason: {f.reason}[/dim]")
        if f.retries > 0:
            console.print(f"     [dim]Retries: {f.retries}[/dim]")

        console.print()

    # Show mode
    if show:
        console.print("[bold green]✓[/bold green] Dry run mode - no files will be processed")
        raise typer.Exit(0)

    # Check if we have file paths
    missing_paths = [f for f in failed_files if not f.filepath or not Path(f.filepath).exists()]
    if missing_paths:
        print_warning(f"{len(missing_paths)} file(s) have missing or invalid paths:")
        for f in missing_paths[:5]:  # Show first 5
            console.print(f"  • {f.filename}: {f.filepath or 'No path stored'}")
        if len(missing_paths) > 5:
            console.print(f"  ... and {len(missing_paths) - 5} more")
        console.print()
        print_error(
            "Cannot retry files without valid paths. "
            "This may be due to an older run format."
        )
        raise typer.Exit(1)

    # Confirm
    console.print(f"[bold]Retry Configuration:[/bold]")
    console.print(f"  • Mode: {mode}")
    if output:
        console.print(f"  • Output: {output}")
    console.print()

    if not typer.confirm("Proceed with retry?"):
        print_info("Retry cancelled")
        raise typer.Exit(0)

    console.print()
    console.print("[bold]🔄 Starting retry...[/bold]")
    console.print()

    # Import here to avoid circular imports
    from src.core.chunk.chunker import chunk_document
    from src.core.pipeline import (
        create_pipeline_manager,
        FileResult,
        PipelineRun,
        retry_with_backoff,
        RetryConfig,
    )

    # Create output directory
    if output:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = Path.cwd() / "output_retry"
        output.mkdir(parents=True, exist_ok=True)

    # Create pipeline manager with specified mode
    auto_continue = (mode == "auto-continue")
    auto_stop = (mode == "auto-stop")
    auto_skip = (mode == "auto-skip")

    pipeline_manager = create_pipeline_manager(
        interactive=(mode == "interactive"),
        auto_continue=auto_continue,
        auto_stop=auto_stop,
        auto_skip=auto_skip,
    )

    # Create new run for retry
    retry_run = history.create_run(
        total_files=len(failed_files),
        mode=mode,
        config={
            "original_run_id": run_id,
            "retry_type": "failed_files",
        }
    )
    history.start_run(retry_run)

    # Process files
    start_time = time.time()
    retry_config = RetryConfig(max_attempts=3)

    for i, file_info in enumerate(failed_files, 1):
        file_path = Path(file_info.filepath)
        console.print(f"📄 [{i}/{len(failed_files)}] Processing: {file_path.name}")

        file_start = time.time()

        try:
            # Wrap chunking in retry
            def _process():
                return chunk_document(
                    str(file_path),
                    strategy=run.config.get("strategy", "semantic"),
                    max_tokens=run.config.get("max_tokens", 400),
                    overlap=run.config.get("overlap", 50),
                )

            result = retry_with_backoff(
                _process,
                config=retry_config,
                operation_name=f"Process {file_path.name}",
            )

            file_duration = time.time() - file_start

            # Success
            file_result = FileResult(
                filename=file_path.name,
                filepath=str(file_path),
                status=FileStatus.SUCCESS,
                chunks_created=len(result.get("chunks", [])),
                duration=file_duration,
                retries=0,  # Would need to track actual retries
            )

            console.print(f"  [green]✅ OK[/green] ({len(result.get('chunks', []))} chunks, {file_duration:.2f}s)")

        except Exception as e:
            file_duration = time.time() - file_start

            # Handle error with pipeline manager
            decision = pipeline_manager.handle_error(
                error=e,
                file_path=file_path,
                attempt=retry_config.max_attempts,
                context={"step": "Chunking"},
            )

            # Create file result based on decision
            if decision.value == "ignore":
                status = FileStatus.SKIPPED
                console.print(f"  [yellow]⏭ Skipped[/yellow]")
            elif decision.value == "stop":
                status = FileStatus.ABORTED
                console.print(f"  [red]🛑 Aborted[/red]")
            else:
                status = FileStatus.FAILED
                console.print(f"  [red]❌ Failed[/red]")

            file_result = FileResult(
                filename=file_path.name,
                filepath=str(file_path),
                status=status,
                error=str(e),
                error_type=type(e).__name__,
                duration=file_duration,
                retries=retry_config.max_attempts,
                user_decision=decision.value,
            )

            # Check if should stop
            if not pipeline_manager.should_continue(decision):
                history.update_run(retry_run, file_result)
                history.abort_run(retry_run, reason="User requested stop")
                console.print()
                print_warning("Pipeline aborted by user")
                break

        # Update history
        history.update_run(retry_run, file_result)
        console.print()

    # Finalize
    total_duration = time.time() - start_time
    history.finalize_run(retry_run, total_duration)

    # Show summary
    console.print()
    console.print("[bold]═══════════════════════════════════════[/bold]")
    console.print("[bold cyan]📊 Retry Summary[/bold cyan]")
    console.print("[bold]═══════════════════════════════════════[/bold]")
    console.print()
    console.print(f"Original run: {run_id}")
    console.print(f"Retry run:    {retry_run.run_id}")
    console.print(f"Duration:     {total_duration:.1f}s")
    console.print()
    console.print(f"✅ Success:  {retry_run.success}/{len(failed_files)}")
    console.print(f"❌ Failed:   {retry_run.failed}/{len(failed_files)}")
    console.print(f"⚠️  Skipped:  {retry_run.skipped}/{len(failed_files)}")
    if retry_run.aborted > 0:
        console.print(f"🛑 Aborted:  {retry_run.aborted}/{len(failed_files)}")
    console.print()

    if retry_run.success == len(failed_files):
        print_success("All files processed successfully! ✨")
    elif retry_run.success > 0:
        print_info(f"Partial success: {retry_run.success}/{len(failed_files)} files processed")
    else:
        print_error("No files processed successfully")

    console.print()
    console.print(f"📁 Output: {output}")
    console.print(f"📜 History: ~/.atlasrag/history/runs/{retry_run.run_id}.json")
    console.print()
