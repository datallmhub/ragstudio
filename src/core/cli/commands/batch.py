"""Batch command for ragctl CLI with retry and interactive error handling."""
import json
import time
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer
import humanfriendly

from src.core.chunk.chunker import chunk_document
from src.core.cli.commands.chunk import ChunkStrategy, Document
from src.core.cli.commands.chunk_helpers import load_document_universal
from src.core.cli.utils.display import (
    console,
    print_success,
    print_error,
    print_warning,
    print_info,
    create_batch_progress,
    display_stats
)
from src.core.cli.utils.validation import validate_directory_exists, validate_output_path
from src.core.cli.utils.output import save_chunks, detect_format_from_extension
from src.core.cli.utils.quality_check import check_chunks_quality
from src.core.cli.utils.security import (
    validate_batch_size,
    validate_total_size,
    validate_pattern_safe,
    validate_disk_space,
    validate_file_size,
    validate_no_symlinks,
    sanitize_metadata,
    get_security_config
)

# Import pipeline components
from src.core.pipeline import (
    create_pipeline_manager,
    HistoryManager,
    FileResult,
    FileStatus,
    RetryConfig,
    retry_with_backoff,
    RetryableError,
)


def batch_command(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Directory containing files to process",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        )
    ],
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern", "-p",
            help="File pattern to match (e.g., '*.txt', '*.pdf', '*.*' for all files)"
        )
    ] = "*.*",
    strategy: Annotated[
        ChunkStrategy,
        typer.Option(
            "--strategy", "-s",
            help="Chunking strategy to use"
        )
    ] = ChunkStrategy.semantic,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens", "-m",
            help="Maximum tokens per chunk",
            min=50,
            max=2000
        )
    ] = 400,
    overlap: Annotated[
        int,
        typer.Option(
            "--overlap", "-ol",
            help="Token overlap between chunks",
            min=0,
            max=500
        )
    ] = 50,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output directory for chunk files (one file per document by default)"
        )
    ] = None,
    single_file: Annotated[
        bool,
        typer.Option(
            "--single-file",
            help="Combine all chunks into a single output file (default: one file per document)"
        )
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive", "-r",
            help="Process subdirectories recursively"
        )
    ] = False,
    advanced_ocr: Annotated[
        bool,
        typer.Option(
            "--advanced-ocr",
            help="Use intelligent OCR routing for scanned PDFs (auto-selects Qwen-VL, Nougat, or classic OCR)"
        )
    ] = False,
    auto_continue: Annotated[
        bool,
        typer.Option(
            "--auto-continue",
            help="Continue automatically on errors (non-interactive, for CI/CD)"
        )
    ] = False,
    auto_stop: Annotated[
        bool,
        typer.Option(
            "--auto-stop",
            help="Stop on first error (non-interactive, for validation)"
        )
    ] = False,
    auto_skip: Annotated[
        bool,
        typer.Option(
            "--auto-skip",
            help="Skip failed files automatically (non-interactive, for batch processing)"
        )
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", "-n",
            help="List files that would be processed without processing them"
        )
    ] = False,
    save_history: Annotated[
        bool,
        typer.Option(
            "--save-history / --no-history",
            help="Save run to history for retry capability (default: True)"
        )
    ] = True,
) -> None:
    """
    Process multiple files in batch mode with automatic retry and error handling.

    This command discovers all files matching the pattern in the specified
    directory and chunks them using the same strategy. Features: automatic
    retry, interactive error handling, and run history tracking.

    \b
    Examples:
        # Process all .txt files in a directory (interactive mode)
        ragctl batch ./documents

        # Process all markdown files recursively
        ragctl batch ./docs --pattern "*.md" --recursive

        # Process and save chunks (one file per document)
        ragctl batch ./data --pattern "*.txt" -o ./output

        # Process and save all chunks in a single file
        ragctl batch ./data --pattern "*.txt" -o all_chunks.jsonl --single-file

        # Process with custom chunking parameters
        ragctl batch ./docs --strategy sentence --max-tokens 300

        # Auto-continue on errors (for CI/CD)
        ragctl batch ./docs --auto-continue

        # Stop on first error (for validation)
        ragctl batch ./docs --auto-stop

        # Skip failed files automatically (for large batches)
        ragctl batch ./docs --auto-skip
    """
    # === SECURITY VALIDATIONS ===
    security_config = get_security_config()

    # Validate pattern safety
    try:
        validate_pattern_safe(pattern, security_config)
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # Validate directory
    try:
        validate_directory_exists(directory)
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # Validate output path
    if output:
        try:
            validate_output_path(output)
        except typer.BadParameter as e:
            print_error(str(e))
            raise typer.Exit(code=1)

    # === FILE DISCOVERY (SOLID refactoring) ===
    from src.core.cli.commands.batch_helpers import (
        discover_files,
        filter_supported_files,
        validate_files_for_batch,
        SUPPORTED_EXTENSIONS
    )

    # Find files using helper
    all_files = discover_files(directory, pattern, recursive)

    if not all_files:
        print_warning(f"No files found matching pattern: {pattern}")
        console.print(f"  Directory: {directory}")
        console.print(f"  Recursive: {recursive}")
        raise typer.Exit(code=0)

    # Filter to supported formats using helper
    files, unsupported_files = filter_supported_files(all_files)

    # Display info about unsupported files
    if unsupported_files:
        console.print()
        print_warning(f"Skipping {len(unsupported_files)} unsupported file(s):")
        for filename, ext in unsupported_files[:5]:  # Show first 5
            console.print(f"  • {filename} [dim](format {ext or 'unknown'} non supporté)[/dim]")
        if len(unsupported_files) > 5:
            console.print(f"  ... et {len(unsupported_files) - 5} autre(s)")
        console.print()

    if not files:
        print_error("No supported files found")
        console.print(f"  Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        raise typer.Exit(code=0)

    # === ADDITIONAL SECURITY VALIDATIONS ===
    try:
        # Validate batch size
        validate_batch_size(files, security_config)

        # Validate total size
        validate_total_size(files, security_config)

        # Validate output disk space if output specified
        if output:
            validate_disk_space(output, security_config)

        # Validate individual files using helper
        files = validate_files_for_batch(files, security_config, validate_no_symlinks)

        if not files:
            print_error("No valid files remaining after security validation")
            raise typer.Exit(code=1)

    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # === DRY RUN ===
    if dry_run:
        console.print(f"Would process {len(files)} files:")
        total_size = 0
        sorted_files = sorted(files, key=lambda p: p.name)
        
        for i, file_path in enumerate(sorted_files):
            try:
                size = file_path.stat().st_size
                total_size += size
                formatted_size = humanfriendly.format_size(size)
                
                prefix = "└── " if i == len(sorted_files) - 1 else "├── "
                console.print(f"  {prefix}{file_path.name} ({formatted_size})")
            except Exception as e:
                console.print(f"  ? {file_path.name} (error getting size: {e})")

        console.print(f"Total: {len(files)} files, {humanfriendly.format_size(total_size)}")
        raise typer.Exit(code=0)

    # Create pipeline manager with specified mode
    pipeline_manager = create_pipeline_manager(
        interactive=(not auto_continue and not auto_stop and not auto_skip),
        auto_continue=auto_continue,
        auto_stop=auto_stop,
        auto_skip=auto_skip,
    )

    # Display header
    console.print(f"\n[bold cyan]📊 ragctl - Batch Processing[/bold cyan]")
    console.print("[bold]═══════════════════════════════════════[/bold]")
    console.print(f"Files to process: {len(files)}")
    console.print(f"Pattern:          {pattern}")
    console.print(f"Strategy:         {strategy.value}")
    console.print(f"Max tokens:       {max_tokens}")
    console.print(f"Mode:             {pipeline_manager.mode.value}")
    if output:
        output_mode = "Single file" if single_file else "One file per document"
        console.print(f"Output:           {output_mode}")
    if advanced_ocr:
        console.print(f"Advanced OCR:     [bold green]Enabled[/bold green]")
    if save_history:
        console.print(f"History:          [bold green]Enabled[/bold green]")
    console.print("[bold]═══════════════════════════════════════[/bold]\n")

    # Create history manager and run
    history = HistoryManager() if save_history else None
    run = None

    if save_history:
        run = history.create_run(
            total_files=len(files),
            mode=pipeline_manager.mode.value,
            config={
                "pattern": pattern,
                "strategy": strategy.value,
                "max_tokens": max_tokens,
                "overlap": overlap,
                "advanced_ocr": advanced_ocr,
                "directory": str(directory),
                "recursive": recursive,
            }
        )
        history.start_run(run)
        console.print(f"[dim]Run ID: {run.run_id}[/dim]\n")

    # Retry configuration
    retry_config = RetryConfig(max_attempts=3)

    # Tracking
    all_chunks = []
    chunks_by_file = {}  # Track chunks per file for individual saving
    success_count = 0
    failed_count = 0
    skipped_count = 0
    aborted_count = 0
    start_time = time.time()

    # Initialize OCR router if needed (lazy loading)
    ocr_router = None
    if advanced_ocr:
        try:
            from src.workflows.router.ocr_router import OCRRouter, OCRRouterConfig
            ocr_router = OCRRouter(OCRRouterConfig())
        except ImportError as e:
            print_warning(f"Advanced OCR dependencies not available: {e}")
            print_warning("PDFs will be processed with standard text extraction")
        except Exception as e:
            print_warning(f"Failed to initialize OCR router: {e}")
            print_warning("PDFs will be processed with standard text extraction")

    # Process files with progress bar
    with create_batch_progress() as progress:
        task = progress.add_task("Processing...", total=len(files))

        for i, file_path in enumerate(files, 1):
            progress.update(task, description=f"[{i}/{len(files)}] {file_path.name}")

            file_start = time.time()
            file_result = None

            try:
                # Define processing function for retry
                def _process_file():
                    # Define formats that support advanced OCR
                    ADVANCED_OCR_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff'}

                    # Read file
                    if advanced_ocr and ocr_router and file_path.suffix.lower() in ADVANCED_OCR_FORMATS:
                        result = ocr_router.process_document(file_path)
                        text = result['extracted_text']
                        if not text:
                            raise ValueError(f"No text extracted: {result.get('errors', [])}")
                    else:
                        text = load_document_universal(file_path, print_info, use_status=False, console=None)

                    # Create document
                    doc = Document(
                        text=text,
                        metadata={"source": str(file_path), "size": len(text)},
                        source_path=str(file_path)
                    )

                    # Chunk document
                    return chunk_document(
                        doc,
                        strategy=strategy.value,
                        max_tokens=max_tokens,
                        overlap=overlap
                    )

                # Execute with retry
                chunks = retry_with_backoff(
                    _process_file,
                    config=retry_config,
                    operation_name=f"{file_path.name}",
                )

                # Convert chunks to dict
                chunks_data = [
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": {**chunk.metadata, "source_file": str(file_path)}
                    }
                    for chunk in chunks
                ]

                # Check chunks quality - detect unreadable documents
                quality_check = check_chunks_quality(chunks_data, min_readable_ratio=0.9)

                file_duration = time.time() - file_start

                # If document is unreadable, mark as failed and don't save
                if not quality_check["is_readable"]:
                    failed_count += 1

                    progress.console.print(
                        f"  [{i}/{len(files)}] [red]❌[/red] {file_path.name} "
                        f"[dim]({file_duration:.1f}s)[/dim] [yellow]Document illisible: {quality_check['reason']}[/yellow]"
                    )

                    file_result = FileResult(
                        filename=file_path.name,
                        filepath=str(file_path),
                        status=FileStatus.FAILED,
                        error=f"Document illisible: {quality_check['reason']}",
                        duration=file_duration,
                        retries=0,
                    )
                else:
                    # Document is readable, proceed normally
                    all_chunks.extend(chunks_data)
                    chunks_by_file[file_path.stem] = chunks_data  # Store by filename
                    success_count += 1

                    # Save individual file if not in single-file mode and output is specified
                    if output and not single_file:
                        try:
                            # Create output directory if it doesn't exist
                            output_dir = output if output.is_dir() or not output.suffix else output.parent
                            output_dir.mkdir(parents=True, exist_ok=True)

                            # Generate output filename
                            output_file = output_dir / f"{file_path.stem}_chunks.jsonl"

                            # Detect format from output filename
                            output_format = detect_format_from_extension(output_file)

                            # Save chunks for this file
                            save_chunks(chunks_data, output_file, output_format)

                            progress.console.print(
                                f"  [{i}/{len(files)}] [green]✅[/green] {file_path.name} "
                                f"({len(chunks_data)} chunks, {file_duration:.1f}s) → {output_file.name}"
                            )
                        except Exception as e:
                            progress.console.print(
                                f"  [{i}/{len(files)}] [green]✅[/green] {file_path.name} "
                                f"({len(chunks_data)} chunks, {file_duration:.1f}s) [yellow]⚠️ Save failed: {e}[/yellow]"
                            )
                    else:
                        progress.console.print(
                            f"  [{i}/{len(files)}] [green]✅[/green] {file_path.name} "
                            f"({len(chunks_data)} chunks, {file_duration:.1f}s)"
                        )

                    # Create success result
                    file_result = FileResult(
                        filename=file_path.name,
                        filepath=str(file_path),
                        status=FileStatus.SUCCESS,
                        chunks_created=len(chunks_data),
                        duration=file_duration,
                        retries=0,
                    )

            except KeyboardInterrupt:
                # User interrupted with Ctrl+C
                progress.console.print("\n[yellow]⚠️[/yellow] Interrupted by user (Ctrl+C)")

                file_result = FileResult(
                    filename=file_path.name,
                    filepath=str(file_path),
                    status=FileStatus.ABORTED,
                    error="User interrupted (Ctrl+C)",
                    duration=time.time() - file_start,
                )

                if run and save_history:
                    history.update_run(run, file_result)
                    history.abort_run(run, "User interrupted (Ctrl+C)")

                break

            except Exception as e:
                file_duration = time.time() - file_start

                # Handle error with pipeline manager
                decision = pipeline_manager.handle_error(
                    error=e,
                    file_path=file_path,
                    attempt=retry_config.max_attempts,
                    context={
                        "step": "Chunking",
                        "strategy": strategy.value,
                    }
                )

                # Create file result based on decision
                if decision.value == "ignore":
                    status = FileStatus.SKIPPED
                    skipped_count += 1
                    emoji = "⏭"
                    color = "yellow"
                elif decision.value == "stop":
                    status = FileStatus.ABORTED
                    aborted_count += 1
                    emoji = "🛑"
                    color = "red"
                else:  # continue
                    status = FileStatus.FAILED
                    failed_count += 1
                    emoji = "❌"
                    color = "red"

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

                progress.console.print(
                    f"  [{i}/{len(files)}] [{color}]{emoji}[/{color}] {file_path.name} "
                    f"({status.value})"
                )

                # Check if should stop
                if not pipeline_manager.should_continue(decision):
                    if run and save_history:
                        history.update_run(run, file_result)
                        history.abort_run(run, "User requested stop")
                    break

            # Update history
            if file_result and run and save_history:
                history.update_run(run, file_result)

            progress.advance(task)

    # Calculate totals
    total_duration = time.time() - start_time
    processed = success_count + failed_count + skipped_count + aborted_count

    # Finalize history
    if run and save_history:
        if pipeline_manager.is_stopped():
            history.abort_run(run, "User requested stop")
        else:
            history.finalize_run(run, total_duration)

    # Display summary
    console.print()
    console.print("[bold]═══════════════════════════════════════[/bold]")
    console.print("[bold cyan]📊 Batch Summary[/bold cyan]")
    console.print("[bold]═══════════════════════════════════════[/bold]")
    console.print()

    if run:
        console.print(f"Run ID:   {run.run_id}")
        console.print(f"Duration: {total_duration:.1f}s")
        console.print(f"Mode:     {pipeline_manager.mode.value}")
        console.print()

    console.print(f"Total:       {len(files)} files")
    console.print(f"✅ Success:  {success_count} ({success_count/len(files)*100:.1f}%)")
    if failed_count > 0:
        console.print(f"❌ Failed:   {failed_count}")
    if skipped_count > 0:
        console.print(f"⚠️  Skipped:  {skipped_count}")
    if aborted_count > 0:
        console.print(f"🛑 Aborted:  {aborted_count}")

    console.print()
    console.print(f"Total chunks: {len(all_chunks)}")

    # Save to single file if requested
    if output and all_chunks and single_file:
        console.print()
        try:
            output_format = detect_format_from_extension(output)
            with console.status(f"[bold green]Saving to {output.name}..."):
                save_chunks(all_chunks, output, output_format)

            console.print(f"✅ Saved: {output} ({len(all_chunks)} chunks, {output_format.value})")
        except Exception as e:
            print_error(f"Error saving output: {e}")
    elif output and all_chunks and not single_file:
        # Files were already saved individually
        console.print()
        console.print(f"✅ Saved: {success_count} files in {output}/ (one file per document)")

    # Show history location
    if run and save_history:
        console.print()
        console.print(f"📜 History: ~/.ragctl/history/runs/{run.run_id}.json")

        if failed_count > 0 or skipped_count > 0:
            console.print()
            console.print("[bold]💡 To retry failed/skipped files:[/bold]")
            console.print(f"   ragctl retry {run.run_id}")

    console.print()

    # Show final status
    if success_count == len(files):
        print_success("All files processed successfully! ✨")
    elif success_count > 0:
        print_info(f"Partial success: {success_count}/{len(files)} files processed")
    else:
        print_error("No files processed successfully")

    console.print()

    # Exit with error code if any files failed
    if failed_count > 0 or aborted_count > 0:
        raise typer.Exit(code=1)
