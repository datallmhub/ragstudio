"""Helper functions for chunk command - following SOLID principles.

Each function has a single responsibility (SRP).
"""
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


def display_routing_decisions(result: Dict[str, Any], console) -> None:
    """
    Display OCR routing decisions in a clear, structured format.

    Single Responsibility: Display OCR routing decisions only.

    Args:
        result: OCR processing result with routing_decisions list
        console: Rich console instance for output
    """
    if not result.get("routing_decisions"):
        return

    console.print("\n[bold cyan]📋 OCR Routing Decision Tree:[/bold cyan]")

    decisions = result["routing_decisions"]
    for i, decision in enumerate(decisions, 1):
        step = decision.get("step", "unknown")

        if step == "ocr_quality_detection":
            quality_category = decision.get("ocr_quality_category", "UNKNOWN")
            quality_score = decision.get("ocr_quality_score", 0.0)
            recommended_engine = decision.get("recommended_engine", "unknown")

            # Color based on quality
            if quality_category == "HIGH":
                quality_color = "green"
            elif quality_category == "MEDIUM":
                quality_color = "yellow"
            else:
                quality_color = "red"

            console.print(f"  [{i}] [bold]OCR Quality Detection[/bold]")
            console.print(f"      • Quality: [{quality_color}]{quality_category}[/{quality_color}] (score: {quality_score:.3f})")
            console.print(f"      • Recommended: [bold]{recommended_engine}[/bold]")
            console.print(f"      → [italic]Reason:[/italic] {'Low quality requires advanced OCR' if quality_category == 'LOW' else 'Standard OCR sufficient'}")

        elif step == "scientific_detection":
            is_scientific = decision.get("is_scientific", False)
            math_density = decision.get("math_density", 0.0)
            recommended_engine = decision.get("recommended_engine", "unknown")

            console.print(f"  [{i}] [bold]Scientific Content Detection[/bold]")
            if is_scientific:
                console.print(f"      • Scientific: [bold green]YES[/bold green] (math density: {math_density:.3f})")
                console.print(f"      • Recommended: [bold]{recommended_engine}[/bold]")
                console.print(f"      → [italic]Reason:[/italic] High mathematical content requires specialized OCR (Nougat)")
            else:
                console.print(f"      • Scientific: [dim]NO[/dim] (math density: {math_density:.3f})")
                console.print(f"      → [italic]Reason:[/italic] No specialized mathematical OCR needed")

        elif step == "complexity_analysis":
            complexity_score = decision.get("complexity_score", 0.0)
            recommended_strategy = decision.get("recommended_strategy", "unknown")

            # Determine complexity level
            if complexity_score >= 0.7:
                complexity_level = "HIGH"
                complexity_color = "red"
                reason = "Complex document requires advanced OCR (Qwen-VL)"
            elif complexity_score >= 0.4:
                complexity_level = "MEDIUM"
                complexity_color = "yellow"
                reason = "Moderate complexity - standard or mid-tier OCR suitable"
            else:
                complexity_level = "LOW"
                complexity_color = "green"
                reason = "Simple document - classic OCR sufficient"

            console.print(f"  [{i}] [bold]Complexity Analysis[/bold]")
            console.print(f"      • Complexity: [{complexity_color}]{complexity_level}[/{complexity_color}] (score: {complexity_score:.3f})")
            console.print(f"      • Recommended: [bold]{recommended_strategy}[/bold]")
            console.print(f"      → [italic]Reason:[/italic] {reason}")

        elif step == "ocr_routing":
            engine_used = decision.get("engine_used", "unknown")
            routing_reason = decision.get("routing_reason", "")

            # Determine engine type for color
            if "qwen" in engine_used.lower():
                engine_color = "magenta"
                engine_type = "Advanced Vision-Language Model"
            elif "nougat" in engine_used.lower():
                engine_color = "blue"
                engine_type = "Scientific OCR Specialist"
            elif "classic" in engine_used.lower():
                engine_color = "cyan"
                engine_type = "Standard OCR"
            else:
                engine_color = "white"
                engine_type = "OCR Engine"

            console.print(f"  [{i}] [bold]Final OCR Engine Selection[/bold]")
            console.print(f"      • Engine: [{engine_color}]{engine_used}[/{engine_color}]")
            console.print(f"      • Type: [dim]{engine_type}[/dim]")
            console.print(f"      → [italic]Reason:[/italic] {routing_reason}")

            # Check for fallback indication
            if "fallback" in routing_reason.lower():
                console.print(f"      [yellow]⚠ Note:[/yellow] Primary engine unavailable, using fallback")

        elif step == "fallback":
            engine_used = decision.get("engine_used", "unknown")
            reason = decision.get("reason", "")

            console.print(f"  [{i}] [bold yellow]Fallback Activated[/bold yellow]")
            console.print(f"      • Fallback engine: [bold]{engine_used}[/bold]")
            console.print(f"      → [italic]Reason:[/italic] {reason}")

    console.print("")


def load_document_universal(file_path: Path, print_info_func, use_status: bool = True, console=None) -> str:
    """
    Universal document loader supporting all formats.

    Single Responsibility: Load documents from any supported format.

    Supports: TXT, MD, PDF, DOCX, DOC, HTML, HTM, PNG, JPG, JPEG, TIFF

    Args:
        file_path: Path to the file
        print_info_func: Function to print info messages
        use_status: Whether to use console.status (disable in batch mode to avoid conflicts)
        console: Rich console instance (optional, needed only if use_status=True)

    Returns:
        Extracted text content
    """
    suffix = file_path.suffix.lower()

    # Fast path for simple text files
    if suffix in {'.txt', '.md'}:
        if use_status and console:
            with console.status(f"[bold green]Reading {file_path.name}..."):
                return file_path.read_text(encoding='utf-8')
        else:
            return file_path.read_text(encoding='utf-8')

    # Use universal loader for all other formats (PDF, Word, HTML, images, etc.)
    try:
        from src.workflows.ingest.loader import ingest_file, SUPPORTED_EXTENSIONS

        # Check if format is supported
        if suffix not in SUPPORTED_EXTENSIONS:
            print_info_func(f"⚠️  Format {suffix} not officially supported, attempting to load...")

        if use_status and console:
            with console.status(f"[bold green]Loading {file_path.name} ({suffix})..."):
                document = ingest_file(str(file_path))
                text = document.text
        else:
            document = ingest_file(str(file_path))
            text = document.text

        if not text:
            raise ValueError(f"No text extracted from {file_path.name}")

        # Show info about extraction (only if not in batch mode)
        if use_status:
            print_info_func(f"Extracted [bold]{len(text)} chars[/bold] from {suffix.upper()} document")

        return text

    except ImportError as e:
        # Fallback to simple text reading if loader not available
        from src.core.cli.utils.display import print_warning
        if use_status:
            print_warning(f"Universal loader not available ({e}), trying simple text read")
        return file_path.read_text(encoding='utf-8')

    except Exception as e:
        raise RuntimeError(f"Failed to load document: {e}")


def generate_processing_summary(
    file_path: Path,
    config: Any,
    processing_data: Dict[str, Any],
    chunks: list,
    success: bool = True,
    errors: list = None
) -> Dict[str, Any]:
    """
    Generate structured JSON summary of the processing pipeline.

    Single Responsibility: Generate processing summary only.

    Args:
        file_path: Path to the processed file
        config: AtlasConfig instance
        processing_data: Dictionary with processing information (OCR results, timings, etc.)
        chunks: List of generated chunks
        success: Whether processing was successful
        errors: List of error messages if any

    Returns:
        Dictionary with complete processing summary

    Examples:
        >>> from pathlib import Path
        >>> from unittest.mock import Mock
        >>> config = Mock()
        >>> config.llm.use_llm = False
        >>> config.ocr.use_advanced_ocr = False
        >>> config.chunking.strategy = "semantic"
        >>> summary = generate_processing_summary(Path("test.txt"), config, {}, [])
        >>> summary["metadata"]["success"]
        True
    """
    summary = {
        "metadata": {
            "atlas_rag_version": "1.0.0",
            "processing_timestamp": datetime.now().isoformat(),
            "success": success,
            "errors": errors or []
        },
        "document": {
            "path": str(file_path),
            "filename": file_path.name,
            "format": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "text_length": processing_data.get("text_length", 0),
            "language": processing_data.get("language", "unknown")
        },
        "configuration": {
            "llm": {
                "enabled": config.llm.use_llm,
                "provider": config.llm.provider if config.llm.use_llm else None,
                "model": config.llm.model if config.llm.use_llm else None,
                "is_local": config.llm.is_local if config.llm.use_llm else None
            },
            "ocr": {
                "advanced_ocr_enabled": config.ocr.use_advanced_ocr,
                "dictionary_threshold": config.ocr.dictionary_threshold,
                "dynamic_threshold": config.ocr.dynamic_threshold,
                "fallback_enabled": config.ocr.enable_fallback
            },
            "chunking": {
                "strategy": config.chunking.strategy,
                "max_tokens": config.chunking.max_tokens,
                "overlap": config.chunking.overlap
            }
        },
        "processing": {
            "total_time_seconds": processing_data.get("total_time", 0),
            "stages": {}
        },
        "results": {
            "chunks": {
                "total_count": len(chunks),
                "average_size_chars": (sum(len(c.text) for c in chunks) // len(chunks)) if len(chunks) > 0 else 0,
                "min_size_chars": min((len(c.text) for c in chunks), default=0),
                "max_size_chars": max((len(c.text) for c in chunks), default=0),
                "total_text_length": sum(len(c.text) for c in chunks)
            }
        }
    }

    # Add OCR-specific data if available
    if processing_data.get("ocr_result"):
        ocr_data = processing_data["ocr_result"]
        summary["processing"]["stages"]["ocr"] = {
            "time_seconds": processing_data.get("ocr_time", 0),
            "engine": ocr_data.get("metadata", {}).get("ocr_engine", "unknown"),
            "success": ocr_data.get("metadata", {}).get("success", False),
            "routing_decisions": ocr_data.get("routing_decisions", []),
            "quality_metrics": ocr_data.get("metadata", {}).get("quality_metrics", {}),
            "fallback_used": ocr_data.get("metadata", {}).get("fallback_from") is not None,
            "fallback_reason": ocr_data.get("metadata", {}).get("fallback_reason")
        }

    # Add strategy selection data if available
    if processing_data.get("strategy_selection"):
        summary["processing"]["stages"]["strategy_selection"] = processing_data["strategy_selection"]

    # Add chunking timing
    if processing_data.get("chunking_time"):
        summary["processing"]["stages"]["chunking"] = {
            "time_seconds": processing_data["chunking_time"]
        }

    return summary
