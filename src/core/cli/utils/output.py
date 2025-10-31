"""Output format utilities for CLI."""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum


class OutputFormat(str, Enum):
    """Available output formats."""
    json = "json"
    jsonl = "jsonl"
    csv = "csv"


def save_chunks_json(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    """Save chunks to JSON format."""
    output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))


def save_chunks_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    """Save chunks to JSONL format (one JSON object per line)."""
    lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    output_path.write_text("\n".join(lines) + "\n")


def save_chunks_csv(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    """Save chunks to CSV format."""
    if not chunks:
        output_path.write_text("id,text,metadata\n")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Determine all possible fields
        fieldnames = ['id', 'text']

        # Add metadata keys
        metadata_keys = set()
        for chunk in chunks:
            if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                metadata_keys.update(chunk['metadata'].keys())

        fieldnames.extend(sorted(metadata_keys))

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for chunk in chunks:
            row = {
                'id': chunk.get('id', ''),
                'text': chunk.get('text', '').replace('\n', ' ')  # Remove newlines for CSV
            }

            # Flatten metadata
            if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                for key, value in chunk['metadata'].items():
                    if isinstance(value, (str, int, float, bool)):
                        row[key] = value
                    else:
                        row[key] = json.dumps(value)

            writer.writerow(row)


def save_chunks(chunks: List[Dict[str, Any]], output_path: Path, format: OutputFormat = OutputFormat.json) -> None:
    """
    Save chunks to file in specified format.

    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output file
        format: Output format (json, jsonl, or csv)
    """
    if format == OutputFormat.json:
        save_chunks_json(chunks, output_path)
    elif format == OutputFormat.jsonl:
        save_chunks_jsonl(chunks, output_path)
    elif format == OutputFormat.csv:
        save_chunks_csv(chunks, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def detect_format_from_extension(path: Path) -> OutputFormat:
    """Detect output format from file extension."""
    ext = path.suffix.lower()

    if ext == '.jsonl':
        return OutputFormat.jsonl
    elif ext == '.csv':
        return OutputFormat.csv
    else:  # Default to JSON
        return OutputFormat.json
