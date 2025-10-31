"""
Helper utilities to read pipeline artifacts from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import Chunk


def read_chunks_jsonl(input_path: Path | str) -> List[Chunk]:
    """
    Read chunks from JSONL format.

    Args:
        input_path: Path to JSONL file containing chunks

    Returns:
        List of Chunk objects

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    chunks = []
    with input_path.open("r", encoding="utf8") as stream:
        for line_num, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                chunk = Chunk.from_dict(data)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                )
            except Exception as e:
                raise ValueError(f"Error parsing chunk on line {line_num}: {e}")

    return chunks
