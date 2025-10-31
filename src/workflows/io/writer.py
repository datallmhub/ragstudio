"""
Helper utilities to persist pipeline artefacts to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Chunk


def write_chunks_jsonl(chunks: Iterable[Chunk], output_path: Path) -> None:
    """
    Persist chunks to JSONL format ready for embedding generation.

    The caller is responsible for ensuring `output_path` lives in a writable directory.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as stream:
        for chunk in chunks:
            json.dump(chunk.to_dict(), stream, ensure_ascii=False)
            stream.write("\n")
