"""
Data schemas for documents and chunks handled by the ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import uuid


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Document:
    """Represents a single source document after ingestion."""

    source_path: Path
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: Optional[str] = None
    content_type: Optional[str] = None
    status: str = "processed"
    id: str = field(default_factory=lambda: _generate_id("doc"))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        return payload


@dataclass
class Chunk:
    """Represents a chunk emitted by the chunking stage."""

    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _generate_id("chunk"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create a Chunk from a dictionary."""
        return cls(
            document_id=data["document_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            id=data.get("id", _generate_id("chunk")),
        )


def make_chunks(
    document: Document,
    parts: Iterable[str],
    *,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Utility to convert raw chunk strings into typed `Chunk` instances."""
    chunks: List[Chunk] = []
    for part in parts:
        metadata = {"source": document.metadata.get("source_name", Path(document.source_path).name)}
        metadata.update(document.metadata)
        if additional_metadata:
            metadata.update(additional_metadata)
        chunks.append(
            Chunk(
                document_id=document.id,
                text=part,
                metadata=metadata,
            )
        )
    return chunks
