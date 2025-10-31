"""
Adaptive chunking powered by LangChain text splitters (v3.0).

Pure LangChain implementation - Chonkie removed.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from src.core.config import chunking as chunking_config
from src.core.chunk.langchain_chunker import AtlasChunker, chunk_document_langchain
from src.workflows.io.schema import Chunk, Document, make_chunks
from src.workflows.ingest.pdf_cleaner import preprocess_before_chunking

LOGGER = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore


DEFAULT_STRATEGY = chunking_config.strategy


def _encoding_length(text: str, model: str) -> int:
    if tiktoken is None:
        return len(text)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:  # pragma: no cover
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _chunk_fallback(
    text: str,
    *,
    max_tokens: int,
    overlap: int,
    model: str,
) -> List[str]:
    """
    Rudimentary token-based chunker used as a last resort.
    """
    if not text:
        return []

    if tiktoken is None:
        LOGGER.warning("Using character window chunking fallback (tiktoken not installed)")
        window = max_tokens * 4
        step = max(window - overlap * 4, 1)
        return [text[i : i + window] for i in range(0, len(text), step)]

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks: List[str] = []
    step = max(max_tokens - overlap, 1)
    for start in range(0, len(tokens), step):
        token_slice = tokens[start : start + max_tokens]
        chunk_text = encoding.decode(token_slice)
        chunks.append(chunk_text)
    return chunks


def _chunk_parent_child(
    text: str,
    *,
    max_tokens: int,
    overlap: int,
    model: str,
) -> List[str]:
    """
    Parent-child chunking strategy: first split by headings, then chunk each section.
    """
    # Split document into sections based on headings
    sections = _split_by_headings(text)
    
    if len(sections) <= 1:
        # No clear structure, fall back to regular chunking
        LOGGER.info("No clear document structure found, using regular chunking")
        return _chunk_fallback(text, max_tokens=max_tokens, overlap=overlap, model=model)
    
    LOGGER.info("Document split into %d sections for parent-child chunking", len(sections))
    
    chunks = []
    for i, (heading, content) in enumerate(sections):
        if not content.strip():
            continue
            
        # Chunk the content of this section using fallback
        section_chunks = _chunk_fallback(content, max_tokens=max_tokens, overlap=overlap, model=model)
        
        # Add heading context to each chunk
        for chunk_text in section_chunks:
            if heading:
                # Prepend heading as context
                contextual_chunk = f"{heading}\n\n{chunk_text}"
            else:
                contextual_chunk = chunk_text
            
            chunks.append(contextual_chunk)
    
    return chunks


def _split_by_headings(text: str) -> List[Tuple[str, str]]:
    """
    Split text into sections based on heading patterns.
    
    Returns:
        List of (heading, content) tuples
    """
    sections = []
    
    # Define heading patterns
    heading_patterns = [
        r'^(#{1,6})\s+(.+)$',  # Markdown headers
        r'^(\d+(?:\.\d+)*)\.\s+(.+)$',  # Numbered sections
        r'^([A-Z][A-Z\s]+)$',  # ALL CAPS lines
        r'^([IVX]+\.\s+.+)$',  # Roman numerals
    ]
    
    lines = text.split('\n')
    current_heading = ""
    current_content = []
    
    for line in lines:
        is_heading = False
        
        for pattern in heading_patterns:
            match = re.match(pattern, line.strip())
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_heading, '\n'.join(current_content)))
                
                # Start new section
                current_heading = line.strip()
                current_content = []
                is_heading = True
                break
        
        if not is_heading:
            current_content.append(line)
    
    # Add final section
    if current_content:
        sections.append((current_heading, '\n'.join(current_content)))
    
    return sections


def chunk_document_adaptive(
    document: Document,
    *,
    strategy_config: Dict[str, any],
    additional_metadata: Optional[dict] = None,
) -> List[Chunk]:
    """
    Chunk document using adaptive strategy selection.
    
    Args:
        document: Document to chunk
        strategy_config: Strategy configuration from select_chunking_strategy()
        additional_metadata: Additional metadata to add to chunks
        
    Returns:
        List of chunks
    """
    if not document.text:
        return []
    
    strategy = strategy_config["strategy"]
    max_tokens = strategy_config["max_tokens"]
    overlap = strategy_config["overlap"]
    model = chunking_config.model  # Use default model
    
    LOGGER.info(
        "Using adaptive chunking: strategy=%s, max_tokens=%d, overlap=%d",
        strategy, max_tokens, overlap
    )
    
    try:
        if strategy == "parent_child":
            chunk_texts = _chunk_parent_child(
                document.text,
                max_tokens=max_tokens,
                overlap=overlap,
                model=model,
            )
        else:
            # Use fallback chunker (LangChain is primary in chunk_document)
            chunk_texts = _chunk_fallback(
                document.text,
                max_tokens=max_tokens,
                overlap=overlap,
                model=model,
            )
    except Exception as exc:  # pragma: no cover - graceful degradation
        LOGGER.warning("Fallback chunker failed: %s", exc)
        chunk_texts = _chunk_fallback(
            document.text,
            max_tokens=max_tokens,
            overlap=overlap,
            model=model,
        )
    
    # Add strategy information to metadata
    if additional_metadata is None:
        additional_metadata = {}
    
    additional_metadata.update({
        "chunking_strategy": strategy,
        "chunking_reason": strategy_config.get("reason", "unknown"),
        "max_tokens": max_tokens,
        "overlap": overlap,
    })
    
    return make_chunks(document, chunk_texts, additional_metadata=additional_metadata)


def chunk_document(
    document: Document,
    *,
    strategy: str = DEFAULT_STRATEGY,
    max_tokens: int = chunking_config.max_tokens,
    overlap: int = chunking_config.overlap,
    model: str = chunking_config.model,
    additional_metadata: Optional[dict] = None,
    clean_pdf: bool = True,
) -> List[Chunk]:
    """
    Chunk document text and return typed chunk objects preserving metadata.

    Uses LangChain text splitters (v3.0) for clean, reliable chunking.

    Args:
        document: Document to chunk
        strategy: Chunking strategy (recursive, token, semantic [legacy])
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        model: Model name for tokenization (unused in v3.0)
        additional_metadata: Additional metadata to add to chunks
        clean_pdf: If True, clean PDF extraction artifacts before chunking

    Returns:
        List of Chunk objects with rich metadata
    """
    if not document.text:
        return []

    # Use AtlasChunker (LangChain) - v3.0 default
    LOGGER.info(f"Using AtlasChunker (LangChain) with strategy='{strategy}'")

    # Map strategy names for compatibility
    if strategy in ("semantic", "sentence"):
        # Legacy strategies → use recursive (best for LangChain)
        actual_strategy = "recursive"
        LOGGER.info(f"Mapping legacy strategy '{strategy}' → 'recursive'")
    elif strategy in ("token", "late", "parent_child"):
        actual_strategy = "token"
    else:
        actual_strategy = "recursive"  # Default

    # Preserve original strategy in metadata (what user requested)
    metadata_with_original_strategy = additional_metadata.copy() if additional_metadata else {}
    metadata_with_original_strategy["requested_strategy"] = strategy

    try:
        # Use LangChain chunker directly
        chunks = chunk_document_langchain(
            document,
            strategy=actual_strategy,
            max_tokens=max_tokens,
            overlap=overlap,
            additional_metadata=metadata_with_original_strategy,
            preprocess=clean_pdf,  # Built-in preprocessing
        )

        LOGGER.info(f"Created {len(chunks)} chunks using LangChain")
        return chunks

    except Exception as exc:  # pragma: no cover - fallback
        LOGGER.warning(f"LangChain chunking failed: {exc}")

        # Fallback: Use simple token-based chunker
        LOGGER.warning("Using token-based fallback chunker")
        text_to_chunk = document.text
        if clean_pdf:
            text_to_chunk = preprocess_before_chunking(document.text, source_type="pdf")

        chunk_texts = _chunk_fallback(
            text_to_chunk,
            max_tokens=max_tokens,
            overlap=overlap,
            model=model,
        )
        return make_chunks(document, chunk_texts, additional_metadata=additional_metadata)
