"""
JSON Export/Import for document chunks.

Allows saving and loading chunks to/from JSON files instead of database.
Useful for:
- Backup and archiving
- Sharing processed datasets
- Version control of data
- Development without database
- Data inspection and debugging
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib

LOGGER = logging.getLogger(__name__)


class ChunksJSONExporter:
    """
    Export and import document chunks to/from JSON files.

    Supports multiple export formats:
    - Single file: All chunks in one JSON file
    - Per document: One file per document
    - JSONL: One chunk per line (streaming friendly)

    Example:
        >>> from src.workflows.io.json_exporter import ChunksJSONExporter
        >>>
        >>> exporter = ChunksJSONExporter(output_dir="./chunks_output")
        >>>
        >>> # Export chunks
        >>> exporter.export_chunks(
        ...     chunks=chunks,
        ...     document_info=doc_info,
        ...     format='single_file',
        ... )
        >>>
        >>> # Import chunks
        >>> chunks = exporter.import_chunks("./chunks_output/document.json")
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize exporter.

        Args:
            output_dir: Base directory for output files (default: ./output)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Initialized ChunksJSONExporter: {self.output_dir}")

    def export_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_info: Optional[Dict[str, Any]] = None,
        format: str = 'single_file',
        filename: Optional[str] = None,
    ) -> Union[Path, List[Path]]:
        """
        Export chunks to JSON.

        Args:
            chunks: List of chunk dictionaries
            document_info: Optional document metadata
            format: Export format ('single_file', 'per_document', 'jsonl')
            filename: Custom filename (auto-generated if None)

        Returns:
            Path to exported file(s)

        Example:
            >>> chunks = [
            ...     {'text': 'Chunk 1', 'chunk_index': 0, 'metadata': {}},
            ...     {'text': 'Chunk 2', 'chunk_index': 1, 'metadata': {}},
            ... ]
            >>> exporter.export_chunks(chunks, format='single_file')
        """
        if not chunks:
            LOGGER.warning("No chunks to export")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_name = self._get_document_name(document_info)
            filename = f"{doc_name}_{timestamp}.json"

        if format == 'single_file':
            return self._export_single_file(chunks, document_info, filename)
        elif format == 'per_document':
            return self._export_per_document(chunks, document_info, filename)
        elif format == 'jsonl':
            return self._export_jsonl(chunks, document_info, filename)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_single_file(
        self,
        chunks: List[Dict[str, Any]],
        document_info: Optional[Dict[str, Any]],
        filename: str,
    ) -> Path:
        """Export all chunks to a single JSON file."""
        output_path = self.output_dir / filename

        export_data = {
            'format_version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'document_info': document_info or {},
            'num_chunks': len(chunks),
            'chunks': chunks,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Exported {len(chunks)} chunks to {output_path}")
        return output_path

    def _export_per_document(
        self,
        chunks: List[Dict[str, Any]],
        document_info: Optional[Dict[str, Any]],
        base_filename: str,
    ) -> List[Path]:
        """Export chunks grouped by document to separate files."""
        # Group chunks by document_id if present
        chunks_by_doc: Dict[str, List[Dict]] = {}

        for chunk in chunks:
            doc_id = chunk.get('document_id', 'unknown')
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)

        # Export each document's chunks
        output_paths = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            filename = f"{Path(base_filename).stem}_{doc_id}.json"
            path = self._export_single_file(doc_chunks, document_info, filename)
            output_paths.append(path)

        LOGGER.info(f"Exported {len(chunks)} chunks to {len(output_paths)} files")
        return output_paths

    def _export_jsonl(
        self,
        chunks: List[Dict[str, Any]],
        document_info: Optional[Dict[str, Any]],
        filename: str,
    ) -> Path:
        """Export chunks as JSONL (one chunk per line)."""
        output_path = self.output_dir / filename.replace('.json', '.jsonl')

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header line with metadata
            header = {
                '_type': 'header',
                'format_version': '1.0',
                'exported_at': datetime.now().isoformat(),
                'document_info': document_info or {},
                'num_chunks': len(chunks),
            }
            f.write(json.dumps(header, ensure_ascii=False) + '\n')

            # Write each chunk as a line
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        LOGGER.info(f"Exported {len(chunks)} chunks to {output_path} (JSONL)")
        return output_path

    def import_chunks(
        self,
        input_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Import chunks from JSON file.

        Args:
            input_path: Path to JSON file
            format: Format ('single_file', 'jsonl'), auto-detected if None

        Returns:
            Dictionary with 'chunks' and 'document_info'

        Example:
            >>> data = exporter.import_chunks("output/document_20250128.json")
            >>> chunks = data['chunks']
            >>> doc_info = data['document_info']
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        # Auto-detect format
        if format is None:
            format = 'jsonl' if input_path.suffix == '.jsonl' else 'single_file'

        if format == 'single_file':
            return self._import_single_file(input_path)
        elif format == 'jsonl':
            return self._import_jsonl(input_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _import_single_file(self, input_path: Path) -> Dict[str, Any]:
        """Import from single JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = data.get('chunks', [])
        document_info = data.get('document_info', {})

        LOGGER.info(f"Imported {len(chunks)} chunks from {input_path}")

        return {
            'chunks': chunks,
            'document_info': document_info,
            'metadata': {
                'format_version': data.get('format_version'),
                'exported_at': data.get('exported_at'),
                'num_chunks': data.get('num_chunks'),
            }
        }

    def _import_jsonl(self, input_path: Path) -> Dict[str, Any]:
        """Import from JSONL file."""
        chunks = []
        document_info = {}
        metadata = {}

        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                data = json.loads(line)

                # First line is header
                if i == 0 and data.get('_type') == 'header':
                    document_info = data.get('document_info', {})
                    metadata = {
                        'format_version': data.get('format_version'),
                        'exported_at': data.get('exported_at'),
                        'num_chunks': data.get('num_chunks'),
                    }
                else:
                    chunks.append(data)

        LOGGER.info(f"Imported {len(chunks)} chunks from {input_path} (JSONL)")

        return {
            'chunks': chunks,
            'document_info': document_info,
            'metadata': metadata,
        }

    def export_atlas_document(
        self,
        atlas_doc,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export an AtlasDocument with all its chunks.

        Args:
            atlas_doc: AtlasDocument instance
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{atlas_doc.source_name}_{timestamp}.json"
            output_path = self.output_dir / filename

        # Convert to dict
        export_data = {
            'format_version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'document_info': {
                'source_name': atlas_doc.source_name,
                'source_path': str(atlas_doc.source_path),
                'content': atlas_doc.content,
                'total_chunks': len(atlas_doc.chunks),
                'metadata': atlas_doc.metadata,
            },
            'chunks': [
                {
                    'chunk_id': chunk.chunk_id,
                    'chunk_index': chunk.chunk_index,
                    'text': chunk.text,
                    'char_start': chunk.char_start,
                    'char_end': chunk.char_end,
                    'token_count': chunk.token_count,
                    'metadata': chunk.metadata,
                }
                for chunk in atlas_doc.chunks
            ],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(
            f"Exported AtlasDocument '{atlas_doc.source_name}' "
            f"with {len(atlas_doc.chunks)} chunks to {output_path}"
        )

        return output_path

    def batch_export(
        self,
        chunks_list: List[List[Dict[str, Any]]],
        document_infos: Optional[List[Dict[str, Any]]] = None,
        format: str = 'per_document',
    ) -> List[Path]:
        """
        Export multiple documents' chunks.

        Args:
            chunks_list: List of chunk lists (one per document)
            document_infos: Optional list of document metadata
            format: Export format

        Returns:
            List of exported file paths
        """
        if document_infos is None:
            document_infos = [{}] * len(chunks_list)

        output_paths = []

        for i, (chunks, doc_info) in enumerate(zip(chunks_list, document_infos)):
            filename = f"batch_export_{i:04d}.json"
            path = self.export_chunks(
                chunks=chunks,
                document_info=doc_info,
                format=format,
                filename=filename,
            )
            if isinstance(path, list):
                output_paths.extend(path)
            else:
                output_paths.append(path)

        LOGGER.info(f"Batch exported {len(chunks_list)} documents to {len(output_paths)} files")

        return output_paths

    @staticmethod
    def _get_document_name(document_info: Optional[Dict[str, Any]]) -> str:
        """Extract document name from info."""
        if not document_info:
            return "document"

        # Try common keys
        for key in ['source_name', 'filename', 'name', 'title']:
            if key in document_info and document_info[key]:
                name = Path(document_info[key]).stem
                # Clean filename
                name = name.replace(' ', '_').replace('/', '_')
                return name

        # Fallback: use hash of document info (not for security)
        info_str = json.dumps(document_info, sort_keys=True)
        hash_str = hashlib.md5(info_str.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"doc_{hash_str}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about exported files."""
        json_files = list(self.output_dir.glob("*.json"))
        jsonl_files = list(self.output_dir.glob("*.jsonl"))

        total_chunks = 0
        for file in json_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_chunks += data.get('num_chunks', 0)
            except Exception as e:
                # Skip corrupted or invalid JSON files in stats
                LOGGER.debug(f"Skipping invalid JSON file {file.name}: {e}")
                pass

        return {
            'output_dir': str(self.output_dir),
            'json_files': len(json_files),
            'jsonl_files': len(jsonl_files),
            'total_files': len(json_files) + len(jsonl_files),
            'total_chunks_exported': total_chunks,
        }


def create_json_exporter(output_dir: Optional[str] = None) -> ChunksJSONExporter:
    """
    Helper function to create JSON exporter.

    Args:
        output_dir: Output directory

    Returns:
        ChunksJSONExporter instance

    Example:
        >>> exporter = create_json_exporter("./my_chunks")
    """
    return ChunksJSONExporter(output_dir=output_dir)


# Quick export functions
def quick_export_chunks(
    chunks: List[Dict[str, Any]],
    output_path: str,
    document_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Quick function to export chunks to JSON.

    Args:
        chunks: List of chunks
        output_path: Output file path
        document_info: Optional document metadata

    Returns:
        Path to exported file

    Example:
        >>> from src.workflows.io.json_exporter import quick_export_chunks
        >>> quick_export_chunks(chunks, "output/my_chunks.json")
    """
    output_path = Path(output_path)
    exporter = ChunksJSONExporter(output_dir=output_path.parent)

    return exporter.export_chunks(
        chunks=chunks,
        document_info=document_info,
        format='single_file',
        filename=output_path.name,
    )


def quick_import_chunks(input_path: str) -> List[Dict[str, Any]]:
    """
    Quick function to import chunks from JSON.

    Args:
        input_path: Input file path

    Returns:
        List of chunks

    Example:
        >>> from src.workflows.io.json_exporter import quick_import_chunks
        >>> chunks = quick_import_chunks("output/my_chunks.json")
    """
    input_path = Path(input_path)
    exporter = ChunksJSONExporter(output_dir=input_path.parent)

    data = exporter.import_chunks(input_path)
    return data['chunks']
