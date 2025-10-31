"""
Complete ingestion pipeline with metadata store tracking.

This module provides a high-level pipeline that tracks all operations
in the PostgreSQL metadata store for full traceability.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.workflows.ingest.langchain_loader import load_document_langchain
from src.core.chunk.langchain_chunker import chunk_document_langchain
from src.core.storage.metadata_store import MetadataStore, get_metadata_store
from src.workflows.io.schema import Document as AtlasDocument
from src.workflows.io.schema import Chunk as AtlasChunk

LOGGER = logging.getLogger(__name__)


class IngestionPipelineWithMetadata:
    """
    Complete ingestion pipeline with metadata store tracking.

    Workflow:
    1. Create document entry in metadata store (status=processing)
    2. Extract document with intelligent orchestrator
    3. Chunk document with LangChain
    4. Store chunks in metadata store
    5. Update document status (done/failed)
    6. Track job metrics
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        ocr_languages: Optional[List[str]] = None,
        chunking_strategy: str = "recursive",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        """
        Initialize pipeline.

        Args:
            metadata_store: MetadataStore instance
            ocr_languages: Languages for OCR (default: ['en', 'fr'])
            chunking_strategy: Chunking strategy (default: 'recursive')
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.metadata_store = metadata_store
        self.ocr_languages = ocr_languages or ['en', 'fr']
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        LOGGER.info("Initialized IngestionPipelineWithMetadata")

    def ingest_document(
        self,
        file_path: Path,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest document with full metadata tracking.

        Args:
            file_path: Path to document
            user_id: User performing ingestion (for audit)

        Returns:
            Dictionary with results:
            {
                'document_id': UUID,
                'chunks_created': int,
                'status': 'done' | 'failed',
                'extraction_time': float,
                'chunking_time': float,
                'total_time': float,
                'error': str (if failed)
            }
        """
        pipeline_start = time.time()

        # Step 1: Create document entry (status=processing)
        try:
            doc_entry = self.metadata_store.create_document(
                filename=file_path.name,
                source_path=file_path,
                file_size_bytes=file_path.stat().st_size,
                file_extension=file_path.suffix.lower(),
                status='processing',
            )
            document_id = str(doc_entry.id)

            LOGGER.info(f"Created document entry {document_id} for {file_path.name}")

            # Create extraction job
            job = self.metadata_store.create_job(
                document_id=document_id,
                job_type='full_pipeline',
                status='pending',
            )
            self.metadata_store.start_job(str(job.id))

        except Exception as e:
            LOGGER.error(f"Failed to create document entry: {e}")
            return {
                'status': 'failed',
                'error': f"Database error: {str(e)}"
            }

        try:
            # Step 2: Extract document
            extraction_start = time.time()

            atlas_doc = load_document_langchain(
                file_path,
                ocr_languages=self.ocr_languages,
            )

            extraction_time = time.time() - extraction_start

            LOGGER.info(
                f"Extracted {len(atlas_doc.text)} chars in {extraction_time:.2f}s "
                f"using {atlas_doc.metadata.get('extraction_strategy', 'unknown')}"
            )

            # Update document with extraction metadata
            self.metadata_store.update_document(
                document_id,
                extraction_engine=atlas_doc.metadata.get('engine', 'unknown'),
                extraction_strategy=atlas_doc.metadata.get('extraction_strategy', 'unknown'),
                extraction_time_seconds=extraction_time,
                ocr_used=atlas_doc.metadata.get('ocr_used', False),
                ocr_languages="+".join(self.ocr_languages),
                total_chars=len(atlas_doc.text),
                total_pages=atlas_doc.metadata.get('total_pages'),
                doc_type=atlas_doc.metadata.get('document_analysis', {}).get('doc_type'),
                extractible_ratio=atlas_doc.metadata.get('document_analysis', {}).get('extractible_ratio'),
            )

            # Step 3: Chunk document
            chunking_start = time.time()

            atlas_chunks = chunk_document_langchain(
                atlas_doc,
                strategy=self.chunking_strategy,
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap,
                preprocess=True,
            )

            chunking_time = time.time() - chunking_start

            LOGGER.info(f"Created {len(atlas_chunks)} chunks in {chunking_time:.2f}s")

            # Step 4: Store chunks in metadata store
            chunks_data = []

            for chunk in atlas_chunks:
                chunk_data = {
                    'chunk_index': chunk.metadata.get('chunk_index', 0),
                    'text': chunk.text,
                    'char_start': chunk.metadata.get('char_start'),
                    'char_end': chunk.metadata.get('char_end'),
                    'chunking_strategy': chunk.metadata.get('chunking_strategy'),
                    'chunk_size': chunk.metadata.get('chunk_size'),
                    'chunk_overlap': chunk.metadata.get('chunk_overlap'),
                    'token_count': chunk.metadata.get('token_count'),
                    'sentence_count': chunk.metadata.get('sentence_count'),
                    'extraction_fixes': chunk.metadata.get('extraction_fixes', 0),
                    'page_numbers_removed': chunk.metadata.get('page_numbers_removed', 0),
                    'metadata': chunk.metadata,  # Store full metadata as JSON
                }
                chunks_data.append(chunk_data)

            db_chunks = self.metadata_store.create_chunks_bulk(
                document_id,
                chunks_data
            )

            LOGGER.info(f"Stored {len(db_chunks)} chunks in metadata store")

            # Step 5: Update document status to done
            total_time = time.time() - pipeline_start

            self.metadata_store.update_document(
                document_id,
                status='done',
                completed_at=time.time(),
            )

            # Complete job
            self.metadata_store.complete_job(
                str(job.id),
                chunks_created=len(db_chunks),
                duration_seconds=total_time,
            )

            # Create audit log
            self.metadata_store.create_audit_log(
                resource_type='document',
                resource_id=document_id,
                action='ingested',
                user_id=user_id or 'system',
                changes={
                    'chunks_created': len(db_chunks),
                    'extraction_engine': atlas_doc.metadata.get('engine'),
                }
            )

            LOGGER.info(
                f"✅ Successfully ingested {file_path.name}: "
                f"{len(db_chunks)} chunks in {total_time:.2f}s"
            )

            return {
                'status': 'done',
                'document_id': document_id,
                'filename': file_path.name,
                'chunks_created': len(db_chunks),
                'extraction_time': round(extraction_time, 2),
                'chunking_time': round(chunking_time, 2),
                'total_time': round(total_time, 2),
                'extraction_engine': atlas_doc.metadata.get('engine'),
                'extraction_strategy': atlas_doc.metadata.get('extraction_strategy'),
                'total_chars': len(atlas_doc.text),
            }

        except Exception as e:
            # Mark as failed
            LOGGER.error(f"Pipeline failed for {file_path.name}: {e}", exc_info=True)

            try:
                self.metadata_store.update_document(
                    document_id,
                    status='failed',
                    error_message=str(e),
                )

                self.metadata_store.fail_job(
                    str(job.id),
                    error_message=str(e),
                )

                self.metadata_store.create_audit_log(
                    resource_type='document',
                    resource_id=document_id,
                    action='failed',
                    user_id=user_id or 'system',
                    changes={'error': str(e)}
                )
            except Exception as db_error:
                LOGGER.error(f"Failed to update database after error: {db_error}")

            return {
                'status': 'failed',
                'document_id': document_id,
                'filename': file_path.name,
                'error': str(e),
                'total_time': round(time.time() - pipeline_start, 2),
            }

    def ingest_batch(
        self,
        file_paths: List[Path],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of file paths
            user_id: User performing ingestion

        Returns:
            List of result dictionaries (one per document)
        """
        results = []

        for file_path in file_paths:
            LOGGER.info(f"Ingesting {file_path.name}...")
            result = self.ingest_document(file_path, user_id=user_id)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r['status'] == 'done')
        failed = sum(1 for r in results if r['status'] == 'failed')

        LOGGER.info(
            f"Batch ingestion complete: {successful} successful, {failed} failed"
        )

        return results


def create_pipeline(
    database_url: str,
    **kwargs
) -> IngestionPipelineWithMetadata:
    """
    Create ingestion pipeline with metadata store.

    Args:
        database_url: PostgreSQL connection string
        **kwargs: Pipeline configuration

    Returns:
        IngestionPipelineWithMetadata instance
    """
    metadata_store = get_metadata_store(database_url)

    # Create tables if not exists
    metadata_store.create_tables()

    return IngestionPipelineWithMetadata(
        metadata_store=metadata_store,
        **kwargs
    )
