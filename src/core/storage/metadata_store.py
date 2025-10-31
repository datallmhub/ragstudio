"""
Metadata Store API for Atlas-RAG.

Provides high-level interface for interacting with SQL metadata store (SQLite).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

from sqlalchemy import create_engine, select, func, and_, or_
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError

from src.core.storage.models import (
    Base,
    Document,
    Chunk,
    ExtractionJob,
    AuditLog,
)

LOGGER = logging.getLogger(__name__)


class MetadataStore:
    """
    High-level API for metadata store operations.

    Manages documents, chunks, jobs, and audit logs in SQLite database.
    """

    def __init__(self, database_url: str):
        """
        Initialize metadata store.

        Args:
            database_url: SQLite connection string
                Example: 'sqlite:///atlas_rag_metadata.db' or 'sqlite:///:memory:' for in-memory
        """
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before using
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        LOGGER.info(f"Initialized MetadataStore with database: {database_url}")

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
        LOGGER.info("Created all database tables")

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        LOGGER.warning("Dropped all database tables")

    @contextmanager
    def get_session(self):
        """Get database session (context manager)."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # =========================================================================
    # Document Operations
    # =========================================================================

    def create_document(
        self,
        filename: str,
        source_path: Path,
        file_size_bytes: int,
        file_extension: str,
        **kwargs
    ) -> Document:
        """
        Create new document entry.

        Args:
            filename: Document filename
            source_path: Full path to source file
            file_size_bytes: File size in bytes
            file_extension: File extension (e.g., '.pdf')
            **kwargs: Additional fields (extraction_engine, ocr_used, etc.)

        Returns:
            Created Document instance
        """
        with self.get_session() as session:
            # Calculate file hash
            file_hash = self._calculate_file_hash(source_path)

            # Check if document already exists
            existing = self.get_document_by_hash(file_hash)
            if existing:
                LOGGER.warning(f"Document with hash {file_hash} already exists: {existing.filename}")
                # Create new version
                version = existing.version + 1
                kwargs['version'] = version
                kwargs['previous_version_id'] = existing.id

            doc = Document(
                filename=filename,
                source_path=str(source_path),
                file_size_bytes=file_size_bytes,
                file_extension=file_extension,
                file_hash=file_hash,
                **kwargs
            )

            session.add(doc)
            session.flush()  # Get ID
            session.refresh(doc)  # Load all attributes before detaching

            LOGGER.info(f"Created document: {doc.filename} (id={doc.id})")
            session.expunge(doc)  # Detach from session to avoid DetachedInstanceError
            return doc

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        with self.get_session() as session:
            stmt = select(Document).where(Document.id == document_id)
            doc = session.scalar(stmt)
            if doc:
                session.expunge(doc)
            return doc

    def get_document_by_filename(self, filename: str) -> Optional[Document]:
        """Get document by filename (latest version)."""
        with self.get_session() as session:
            stmt = select(Document).where(
                Document.filename == filename
            ).order_by(Document.version.desc())
            doc = session.scalar(stmt)
            if doc:
                session.expunge(doc)
            return doc

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get document by file hash (latest version)."""
        with self.get_session() as session:
            stmt = select(Document).where(
                Document.file_hash == file_hash
            ).order_by(Document.version.desc())
            doc = session.scalar(stmt)
            if doc:
                session.expunge(doc)
            return doc

    def update_document(
        self,
        document_id: str,
        **kwargs
    ) -> Document:
        """
        Update document fields.

        Args:
            document_id: Document ID
            **kwargs: Fields to update (status, extraction_engine, etc.)

        Returns:
            Updated Document instance
        """
        with self.get_session() as session:
            doc = session.get(Document, document_id)
            if not doc:
                raise ValueError(f"Document {document_id} not found")

            for key, value in kwargs.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)

            session.flush()
            session.refresh(doc)
            session.expunge(doc)
            LOGGER.info(f"Updated document {document_id}: {kwargs}")
            return doc

    def delete_document(self, document_id: str):
        """Delete document and all related chunks (CASCADE)."""
        with self.get_session() as session:
            doc = session.get(Document, document_id)
            if doc:
                session.delete(doc)
                LOGGER.info(f"Deleted document {document_id}")
            else:
                LOGGER.warning(f"Document {document_id} not found")

    def list_documents(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Document], int]:
        """
        List documents with optional filtering.

        Args:
            status: Filter by status ('processing', 'done', 'failed')
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            Tuple of (List of Document instances, total count)
        """
        with self.get_session() as session:
            # Count query (without limit/offset)
            count_stmt = select(func.count(Document.id))
            if status:
                count_stmt = count_stmt.where(Document.status == status)
            total = session.scalar(count_stmt) or 0

            # Data query (with limit/offset)
            stmt = select(Document)
            if status:
                stmt = stmt.where(Document.status == status)
            stmt = stmt.order_by(Document.created_at.desc())
            stmt = stmt.limit(limit).offset(offset)

            docs = list(session.scalars(stmt))
            for doc in docs:
                session.expunge(doc)
            return docs, total

    # =========================================================================
    # Chunk Operations
    # =========================================================================

    def create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text: str,
        **kwargs
    ) -> Chunk:
        """
        Create new chunk.

        Args:
            document_id: Parent document ID
            chunk_index: Index of chunk in document
            text: Chunk text content
            **kwargs: Additional fields (char_start, token_count, etc.)

        Returns:
            Created Chunk instance
        """
        with self.get_session() as session:
            # Calculate chunk hash
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()

            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                chunk_hash=chunk_hash,
                char_length=len(text),
                **kwargs
            )

            session.add(chunk)
            session.flush()
            session.refresh(chunk)
            session.expunge(chunk)

            LOGGER.debug(f"Created chunk {chunk_index} for document {document_id}")
            return chunk

    def create_chunks_bulk(
        self,
        document_id: str,
        chunks_data: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Create multiple chunks in bulk.

        Args:
            document_id: Parent document ID
            chunks_data: List of chunk dictionaries

        Returns:
            List of created Chunk instances
        """
        with self.get_session() as session:
            chunks = []

            for data in chunks_data:
                text = data['text']
                chunk_hash = hashlib.sha256(text.encode()).hexdigest()

                chunk = Chunk(
                    document_id=document_id,
                    chunk_hash=chunk_hash,
                    char_length=len(text),
                    **data
                )
                chunks.append(chunk)

            session.add_all(chunks)
            session.flush()
            for chunk in chunks:
                session.refresh(chunk)
                session.expunge(chunk)

            LOGGER.info(f"Created {len(chunks)} chunks for document {document_id}")
            return chunks

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        with self.get_session() as session:
            chunk = session.get(Chunk, chunk_id)
            if chunk:
                session.expunge(chunk)
            return chunk

    def get_chunks_by_document(
        self,
        document_id: str,
        limit: Optional[int] = None
    ) -> List[Chunk]:
        """Get all chunks for a document."""
        with self.get_session() as session:
            stmt = select(Chunk).where(
                Chunk.document_id == document_id
            ).order_by(Chunk.chunk_index)

            if limit:
                stmt = stmt.limit(limit)

            chunks = list(session.scalars(stmt))
            for chunk in chunks:
                session.expunge(chunk)
            return chunks

    def update_chunk_vector_id(
        self,
        chunk_id: str,
        vector_id: str,
        embedding_model: Optional[str] = None
    ) -> Optional[Chunk]:
        """Update chunk with vector store reference."""
        with self.get_session() as session:
            chunk = session.get(Chunk, chunk_id)
            if chunk:
                chunk.vector_id = vector_id
                if embedding_model:
                    chunk.embedding_model = embedding_model
                session.flush()
                session.refresh(chunk)
                LOGGER.debug(f"Updated chunk {chunk_id} with vector_id {vector_id}")
                session.expunge(chunk)
                return chunk
            return None

    # =========================================================================
    # Extraction Job Operations
    # =========================================================================

    def create_job(
        self,
        document_id: str,
        job_type: str,
        **kwargs
    ) -> ExtractionJob:
        """
        Create new extraction job.

        Args:
            document_id: Document being processed
            job_type: Type of job ('extraction', 'chunking', 'embedding', 'full_pipeline')
            **kwargs: Additional fields

        Returns:
            Created ExtractionJob instance
        """
        with self.get_session() as session:
            job = ExtractionJob(
                document_id=document_id,
                job_type=job_type,
                **kwargs
            )

            session.add(job)
            session.flush()
            session.refresh(job)
            session.expunge(job)

            LOGGER.info(f"Created job {job.id} (type={job_type}) for document {document_id}")
            return job

    def start_job(self, job_id: str):
        """Mark job as started."""
        with self.get_session() as session:
            job = session.get(ExtractionJob, job_id)
            if job:
                job.status = 'running'
                job.started_at = datetime.utcnow()
                session.flush()

    def complete_job(
        self,
        job_id: str,
        **kwargs
    ):
        """
        Mark job as completed with results.

        Args:
            job_id: Job ID
            **kwargs: Results (chunks_created, vectors_created, etc.)
        """
        with self.get_session() as session:
            job = session.get(ExtractionJob, job_id)
            if job:
                job.status = 'completed'
                job.completed_at = datetime.utcnow()

                if job.started_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    job.duration_seconds = duration

                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)

                session.flush()
                LOGGER.info(f"Completed job {job_id}")

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        error_traceback: Optional[str] = None
    ):
        """Mark job as failed with error details."""
        with self.get_session() as session:
            job = session.get(ExtractionJob, job_id)
            if job:
                job.status = 'failed'
                job.completed_at = datetime.utcnow()
                job.error_message = error_message
                job.error_traceback = error_traceback

                if job.started_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    job.duration_seconds = duration

                session.flush()
                LOGGER.error(f"Failed job {job_id}: {error_message}")

    # =========================================================================
    # Audit Log Operations
    # =========================================================================

    def create_audit_log(
        self,
        resource_type: str,
        resource_id: str,
        action: str,
        user_id: Optional[str] = None,
        changes: Optional[Dict] = None,
        **kwargs
    ) -> AuditLog:
        """Create audit log entry."""
        with self.get_session() as session:
            log = AuditLog(
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                user_id=user_id or 'system',
                changes=changes,
                **kwargs
            )

            session.add(log)
            session.flush()
            session.refresh(log)
            session.expunge(log)

            return log

    # =========================================================================
    # Analytics & Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        with self.get_session() as session:
            total_docs = session.scalar(select(func.count(Document.id))) or 0
            done_docs = session.scalar(
                select(func.count(Document.id)).where(Document.status == 'done')
            ) or 0
            failed_docs = session.scalar(
                select(func.count(Document.id)).where(Document.status == 'failed')
            ) or 0
            total_chunks = session.scalar(select(func.count(Chunk.id))) or 0
            total_jobs = session.scalar(select(func.count(ExtractionJob.id))) or 0
            total_audit_logs = session.scalar(select(func.count(AuditLog.id))) or 0

            success_rate = (done_docs / total_docs * 100) if total_docs > 0 else 0

            return {
                'total_documents': total_docs,
                'successful_documents': done_docs,
                'failed_documents': failed_docs,
                'total_chunks': total_chunks,
                'total_jobs': total_jobs,
                'total_audit_logs': total_audit_logs,
                'success_rate': round(success_rate, 2),
            }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


# Singleton instance
_metadata_store_instance: Optional[MetadataStore] = None


def get_metadata_store(database_url: Optional[str] = None) -> MetadataStore:
    """
    Get or create metadata store singleton instance.

    Args:
        database_url: SQLite connection string (required on first call)

    Returns:
        MetadataStore instance
    """
    global _metadata_store_instance

    if _metadata_store_instance is None:
        if database_url is None:
            raise ValueError("database_url required for first initialization")
        _metadata_store_instance = MetadataStore(database_url)

    return _metadata_store_instance
