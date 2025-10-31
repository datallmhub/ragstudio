"""
SQLAlchemy ORM models for Atlas-RAG metadata store.

These models correspond to the PostgreSQL schema defined in
scripts/migrations/init_db.sql
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, Text,
    DateTime, ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Document(Base):
    """
    Represents an ingested document with extraction metadata.

    Tracks files that have been ingested into the system, including
    extraction engine used, performance metrics, and processing status.
    """
    __tablename__ = "documents"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # File information
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_extension: Mapped[str] = mapped_column(String(10), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Extraction metadata
    extraction_engine: Mapped[Optional[str]] = mapped_column(String(50))
    extraction_strategy: Mapped[Optional[str]] = mapped_column(String(50))
    extraction_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_languages: Mapped[Optional[str]] = mapped_column(String(100))

    # Content statistics
    total_chars: Mapped[Optional[int]] = mapped_column(Integer)
    total_pages: Mapped[Optional[int]] = mapped_column(Integer)
    total_elements: Mapped[Optional[int]] = mapped_column(Integer)

    # Document analysis
    doc_type: Mapped[Optional[str]] = mapped_column(String(20))
    extractible_ratio: Mapped[Optional[float]] = mapped_column(Float)

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='processing'
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1)
    previous_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('documents.id', ondelete='SET NULL')
    )

    # Additional metadata (JSON)
    # Note: Using 'extra_metadata' as attribute name because 'metadata' is reserved in SQLAlchemy
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    extraction_jobs: Mapped[List["ExtractionJob"]] = relationship(
        "ExtractionJob",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    previous_version: Mapped[Optional["Document"]] = relationship(
        "Document",
        remote_side=[id],
        foreign_keys=[previous_version_id]
    )

    # Table constraints
    __table_args__ = (
        UniqueConstraint('file_hash', 'version', name='unique_file_hash_version'),
        Index('idx_documents_filename', 'filename'),
        Index('idx_documents_file_hash', 'file_hash'),
        Index('idx_documents_status', 'status'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_extraction_engine', 'extraction_engine'),
        Index('idx_documents_doc_type', 'doc_type'),
        Index('idx_documents_metadata', 'metadata', postgresql_using='gin'),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, filename='{self.filename}', "
            f"status='{self.status}', engine='{self.extraction_engine}')>"
        )


class Chunk(Base):
    """
    Represents a text chunk extracted from a document.

    Stores individual chunks with their text content, position in document,
    chunking metadata, and reference to vector store embedding.
    """
    __tablename__ = "chunks"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('documents.id', ondelete='CASCADE'),
        nullable=False
    )

    # Chunk identification
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Position in document
    char_start: Mapped[Optional[int]] = mapped_column(Integer)
    char_end: Mapped[Optional[int]] = mapped_column(Integer)
    char_length: Mapped[Optional[int]] = mapped_column(Integer)

    # Chunking metadata
    chunking_strategy: Mapped[Optional[str]] = mapped_column(String(50))
    chunk_size: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_overlap: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    sentence_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Quality metrics
    extraction_fixes: Mapped[int] = mapped_column(Integer, default=0)
    page_numbers_removed: Mapped[int] = mapped_column(Integer, default=0)

    # Vector store reference
    vector_id: Mapped[Optional[str]] = mapped_column(String(255))
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100))

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    # Additional metadata (JSON)
    # Note: Using 'extra_metadata' as attribute name because 'metadata' is reserved in SQLAlchemy
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks"
    )

    # Table constraints
    __table_args__ = (
        UniqueConstraint('document_id', 'chunk_index', name='unique_document_chunk_index'),
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_chunk_index', 'chunk_index'),
        Index('idx_chunks_vector_id', 'vector_id'),
        Index('idx_chunks_embedding_model', 'embedding_model'),
        Index('idx_chunks_created_at', 'created_at'),
        Index('idx_chunks_metadata', 'metadata', postgresql_using='gin'),
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk(id={self.id}, document_id={self.document_id}, "
            f"index={self.chunk_index}, length={self.char_length})>"
        )


class ExtractionJob(Base):
    """
    Tracks extraction jobs for monitoring and analytics.

    Records performance metrics, resource usage, and errors for each
    extraction/chunking job.
    """
    __tablename__ = "extraction_jobs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('documents.id', ondelete='CASCADE'),
        nullable=False
    )

    # Job information
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default='pending')

    # Performance metrics
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    # Resource usage
    cpu_usage_percent: Mapped[Optional[float]] = mapped_column(Float)
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float)

    # Results
    chunks_created: Mapped[Optional[int]] = mapped_column(Integer)
    vectors_created: Mapped[Optional[int]] = mapped_column(Integer)

    # Errors
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Additional metadata
    # Note: Using 'extra_metadata' as attribute name because 'metadata' is reserved in SQLAlchemy
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="extraction_jobs"
    )

    # Table constraints
    __table_args__ = (
        Index('idx_extraction_jobs_document_id', 'document_id'),
        Index('idx_extraction_jobs_status', 'status'),
        Index('idx_extraction_jobs_job_type', 'job_type'),
        Index('idx_extraction_jobs_created_at', 'created_at'),
    )

    def __repr__(self) -> str:
        return (
            f"<ExtractionJob(id={self.id}, document_id={self.document_id}, "
            f"type='{self.job_type}', status='{self.status}')>"
        )


class AuditLog(Base):
    """
    Audit trail for compliance and debugging.

    Records all actions performed on resources (documents, chunks, jobs)
    for traceability and compliance.
    """
    __tablename__ = "audit_logs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Resource information
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Action information
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Changes (before/after)
    changes: Mapped[Optional[dict]] = mapped_column(JSON)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    # Additional context
    # Note: Using 'extra_metadata' as attribute name because 'metadata' is reserved in SQLAlchemy
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    # Table constraints
    __table_args__ = (
        Index('idx_audit_logs_resource_type', 'resource_type'),
        Index('idx_audit_logs_resource_id', 'resource_id'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_created_at', 'created_at'),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, resource_type='{self.resource_type}', "
            f"action='{self.action}', created_at={self.created_at})>"
        )
