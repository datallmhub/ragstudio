"""
Tests for MetadataStore.

Tests the high-level API for metadata store operations.
"""

import pytest
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

from src.core.storage.metadata_store import MetadataStore
from src.core.storage.models import Base, Document, Chunk, ExtractionJob, AuditLog


@pytest.fixture
def temp_db():
    """Create temporary SQLite database for testing."""
    # Use in-memory SQLite for fast tests
    db_url = "sqlite:///:memory:"
    yield db_url


@pytest.fixture
def metadata_store(temp_db):
    """Create MetadataStore instance with temp database."""
    store = MetadataStore(temp_db)
    store.create_tables()
    yield store


@pytest.fixture
def sample_document_data(tmp_path):
    """Sample document data for testing."""
    # Create a real temporary file for hash calculation
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("This is a test document for metadata store testing.")

    return {
        "filename": "test_doc.txt",
        "source_path": str(test_file),  # Convert Path to string for SQLAlchemy
        "file_size_bytes": test_file.stat().st_size,
        "file_extension": ".txt",
    }


@pytest.fixture
def sample_chunk_data():
    """Sample chunk data for testing."""
    return {
        "text": "This is a test chunk",
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 20,
        "token_count": 5,
    }


class TestMetadataStoreInit:
    """Test MetadataStore initialization."""

    def test_init_creates_engine(self, temp_db):
        """Test that init creates SQLAlchemy engine."""
        store = MetadataStore(temp_db)

        assert store.engine is not None
        assert store.SessionLocal is not None


class TestMetadataStoreTableOps:
    """Test table creation and deletion operations."""

    def test_create_tables(self, temp_db):
        """Test creating all tables."""
        store = MetadataStore(temp_db)
        store.create_tables()

        # Verify tables exist by checking metadata
        inspector = create_engine(temp_db).connect()
        tables = [table.name for table in Base.metadata.sorted_tables]
        assert "documents" in tables
        assert "chunks" in tables
        assert "extraction_jobs" in tables
        assert "audit_logs" in tables

    def test_drop_tables(self, metadata_store):
        """Test dropping all tables."""
        # Tables exist after fixture setup
        with metadata_store.get_session() as session:
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            assert len(tables) > 0

        # Drop tables
        metadata_store.drop_tables()

        # Verify tables are gone
        with metadata_store.get_session() as session:
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            # Only sqlite internal tables remain
            assert "documents" not in tables


class TestMetadataStoreSession:
    """Test session management."""

    def test_get_session_context_manager(self, metadata_store):
        """Test session context manager."""
        with metadata_store.get_session() as session:
            assert session is not None
            # Session should be active
            assert session.is_active

    def test_get_session_commits_on_success(self, metadata_store, sample_document_data):
        """Test that session commits on successful exit."""
        with metadata_store.get_session() as session:
            # Add required file_hash field for direct Document creation
            doc_data = {**sample_document_data, "file_hash": "abc123"}
            doc = Document(**doc_data)
            session.add(doc)

        # Verify commit happened by querying in new session
        with metadata_store.get_session() as session:
            docs = session.query(Document).all()
            assert len(docs) == 1

    def test_get_session_rolls_back_on_error(self, metadata_store):
        """Test that session rolls back on error."""
        try:
            with metadata_store.get_session() as session:
                # Create document with missing required field (will fail)
                doc = Document(filename="test")  # Missing required fields
                session.add(doc)
                session.flush()  # Force the error
        except:
            pass

        # Verify rollback happened
        with metadata_store.get_session() as session:
            docs = session.query(Document).all()
            assert len(docs) == 0


class TestMetadataStoreDocuments:
    """Test document CRUD operations."""

    def test_create_document(self, metadata_store, sample_document_data):
        """Test creating a document."""
        doc = metadata_store.create_document(**sample_document_data)

        assert doc.id is not None
        assert doc.filename == "test_doc.txt"
        assert doc.file_size_bytes == sample_document_data["file_size_bytes"]
        assert doc.created_at is not None

    def test_create_document_with_optional_fields(self, metadata_store, sample_document_data):
        """Test creating document with optional metadata."""
        sample_document_data["extra_metadata"] = {"language": "en", "author": "test"}

        doc = metadata_store.create_document(**sample_document_data)

        assert doc.extra_metadata is not None
        assert doc.extra_metadata["language"] == "en"

    def test_get_document(self, metadata_store, sample_document_data):
        """Test getting document by ID."""
        created_doc = metadata_store.create_document(**sample_document_data)

        retrieved_doc = metadata_store.get_document(created_doc.id)

        assert retrieved_doc is not None
        assert retrieved_doc.id == created_doc.id
        assert retrieved_doc.filename == "test_doc.txt"

    def test_get_document_not_found(self, metadata_store):
        """Test getting non-existent document returns None."""
        # Use a valid UUID that doesn't exist in the database
        nonexistent_id = uuid.uuid4()
        doc = metadata_store.get_document(nonexistent_id)

        assert doc is None

    def test_get_document_by_filename(self, metadata_store, sample_document_data):
        """Test getting document by filename."""
        metadata_store.create_document(**sample_document_data)

        doc = metadata_store.get_document_by_filename("test_doc.txt")

        assert doc is not None
        assert doc.filename == "test_doc.txt"

    def test_get_document_by_hash(self, metadata_store, sample_document_data):
        """Test getting document by file hash."""
        created_doc = metadata_store.create_document(**sample_document_data)

        # Use the actual file hash that was created
        doc = metadata_store.get_document_by_hash(created_doc.file_hash)

        assert doc is not None
        assert doc.file_hash == created_doc.file_hash

    def test_update_document(self, metadata_store, sample_document_data):
        """Test updating document fields."""
        doc = metadata_store.create_document(**sample_document_data)

        updated_doc = metadata_store.update_document(
            doc.id,
            file_size_bytes=2048,
            extra_metadata={"updated": True}
        )

        assert updated_doc.file_size_bytes == 2048
        assert updated_doc.extra_metadata["updated"] is True

    def test_delete_document(self, metadata_store, sample_document_data):
        """Test deleting a document."""
        doc = metadata_store.create_document(**sample_document_data)
        doc_id = doc.id

        metadata_store.delete_document(doc_id)

        # Verify document is gone
        deleted_doc = metadata_store.get_document(doc_id)
        assert deleted_doc is None

    def test_list_documents_empty(self, metadata_store):
        """Test listing documents when none exist."""
        docs, total = metadata_store.list_documents()

        assert docs == []
        assert total == 0

    def test_list_documents_with_multiple(self, metadata_store, tmp_path):
        """Test listing multiple documents."""
        # Create 3 documents
        for i in range(3):
            test_file = tmp_path / f"doc{i}.txt"
            test_file.write_text(f"Content {i}")
            data = {
                "filename": f"doc{i}.txt",
                "source_path": test_file,
                "file_size_bytes": test_file.stat().st_size,
                "file_extension": ".txt",
            }
            metadata_store.create_document(**data)

        docs, total = metadata_store.list_documents()

        assert len(docs) == 3
        assert total == 3

    def test_list_documents_with_limit(self, metadata_store, tmp_path):
        """Test listing documents with limit."""
        # Create 5 documents
        for i in range(5):
            test_file = tmp_path / f"doc{i}.txt"
            test_file.write_text(f"Content {i}")
            data = {
                "filename": f"doc{i}.txt",
                "source_path": test_file,
                "file_size_bytes": test_file.stat().st_size,
                "file_extension": ".txt",
            }
            metadata_store.create_document(**data)

        docs, total = metadata_store.list_documents(limit=3)

        assert len(docs) == 3
        assert total == 5  # Total count should still be 5

    def test_list_documents_with_offset(self, metadata_store, tmp_path):
        """Test listing documents with offset."""
        # Create 3 documents
        for i in range(3):
            test_file = tmp_path / f"doc{i}.txt"
            test_file.write_text(f"Content {i}")
            data = {
                "filename": f"doc{i}.txt",
                "source_path": test_file,
                "file_size_bytes": test_file.stat().st_size,
                "file_extension": ".txt",
            }
            metadata_store.create_document(**data)

        docs, total = metadata_store.list_documents(offset=1, limit=10)

        assert len(docs) == 2  # Skip first, get remaining 2
        assert total == 3


class TestMetadataStoreChunks:
    """Test chunk operations."""

    def test_create_chunk(self, metadata_store, sample_document_data, sample_chunk_data):
        """Test creating a chunk."""
        doc = metadata_store.create_document(**sample_document_data)

        chunk = metadata_store.create_chunk(
            document_id=doc.id,
            **sample_chunk_data
        )

        assert chunk.id is not None
        assert chunk.document_id == doc.id
        assert chunk.text == "This is a test chunk"
        assert chunk.chunk_index == 0

    def test_create_chunks_bulk(self, metadata_store, sample_document_data):
        """Test bulk chunk creation."""
        doc = metadata_store.create_document(**sample_document_data)

        chunks_data = [
            {
                "text": f"Chunk {i}",
                "chunk_index": i,
                "char_start": i * 10,
                "char_end": (i + 1) * 10,
                "token_count": 2,
            }
            for i in range(3)
        ]

        chunks = metadata_store.create_chunks_bulk(doc.id, chunks_data)

        assert len(chunks) == 3
        assert all(c.document_id == doc.id for c in chunks)

    def test_get_chunk(self, metadata_store, sample_document_data, sample_chunk_data):
        """Test getting chunk by ID."""
        doc = metadata_store.create_document(**sample_document_data)
        created_chunk = metadata_store.create_chunk(doc.id, **sample_chunk_data)

        retrieved_chunk = metadata_store.get_chunk(created_chunk.id)

        assert retrieved_chunk is not None
        assert retrieved_chunk.id == created_chunk.id

    def test_get_chunks_by_document(self, metadata_store, sample_document_data):
        """Test getting all chunks for a document."""
        doc = metadata_store.create_document(**sample_document_data)

        # Create 3 chunks
        for i in range(3):
            metadata_store.create_chunk(
                doc.id,
                chunk_index=i,
                text=f"Chunk {i}",
                char_start=i * 10,
                char_end=(i + 1) * 10,
                token_count=2,
            )

        chunks = metadata_store.get_chunks_by_document(doc.id)

        assert len(chunks) == 3
        # Should be ordered by chunk_index
        assert chunks[0].chunk_index == 0
        assert chunks[2].chunk_index == 2

    def test_update_chunk_vector_id(self, metadata_store, sample_document_data, sample_chunk_data):
        """Test updating chunk's vector ID."""
        doc = metadata_store.create_document(**sample_document_data)
        chunk = metadata_store.create_chunk(doc.id, **sample_chunk_data)

        updated_chunk = metadata_store.update_chunk_vector_id(chunk.id, "vector_123")

        assert updated_chunk.vector_id == "vector_123"


class TestMetadataStoreJobs:
    """Test extraction job operations."""

    def test_create_job(self, metadata_store, sample_document_data):
        """Test creating extraction job."""
        doc = metadata_store.create_document(**sample_document_data)

        job = metadata_store.create_job(
            document_id=doc.id,
            job_type="chunking",
            extra_metadata={"strategy": "semantic"}
        )

        assert job.id is not None
        assert job.document_id == doc.id
        assert job.status == "pending"
        assert job.job_type == "chunking"

    def test_start_job(self, metadata_store, sample_document_data):
        """Test starting a job."""
        doc = metadata_store.create_document(**sample_document_data)
        job = metadata_store.create_job(doc.id, "chunking")

        metadata_store.start_job(job.id)

        # Verify job status updated
        with metadata_store.get_session() as session:
            updated_job = session.query(ExtractionJob).filter_by(id=job.id).first()
            assert updated_job.status == "running"
            assert updated_job.started_at is not None

    def test_complete_job(self, metadata_store, sample_document_data):
        """Test completing a job."""
        doc = metadata_store.create_document(**sample_document_data)
        job = metadata_store.create_job(doc.id, "chunking")
        metadata_store.start_job(job.id)

        metadata_store.complete_job(
            job.id,
            chunks_created=10,
            result={"success": True}
        )

        # Verify job completed
        with metadata_store.get_session() as session:
            updated_job = session.query(ExtractionJob).filter_by(id=job.id).first()
            assert updated_job.status == "completed"
            assert updated_job.completed_at is not None
            assert updated_job.chunks_created == 10

    def test_fail_job(self, metadata_store, sample_document_data):
        """Test failing a job."""
        doc = metadata_store.create_document(**sample_document_data)
        job = metadata_store.create_job(doc.id, "chunking")
        metadata_store.start_job(job.id)

        metadata_store.fail_job(job.id, "Test error message")

        # Verify job failed
        with metadata_store.get_session() as session:
            updated_job = session.query(ExtractionJob).filter_by(id=job.id).first()
            assert updated_job.status == "failed"
            assert updated_job.error_message == "Test error message"
            assert updated_job.completed_at is not None


class TestMetadataStoreAudit:
    """Test audit log operations."""

    def test_create_audit_log(self, metadata_store, sample_document_data):
        """Test creating audit log entry."""
        doc = metadata_store.create_document(**sample_document_data)

        log = metadata_store.create_audit_log(
            resource_type="document",
            resource_id=doc.id,
            action="create",
            changes={"user": "test_user"}
        )

        assert log.id is not None
        assert log.resource_id == doc.id
        assert log.action == "create"
        assert log.changes["user"] == "test_user"
        assert log.created_at is not None


class TestMetadataStoreStatistics:
    """Test statistics operations."""

    def test_get_statistics_empty(self, metadata_store):
        """Test statistics when database is empty."""
        stats = metadata_store.get_statistics()

        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0
        assert stats["total_jobs"] == 0
        assert stats["total_audit_logs"] == 0

    def test_get_statistics_with_data(self, metadata_store, sample_document_data):
        """Test statistics with actual data."""
        # Create 2 documents
        doc1 = metadata_store.create_document(**sample_document_data)
        sample_document_data["filename"] = "doc2.txt"
        doc2 = metadata_store.create_document(**sample_document_data)

        # Create 3 chunks for doc1
        for i in range(3):
            metadata_store.create_chunk(
                doc1.id,
                chunk_index=i,
                text=f"Chunk {i}",
                char_start=0,
                char_end=10,
                token_count=2,
            )

        # Create 1 job
        metadata_store.create_job(doc1.id, "chunking")

        # Create 1 audit log
        metadata_store.create_audit_log("document", doc1.id, "create")

        stats = metadata_store.get_statistics()

        assert stats["total_documents"] == 2
        assert stats["total_chunks"] == 3
        assert stats["total_jobs"] == 1
        assert stats["total_audit_logs"] == 1


class TestMetadataStoreFileHash:
    """Test file hash calculation."""

    def test_calculate_file_hash(self, tmp_path):
        """Test calculating SHA-256 hash of file."""
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        file_hash = MetadataStore._calculate_file_hash(test_file)

        assert file_hash is not None
        assert len(file_hash) == 64  # SHA-256 produces 64 hex characters
        # Verify consistent hash
        file_hash2 = MetadataStore._calculate_file_hash(test_file)
        assert file_hash == file_hash2

    def test_calculate_file_hash_different_content(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = MetadataStore._calculate_file_hash(file1)
        hash2 = MetadataStore._calculate_file_hash(file2)

        assert hash1 != hash2
