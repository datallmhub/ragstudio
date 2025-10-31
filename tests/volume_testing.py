"""
ChunkForge Test Strategy - Volume Testing Framework

This module provides comprehensive testing capabilities for ChunkForge RAG pipeline
with large volumes of diverse document types (PDF, DOC, TXT, etc.).
"""

import asyncio
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from src.core.config import chunking, languages
from pipeline.chunkforge_flow import chunkforge_flow, ChunkforgeParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestDocument:
    """Represents a test document with metadata."""
    path: str
    file_type: str
    size_bytes: int
    expected_pages: Optional[int] = None
    expected_language: Optional[str] = None
    complexity_score: Optional[float] = None
    ocr_required: bool = False


@dataclass
class TestResult:
    """Results from processing a test document."""
    document: TestDocument
    success: bool
    processing_time: float
    chunks_generated: int
    text_length: int
    language_detected: Optional[str] = None
    ocr_engine_used: Optional[str] = None
    error_message: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class TestSuiteResults:
    """Aggregated results from a test suite."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_processing_time: float
    average_processing_time: float
    total_chunks_generated: int
    average_chunks_per_document: float
    success_rate: float
    documents_by_type: Dict[str, int]
    errors_by_type: Dict[str, int]
    performance_metrics: Dict[str, float]


class TestDataGenerator:
    """Generates test datasets for comprehensive testing."""
    
    def __init__(self, output_dir: str = "data/test_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_test_corpus(self, 
                           num_documents: int = 1000,
                           document_types: List[str] = None,
                           size_distribution: Dict[str, float] = None) -> List[TestDocument]:
        """Generate a comprehensive test corpus."""
        
        if document_types is None:
            document_types = ["pdf", "docx", "txt", "html", "md"]
            
        if size_distribution is None:
            size_distribution = {
                "small": 0.4,    # < 100KB
                "medium": 0.4,   # 100KB - 1MB
                "large": 0.2     # > 1MB
            }
        
        logger.info(f"Generating test corpus with {num_documents} documents")
        
        documents = []
        
        for i in range(num_documents):
            doc_type = random.choice(document_types)
            size_category = random.choices(
                list(size_distribution.keys()),
                weights=list(size_distribution.values())
            )[0]
            
            doc = self._create_test_document(i, doc_type, size_category)
            documents.append(doc)
            
        # Save corpus metadata
        self._save_corpus_metadata(documents)
        
        logger.info(f"Generated {len(documents)} test documents")
        return documents
    
    def _create_test_document(self, index: int, doc_type: str, size_category: str) -> TestDocument:
        """Create a single test document."""
        
        # Generate realistic content based on type and size
        content = self._generate_content(doc_type, size_category)
        
        # Create file path
        filename = f"test_doc_{index:04d}.{doc_type}"
        filepath = self.output_dir / filename
        
        # Write content to file
        if doc_type == "pdf":
            self._create_pdf_file(filepath, content)
        elif doc_type == "docx":
            self._create_docx_file(filepath, content)
        elif doc_type == "txt":
            self._create_txt_file(filepath, content)
        elif doc_type == "html":
            self._create_html_file(filepath, content)
        elif doc_type == "md":
            self._create_markdown_file(filepath, content)
        
        # Get file size
        size_bytes = filepath.stat().st_size
        
        # Determine if OCR is required (simulate scanned PDFs)
        ocr_required = doc_type == "pdf" and random.random() < 0.3
        
        return TestDocument(
            path=str(filepath),
            file_type=doc_type,
            size_bytes=size_bytes,
            ocr_required=ocr_required,
            complexity_score=random.uniform(0.1, 0.9)
        )
    
    def _generate_content(self, doc_type: str, size_category: str) -> str:
        """Generate realistic content based on document type and size."""
        
        # Base content templates
        templates = {
            "pdf": self._get_pdf_content_template(),
            "docx": self._get_docx_content_template(),
            "txt": self._get_txt_content_template(),
            "html": self._get_html_content_template(),
            "md": self._get_markdown_content_template()
        }
        
        base_content = templates.get(doc_type, templates["txt"])
        
        # Scale content based on size category
        size_multipliers = {
            "small": 1,
            "medium": 5,
            "large": 20
        }
        
        multiplier = size_multipliers.get(size_category, 1)
        
        # Repeat and vary content
        content = base_content
        for _ in range(multiplier):
            content += "\n\n" + self._vary_content(base_content)
            
        return content
    
    def _get_pdf_content_template(self) -> str:
        """PDF content template."""
        return """
        # Document Title
        
        ## Abstract
        This is a comprehensive document containing multiple sections with detailed information.
        
        ## Introduction
        The purpose of this document is to demonstrate the capabilities of the ChunkForge RAG pipeline.
        
        ## Methodology
        We employ advanced natural language processing techniques combined with machine learning algorithms.
        
        ## Results
        The results show significant improvements in document processing accuracy and speed.
        
        ## Conclusion
        This document serves as a test case for the ChunkForge pipeline validation.
        """
    
    def _get_docx_content_template(self) -> str:
        """DOCX content template."""
        return """
        Business Report
        
        Executive Summary
        This report analyzes the current market conditions and provides recommendations for future growth.
        
        Market Analysis
        The market shows strong growth potential with emerging opportunities in various sectors.
        
        Financial Performance
        Revenue increased by 15% compared to the previous quarter, indicating positive momentum.
        
        Strategic Recommendations
        1. Expand into new markets
        2. Invest in technology infrastructure
        3. Develop strategic partnerships
        
        Conclusion
        The company is well-positioned for continued success in the coming quarters.
        """
    
    def _get_txt_content_template(self) -> str:
        """Plain text content template."""
        return """
        Technical Documentation
        
        Overview
        This document provides technical specifications and implementation details.
        
        System Requirements
        - Operating System: Linux/Windows/macOS
        - Memory: Minimum 8GB RAM
        - Storage: 50GB available space
        
        Installation Instructions
        1. Download the software package
        2. Extract the archive
        3. Run the installation script
        4. Configure system settings
        
        Usage Examples
        Here are some common usage patterns and examples.
        
        Troubleshooting
        Common issues and their solutions are documented here.
        """
    
    def _get_html_content_template(self) -> str:
        """HTML content template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Web Document</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Web Document Title</h1>
            <h2>Introduction</h2>
            <p>This is a web document designed to test HTML processing capabilities.</p>
            
            <h2>Content Section</h2>
            <p>This section contains various HTML elements and formatting.</p>
            
            <ul>
                <li>List item 1</li>
                <li>List item 2</li>
                <li>List item 3</li>
            </ul>
            
            <h2>Conclusion</h2>
            <p>This document serves as a test case for HTML processing.</p>
        </body>
        </html>
        """
    
    def _get_markdown_content_template(self) -> str:
        """Markdown content template."""
        return """
        # Markdown Document
        
        ## Overview
        This is a markdown document with various formatting elements.
        
        ### Features
        - **Bold text**
        - *Italic text*
        - `Code snippets`
        
        ### Code Example
        ```python
        def hello_world():
            print("Hello, World!")
        ```
        
        ### Table
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | Data 1   | Data 2   | Data 3   |
        | Data 4   | Data 5   | Data 6   |
        
        ## Conclusion
        This markdown document tests various formatting capabilities.
        """
    
    def _vary_content(self, base_content: str) -> str:
        """Add variation to content."""
        variations = [
            "Additional information and details.",
            "Further analysis and insights.",
            "Extended discussion of key points.",
            "Supplementary data and examples.",
            "Additional context and background."
        ]
        
        return random.choice(variations) + "\n" + base_content
    
    def _create_pdf_file(self, filepath: Path, content: str):
        """Create a PDF file (simplified - would use reportlab in real implementation)."""
        # For now, create a text file with .pdf extension
        # In production, use reportlab or similar to create actual PDFs
        filepath.write_text(content)
    
    def _create_docx_file(self, filepath: Path, content: str):
        """Create a DOCX file (simplified - would use python-docx in real implementation)."""
        # For now, create a text file with .docx extension
        filepath.write_text(content)
    
    def _create_txt_file(self, filepath: Path, content: str):
        """Create a plain text file."""
        filepath.write_text(content)
    
    def _create_html_file(self, filepath: Path, content: str):
        """Create an HTML file."""
        filepath.write_text(content)
    
    def _create_markdown_file(self, filepath: Path, content: str):
        """Create a markdown file."""
        filepath.write_text(content)
    
    def _save_corpus_metadata(self, documents: List[TestDocument]):
        """Save corpus metadata to JSON."""
        metadata = {
            "total_documents": len(documents),
            "document_types": {},
            "size_distribution": {},
            "ocr_required_count": 0,
            "documents": [asdict(doc) for doc in documents]
        }
        
        # Calculate statistics
        for doc in documents:
            doc_type = doc.file_type
            metadata["document_types"][doc_type] = metadata["document_types"].get(doc_type, 0) + 1
            
            if doc.ocr_required:
                metadata["ocr_required_count"] += 1
        
        # Save metadata
        metadata_path = self.output_dir / "corpus_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved corpus metadata to {metadata_path}")


class VolumeTestRunner:
    """Runs volume tests with comprehensive metrics collection."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.results: List[TestResult] = []
        
    @task
    async def process_single_document(self, document: TestDocument) -> TestResult:
        """Process a single document and return results."""
        logger = get_run_logger()
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing document: {document.path}")
            
            # Create ChunkForge parameters
            params = ChunkforgeParams(
                input_path=document.path,
                output_dir="data/test_output",
                chunking_strategy="semantic",
                chunk_size=512,
                chunk_overlap=50,
                languages=["en", "fr"]
            )
            
            # Process document
            result = await chunkforge_flow(params)
            
            processing_time = time.time() - start_time
            
            # Extract metrics from result
            chunks_generated = len(result.get("chunks", []))
            text_length = len(result.get("text", ""))
            
            return TestResult(
                document=document,
                success=True,
                processing_time=processing_time,
                chunks_generated=chunks_generated,
                text_length=text_length,
                language_detected=result.get("language"),
                ocr_engine_used=result.get("ocr_engine"),
                quality_score=self._calculate_quality_score(result)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process {document.path}: {str(e)}")
            
            return TestResult(
                document=document,
                success=False,
                processing_time=processing_time,
                chunks_generated=0,
                text_length=0,
                error_message=str(e)
            )
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Calculate a quality score for the processing result."""
        score = 1.0
        
        # Penalize for very short text
        text_length = len(result.get("text", ""))
        if text_length < 100:
            score -= 0.3
        elif text_length < 500:
            score -= 0.1
        
        # Penalize for very few chunks
        chunks_count = len(result.get("chunks", []))
        if chunks_count == 0:
            score -= 0.5
        elif chunks_count < 3:
            score -= 0.2
        
        # Bonus for successful OCR
        if result.get("ocr_engine") and result.get("ocr_confidence", 0) > 0.8:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    @flow(task_runner=ConcurrentTaskRunner())
    async def run_volume_test(self, 
                            documents: List[TestDocument],
                            batch_size: int = 50) -> TestSuiteResults:
        """Run volume tests on a batch of documents."""
        logger = get_run_logger()
        
        logger.info(f"Starting volume test with {len(documents)} documents")
        
        # Process documents in batches
        all_results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self.process_single_document(doc) for doc in batch],
                return_exceptions=True
            )
            
            # Filter out exceptions and add to results
            for result in batch_results:
                if isinstance(result, TestResult):
                    all_results.append(result)
                else:
                    logger.error(f"Exception in batch processing: {result}")
        
        # Calculate aggregated results
        suite_results = self._calculate_suite_results(all_results)
        
        # Save detailed results
        self._save_test_results(all_results, suite_results)
        
        logger.info(f"Volume test completed. Success rate: {suite_results.success_rate:.2%}")
        
        return suite_results
    
    def _calculate_suite_results(self, results: List[TestResult]) -> TestSuiteResults:
        """Calculate aggregated results from individual test results."""
        
        total_docs = len(results)
        successful_docs = sum(1 for r in results if r.success)
        failed_docs = total_docs - successful_docs
        
        processing_times = [r.processing_time for r in results]
        total_processing_time = sum(processing_times)
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0
        
        total_chunks = sum(r.chunks_generated for r in results)
        avg_chunks = total_chunks / total_docs if total_docs > 0 else 0
        
        success_rate = successful_docs / total_docs if total_docs > 0 else 0
        
        # Document type distribution
        doc_types = {}
        for result in results:
            doc_type = result.document.file_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Error type distribution
        error_types = {}
        for result in results:
            if not result.success and result.error_message:
                error_type = self._categorize_error(result.error_message)
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Performance metrics
        performance_metrics = {
            "min_processing_time": min(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
            "median_processing_time": statistics.median(processing_times) if processing_times else 0,
            "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
            "throughput_docs_per_minute": (total_docs / total_processing_time * 60) if total_processing_time > 0 else 0
        }
        
        return TestSuiteResults(
            total_documents=total_docs,
            successful_documents=successful_docs,
            failed_documents=failed_docs,
            total_processing_time=total_processing_time,
            average_processing_time=avg_processing_time,
            total_chunks_generated=total_chunks,
            average_chunks_per_document=avg_chunks,
            success_rate=success_rate,
            documents_by_type=doc_types,
            errors_by_type=error_types,
            performance_metrics=performance_metrics
        )
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages for analysis."""
        error_lower = error_message.lower()
        
        if "ocr" in error_lower or "tesseract" in error_lower:
            return "OCR_ERROR"
        elif "pdf" in error_lower or "parsing" in error_lower:
            return "PARSING_ERROR"
        elif "memory" in error_lower or "timeout" in error_lower:
            return "RESOURCE_ERROR"
        elif "language" in error_lower or "encoding" in error_lower:
            return "LANGUAGE_ERROR"
        else:
            return "OTHER_ERROR"
    
    def _save_test_results(self, results: List[TestResult], suite_results: TestSuiteResults):
        """Save test results to files."""
        
        # Create results directory
        results_dir = Path("data/test_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "document_path": result.document.path,
                "file_type": result.document.file_type,
                "size_bytes": result.document.size_bytes,
                "success": result.success,
                "processing_time": result.processing_time,
                "chunks_generated": result.chunks_generated,
                "text_length": result.text_length,
                "language_detected": result.language_detected,
                "ocr_engine_used": result.ocr_engine_used,
                "quality_score": result.quality_score,
                "error_message": result.error_message
            })
        
        detailed_path = results_dir / f"detailed_results_{timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save summary results
        summary_path = results_dir / f"summary_results_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(asdict(suite_results), f, indent=2)
        
        # Save CSV for analysis
        df = pd.DataFrame(detailed_results)
        csv_path = results_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Test results saved to {results_dir}")


@flow
async def run_comprehensive_volume_test(
    num_documents: int = 1000,
    document_types: List[str] = None,
    batch_size: int = 50,
    max_workers: int = 10
) -> TestSuiteResults:
    """Main flow for running comprehensive volume tests."""
    
    logger = get_run_logger()
    logger.info("Starting comprehensive volume test")
    
    # Generate test corpus
    generator = TestDataGenerator()
    documents = generator.generate_test_corpus(
        num_documents=num_documents,
        document_types=document_types
    )
    
    # Run volume tests
    runner = VolumeTestRunner(max_workers=max_workers)
    results = await runner.run_volume_test(documents, batch_size=batch_size)
    
    # Generate report
    report_path = Path("data/test_results") / f"volume_test_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    generate_test_report(results, report_path)
    
    logger.info(f"Volume test completed. Report saved to {report_path}")
    
    return results


def generate_test_report(results: TestSuiteResults, report_path: Path):
    """Generate a comprehensive test report."""
    
    report_content = f"""
# ChunkForge Volume Test Report

## Test Summary

- **Total Documents**: {results.total_documents:,}
- **Successful**: {results.successful_documents:,} ({results.success_rate:.2%})
- **Failed**: {results.failed_documents:,} ({1-results.success_rate:.2%})
- **Total Processing Time**: {results.total_processing_time:.2f} seconds
- **Average Processing Time**: {results.average_processing_time:.2f} seconds per document
- **Throughput**: {results.performance_metrics['throughput_docs_per_minute']:.1f} documents/minute

## Document Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
"""
    
    for doc_type, count in results.documents_by_type.items():
        percentage = (count / results.total_documents) * 100
        report_content += f"| {doc_type} | {count:,} | {percentage:.1f}% |\n"
    
    report_content += f"""
## Error Analysis

| Error Type | Count | Percentage |
|------------|-------|------------|
"""
    
    for error_type, count in results.errors_by_type.items():
        percentage = (count / results.failed_documents) * 100 if results.failed_documents > 0 else 0
        report_content += f"| {error_type} | {count:,} | {percentage:.1f}% |\n"
    
    report_content += f"""
## Performance Metrics

- **Min Processing Time**: {results.performance_metrics['min_processing_time']:.3f}s
- **Max Processing Time**: {results.performance_metrics['max_processing_time']:.3f}s
- **Median Processing Time**: {results.performance_metrics['median_processing_time']:.3f}s
- **Standard Deviation**: {results.performance_metrics['std_processing_time']:.3f}s

## Chunking Results

- **Total Chunks Generated**: {results.total_chunks_generated:,}
- **Average Chunks per Document**: {results.average_chunks_per_document:.1f}

## Recommendations

Based on the test results:

1. **Success Rate**: {'✅ Excellent' if results.success_rate > 0.95 else '⚠️ Needs improvement' if results.success_rate > 0.8 else '❌ Critical issues'}
2. **Performance**: {'✅ Good' if results.average_processing_time < 5 else '⚠️ Slow' if results.average_processing_time < 10 else '❌ Too slow'}
3. **Throughput**: {'✅ High' if results.performance_metrics['throughput_docs_per_minute'] > 100 else '⚠️ Moderate' if results.performance_metrics['throughput_docs_per_minute'] > 50 else '❌ Low'}

## Next Steps

1. Investigate failed documents
2. Optimize processing for slow document types
3. Scale infrastructure if needed
4. Implement additional error handling
"""
    
    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Test report generated: {report_path}")


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Run a small test first
        results = await run_comprehensive_volume_test(
            num_documents=100,
            document_types=["pdf", "txt", "docx"],
            batch_size=20,
            max_workers=5
        )
        
        print(f"Test completed with {results.success_rate:.2%} success rate")
    
    asyncio.run(main())
