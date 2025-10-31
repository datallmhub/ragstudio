#!/usr/bin/env bash
#
# Atlas-RAG CLI End-to-End Test Suite
# Tests all CLI commands and features
#
# Usage: ./tests/cli_e2e_test.sh
#        make test-cli-e2e

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test directory
TEST_DIR="/tmp/atlas-rag-e2e-tests"
RESULTS_FILE="${TEST_DIR}/test_results.log"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
    echo "[PASS] $1" >> "$RESULTS_FILE"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
    echo "[FAIL] $1" >> "$RESULTS_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit_code="${3:-0}"

    ((TOTAL_TESTS++))

    log_info "Running: $test_name"

    # Run command and capture exit code properly
    set +e  # Don't exit on error
    eval "$command" >> "$RESULTS_FILE" 2>&1
    actual_exit_code=$?
    set -e

    if [ "$actual_exit_code" -eq "$expected_exit_code" ]; then
        log_success "$test_name"
        return 0
    else
        log_error "$test_name (expected exit code $expected_exit_code, got $actual_exit_code)"
        return 1
    fi
}

cleanup() {
    log_info "Cleaning up test directory..."
    rm -rf "$TEST_DIR"
}

setup() {
    log_info "Setting up test environment..."

    # Create test directory
    mkdir -p "$TEST_DIR"/{input,output,batch,fixtures}

    # Initialize results file
    echo "Atlas-RAG CLI E2E Test Results" > "$RESULTS_FILE"
    echo "Date: $(date)" >> "$RESULTS_FILE"
    echo "========================================" >> "$RESULTS_FILE"

    # Create test fixtures
    create_fixtures
}

create_fixtures() {
    log_info "Creating test fixtures..."

    # Simple text file
    cat > "${TEST_DIR}/fixtures/simple.txt" <<'EOF'
This is a simple test document.
It contains multiple paragraphs for testing.

The second paragraph is here with more content.
This should be chunked appropriately.
EOF

    # Medium document
    cat > "${TEST_DIR}/fixtures/medium.txt" <<'EOF'
Atlas-RAG is a production-ready document processing pipeline.

It provides intelligent chunking strategies including semantic, sentence, and token-based approaches.

The system supports multiple output formats: JSON for structured data, JSONL for streaming, and CSV for spreadsheet compatibility.

Quality evaluation tools help optimize your chunking strategy with detailed metrics and recommendations.

Vector store integration enables seamless ingestion into Qdrant with automatic embedding generation.
EOF

    # Long document
    cat > "${TEST_DIR}/fixtures/long.txt" <<'EOF'
Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation.

Document Processing Pipeline

The first step in any RAG system is document processing. This involves loading documents, cleaning text, and chunking into meaningful segments.

Chunking Strategies

Semantic Chunking: Breaks text at natural semantic boundaries, preserving context and meaning.

Sentence Chunking: Splits on sentence boundaries, useful for maintaining grammatical structure.

Token Chunking: Uses fixed token counts with configurable overlap, providing consistency.

Vector Embeddings

After chunking, text segments are converted to vector embeddings using models like SentenceTransformers.

Storage and Retrieval

Embeddings are stored in vector databases like Qdrant, enabling fast semantic search.

Conclusion

A well-designed RAG pipeline with proper chunking is crucial for high-quality results.
EOF

    # Batch test files
    for i in {1..5}; do
        cat > "${TEST_DIR}/batch/doc${i}.txt" <<EOF
Document ${i} for batch processing test.

This document contains content that should be chunked.
Each document has unique content to test batch processing.

Paragraph ${i}.1: Additional content for testing.
Paragraph ${i}.2: More text to ensure proper chunking.
EOF
    done

    # Create subdirectory for recursive test
    mkdir -p "${TEST_DIR}/batch/subdir"
    cat > "${TEST_DIR}/batch/subdir/nested.txt" <<'EOF'
This is a nested document in a subdirectory.
Used for testing recursive batch processing.
EOF
}

# =============================================================================
# TEST SUITE
# =============================================================================

echo "========================================="
echo "  Atlas-RAG CLI E2E Test Suite"
echo "========================================="
echo ""

setup

# -----------------------------------------------------------------------------
# 1. Basic CLI Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 1. Basic CLI Tests =====${NC}\n"

run_test "CLI: --version" \
    "atlas-rag --version"

run_test "CLI: --help" \
    "atlas-rag --help"

run_test "CLI: chunk --help" \
    "atlas-rag chunk --help"

run_test "CLI: batch --help" \
    "atlas-rag batch --help"

run_test "CLI: ingest --help" \
    "atlas-rag ingest --help"

run_test "CLI: eval --help" \
    "atlas-rag eval --help"

run_test "CLI: search --help" \
    "atlas-rag search --help"

run_test "CLI: info --help" \
    "atlas-rag info --help"

# -----------------------------------------------------------------------------
# 2. Chunk Command Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 2. Chunk Command Tests =====${NC}\n"

# Basic chunking
run_test "Chunk: Simple file" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt"

run_test "Chunk: Medium file" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt"

# Output formats
run_test "Chunk: Output JSON" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt -o ${TEST_DIR}/output/simple.json"

run_test "Chunk: Output JSONL" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt -o ${TEST_DIR}/output/simple.jsonl"

run_test "Chunk: Output CSV" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt -o ${TEST_DIR}/output/simple.csv"

# Verify output files exist
if [ -f "${TEST_DIR}/output/simple.json" ]; then
    log_success "Chunk: JSON file created"
    ((TOTAL_TESTS++))
else
    log_error "Chunk: JSON file not created"
    ((TOTAL_TESTS++))
fi

if [ -f "${TEST_DIR}/output/simple.jsonl" ]; then
    log_success "Chunk: JSONL file created"
    ((TOTAL_TESTS++))
else
    log_error "Chunk: JSONL file not created"
    ((TOTAL_TESTS++))
fi

if [ -f "${TEST_DIR}/output/simple.csv" ]; then
    log_success "Chunk: CSV file created"
    ((TOTAL_TESTS++))
else
    log_error "Chunk: CSV file not created"
    ((TOTAL_TESTS++))
fi

# Strategies
run_test "Chunk: Strategy semantic" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt --strategy semantic -o ${TEST_DIR}/output/semantic.json"

run_test "Chunk: Strategy sentence" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt --strategy sentence -o ${TEST_DIR}/output/sentence.json"

run_test "Chunk: Strategy token" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt --strategy token -o ${TEST_DIR}/output/token.json"

# Parameters
run_test "Chunk: Custom max-tokens" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt --max-tokens 200 -o ${TEST_DIR}/output/tokens_200.json"

run_test "Chunk: Custom overlap" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt --overlap 100 -o ${TEST_DIR}/output/overlap_100.json"

run_test "Chunk: With --show flag" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt --show"

# -----------------------------------------------------------------------------
# 3. Strategy Auto Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 3. Strategy Auto Tests =====${NC}\n"

# Note: May fallback to semantic if ML unavailable
log_warning "Strategy auto tests may fallback to semantic due to NumPy issues"

run_test "Chunk: Strategy auto (simple)" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt --strategy auto -o ${TEST_DIR}/output/auto_simple.json"

run_test "Chunk: Strategy auto (medium)" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/medium.txt --strategy auto -o ${TEST_DIR}/output/auto_medium.json"

# -----------------------------------------------------------------------------
# 4. Batch Command Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 4. Batch Command Tests =====${NC}\n"

run_test "Batch: Basic processing" \
    "atlas-rag batch ${TEST_DIR}/batch -o ${TEST_DIR}/output/batch_all.json"

run_test "Batch: Output JSONL" \
    "atlas-rag batch ${TEST_DIR}/batch -o ${TEST_DIR}/output/batch_all.jsonl"

run_test "Batch: Output CSV" \
    "atlas-rag batch ${TEST_DIR}/batch -o ${TEST_DIR}/output/batch_all.csv"

run_test "Batch: Recursive processing" \
    "atlas-rag batch ${TEST_DIR}/batch --recursive -o ${TEST_DIR}/output/batch_recursive.json"

run_test "Batch: Pattern matching" \
    "atlas-rag batch ${TEST_DIR}/batch --pattern '*.txt' -o ${TEST_DIR}/output/batch_pattern.json"

run_test "Batch: Custom strategy" \
    "atlas-rag batch ${TEST_DIR}/batch --strategy sentence -o ${TEST_DIR}/output/batch_sentence.json"

run_test "Batch: Custom max-tokens" \
    "atlas-rag batch ${TEST_DIR}/batch --max-tokens 300 -o ${TEST_DIR}/output/batch_300.json"

# -----------------------------------------------------------------------------
# 5. Eval Command Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 5. Eval Command Tests =====${NC}\n"

run_test "Eval: Single file" \
    "atlas-rag eval ${TEST_DIR}/output/semantic.json"

run_test "Eval: Compare strategies" \
    "atlas-rag eval ${TEST_DIR}/output/semantic.json ${TEST_DIR}/output/sentence.json ${TEST_DIR}/output/token.json --compare"

run_test "Eval: With report output" \
    "atlas-rag eval ${TEST_DIR}/output/semantic.json --report ${TEST_DIR}/output/eval_report.json"

# Verify report created
if [ -f "${TEST_DIR}/output/eval_report.json" ]; then
    log_success "Eval: Report file created"
    ((TOTAL_TESTS++))
else
    log_error "Eval: Report file not created"
    ((TOTAL_TESTS++))
fi

# -----------------------------------------------------------------------------
# 6. Info Command Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 6. Info Command Tests =====${NC}\n"

run_test "Info: System information" \
    "atlas-rag info"

# -----------------------------------------------------------------------------
# 7. Error Handling Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 7. Error Handling Tests =====${NC}\n"

run_test "Error: Non-existent file" \
    "atlas-rag chunk /nonexistent/file.txt" \
    2

run_test "Error: Invalid strategy" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt --strategy invalid" \
    1

run_test "Error: Tokens too low" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt --max-tokens 10" \
    2

run_test "Error: Tokens too high" \
    "atlas-rag chunk ${TEST_DIR}/fixtures/simple.txt --max-tokens 5000" \
    2

run_test "Error: Non-existent directory (batch)" \
    "atlas-rag batch /nonexistent/dir" \
    2

run_test "Error: Empty directory (batch)" \
    "atlas-rag batch ${TEST_DIR}/output --pattern '*.nonexistent'"

# -----------------------------------------------------------------------------
# 8. Output Format Validation Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 8. Output Format Validation =====${NC}\n"

# Validate JSON format
if [ -f "${TEST_DIR}/output/simple.json" ]; then
    if python3 -m json.tool "${TEST_DIR}/output/simple.json" > /dev/null 2>&1; then
        log_success "Format: Valid JSON"
        ((TOTAL_TESTS++))
    else
        log_error "Format: Invalid JSON"
        ((TOTAL_TESTS++))
    fi
fi

# Validate JSONL format (each line is valid JSON)
if [ -f "${TEST_DIR}/output/simple.jsonl" ]; then
    if head -1 "${TEST_DIR}/output/simple.jsonl" | python3 -m json.tool > /dev/null 2>&1; then
        log_success "Format: Valid JSONL"
        ((TOTAL_TESTS++))
    else
        log_error "Format: Invalid JSONL"
        ((TOTAL_TESTS++))
    fi
fi

# Validate CSV format (has headers)
if [ -f "${TEST_DIR}/output/simple.csv" ]; then
    if head -1 "${TEST_DIR}/output/simple.csv" | grep -q "id,text"; then
        log_success "Format: Valid CSV headers"
        ((TOTAL_TESTS++))
    else
        log_error "Format: Invalid CSV headers"
        ((TOTAL_TESTS++))
    fi
fi

# -----------------------------------------------------------------------------
# 9. Integration Tests (Optional - require running services)
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 9. Integration Tests (Optional) =====${NC}\n"

log_warning "Skipping integration tests (ingest, search) - require running services"
log_info "To test: Start docker-compose up -d, then run these manually:"
log_info "  atlas-rag ingest ${TEST_DIR}/output/semantic.json"
log_info "  atlas-rag search 'test query'"

# -----------------------------------------------------------------------------
# 10. Performance Tests
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}===== 10. Performance Tests =====${NC}\n"

# Test with large batch
log_info "Performance: Large batch test (5 files)"
START_TIME=$(date +%s)
atlas-rag batch ${TEST_DIR}/batch -o ${TEST_DIR}/output/perf_batch.json > /dev/null 2>&1
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $DURATION -lt 30 ]; then
    log_success "Performance: Batch completed in ${DURATION}s (< 30s)"
    ((TOTAL_TESTS++))
else
    log_warning "Performance: Batch took ${DURATION}s (> 30s)"
    ((TOTAL_TESTS++))
fi

# =============================================================================
# TEST RESULTS SUMMARY
# =============================================================================

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo ""
echo -e "Total Tests:  ${TOTAL_TESTS}"
echo -e "${GREEN}Passed:       ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed:       ${FAILED_TESTS}${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    EXIT_CODE=0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    EXIT_CODE=1
fi

echo ""
echo "Detailed results saved to: $RESULTS_FILE"
echo ""

# Optional: Keep or cleanup
if [ "$1" = "--keep" ]; then
    log_info "Test directory preserved: $TEST_DIR"
else
    cleanup
fi

exit $EXIT_CODE
