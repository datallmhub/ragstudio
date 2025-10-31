#!/bin/bash
# Setup test data for ragctl functional tests

set -e

echo "🔧 Setting up test data for functional tests..."

# Create base directory
TEST_DATA_DIR="./test_data"
mkdir -p "$TEST_DATA_DIR"

# 1. Create simple text file
echo "📄 Creating test.txt..."
cat > "$TEST_DATA_DIR/test.txt" << 'EOF'
# Sample Document for Testing

This is a sample document used for testing the ragctl chunking functionality.

## Section 1: Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Section 2: Content

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

## Section 3: Conclusion

Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.
EOF

# 2. Create empty file
echo "📄 Creating empty.txt..."
touch "$TEST_DATA_DIR/empty.txt"

# 3. Create large text file
echo "📄 Creating large.txt..."
{
  for i in {1..1000}; do
    echo "## Section $i"
    echo ""
    echo "This is section $i with some content. Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    echo "More text for section $i to make it substantial. Sed do eiusmod tempor incididunt ut labore."
    echo ""
  done
} > "$TEST_DATA_DIR/large.txt"

# 4. Create valid chunks JSON
echo "📄 Creating chunks.json..."
cat > "$TEST_DATA_DIR/chunks.json" << 'EOF'
{
  "chunks": [
    {
      "id": "chunk_001",
      "text": "This is the first chunk of text.",
      "metadata": {
        "source": "test.txt",
        "chunk_index": 0
      }
    },
    {
      "id": "chunk_002",
      "text": "This is the second chunk of text.",
      "metadata": {
        "source": "test.txt",
        "chunk_index": 1
      }
    },
    {
      "id": "chunk_003",
      "text": "This is the third chunk of text.",
      "metadata": {
        "source": "test.txt",
        "chunk_index": 2
      }
    }
  ]
}
EOF

# 5. Create valid chunks JSONL
echo "📄 Creating chunks.jsonl..."
cat > "$TEST_DATA_DIR/chunks.jsonl" << 'EOF'
{"id": "chunk_001", "text": "This is the first chunk.", "metadata": {"source": "test.txt", "chunk_index": 0}}
{"id": "chunk_002", "text": "This is the second chunk.", "metadata": {"source": "test.txt", "chunk_index": 1}}
{"id": "chunk_003", "text": "This is the third chunk.", "metadata": {"source": "test.txt", "chunk_index": 2}}
EOF

# 6. Create invalid JSON
echo "📄 Creating invalid.json..."
cat > "$TEST_DATA_DIR/invalid.json" << 'EOF'
{
  "chunks": [
    {"id": "chunk_001", "text": "Missing closing brace"
  ]
EOF

# 7. Create docs directory with mixed files
echo "📁 Creating docs directory..."
mkdir -p "$TEST_DATA_DIR/docs"

cat > "$TEST_DATA_DIR/docs/doc1.txt" << 'EOF'
Document 1 content.
This is a simple text document.
EOF

cat > "$TEST_DATA_DIR/docs/doc2.txt" << 'EOF'
Document 2 content.
Another text document for batch processing.
EOF

cat > "$TEST_DATA_DIR/docs/doc3.md" << 'EOF'
# Markdown Document

This is a markdown document with **bold** and *italic* text.

## Features
- Feature 1
- Feature 2
- Feature 3
EOF

# 8. Create docs subdirectory for recursive test
mkdir -p "$TEST_DATA_DIR/docs/subdir"

cat > "$TEST_DATA_DIR/docs/subdir/nested.txt" << 'EOF'
Nested document in subdirectory.
This tests recursive processing.
EOF

# 9. Create empty directory
echo "📁 Creating empty directory..."
mkdir -p "$TEST_DATA_DIR/empty"

# 10. Create mixed file types directory
echo "📁 Creating mixed directory..."
mkdir -p "$TEST_DATA_DIR/mixed"

cp "$TEST_DATA_DIR/test.txt" "$TEST_DATA_DIR/mixed/file1.txt"
cp "$TEST_DATA_DIR/docs/doc3.md" "$TEST_DATA_DIR/mixed/file2.md"
echo "CSV,File,Test" > "$TEST_DATA_DIR/mixed/file3.csv"
echo '{"key": "value"}' > "$TEST_DATA_DIR/mixed/file4.json"

# 11. Copy existing PDF test files from tests/data/
if [ -f "tests/data/pb-kafka.pdf" ]; then
  echo "📄 Copying test PDFs from tests/data/..."
  cp tests/data/pb-kafka.pdf "$TEST_DATA_DIR/test.pdf"
  cp tests/data/grammaire-francaise.pdf "$TEST_DATA_DIR/large.pdf" 2>/dev/null || true
  echo "✅ test.pdf created (pb-kafka.pdf)"
  if [ -f "$TEST_DATA_DIR/large.pdf" ]; then
    echo "✅ large.pdf created (grammaire-francaise.pdf)"
  fi
else
  echo "⚠️  tests/data/pb-kafka.pdf not found"
  echo "   PDF tests will be skipped"
fi

echo ""
echo "✅ Test data setup complete!"
echo ""
echo "📊 Created files:"
echo "   - $TEST_DATA_DIR/test.txt (simple text)"
echo "   - $TEST_DATA_DIR/empty.txt (empty file)"
echo "   - $TEST_DATA_DIR/large.txt (large text, ~100KB)"
echo "   - $TEST_DATA_DIR/chunks.json (valid JSON)"
echo "   - $TEST_DATA_DIR/chunks.jsonl (valid JSONL)"
echo "   - $TEST_DATA_DIR/invalid.json (malformed JSON)"
echo "   - $TEST_DATA_DIR/docs/ (3 files + 1 subdir)"
echo "   - $TEST_DATA_DIR/empty/ (empty directory)"
echo "   - $TEST_DATA_DIR/mixed/ (4 different file types)"
if [ -f "$TEST_DATA_DIR/test.pdf" ]; then
  echo "   - $TEST_DATA_DIR/test.pdf (PDF document)"
fi
echo ""
echo "🚀 Ready to run functional tests!"
echo "   Run: ./tests/functional/test_ragctl.sh"
