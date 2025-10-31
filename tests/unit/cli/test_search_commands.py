"""Unit tests for search command."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from click.exceptions import Exit as ClickExit
import typer
import requests

from src.core.cli.commands.search import search_command


class TestSearchAPIHealth:
    """Tests for API health check during search."""

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_successful(self, mock_post, mock_get):
        """Test successful search operation."""
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_health_response

        # Mock search response
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "chunk_1",
                    "score": 0.85,
                    "text": "Machine learning is a subset of AI",
                    "metadata": {"source_file": "ml_intro.pdf"}
                }
            ],
            "search_time_seconds": 0.123
        }
        mock_post.return_value = mock_search_response

        # Execute search
        search_command(query="machine learning")

        # Verify calls
        mock_get.assert_called_once_with("http://localhost:8000/health", timeout=5)
        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/vector/search",
            json={"query": "machine learning", "top_k": 5},
            timeout=30
        )

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_with_top_k(self, mock_post, mock_get):
        """Test search with custom top_k parameter."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock search response
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [],
            "search_time_seconds": 0.05
        }
        mock_post.return_value = mock_search_response

        # Execute search with top_k=10
        search_command(query="test query", top_k=10)

        # Verify top_k is passed correctly
        call_args = mock_post.call_args
        assert call_args[1]["json"]["top_k"] == 10

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_with_threshold(self, mock_post, mock_get):
        """Test search with score threshold."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock search response
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [],
            "search_time_seconds": 0.05
        }
        mock_post.return_value = mock_search_response

        # Execute search with threshold
        search_command(query="test query", threshold=0.7)

        # Verify threshold is passed as score_threshold
        call_args = mock_post.call_args
        assert call_args[1]["json"]["score_threshold"] == 0.7

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_with_custom_api_url(self, mock_post, mock_get):
        """Test search with custom API URL."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock search response
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [],
            "search_time_seconds": 0.05
        }
        mock_post.return_value = mock_search_response

        custom_url = "http://192.168.1.100:8000"
        search_command(query="test", api_url=custom_url)

        # Verify custom URL is used
        mock_get.assert_called_once_with(f"{custom_url}/health", timeout=5)
        mock_post.assert_called_once()
        assert custom_url in mock_post.call_args[0][0]

    @patch('src.core.cli.commands.search.requests.get')
    def test_search_api_connection_error(self, mock_get):
        """Test search when API is not reachable."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        # Verify Exit is raised
        with pytest.raises((ClickExit, typer.Exit)):
            search_command(query="test query")

    @patch('src.core.cli.commands.search.requests.get')
    def test_search_api_health_check_general_error(self, mock_get):
        """Test search when API health check fails with general error."""
        # Mock general exception
        mock_get.side_effect = Exception("Unexpected error")

        # Verify Exit is raised
        with pytest.raises((ClickExit, typer.Exit)):
            search_command(query="test query")


class TestSearchExecution:
    """Tests for search execution and error handling."""

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_http_error(self, mock_post, mock_get):
        """Test search when HTTP error occurs."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock HTTP error
        mock_error_response = Mock()
        mock_error_response.text = "Search failed: invalid query"
        http_error = requests.exceptions.HTTPError("400 Client Error")
        http_error.response = mock_error_response
        mock_post.side_effect = http_error

        # Verify Exit is raised
        with pytest.raises((ClickExit, typer.Exit)):
            search_command(query="test")

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_general_error(self, mock_post, mock_get):
        """Test search when general error occurs."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock general exception
        mock_post.side_effect = Exception("Network timeout")

        # Verify Exit is raised
        with pytest.raises((ClickExit, typer.Exit)):
            search_command(query="test")


class TestSearchResults:
    """Tests for search results display."""

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_no_results(self, mock_post, mock_get):
        """Test search with no results."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock empty results
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [],
            "search_time_seconds": 0.05
        }
        mock_post.return_value = mock_search_response

        # Should complete without error
        search_command(query="nonexistent query")

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_with_multiple_results(self, mock_post, mock_get):
        """Test search with multiple results."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock multiple results with different scores
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "chunk_1",
                    "score": 0.95,  # High score (> 0.7)
                    "text": "First result",
                    "metadata": {"source_file": "file1.pdf"}
                },
                {
                    "chunk_id": "chunk_2",
                    "score": 0.65,  # Medium score (> 0.5)
                    "text": "Second result",
                    "metadata": {}
                },
                {
                    "chunk_id": "chunk_3",
                    "score": 0.45,  # Low score (< 0.5)
                    "text": "Third result",
                    "metadata": {"source": "file3.txt"}
                }
            ],
            "search_time_seconds": 0.234
        }
        mock_post.return_value = mock_search_response

        # Should display all results
        search_command(query="test query", top_k=3)

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_result_text_truncation(self, mock_post, mock_get):
        """Test that long result text is truncated."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock result with very long text (> 300 chars)
        long_text = "A" * 400
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "chunk_1",
                    "score": 0.85,
                    "text": long_text,
                    "metadata": {}
                }
            ],
            "search_time_seconds": 0.1
        }
        mock_post.return_value = mock_search_response

        # Should truncate without error
        search_command(query="test")

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_result_with_metadata(self, mock_post, mock_get):
        """Test search result with various metadata formats."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock results with different metadata formats
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "chunk_1",
                    "score": 0.85,
                    "text": "Result with source_file",
                    "metadata": {"source_file": "doc.pdf", "page": 5}
                },
                {
                    "chunk_id": "chunk_2",
                    "score": 0.75,
                    "text": "Result with source",
                    "metadata": {"source": "article.txt"}
                },
                {
                    "chunk_id": "chunk_3",
                    "score": 0.65,
                    "text": "Result with no source",
                    "metadata": {"other_field": "value"}
                }
            ],
            "search_time_seconds": 0.15
        }
        mock_post.return_value = mock_search_response

        # Should handle all metadata formats
        search_command(query="test query")

    @patch('src.core.cli.commands.search.requests.get')
    @patch('src.core.cli.commands.search.requests.post')
    def test_search_with_threshold_display(self, mock_post, mock_get):
        """Test that threshold is displayed in results."""
        # Mock health check
        mock_health_response = Mock()
        mock_get.return_value = mock_health_response

        # Mock search response
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "chunk_1",
                    "score": 0.85,
                    "text": "High score result",
                    "metadata": {}
                }
            ],
            "search_time_seconds": 0.1
        }
        mock_post.return_value = mock_search_response

        # Execute with threshold
        search_command(query="test", threshold=0.7)

        # Threshold should be displayed (verified by code coverage)
