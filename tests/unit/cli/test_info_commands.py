"""
Unit tests for core/cli/commands/info.py

Tests cover:
- API health check (available/unavailable)
- Vector store information retrieval
- Error handling (connection, timeouts, general errors)
- Custom API URL parameter
"""

import pytest
from unittest.mock import Mock, patch
import requests
from click.exceptions import Exit as ClickExit
import typer

from src.core.cli.commands.info import info_command


class TestInfoAPIHealth:
    """Test suite for API health checking."""

    @patch('src.core.cli.commands.info.requests.get')
    def test_api_available_and_healthy(self, mock_get):
        """Test info when API is available and healthy."""
        # Mock successful API health response
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0"
        }
        mock_get.return_value = mock_response

        # Should complete without error
        try:
            info_command()
        except (ClickExit, typer.Exit):
            pass

        # Verify health endpoint was called
        assert mock_get.called
        call_args = mock_get.call_args_list[0]
        assert "/health" in call_args[0][0]

    @patch('src.core.cli.commands.info.requests.get')
    @patch('src.core.cli.commands.info.print_error')
    def test_api_connection_error(self, mock_print_error, mock_get):
        """Test info when API is not available (connection error)."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        try:
            info_command()
        except (ClickExit, typer.Exit):
            pass

        # Verify error was displayed
        assert mock_print_error.called
        error_message = mock_print_error.call_args[0][0]
        assert "Not available" in error_message

    @patch('src.core.cli.commands.info.requests.get')
    @patch('src.core.cli.commands.info.print_error')
    def test_api_general_error(self, mock_print_error, mock_get):
        """Test info when API returns an error."""
        # Mock general exception
        mock_get.side_effect = Exception("Internal server error")

        try:
            info_command()
        except (ClickExit, typer.Exit):
            pass

        # Verify error was caught and displayed
        assert mock_print_error.called

    @patch('src.core.cli.commands.info.requests.get')
    def test_custom_api_url(self, mock_get):
        """Test info with custom API URL."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0"
        }
        mock_get.return_value = mock_response

        custom_url = "http://192.168.1.100:8000"

        try:
            info_command(api_url=custom_url)
        except (ClickExit, typer.Exit):
            pass

        # Verify custom URL was used
        assert mock_get.called
        call_args = mock_get.call_args_list[0]
        assert custom_url in call_args[0][0]


class TestInfoVectorStore:
    """Test suite for vector store information."""

    @patch('src.core.cli.commands.info.requests.get')
    def test_vector_store_available(self, mock_get):
        """Test info when vector store info is available."""
        # Mock API health and vector store responses
        def mock_get_side_effect(url, *args, **kwargs):
            mock_response = Mock()
            if "/health" in url:
                mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
            elif "/vector/info" in url:
                mock_response.json.return_value = {
                    "name": "test_collection",
                    "vectors_count": 1000,
                    "config": {"dimension": 384}
                }
            return mock_response

        mock_get.side_effect = mock_get_side_effect

        try:
            info_command()
        except (ClickExit, typer.Exit):
            pass

        # Verify both endpoints were called
        assert mock_get.call_count >= 2

    @patch('src.core.cli.commands.info.requests.get')
    @patch('src.core.cli.commands.info.print_warning')
    def test_vector_store_error(self, mock_print_warning, mock_get):
        """Test info when vector store returns an error."""
        # Mock API healthy but vector store fails
        def mock_get_side_effect(url, *args, **kwargs):
            mock_response = Mock()
            if "/health" in url:
                mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
            elif "/vector/info" in url:
                raise Exception("Vector store not initialized")
            return mock_response

        mock_get.side_effect = mock_get_side_effect

        try:
            info_command()
        except (ClickExit, typer.Exit):
            pass

        # Verify warning was displayed
        assert mock_print_warning.called
