"""Tests for CourseSearchTool functionality"""

from unittest.mock import Mock

import pytest
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly structured"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Check required parameters
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_successful_search(self, course_search_tool):
        """Test successful search with results"""
        result = course_search_tool.execute("introduction")

        assert "MCP: Build Rich-Context AI Apps with Anthropic" in result
        assert "Introduction to Model Context Protocol" in result
        assert course_search_tool.last_sources is not None
        assert len(course_search_tool.last_sources) > 0

    def test_execute_empty_results(self, course_search_tool):
        """Test search with no results"""
        result = course_search_tool.execute("empty_results")

        assert "No relevant content found" in result
        assert course_search_tool.last_sources == []

    def test_execute_with_course_filter(self, course_search_tool):
        """Test search with course name filter"""
        result = course_search_tool.execute("introduction", course_name="MCP")

        # Should still get results since our mock returns them
        assert len(result) > 0
        # Verify the search was called with course_name parameter
        course_search_tool.store.search.assert_called_with(
            query="introduction", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test search with lesson number filter"""
        result = course_search_tool.execute("introduction", lesson_number=1)

        assert len(result) > 0
        # Verify the search was called with lesson_number parameter
        course_search_tool.store.search.assert_called_with(
            query="introduction", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, course_search_tool):
        """Test search with both course and lesson filters"""
        result = course_search_tool.execute(
            "introduction", course_name="MCP", lesson_number=1
        )

        assert len(result) > 0
        course_search_tool.store.search.assert_called_with(
            query="introduction", course_name="MCP", lesson_number=1
        )

    def test_execute_search_error(self, course_search_tool):
        """Test handling of search errors"""
        result = course_search_tool.execute("error_test")

        assert "Search error: Test error" in result

    def test_execute_empty_results_with_filters(self, course_search_tool):
        """Test empty results with filter information"""
        result = course_search_tool.execute(
            "empty_results", course_name="MCP", lesson_number=1
        )

        assert "No relevant content found in course 'MCP' in lesson 1" in result

    def test_format_results_with_lesson(self, course_search_tool):
        """Test result formatting with lesson information"""
        # Create mock results with lesson data
        results = SearchResults(
            documents=["Test content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 5, "chunk_index": 0}
            ],
            distances=[0.1],
        )

        formatted = course_search_tool._format_results(results)

        assert "[Test Course - Lesson 5]" in formatted
        assert "Test content" in formatted

    def test_format_results_without_lesson(self, course_search_tool):
        """Test result formatting without lesson information"""
        results = SearchResults(
            documents=["Test content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": None, "chunk_index": 0}
            ],
            distances=[0.1],
        )

        formatted = course_search_tool._format_results(results)

        assert "[Test Course]" in formatted
        assert "Test content" in formatted

    def test_source_tracking(self, course_search_tool):
        """Test that sources are properly tracked"""
        course_search_tool.execute("introduction")

        sources = course_search_tool.last_sources
        assert len(sources) > 0

        source = sources[0]
        assert "text" in source
        assert "url" in source

    def test_source_tracking_with_lesson_links(self, course_search_tool):
        """Test source tracking includes lesson links when available"""
        # Mock a search result with lesson number to trigger link lookup
        results = SearchResults(
            documents=["Test content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}
            ],
            distances=[0.1],
        )

        course_search_tool.store.search.return_value = results
        course_search_tool.store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        course_search_tool.execute("test")

        # Verify lesson link was requested
        course_search_tool.store.get_lesson_link.assert_called_with("Test Course", 1)

        # Check source includes URL
        sources = course_search_tool.last_sources
        assert sources[0]["url"] == "https://example.com/lesson1"


class TestCourseSearchToolWithEmptyStore:
    """Test CourseSearchTool with empty vector store"""

    def test_search_with_empty_store(self, mock_empty_vector_store):
        """Test search behavior when vector store is empty"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute("test query")

        assert "No relevant content found" in result

    def test_search_with_course_filter_empty_store(self, mock_empty_vector_store):
        """Test search with course filter on empty store"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute("test query", course_name="Nonexistent Course")

        assert "No relevant content found in course 'Nonexistent Course'" in result


class TestCourseSearchToolWithErrors:
    """Test CourseSearchTool error handling"""

    def test_search_with_database_error(self, mock_error_vector_store):
        """Test search behavior when vector store raises exceptions"""
        tool = CourseSearchTool(mock_error_vector_store)
        result = tool.execute("test query")

        # Should return the error message from SearchResults.empty()
        assert "Database connection error" in result

    def test_lesson_link_error_handling(self, mock_vector_store):
        """Test handling of lesson link lookup errors"""
        # Make lesson link lookup fail
        mock_vector_store.get_lesson_link.side_effect = Exception("Link lookup failed")

        # Override the mock search to return our specific result
        results = SearchResults(
            documents=["Test content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}
            ],
            distances=[0.1],
        )
        mock_vector_store.search.side_effect = None  # Clear existing side_effect
        mock_vector_store.search.return_value = results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test")

        # Should still work, just without the link
        assert "Test content" in result
        # Source should have None URL due to error
        assert tool.last_sources[0]["url"] is None
