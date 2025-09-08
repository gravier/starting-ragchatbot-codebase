"""Shared test fixtures and utilities for RAG system tests"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from config import Config

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock(spec=Config)
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config

@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to MCP", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Setting up MCP", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Advanced MCP Features", lesson_link="https://example.com/lesson3")
    ]
    
    return Course(
        title="MCP: Build Rich-Context AI Apps with Anthropic",
        instructor="DeepLearning.AI",
        course_link="https://example.com/course",
        lessons=lessons
    )

@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course content chunks for testing"""
    return [
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
            content="Introduction to Model Context Protocol (MCP). MCP enables AI applications to securely connect to external data sources and tools."
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1,
            content="MCP provides a standardized way for AI applications to interact with various data sources including databases, APIs, and file systems."
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=0,
            content="Setting up MCP requires installing the MCP SDK and configuring your development environment with the necessary dependencies."
        )
    ]

@pytest.fixture
def mock_vector_store():
    """Mock vector store with test data"""
    store = Mock()
    
    # Mock search method
    def mock_search(query: str, course_name: str = None, lesson_number: int = None, limit: int = None):
        if query == "introduction":
            return SearchResults(
                documents=["Introduction to Model Context Protocol (MCP). MCP enables AI applications to securely connect to external data sources and tools."],
                metadata=[{"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 1, "chunk_index": 0}],
                distances=[0.1]
            )
        elif query == "empty_results":
            return SearchResults(documents=[], metadata=[], distances=[])
        elif query == "error_test":
            return SearchResults.empty("Search error: Test error")
        else:
            return SearchResults(
                documents=["Sample content about the query"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}],
                distances=[0.2]
            )
    
    store.search.side_effect = mock_search
    
    # Mock other methods
    store._resolve_course_name.return_value = "MCP: Build Rich-Context AI Apps with Anthropic"
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store.get_existing_course_titles.return_value = ["MCP: Build Rich-Context AI Apps with Anthropic", "Another Course"]
    store.get_course_count.return_value = 2
    store.get_course_titles.return_value = ["MCP: Build Rich-Context AI Apps with Anthropic", "Another Course"]
    
    # Mock course catalog for outline tool
    catalog_mock = Mock()
    catalog_mock.get.return_value = {
        'metadatas': [{
            'title': 'MCP: Build Rich-Context AI Apps with Anthropic',
            'instructor': 'DeepLearning.AI',
            'course_link': 'https://example.com/course',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction to MCP", "lesson_link": "https://example.com/lesson1"}, {"lesson_number": 2, "lesson_title": "Setting up MCP", "lesson_link": "https://example.com/lesson2"}]',
            'lesson_count': 2
        }]
    }
    store.course_catalog = catalog_mock
    
    return store

@pytest.fixture
def mock_empty_vector_store():
    """Mock vector store with no data (simulates empty database)"""
    store = Mock()
    
    # All searches return empty results
    store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    store._resolve_course_name.return_value = None
    store.get_existing_course_titles.return_value = []
    store.get_lesson_link.return_value = None
    store.get_course_count.return_value = 0
    store.get_course_titles.return_value = []
    
    # Empty catalog
    catalog_mock = Mock()
    catalog_mock.get.return_value = {'metadatas': []}
    store.course_catalog = catalog_mock
    
    return store

@pytest.fixture
def mock_error_vector_store():
    """Mock vector store that raises errors (simulates database issues)"""
    store = Mock()
    
    # Search returns error results instead of raising exceptions
    store.search.return_value = SearchResults.empty("Database connection error")
    store._resolve_course_name.side_effect = Exception("Database connection error")
    store.get_existing_course_titles.side_effect = Exception("Database connection error")
    store.get_lesson_link.side_effect = Exception("Database connection error")
    
    catalog_mock = Mock()
    catalog_mock.get.side_effect = Exception("Database connection error")
    store.course_catalog = catalog_mock
    
    return store

@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)

@pytest.fixture
def course_outline_tool(mock_vector_store):
    """CourseOutlineTool with mock vector store"""
    return CourseOutlineTool(mock_vector_store)

@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager

@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response for testing"""
    response = Mock()
    response.content = [Mock()]
    response.content[0].text = "This is a test response about MCP."
    response.content[0].type = "text"
    response.stop_reason = "end_turn"
    return response

@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic API response with tool use"""
    response = Mock()
    
    # Tool use block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "introduction"}
    tool_block.id = "tool_123"
    
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response

@pytest.fixture
def mock_anthropic_final_response():
    """Mock final Anthropic API response after tool execution"""
    response = Mock()
    response.content = [Mock()]
    response.content[0].text = "Based on the course materials, MCP (Model Context Protocol) enables AI applications to securely connect to external data sources."
    response.stop_reason = "end_turn"
    return response