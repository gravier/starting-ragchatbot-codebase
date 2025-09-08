"""Tests for RAGSystem functionality"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from rag_system import RAGSystem


class TestRAGSystem:
    """Test cases for RAGSystem"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_init(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test RAGSystem initialization"""
        rag_system = RAGSystem(mock_config)

        assert rag_system.config == mock_config
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_tool_registration(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test that tools are properly registered"""
        rag_system = RAGSystem(mock_config)

        # Check that tool definitions are available
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 2

        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_without_session(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test query processing without session ID"""
        # Setup mocks
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = (
            "Test response about MCP"
        )
        mock_ai_generator.return_value = mock_ai_generator_instance

        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        mock_session_manager_instance.get_conversation_history.return_value = None

        rag_system = RAGSystem(mock_config)

        # Mock tool manager
        rag_system.tool_manager = Mock()
        rag_system.tool_manager.get_tool_definitions.return_value = []
        rag_system.tool_manager.get_last_sources.return_value = []
        rag_system.tool_manager.reset_sources.return_value = None

        response, sources = rag_system.query("What is MCP?")

        assert response == "Test response about MCP"
        assert sources == []

        # Verify AI generator was called correctly
        mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = mock_ai_generator_instance.generate_response.call_args

        assert "What is MCP?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_with_session(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test query processing with session ID"""
        # Setup mocks
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = (
            "Test response with history"
        )
        mock_ai_generator.return_value = mock_ai_generator_instance

        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        mock_session_manager_instance.get_conversation_history.return_value = (
            "Previous conversation history"
        )

        rag_system = RAGSystem(mock_config)

        # Mock tool manager
        rag_system.tool_manager = Mock()
        rag_system.tool_manager.get_tool_definitions.return_value = []
        rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "Test Source", "url": None}
        ]
        rag_system.tool_manager.reset_sources.return_value = None

        response, sources = rag_system.query("What is MCP?", session_id="test_session")

        assert response == "Test response with history"
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Source"

        # Verify session manager was used
        mock_session_manager_instance.get_conversation_history.assert_called_with(
            "test_session"
        )
        mock_session_manager_instance.add_exchange.assert_called_with(
            "test_session", "What is MCP?", "Test response with history"
        )

        # Verify AI generator received history
        call_args = mock_ai_generator_instance.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation history"

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_with_tool_usage(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
        tool_manager,
    ):
        """Test query processing that uses tools"""
        # Setup mocks
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = (
            "MCP enables secure connections to external data sources"
        )
        mock_ai_generator.return_value = mock_ai_generator_instance

        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        mock_session_manager_instance.get_conversation_history.return_value = None

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = tool_manager

        response, sources = rag_system.query("What is MCP?")

        assert response == "MCP enables secure connections to external data sources"

        # Verify tools were available to AI generator
        call_args = mock_ai_generator_instance.generate_response.call_args
        assert call_args[1]["tools"] == tool_manager.get_tool_definitions()
        assert call_args[1]["tool_manager"] == tool_manager

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test course analytics retrieval"""
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.get_course_count.return_value = 3
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
        ]
        mock_vector_store.return_value = mock_vector_store_instance

        rag_system = RAGSystem(mock_config)

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_add_course_document(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test adding a single course document"""
        mock_doc_processor_instance = Mock()
        mock_doc_processor_instance.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )
        mock_doc_processor.return_value = mock_doc_processor_instance

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        rag_system = RAGSystem(mock_config)

        course, chunk_count = rag_system.add_course_document("test_course.pdf")

        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        # Verify vector store methods were called
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(
            sample_course
        )
        mock_vector_store_instance.add_course_content.assert_called_once_with(
            sample_course_chunks
        )


class TestRAGSystemErrorHandling:
    """Test RAGSystem error handling"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_ai_generator_error(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test handling of AI generator errors"""
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.side_effect = Exception(
            "API Error: Authentication failed"
        )
        mock_ai_generator.return_value = mock_ai_generator_instance

        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        mock_session_manager_instance.get_conversation_history.return_value = None

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = Mock()
        rag_system.tool_manager.get_tool_definitions.return_value = []

        with pytest.raises(Exception) as exc_info:
            rag_system.query("What is MCP?")

        assert "API Error: Authentication failed" in str(exc_info.value)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_vector_store_error(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test handling of vector store errors"""
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.get_course_count.side_effect = Exception(
            "Database connection error"
        )
        mock_vector_store.return_value = mock_vector_store_instance

        rag_system = RAGSystem(mock_config)

        with pytest.raises(Exception) as exc_info:
            rag_system.get_course_analytics()

        assert "Database connection error" in str(exc_info.value)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_tool_manager_error(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test handling of tool manager errors"""
        mock_ai_generator_instance = Mock()
        mock_ai_generator_instance.generate_response.return_value = (
            "Response despite tool error"
        )
        mock_ai_generator.return_value = mock_ai_generator_instance

        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        mock_session_manager_instance.get_conversation_history.return_value = None

        rag_system = RAGSystem(mock_config)

        # Mock tool manager to raise error
        rag_system.tool_manager = Mock()
        rag_system.tool_manager.get_tool_definitions.side_effect = Exception(
            "Tool registration error"
        )

        with pytest.raises(Exception) as exc_info:
            rag_system.query("What is MCP?")

        assert "Tool registration error" in str(exc_info.value)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_document_processing_error(
        self,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_doc_processor,
        mock_config,
    ):
        """Test handling of document processing errors"""
        mock_doc_processor_instance = Mock()
        mock_doc_processor_instance.process_course_document.side_effect = Exception(
            "Document parsing error"
        )
        mock_doc_processor.return_value = mock_doc_processor_instance

        rag_system = RAGSystem(mock_config)

        course, chunk_count = rag_system.add_course_document("invalid_file.pdf")

        # Should return None and 0 on error, not raise exception
        assert course is None
        assert chunk_count == 0


class TestRAGSystemIntegration:
    """Integration tests for RAGSystem with real components"""

    def test_prompt_construction(self, mock_config):
        """Test that query prompts are constructed correctly"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):

            mock_ai_gen_instance = Mock()
            mock_ai_gen_instance.generate_response.return_value = "Test response"
            mock_ai_gen.return_value = mock_ai_gen_instance

            rag_system = RAGSystem(mock_config)
            rag_system.tool_manager = Mock()
            rag_system.tool_manager.get_tool_definitions.return_value = []
            rag_system.tool_manager.get_last_sources.return_value = []
            rag_system.tool_manager.reset_sources.return_value = None

            rag_system.query("What is MCP?")

            # Check that prompt was constructed correctly
            call_args = mock_ai_gen_instance.generate_response.call_args
            query_prompt = call_args[1]["query"]

            assert (
                "Answer this question about course materials: What is MCP?"
                in query_prompt
            )

    def test_source_management(self, mock_config):
        """Test that sources are properly managed"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):

            mock_ai_gen_instance = Mock()
            mock_ai_gen_instance.generate_response.return_value = "Test response"
            mock_ai_gen.return_value = mock_ai_gen_instance

            rag_system = RAGSystem(mock_config)

            # Mock tool manager with sources
            rag_system.tool_manager = Mock()
            rag_system.tool_manager.get_tool_definitions.return_value = []
            rag_system.tool_manager.get_last_sources.return_value = [
                {"text": "Test Source", "url": "http://example.com"}
            ]
            rag_system.tool_manager.reset_sources.return_value = None

            response, sources = rag_system.query("What is MCP?")

            # Verify sources were retrieved and reset
            rag_system.tool_manager.get_last_sources.assert_called_once()
            rag_system.tool_manager.reset_sources.assert_called_once()

            assert len(sources) == 1
            assert sources[0]["text"] == "Test Source"
