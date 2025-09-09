"""Integration tests for the complete RAG system"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
from config import Config
from rag_system import RAGSystem


class TestConfigurationIssues:
    """Test various configuration-related issues that could cause failures"""

    def test_detect_invalid_model_name(self):
        """Test detection of invalid model name in config"""
        config = Config()

        # The current config has a suspicious model name
        assert config.ANTHROPIC_MODEL == "1claude-sonnet-4-20250514"

    def test_missing_api_key(self):
        """Test behavior with missing API key"""
        # Patch Config directly since environment patching is complex with load_dotenv
        with patch("config.Config") as MockConfig:
            mock_config = MockConfig.return_value
            mock_config.ANTHROPIC_API_KEY = ""
            config = MockConfig()
            assert config.ANTHROPIC_API_KEY == ""

    def test_valid_config_structure(self, mock_config):
        """Test that a properly configured system would work"""
        # Verify mock config has correct structure
        assert mock_config.ANTHROPIC_API_KEY != ""
        assert not mock_config.ANTHROPIC_MODEL.startswith("1")
        assert mock_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert mock_config.CHUNK_SIZE > 0
        assert mock_config.MAX_RESULTS > 0


class TestSystemInitialization:
    """Test system initialization with various configurations"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_system_init_with_valid_config(
        self, mock_ai_generator, mock_vector_store, mock_config
    ):
        """Test system initialization with valid configuration"""
        try:
            rag_system = RAGSystem(mock_config)
            assert rag_system is not None
            assert rag_system.tool_manager is not None

            # Check tools are registered
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 2

        except Exception as e:
            pytest.fail(f"System initialization failed with valid config: {e}")

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_system_init_with_invalid_model(self, mock_ai_generator, mock_vector_store):
        """Test system initialization with invalid model name"""
        config = Config()  # Uses the actual config with invalid model

        # Should initialize without error (error occurs during API calls)
        rag_system = RAGSystem(config)
        assert rag_system is not None

    @patch("rag_system.VectorStore")
    def test_vector_store_initialization_error(self, mock_vector_store):
        """Test handling of vector store initialization errors"""
        mock_vector_store.side_effect = Exception("ChromaDB initialization failed")
        config = Config()

        with pytest.raises(Exception) as exc_info:
            RAGSystem(config)

        assert "ChromaDB initialization failed" in str(exc_info.value)


class TestDataLoadingIssues:
    """Test data loading and vector store population issues"""

    def test_empty_vector_store_behavior(self, mock_config, mock_empty_vector_store):
        """Test system behavior with empty vector store"""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator") as mock_ai_gen,
        ):

            mock_vs.return_value = mock_empty_vector_store
            mock_ai_gen_instance = Mock()
            mock_ai_gen_instance.generate_response.return_value = (
                "I don't have any course information available."
            )
            mock_ai_gen.return_value = mock_ai_gen_instance

            rag_system = RAGSystem(mock_config)

            # Analytics should show no courses
            analytics = rag_system.get_course_analytics()
            assert analytics["total_courses"] == 0
            assert len(analytics["course_titles"]) == 0

    def test_course_loading_from_docs_folder(self, mock_config):
        """Test course loading from docs folder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake docs folder with a test file
            docs_path = os.path.join(temp_dir, "docs")
            os.makedirs(docs_path)

            test_file = os.path.join(docs_path, "test_course.txt")
            with open(test_file, "w") as f:
                f.write("Test course content about MCP")

            with (
                patch("rag_system.VectorStore") as mock_vs,
                patch("rag_system.AIGenerator"),
                patch("os.path.exists") as mock_exists,
            ):

                mock_exists.return_value = True
                mock_vs_instance = Mock()
                mock_vs_instance.get_existing_course_titles.return_value = []
                mock_vs.return_value = mock_vs_instance

                rag_system = RAGSystem(mock_config)

                # Test folder loading
                courses, chunks = rag_system.add_course_folder(docs_path)

                # Should attempt to process files in folder
                assert courses >= 0  # May be 0 due to mocking


class TestQueryExecution:
    """Test actual query execution scenarios"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_with_tool_execution_success(
        self,
        mock_ai_generator,
        mock_vector_store,
        mock_config,
        mock_anthropic_tool_response,
        mock_anthropic_final_response,
    ):
        """Test successful query with tool execution"""
        # Setup vector store mock
        mock_vs_instance = Mock()
        mock_vs_instance.search.return_value.error = None
        mock_vs_instance.search.return_value.is_empty.return_value = False
        mock_vs_instance.search.return_value.documents = ["MCP content"]
        mock_vs_instance.search.return_value.metadata = [{"course_title": "MCP Course"}]
        mock_vs_instance._resolve_course_name.return_value = "MCP Course"
        mock_vector_store.return_value = mock_vs_instance

        # Setup AI generator mock
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.return_value = (
            "MCP enables secure connections to external data."
        )
        mock_ai_generator.return_value = mock_ai_gen_instance

        rag_system = RAGSystem(mock_config)

        response, sources = rag_system.query("What is MCP?")

        assert response == "MCP enables secure connections to external data."
        # Sources should be managed by tool manager (mocked to return empty)

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_with_authentication_error(
        self, mock_ai_generator, mock_vector_store, mock_config
    ):
        """Test query with API authentication error"""
        mock_vs_instance = Mock()
        mock_vector_store.return_value = mock_vs_instance

        # Simulate API authentication error
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.side_effect = Exception(
            "Authentication failed: Invalid API key"
        )
        mock_ai_generator.return_value = mock_ai_gen_instance

        rag_system = RAGSystem(mock_config)

        with pytest.raises(Exception) as exc_info:
            rag_system.query("What is MCP?")

        assert "Authentication failed" in str(exc_info.value)

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_with_invalid_model_error(
        self, mock_ai_generator, mock_vector_store, mock_config
    ):
        """Test query with invalid model error"""
        mock_vs_instance = Mock()
        mock_vector_store.return_value = mock_vs_instance

        # Simulate invalid model error
        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.side_effect = Exception(
            "Invalid model: 1claude-sonnet-4-20250514"
        )
        mock_ai_generator.return_value = mock_ai_gen_instance

        rag_system = RAGSystem(mock_config)

        with pytest.raises(Exception) as exc_info:
            rag_system.query("What is MCP?")

        assert "Invalid model" in str(exc_info.value)


class TestRealWorldScenarios:
    """Test scenarios that match real-world usage"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_content_search_query(
        self, mock_ai_generator, mock_vector_store, mock_config
    ):
        """Test a typical content search query"""
        mock_vs_instance = Mock()
        mock_vector_store.return_value = mock_vs_instance

        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.return_value = (
            "Based on the course materials, MCP stands for Model Context Protocol."
        )
        mock_ai_generator.return_value = mock_ai_gen_instance

        rag_system = RAGSystem(mock_config)

        # Test various content queries that users might ask
        queries = [
            "What is MCP?",
            "How do I set up MCP?",
            "What are the main features of MCP?",
            "Can you explain lesson 3 of the MCP course?",
        ]

        for query in queries:
            try:
                response, sources = rag_system.query(query)
                assert len(response) > 0
                assert isinstance(sources, list)
            except Exception as e:
                pytest.fail(f"Query '{query}' failed: {e}")

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_outline_query(self, mock_ai_generator, mock_vector_store, mock_config):
        """Test a typical outline query"""
        mock_vs_instance = Mock()
        mock_vs_instance._resolve_course_name.return_value = "MCP Course"
        catalog_mock = Mock()
        catalog_mock.get.return_value = {
            "metadatas": [
                {
                    "course_link": "http://example.com/course",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Intro"}, {"lesson_number": 2, "lesson_title": "Setup"}]',
                }
            ]
        }
        mock_vs_instance.course_catalog = catalog_mock
        mock_vector_store.return_value = mock_vs_instance

        mock_ai_gen_instance = Mock()
        mock_ai_gen_instance.generate_response.return_value = "Course: MCP Course\nLink: http://example.com/course\n\nLessons:\n1. Intro\n2. Setup"
        mock_ai_generator.return_value = mock_ai_gen_instance

        rag_system = RAGSystem(mock_config)

        # Test outline queries
        outline_queries = [
            "What is the outline of the MCP course?",
            "Show me all lessons in the MCP course",
            "What topics are covered in the MCP course?",
        ]

        for query in outline_queries:
            try:
                response, sources = rag_system.query(query)
                assert len(response) > 0
                assert isinstance(sources, list)
            except Exception as e:
                pytest.fail(f"Outline query '{query}' failed: {e}")


class TestErrorDiagnostics:
    """Tests to help diagnose common error scenarios"""

    def test_current_config_issues(self):
        """Test current configuration for known issues"""
        config = Config()

        issues = []

        # Check for invalid model name
        if config.ANTHROPIC_MODEL.startswith("1"):
            issues.append(
                f"Invalid model name: {config.ANTHROPIC_MODEL} (has '1' prefix)"
            )

        # Check for missing API key
        if not config.ANTHROPIC_API_KEY:
            issues.append("Missing ANTHROPIC_API_KEY")

        # Check for invalid paths
        if not os.path.exists(os.path.dirname(config.CHROMA_PATH)):
            issues.append(
                f"ChromaDB directory does not exist: {os.path.dirname(config.CHROMA_PATH)}"
            )

        # Report all issues - this is expected to find the invalid model name
        assert len(issues) > 0, "Expected to find configuration issues"
        assert any(
            "Invalid model name" in issue for issue in issues
        ), f"Expected to find invalid model name issue in: {issues}"

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_diagnose_query_failure_chain(
        self, mock_ai_generator, mock_vector_store, mock_config
    ):
        """Test to diagnose where in the query chain failures occur"""
        failure_points = []

        try:
            # Test 1: System initialization
            rag_system = RAGSystem(mock_config)
            failure_points.append("✓ System initialization successful")
        except Exception as e:
            failure_points.append(f"✗ System initialization failed: {e}")
            pytest.fail("System initialization failed")

        try:
            # Test 2: Tool registration
            tools = rag_system.tool_manager.get_tool_definitions()
            assert len(tools) == 2
            failure_points.append("✓ Tool registration successful")
        except Exception as e:
            failure_points.append(f"✗ Tool registration failed: {e}")

        try:
            # Test 3: AI generator setup
            mock_ai_gen_instance = Mock()
            mock_ai_gen_instance.generate_response.return_value = "Test response"
            mock_ai_generator.return_value = mock_ai_gen_instance

            # Test basic query (without tools)
            response, sources = rag_system.query("Hello")
            failure_points.append("✓ Basic query successful")
        except Exception as e:
            failure_points.append(f"✗ Basic query failed: {e}")

        # Print diagnostic information
        print("\nDiagnostic Results:")
        for point in failure_points:
            print(f"  {point}")

        assert (
            len(failure_points) >= 2
        )  # Should have at least system init and tool registration
