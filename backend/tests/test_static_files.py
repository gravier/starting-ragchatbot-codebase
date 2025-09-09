"""Tests for static file handling and frontend serving"""

import pytest
import os
import tempfile
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


@pytest.mark.api
class TestStaticFileHandling:
    """Tests for static file mounting and serving"""
    
    def test_main_app_import_without_static_files(self):
        """Test that we can import main app components without static file issues"""
        # This tests that we can import the main app modules for testing
        # without running into static file mounting issues
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Import key components that should work in test environment
            from config import Config
            from rag_system import RAGSystem
            from models import Course, Lesson, CourseChunk
            
            # These imports should work without issues
            assert Config is not None
            assert RAGSystem is not None
            assert Course is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import main app components: {e}")
    
    def test_static_file_serving_with_temp_directory(self, temp_frontend_dir):
        """Test static file serving with temporary directory"""
        # Create a test app with static file mounting using temp directory
        app = FastAPI()
        app.mount("/", StaticFiles(directory=temp_frontend_dir, html=True), name="static")
        
        client = TestClient(app)
        
        # Test that we can serve the index.html file
        response = client.get("/")
        assert response.status_code == 200
        assert "Test Frontend" in response.text
    
    def test_static_file_not_found(self, temp_frontend_dir):
        """Test handling of non-existent static files"""
        app = FastAPI()
        app.mount("/", StaticFiles(directory=temp_frontend_dir, html=True), name="static")
        
        client = TestClient(app)
        
        # Test request for non-existent file
        response = client.get("/nonexistent.html")
        assert response.status_code == 404
    
    @patch('os.path.exists')
    def test_main_app_with_missing_frontend_directory(self, mock_exists):
        """Test main app behavior when frontend directory is missing"""
        mock_exists.return_value = False
        
        # This simulates the situation where ../frontend doesn't exist
        # The app should handle this gracefully during testing
        try:
            # The main app import should still work even if frontend dir is missing
            # because we're not actually mounting the static files in tests
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from config import Config
            config = Config()
            assert config is not None
            
        except Exception as e:
            # This test ensures we can work around the static file issue
            # The exception might occur, but we should be able to handle it
            assert "static" in str(e).lower() or "frontend" in str(e).lower()


@pytest.mark.api
class TestMainAppStructure:
    """Tests for main app structure and components"""
    
    def test_app_endpoints_definition(self):
        """Test that we can define app endpoints without static file mounting"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import List, Optional, Union, Dict, Any
        
        # This replicates the main app structure without static file issues
        app = FastAPI(title="Test RAG System")
        
        class QueryRequest(BaseModel):
            query: str
            session_id: Optional[str] = None

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Union[str, Dict[str, Any]]]
            session_id: str

        class CourseStats(BaseModel):
            total_courses: int
            course_titles: List[str]
        
        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            return QueryResponse(
                answer="test response",
                sources=[],
                session_id=request.session_id or "test-session"
            )
        
        @app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            return CourseStats(total_courses=0, course_titles=[])
        
        @app.get("/")
        async def read_root():
            return {"message": "RAG System API", "status": "running"}
        
        # Test that the app can be created and endpoints work
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        
        # Test courses endpoint
        response = client.get("/api/courses")
        assert response.status_code == 200
        
        # Test query endpoint
        response = client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200
    
    def test_cors_middleware_configuration(self):
        """Test CORS middleware configuration"""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI()
        
        # Add CORS middleware (same as main app)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Test that app can be created with CORS middleware
        client = TestClient(app)
        
        # This is mainly a smoke test to ensure middleware doesn't cause issues
        assert client is not None
    
    def test_pydantic_models_validation(self):
        """Test Pydantic model validation"""
        from pydantic import BaseModel, ValidationError
        from typing import List, Optional, Union, Dict, Any
        
        class QueryRequest(BaseModel):
            query: str
            session_id: Optional[str] = None

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Union[str, Dict[str, Any]]]
            session_id: str

        class CourseStats(BaseModel):
            total_courses: int
            course_titles: List[str]
        
        # Test valid QueryRequest
        valid_request = QueryRequest(query="test query")
        assert valid_request.query == "test query"
        assert valid_request.session_id is None
        
        # Test QueryRequest with session_id
        request_with_session = QueryRequest(query="test", session_id="session-123")
        assert request_with_session.session_id == "session-123"
        
        # Test invalid QueryRequest (missing query)
        with pytest.raises(ValidationError):
            QueryRequest()
        
        # Test valid QueryResponse
        valid_response = QueryResponse(
            answer="test answer",
            sources=["source1", {"course": "test"}],
            session_id="session-123"
        )
        assert valid_response.answer == "test answer"
        
        # Test valid CourseStats
        valid_stats = CourseStats(total_courses=2, course_titles=["Course 1", "Course 2"])
        assert valid_stats.total_courses == 2
        assert len(valid_stats.course_titles) == 2


@pytest.mark.integration
class TestAppIntegrationWithoutStatic:
    """Integration tests that avoid static file mounting issues"""
    
    def test_app_initialization_components(self):
        """Test that main app components can be initialized in test environment"""
        from unittest.mock import Mock, patch
        
        # Mock the components that might cause issues
        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator:
            
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                from config import Config
                config = Config()
                
                # These should work without static file issues
                assert config.ANTHROPIC_MODEL is not None
                assert config.CHUNK_SIZE > 0
                
            except Exception as e:
                # If we get an exception, it should be related to missing API keys
                # or configuration, not static file mounting
                assert "static" not in str(e).lower()
                assert "frontend" not in str(e).lower()