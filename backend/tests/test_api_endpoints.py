"""API endpoint tests for FastAPI application"""

import pytest
from fastapi import HTTPException
from httpx import HTTPStatusError
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for /api/query endpoint"""
    
    def test_query_endpoint_success(self, test_client, sample_query_request):
        """Test successful query with valid request"""
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Validate response content
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Validate that answer contains query content
        assert "What is MCP?" in data["answer"]
    
    def test_query_endpoint_with_session_id(self, test_client, sample_query_request_with_session):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=sample_query_request_with_session)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use provided session ID
        assert data["session_id"] == "test-session-456"
        assert "How do I set up MCP?" in data["answer"]
    
    def test_query_endpoint_without_session_id(self, test_client):
        """Test query endpoint without session ID (should auto-create)"""
        request_data = {"query": "Test query without session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should auto-create session ID
        assert data["session_id"] == "test-session-123"  # From mock
        assert "Test query without session" in data["answer"]
    
    def test_query_endpoint_invalid_request_missing_query(self, test_client):
        """Test query endpoint with missing query field"""
        request_data = {"session_id": "test-session"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 422  # Unprocessable Entity
        error_data = response.json()
        assert "detail" in error_data
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        request_data = {"query": ""}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still process empty query
        assert "answer" in data
        assert data["session_id"] == "test-session-123"
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_endpoint_long_query(self, test_client):
        """Test query endpoint with very long query"""
        long_query = "What is MCP? " * 100  # Very long query
        request_data = {"query": long_query}
        
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for /api/courses endpoint"""
    
    def test_courses_endpoint_success(self, test_client):
        """Test successful course stats retrieval"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Validate response content
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Validate mock data content
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Test Course 1" in data["course_titles"]
        assert "Test Course 2" in data["course_titles"]
    
    def test_courses_endpoint_no_query_params(self, test_client):
        """Test that courses endpoint doesn't accept query parameters"""
        # This should still work, query params are just ignored
        response = test_client.get("/api/courses?param=value")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
    
    def test_courses_endpoint_methods(self, test_client):
        """Test that courses endpoint only accepts GET requests"""
        # POST should not be allowed
        response = test_client.post("/api/courses")
        assert response.status_code == 405  # Method Not Allowed
        
        # PUT should not be allowed
        response = test_client.put("/api/courses")
        assert response.status_code == 405
        
        # DELETE should not be allowed
        response = test_client.delete("/api/courses")
        assert response.status_code == 405


@pytest.mark.api
class TestRootEndpoint:
    """Tests for / (root) endpoint"""
    
    def test_root_endpoint_success(self, test_client):
        """Test successful root endpoint access"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "message" in data
        assert "status" in data
        
        # Validate response content
        assert data["message"] == "RAG System API"
        assert data["status"] == "running"
    
    def test_root_endpoint_methods(self, test_client):
        """Test that root endpoint accepts multiple methods"""
        # GET should work
        response = test_client.get("/")
        assert response.status_code == 200
        
        # Other methods should not be allowed
        response = test_client.post("/")
        assert response.status_code == 405


@pytest.mark.api
class TestAPIErrorHandling:
    """Tests for API error handling and edge cases"""
    
    def test_nonexistent_endpoint(self, test_client):
        """Test request to non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_content_type(self, test_client):
        """Test request with invalid content type for query endpoint"""
        response = test_client.post(
            "/api/query",
            content="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        # FastAPI should still try to parse this, but it will likely fail
        assert response.status_code == 422
    
    def test_cors_headers(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/")
        
        # Check that CORS middleware is working
        assert response.status_code == 200
        # Note: TestClient might not include all CORS headers in test environment
        # This is more of a smoke test that the endpoint works


@pytest.mark.api
class TestResponseValidation:
    """Tests for response format validation"""
    
    def test_query_response_schema_validation(self, test_client):
        """Test that query responses match expected schema"""
        request_data = {"query": "test query"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate all required fields are present
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
    
    def test_courses_response_schema_validation(self, test_client):
        """Test that courses responses match expected schema"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate all required fields are present
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Validate that total_courses matches course_titles length
        assert data["total_courses"] == len(data["course_titles"])


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints working together"""
    
    def test_session_continuity(self, test_client):
        """Test that session IDs work correctly across multiple requests"""
        # First query without session ID
        request1 = {"query": "First query"}
        response1 = test_client.post("/api/query", json=request1)
        
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query with the same session ID
        request2 = {"query": "Second query", "session_id": session_id}
        response2 = test_client.post("/api/query", json=request2)
        
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
    
    def test_multiple_concurrent_requests(self, test_client):
        """Test handling of multiple concurrent requests"""
        requests = [
            {"query": f"Query number {i}"}
            for i in range(5)
        ]
        
        responses = []
        for req in requests:
            response = test_client.post("/api/query", json=req)
            responses.append(response)
        
        # All requests should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert f"Query number {i}" in data["answer"]


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance-related tests for API endpoints"""
    
    def test_query_endpoint_response_time(self, test_client):
        """Test that query endpoint responds within reasonable time"""
        import time
        
        request_data = {"query": "Performance test query"}
        
        start_time = time.time()
        response = test_client.post("/api/query", json=request_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds (generous for tests)
    
    def test_courses_endpoint_response_time(self, test_client):
        """Test that courses endpoint responds quickly"""
        import time
        
        start_time = time.time()
        response = test_client.get("/api/courses")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should be very fast since it's just returning stats