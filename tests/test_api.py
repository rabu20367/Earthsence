"""
Test cases for the EarthSense API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app import app

# Initialize test client
client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "EarthSense AI" in response.text

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.parametrize("endpoint", [
    "/api/v1/satellite/images",
    "/api/v1/iot/readings",
    "/api/v1/detections"
])
def test_protected_endpoints(endpoint):
    """Test that protected endpoints require authentication."""
    response = client.get(endpoint)
    assert response.status_code == 401  # Unauthorized

def test_invalid_endpoint():
    """Test accessing a non-existent endpoint."""
    response = client.get("/api/non-existent")
    assert response.status_code == 404

# Add more test cases as needed
