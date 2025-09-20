import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_endpoint():
    # Use a sample image from samples/ folder
    with open("samples/sample1.jpeg", "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample1.jpeg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    # Check required fields
    assert "observation" in data
    assert "confidence" in data
    assert "confidence_display" in data
    assert "severity" in data
    assert "advice" in data
    assert "generation_explanation" in data
    assert "regions_of_concern" in data
    assert "raw_model_response" in data
