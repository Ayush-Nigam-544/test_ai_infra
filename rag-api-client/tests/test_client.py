import pytest
import requests
from src.clients.bentoml_client import RagClient

@pytest.fixture
def rag_client():
    return RagClient(base_url="http://localhost:8000")

def test_rag_query(rag_client):
    query = "What is the capital of France?"
    response = rag_client.rag_query(query)
    
    assert response is not None
    assert "answer" in response
    assert "confidence" in response
    assert response["query"] == query

def test_health_check(rag_client):
    response = rag_client.health_check()
    
    assert response is not None
    assert response["status"] == "healthy"
    assert "documents_count" in response
    assert "faiss_index_size" in response
    assert "aws_connected" in response
    assert "aws_services" in response