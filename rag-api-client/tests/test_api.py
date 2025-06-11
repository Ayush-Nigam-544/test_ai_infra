from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service_type": "lightweight_rag",
        "documents_count": 0,  # Adjust based on initial state
        "faiss_index_size": 0,  # Adjust based on initial state
        "aws_connected": False,  # Adjust based on initial state
        "aws_services": {
            "s3_bucket": "bentoml-rag-storage",  # Adjust based on your settings
            "dynamodb_table": "rag-document-metadata"  # Adjust based on your settings
        }
    }

def test_rag_query():
    query = "What is the capital of France?"
    response = client.post("/rag_query", json={"query": query})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "confidence" in response.json()
    assert "retrieved_documents" in response.json()