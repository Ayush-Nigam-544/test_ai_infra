import requests
from typing import Any, Dict, List

class BentoMLClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict[str, Any]:
        """Check if the BentoML service is healthy."""
        response = requests.post(f"{self.base_url}/health_check")
        response.raise_for_status()
        return response.json()

    def rag_query(self, query: str) -> Dict[str, Any]:
        """Query the RAG service with the correct request format."""
        payload = {"request": {"query": query}}
        response = requests.post(f"{self.base_url}/rag_query", json=payload)
        response.raise_for_status()
        return response.json()

    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Add documents to the RAG service with the correct request format."""
        payload = {"request": {"documents": documents}}
        response = requests.post(f"{self.base_url}/add_documents", json=payload)
        response.raise_for_status()
        return response.json()