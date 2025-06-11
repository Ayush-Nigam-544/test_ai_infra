from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import os
from config.settings import settings
from clients.bentoml_client import BentoMLClient

router = APIRouter()

# Initialize BentoML client
bentoml_client = BentoMLClient(settings.bentoml_endpoint)

# Define Pydantic models for request and response validation
class RAGQueryRequest(BaseModel):
    query: str

class DocumentInput(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any] = {}

class AddDocumentsRequest(BaseModel):
    documents: List[DocumentInput]

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    retrieved_documents: List[str]  # BentoML returns strings, not dicts
    context_length: int

@router.post("/health")
async def health_check():
    """Check if the BentoML service is healthy."""
    try:
        response = bentoml_client.health_check()
        return {"status": "healthy", "bentoml_service": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BentoML service unavailable: {str(e)}")

@router.post("/rag_query")
async def rag_query(request: RAGQueryRequest):
    """Endpoint to query the RAG model using the deployed BentoML service."""
    try:
        response = bentoml_client.rag_query(request.query)
        return response
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BentoML service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    """Endpoint to add documents to the RAG service."""
    try:
        # Convert Pydantic models to dictionaries for the BentoML client
        documents_data = [doc.dict() for doc in request.documents]
        response = bentoml_client.add_documents(documents_data)
        return response
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BentoML service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")