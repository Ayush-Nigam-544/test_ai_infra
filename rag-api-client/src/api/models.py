from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    doc_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentInput(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any] = {}

class AddDocumentsRequest(BaseModel):
    documents: List[DocumentInput]

class RAGQueryRequest(BaseModel):
    query: str

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    retrieved_documents: List[str]  # BentoML returns document content as strings
    context_length: int