from __future__ import annotations

import bentoml
from typing import List, Dict, Any
import json
import os
import time
from pydantic import BaseModel

# Input models
class DocumentInput(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class AddDocumentsRequest(BaseModel):
    documents: List[DocumentInput]

class QueryRequest(BaseModel):
    query: str

@bentoml.service(
    resources={
        "cpu": "1",
        "memory": "2Gi"
    },
    scaling={
        "min_replicas": 1,
        "max_replicas": 2
    }
)
class RAGService:
    
    def __init__(self) -> None:
        print("🚀 Initializing RAG Service...")

        try:
            # Initialize AWS clients with error handling
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            region = os.getenv('AWS_REGION', 'us-east-1')
            
            print(f"Connecting to AWS region: {region}")
            
            import boto3
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            
            self.dynamodb = boto3.resource(
                'dynamodb',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            
            print("✅ AWS connections initialized")
            
        except Exception as e:
            print(f"⚠️ AWS initialization warning: {e}")
            self.s3_client = None
            self.dynamodb = None

        # AWS configuration
        self.s3_bucket = os.getenv('S3_BUCKET_NAME', 'bentoml-rag-storage-ayush-2024')
        self.dynamodb_table_name = os.getenv('DYNAMODB_TABLE_NAME', 'rag-document-metadata')
        
        # Force CPU usage to minimize costs
        device = "cpu"
        print(f"Using device: {device}")
        
        try:
            print("📥 Loading embedding model...")
            
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
            import torch
            
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=device
            )
            print("✅ Embedding model loaded")
            
            print("📥 Loading QA model...")
            self.qa_pipeline = pipeline(
                'question-answering',
                model="distilbert-base-cased-distilled-squad",
                device=device,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("✅ QA model loaded")
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            # Fallback: continue without models for basic functionality
            self.embedding_model = None
            self.qa_pipeline = None
        
        # Initialize FAISS index
        try:
            import faiss
            self.vector_dimension = 384
            self.faiss_index = faiss.IndexFlatL2(self.vector_dimension)
            print("✅ FAISS index initialized")
        except Exception as e:
            print(f"❌ FAISS initialization error: {e}")
            self.faiss_index = None
        
        # Document storage
        self.documents = []
        
        # Load existing data (now with proper methods)
        self._load_vectors_from_s3()
        self._load_documents_from_dynamodb()
        
        print(f"✅ RAG Service initialized with {len(self.documents)} documents")

    def _load_vectors_from_s3(self):
        """Load FAISS index from S3"""
        if not self.s3_client or not self.faiss_index:
            print("S3 or FAISS not available, skipping vector loading")
            return
            
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key='faiss_index/index.faiss'
            )
            
            # Save to temporary file
            with open('/tmp/index.faiss', 'wb') as f:
                f.write(response['Body'].read())
            
            # Load FAISS index
            import faiss
            self.faiss_index = faiss.read_index('/tmp/index.faiss')
            print("✅ FAISS index loaded from S3")
            
        except Exception as e:
            print(f"ℹ️ No existing FAISS index found in S3: {e}")

    def _save_vectors_to_s3(self):
        """Save FAISS index to S3"""
        if not self.s3_client or not self.faiss_index or self.faiss_index.ntotal == 0:
            print("S3 or FAISS not available, skipping vector save")
            return
            
        try:
            import faiss
            
            # Write to temporary file
            faiss.write_index(self.faiss_index, '/tmp/index.faiss')
            
            # Upload to S3
            self.s3_client.upload_file(
                '/tmp/index.faiss',
                self.s3_bucket,
                'faiss_index/index.faiss'
            )
            print("✅ FAISS index saved to S3")
            
        except Exception as e:
            print(f"❌ Error saving FAISS index to S3: {e}")

    def _load_documents_from_dynamodb(self):
        """Load document metadata from DynamoDB"""
        if not self.dynamodb:
            print("DynamoDB not available, skipping document loading")
            return
            
        try:
            table = self.dynamodb.Table(self.dynamodb_table_name)
            response = table.scan(Limit=100)
            
            for item in response['Items']:
                content = item.get('content', '')
                self.documents.append(content)
            
            print(f"✅ Loaded {len(self.documents)} documents from DynamoDB")
            
        except Exception as e:
            print(f"ℹ️ No existing documents in DynamoDB: {e}")

    def _save_document_to_dynamodb(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Save document metadata to DynamoDB"""
        if not self.dynamodb:
            print("DynamoDB not available, skipping save")
            return
            
        try:
            table = self.dynamodb.Table(self.dynamodb_table_name)
            table.put_item(
                Item={
                    'doc_id': doc_id,
                    'content': content[:1000],
                    'metadata': json.dumps(metadata),
                    'timestamp': str(int(time.time()))
                }
            )
            print(f"✅ Document {doc_id} saved to DynamoDB")
            
        except Exception as e:
            print(f"❌ Error saving document to DynamoDB: {e}")

    def _store_document_in_s3(self, doc_id: str, content: str):
        """Store full document content in S3"""
        if not self.s3_client:
            print("S3 not available, skipping storage")
            return
            
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f'documents/{doc_id}.txt',
                Body=content.encode('utf-8')
            )
            print(f"✅ Document {doc_id} stored in S3")
            
        except Exception as e:
            print(f"❌ Error storing document in S3: {e}")

    def _retrieve_similar_documents(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieve similar documents using vector search"""
        if not self.faiss_index or self.faiss_index.ntotal == 0 or not self.embedding_model:
            print("Vector search not available, returning recent documents")
            return self.documents[-top_k:] if self.documents else []
        
        try:
            import numpy as np
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search for similar vectors
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # Return corresponding documents
            retrieved_docs = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    if len(doc) > 500:
                        doc = doc[:500] + "..."
                    retrieved_docs.append(doc)
            
            return retrieved_docs
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return self.documents[-top_k:] if self.documents else []

    @bentoml.api
    def add_documents(self, request: AddDocumentsRequest) -> Dict[str, str]:
        """Add documents to the vector store"""
        import uuid
        
        documents = request.documents[:5]  # Limit to 5 documents
        
        added_count = 0
        for doc in documents:
            try:
                doc_id = str(uuid.uuid4())
                content = doc.content
                metadata = doc.metadata
                
                # Truncate long content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                # Generate embedding if model available
                if self.embedding_model and self.faiss_index:
                    try:
                        embedding = self.embedding_model.encode([content])
                        self.faiss_index.add(embedding.astype('float32'))
                        print("✅ Embedding added to FAISS")
                    except Exception as e:
                        print(f"⚠️ Embedding failed: {e}")
                
                # Store document
                self.documents.append(content)
                
                # Save to AWS services
                self._save_document_to_dynamodb(doc_id, content, metadata)
                self._store_document_in_s3(doc_id, content)
                
                added_count += 1
                print(f"✅ Added document {added_count}: {content[:50]}...")
                
            except Exception as e:
                print(f"❌ Error adding document: {e}")
        
        # Save updated FAISS index
        if added_count > 0:
            self._save_vectors_to_s3()
        
        return {"message": f"Added {added_count} documents successfully"}

    @bentoml.api
    def rag_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Perform RAG query"""
        query = request.query
        
        # Truncate long queries
        if len(query) > 200:
            query = query[:200]
        
        print(f"Processing query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self._retrieve_similar_documents(query, top_k=2)
        
        # Prepare context
        context = " ".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
        if len(context) > 512:
            context = context[:512] + "..."
        
        # Generate answer
        try:
            if retrieved_docs and self.qa_pipeline:
                qa_result = self.qa_pipeline(question=query, context=context)
                answer = qa_result['answer']
                confidence = qa_result['score']
            else:
                # Fallback response
                if retrieved_docs:
                    answer = f"Based on the available documents: {context[:100]}..."
                else:
                    answer = "No relevant information found in the knowledge base."
                confidence = 0.5
                
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            answer = f"Found some relevant information but couldn't process it properly."
            confidence = 0.3
        
        return {
            "query": query,
            "answer": answer,
            "confidence": round(confidence, 3),
            "retrieved_documents": retrieved_docs,
            "context_length": len(context)
        }

    @bentoml.api
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service_type": "rag_service",
            "documents_count": len(self.documents),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "aws_connected": self.s3_client is not None,
            "models_loaded": {
                "embedding_model": self.embedding_model is not None,
                "qa_model": self.qa_pipeline is not None
            },
            "aws_services": {
                "s3_bucket": self.s3_bucket,
                "dynamodb_table": self.dynamodb_table_name
            }
        }