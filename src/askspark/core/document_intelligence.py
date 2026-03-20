import os
import io
import PyPDF2
from docx import Document
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from .ai_providers import UnifiedAIClient
from ..config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    content: str
    source: str
    page_number: Optional[int] = None
    chunk_id: str = ""
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Represents a search result from RAG"""
    content: str
    source: str
    score: float
    page_number: Optional[int] = None

class DocumentProcessor:
    """Handles document processing for multiple formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF file with page numbers"""
        text_by_page = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_by_page.append((text, page_num + 1))
                        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            
        return text_by_page
    
    def extract_text_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from DOCX file"""
        text_by_page = []
        
        try:
            doc = Document(file_path)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            # Combine all text as single page for DOCX
            if full_text:
                text_by_page.append(('\n'.join(full_text), 1))
                
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            
        return text_by_page
    
    def extract_text_from_txt(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from TXT file"""
        text_by_page = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if text.strip():
                    text_by_page.append((text, 1))
                    
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {str(e)}")
            
        return text_by_page
    
    def process_document(self, file_path: str) -> List[Tuple[str, int]]:
        """Process document based on file extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

class RAGEngine:
    """Retrieval-Augmented Generation engine for document Q&A"""
    
    def __init__(self):
        self.client = UnifiedAIClient()
        self.processor = DocumentProcessor()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="documents")
        self.processed_documents = {}
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def process_and_store_document(self, file_path: str, document_id: str = None) -> str:
        """Process document and store in vector database"""
        if document_id is None:
            document_id = os.path.basename(file_path)
        
        # Extract text from document
        text_pages = self.processor.process_document(file_path)
        
        if not text_pages:
            raise ValueError(f"No text extracted from {file_path}")
        
        # Create chunks
        all_chunks = []
        chunk_id = 0
        
        for page_text, page_num in text_pages:
            chunks = self.chunk_text(page_text)
            
            for chunk in chunks:
                chunk_obj = DocumentChunk(
                    content=chunk,
                    source=document_id,
                    page_number=page_num,
                    chunk_id=f"{document_id}_{chunk_id}"
                )
                all_chunks.append(chunk_obj)
                chunk_id += 1
        
        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Store in ChromaDB
        documents = [chunk.content for chunk in all_chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id
            }
            for chunk in all_chunks
        ]
        ids = [chunk.chunk_id for chunk in all_chunks]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        # Store in memory
        self.processed_documents[document_id] = all_chunks
        
        logger.info(f"Processed and stored {len(all_chunks)} chunks from {document_id}")
        return document_id
    
    def search_documents(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Search for relevant document chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        for i in range(len(results['documents'][0])):
            result = SearchResult(
                content=results['documents'][0][i],
                source=results['metadatas'][0][i]['source'],
                score=results['distances'][0][i],
                page_number=results['metadatas'][0][i].get('page_number')
            )
            search_results.append(result)
        
        return search_results
    
    def generate_answer(self, query: str, context_results: List[SearchResult], 
                       provider: str = "openai", model: str = "gpt-3.5-turbo") -> str:
        """Generate answer using retrieved context"""
        # Format context
        context_text = "\n\n".join([
            f"From {result.source} (Page {result.page_number}): {result.content}"
            for result in context_results
        ])
        
        # Create prompt
        prompt = f"""You are an AI assistant helping with document analysis. 
Use the following context to answer the user's question accurately and comprehensively.

Context:
{context_text}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to answer the question completely, please indicate what information is available and what might be missing."""

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specialized in document analysis."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.call_model(provider, model, messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def query_documents(self, query: str, provider: str = "openai", 
                       model: str = "gpt-3.5-turbo", n_results: int = 5) -> Dict:
        """Complete RAG pipeline: search and generate answer"""
        # Search for relevant documents
        search_results = self.search_documents(query, n_results)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "query": query
            }
        
        # Generate answer
        answer = self.generate_answer(query, search_results, provider, model)
        
        # Format sources
        sources = [
            {
                "source": result.source,
                "page": result.page_number,
                "score": round(result.score, 3),
                "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            for result in search_results
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }
    
    def summarize_document(self, document_id: str, provider: str = "openai", 
                          model: str = "gpt-3.5-turbo") -> Dict:
        """Generate comprehensive summary of a document"""
        if document_id not in self.processed_documents:
            raise ValueError(f"Document {document_id} not found")
        
        chunks = self.processed_documents[document_id]
        
        # Combine chunks for summary
        full_text = "\n\n".join([chunk.content for chunk in chunks])
        
        # Truncate if too long
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."
        
        prompt = f"""Please provide a comprehensive summary of the following document. 
Include:
1. Main topics and themes
2. Key points and findings
3. Important conclusions or recommendations
4. Overall purpose and audience

Document content:
{full_text}"""

        messages = [
            {"role": "system", "content": "You are an expert at document analysis and summarization."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.call_model(provider, model, messages)
            
            return {
                "document_id": document_id,
                "summary": response.content,
                "total_chunks": len(chunks),
                "provider": provider,
                "model": model
            }
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return {"error": str(e)}
    
    def get_document_insights(self, document_id: str, provider: str = "openai", 
                             model: str = "gpt-3.5-turbo") -> Dict:
        """Extract key insights from document"""
        if document_id not in self.processed_documents:
            raise ValueError(f"Document {document_id} not found")
        
        chunks = self.processed_documents[document_id]
        
        # Combine chunks for analysis
        full_text = "\n\n".join([chunk.content for chunk in chunks])
        
        # Truncate if too long
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."
        
        prompt = f"""Analyze the following document and provide key insights:
1. Main themes and topics
2. Key entities mentioned (people, organizations, dates)
3. Important data points or statistics
4. Action items or recommendations
5. Potential questions this document answers

Document content:
{full_text}"""

        messages = [
            {"role": "system", "content": "You are an expert at extracting insights from business documents."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.call_model(provider, model, messages)
            
            return {
                "document_id": document_id,
                "insights": response.content,
                "total_chunks": len(chunks),
                "provider": provider,
                "model": model
            }
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return {"error": str(e)}
    
    def list_documents(self) -> List[str]:
        """List all processed documents"""
        return list(self.processed_documents.keys())
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the system"""
        try:
            # Get chunk IDs for the document
            if document_id in self.processed_documents:
                chunk_ids = [chunk.chunk_id for chunk in self.processed_documents[document_id]]
                
                # Delete from ChromaDB
                self.collection.delete(ids=chunk_ids)
                
                # Delete from memory
                del self.processed_documents[document_id]
                
                logger.info(f"Deleted document {document_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
