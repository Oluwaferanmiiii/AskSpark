"""
Unit tests for Document Intelligence (RAG Engine)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.askspark.core.document_intelligence import RAGEngine, DocumentChunk


class TestDocumentChunk:
    """Test DocumentChunk dataclass"""
    
    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk"""
        chunk = DocumentChunk(
            content="This is a test document chunk about AI.",
            source="test_document.pdf",
            page_number=1,
            chunk_id="chunk_1"
        )
        
        assert chunk.content == "This is a test document chunk about AI."
        assert chunk.source == "test_document.pdf"
        assert chunk.page_number == 1
        assert chunk.chunk_id == "chunk_1"
        assert chunk.embedding is None


class TestRAGEngine:
    """Test RAG Engine"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_pdf_path(self, temp_dir):
        """Create a sample PDF file for testing"""
        pdf_path = temp_dir / "test_document.pdf"
        
        # Create a mock PDF content
        mock_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000261 00000 n\ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n345\n%%EOF"
        
        with open(pdf_path, 'wb') as f:
            f.write(mock_pdf_content)
        
        return pdf_path
    
    @pytest.fixture
    def sample_txt_path(self, temp_dir):
        """Create a sample TXT file for testing"""
        txt_path = temp_dir / "test_document.txt"
        
        content = """Artificial Intelligence (AI) is transforming businesses worldwide.
        
Machine Learning, a subset of AI, enables systems to learn from data.
Deep Learning uses neural networks to process complex patterns.
Natural Language Processing allows computers to understand human language.
Computer Vision enables machines to interpret visual information.
        
These technologies are revolutionizing industries from healthcare to finance."""
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return txt_path
    
    @pytest.fixture
    def rag_engine_with_mocks(self):
        """Create RAGEngine with mocked dependencies"""
        with patch('src.askspark.core.document_intelligence.chromadb'), \
             patch('src.askspark.core.document_intelligence.SentenceTransformer'), \
             patch('src.askspark.core.document_intelligence.UnifiedAIClient'):
            
            engine = RAGEngine()
            
            # Mock the vector store and embedding model
            engine.vector_store = Mock()
            engine.embedding_model = Mock()
            engine.ai_client = Mock()
            
            # Mock embedding generation
            engine.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] * 34]  # 1024 dimensions
            
            return engine
    
    def test_engine_initialization(self, rag_engine_with_mocks):
        """Test engine initialization"""
        assert rag_engine_with_mocks.vector_store is not None
        assert rag_engine_with_mocks.embedding_model is not None
        assert rag_engine_with_mocks.ai_client is not None
        assert rag_engine_with_mocks.processed_documents == {}
    
    def test_process_text_file(self, rag_engine_with_mocks, sample_txt_path):
        """Test processing a text file"""
        doc_id = rag_engine_with_mocks.process_and_store_document(str(sample_txt_path))
        
        assert doc_id is not None
        assert doc_id in rag_engine_with_mocks.processed_documents
        
        # Check that chunks were created
        doc_info = rag_engine_with_mocks.processed_documents[doc_id]
        assert len(doc_info['chunks']) > 0
        
        # Check that embeddings were generated
        for chunk in doc_info['chunks']:
            assert chunk.embedding is not None
    
    def test_process_pdf_file(self, rag_engine_with_mocks, sample_pdf_path):
        """Test processing a PDF file"""
        with patch('src.askspark.core.document_intelligence.PyPDF2.PdfReader') as mock_pdf_reader:
            # Mock PDF reader
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = "This is a test PDF content about AI and machine learning."
            mock_reader.pages = [mock_page]
            mock_reader.num_pages = 1
            mock_pdf_reader.return_value = mock_reader
            
            doc_id = rag_engine_with_mocks.process_and_store_document(str(sample_pdf_path))
            
            assert doc_id is not None
            assert doc_id in rag_engine_with_mocks.processed_documents
    
    def test_chunk_text(self, rag_engine_with_mocks):
        """Test text chunking functionality"""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        
        chunks = rag_engine_with_mocks._chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 50 + 20 for chunk in chunks)  # Allow some overlap
    
    def test_generate_embeddings(self, rag_engine_with_mocks):
        """Test embedding generation"""
        texts = ["This is a test text about AI.", "Machine learning is a subset of AI."]
        
        embeddings = rag_engine_with_mocks._generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 1024 for emb in embeddings)  # Standard embedding size
    
    def test_store_chunks(self, rag_engine_with_mocks):
        """Test storing chunks in vector database"""
        chunks = [
            DocumentChunk(
                content="AI is transforming technology.",
                source="test.txt",
                chunk_id="chunk_1",
                embedding=[[0.1, 0.2, 0.3] * 34]
            ),
            DocumentChunk(
                content="Machine learning enables data analysis.",
                source="test.txt",
                chunk_id="chunk_2",
                embedding=[[0.4, 0.5, 0.6] * 34]
            )
        ]
        
        # Mock vector store operations
        rag_engine_with_mocks.vector_store.add.return_value = ["id1", "id2"]
        
        chunk_ids = rag_engine_with_mocks._store_chunks(chunks)
        
        assert len(chunk_ids) == 2
        rag_engine_with_mocks.vector_store.add.assert_called_once()
    
    def test_query_documents(self, rag_engine_with_mocks):
        """Test querying documents"""
        # Setup mock data
        rag_engine_with_mocks.processed_documents = {
            "doc1": {
                "source": "test.txt",
                "chunks": [
                    DocumentChunk(
                        content="AI is revolutionizing healthcare with diagnostic tools.",
                        source="test.txt",
                        chunk_id="chunk_1",
                        embedding=[[0.1, 0.2, 0.3] * 34]
                    )
                ]
            }
        }
        
        # Mock similarity search
        mock_results = [
            {
                "id": "chunk_1",
                "metadata": {"source": "test.txt", "chunk_id": "chunk_1"},
                "document": "AI is revolutionizing healthcare with diagnostic tools.",
                "distance": 0.1
            }
        ]
        
        rag_engine_with_mocks.vector_store.query.return_value = mock_results
        
        # Mock AI response
        rag_engine_with_mocks.ai_client.generate_response.return_value = Mock(
            content="AI is transforming healthcare through improved diagnostic accuracy, personalized treatment plans, and drug discovery acceleration.",
            model="gpt-3.5-turbo",
            provider="openai",
            response_time=1.0,
            tokens_used=50,
            cost=0.001
        )
        
        query = "How is AI impacting healthcare?"
        result = rag_engine_with_mocks.query_documents(query)
        
        assert result is not None
        assert 'answer' in result
        assert 'sources' in result
        assert len(result['sources']) > 0
        assert result['answer'] is not None
    
    def test_query_documents_no_results(self, rag_engine_with_mocks):
        """Test querying when no documents are processed"""
        result = rag_engine_with_mocks.query_documents("Any query")
        
        assert result is not None
        assert result['answer'] == "No documents have been processed yet."
        assert result['sources'] == []
    
    def test_delete_document(self, rag_engine_with_mocks):
        """Test deleting a document"""
        # Setup mock data
        doc_id = "doc1"
        rag_engine_with_mocks.processed_documents[doc_id] = {
            "source": "test.txt",
            "chunks": ["chunk1", "chunk2"]
        }
        
        # Mock vector store delete
        rag_engine_with_mocks.vector_store.delete.return_value = True
        
        success = rag_engine_with_mocks.delete_document(doc_id)
        
        assert success is True
        assert doc_id not in rag_engine_with_mocks.processed_documents
    
    def test_get_document_summary(self, rag_engine_with_mocks):
        """Test getting document summary"""
        # Setup mock data
        doc_id = "doc1"
        rag_engine_with_mocks.processed_documents[doc_id] = {
            "source": "test.txt",
            "chunks": [
                DocumentChunk(
                    content="First chunk about AI.",
                    source="test.txt",
                    chunk_id="chunk_1"
                ),
                DocumentChunk(
                    content="Second chunk about machine learning.",
                    source="test.txt",
                    chunk_id="chunk_2"
                )
            ],
            "processed_at": "2024-01-01T00:00:00Z"
        }
        
        summary = rag_engine_with_mocks.get_document_summary(doc_id)
        
        assert summary is not None
        assert summary['source'] == "test.txt"
        assert summary['total_chunks'] == 2
        assert summary['processed_at'] == "2024-01-01T00:00:00Z"
    
    def test_search_similar_chunks(self, rag_engine_with_mocks):
        """Test searching for similar chunks"""
        # Setup mock data
        rag_engine_with_mocks.processed_documents = {
            "doc1": {
                "source": "test.txt",
                "chunks": [
                    DocumentChunk(
                        content="AI and machine learning are related fields.",
                        source="test.txt",
                        chunk_id="chunk_1",
                        embedding=[[0.1, 0.2, 0.3] * 34]
                    )
                ]
            }
        }
        
        # Mock similarity search
        mock_results = [
            {
                "id": "chunk_1",
                "metadata": {"source": "test.txt", "chunk_id": "chunk_1"},
                "document": "AI and machine learning are related fields.",
                "distance": 0.1
            }
        ]
        
        rag_engine_with_mocks.vector_store.query.return_value = mock_results
        
        query = "artificial intelligence"
        results = rag_engine_with_mocks.search_similar_chunks(query, top_k=5)
        
        assert len(results) <= 5
        assert all('content' in result for result in results)
        assert all('source' in result for result in results)
        assert all('similarity_score' in result for result in results)
