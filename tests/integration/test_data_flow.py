"""
Integration tests for complete data flow from OS level to final output
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path

from src.askspark.config.settings import Config
from src.askspark.config.logging import setup_logging
from src.askspark.core.ai_providers import UnifiedAIClient, ModelResponse
from src.askspark.core.model_comparison import ModelComparisonEngine
from src.askspark.core.document_intelligence import RAGEngine
from src.askspark.workflows.engine import WorkflowEngine
from src.askspark.notifications.service import NotificationService


class TestDataFlowOSLevel:
    """Test data flow starting from OS level (environment variables, file system)"""
    
    @pytest.fixture
    def temp_env_dir(self):
        """Create temporary directory with environment files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_content = """
OPENAI_API_KEY=test_openai_key_from_env
ANTHROPIC_API_KEY=test_anthropic_key_from_env
GOOGLE_API_KEY=test_google_key_from_env
GROQ_API_KEY=test_groq_key_from_env
PUSHOVER_USER_KEY=test_pushover_user_key
PUSHOVER_APP_TOKEN=test_pushover_app_token
DEBUG=true
"""
            env_path.write_text(env_content)
            yield tmpdir
    
    @pytest.fixture
    def mock_environment(self, temp_env_dir):
        """Mock environment variables"""
        env_vars = {
            'OPENAI_API_KEY': 'test_openai_key_from_env',
            'ANTHROPIC_API_KEY': 'test_anthropic_key_from_env',
            'GOOGLE_API_KEY': 'test_google_key_from_env',
            'GROQ_API_KEY': 'test_groq_key_from_env',
            'PUSHOVER_USER_KEY': 'test_pushover_user_key',
            'PUSHOVER_APP_TOKEN': 'test_pushover_app_token',
            'DEBUG': 'true'
        }
        
        with patch.dict(os.environ, env_vars):
            yield env_vars
    
    def test_environment_variable_loading(self, mock_environment):
        """Test that environment variables are loaded correctly"""
        # Reload config to pick up new environment
        from importlib import reload
        reload(Config)
        
        assert Config.OPENAI_API_KEY == 'test_openai_key_from_env'
        assert Config.ANTHROPIC_API_KEY == 'test_anthropic_key_from_env'
        assert Config.GOOGLE_API_KEY == 'test_google_key_from_env'
        assert Config.GROQ_API_KEY == 'test_groq_key_from_env'
        assert Config.PUSHOVER_USER_KEY == 'test_pushover_user_key'
        assert Config.PUSHOVER_APP_TOKEN == 'test_pushover_app_token'
        assert Config.DEBUG is True
    
    def test_file_system_document_processing(self, mock_environment):
        """Test document processing from file system"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test document
            doc_path = Path(tmpdir) / "test_document.txt"
            doc_content = """
            Artificial Intelligence (AI) is revolutionizing healthcare.
            Machine Learning algorithms help in disease diagnosis.
            Deep Learning enables image recognition for medical imaging.
            Natural Language Processing powers chatbots for patient care.
            """
            doc_path.write_text(doc_content)
            
            # Mock the AI client and dependencies
            with patch('src.askspark.core.document_intelligence.chromadb'), \
                 patch('src.askspark.core.document_intelligence.SentenceTransformer'), \
                 patch('src.askspark.core.document_intelligence.UnifiedAIClient'):
                
                rag_engine = RAGEngine()
                rag_engine.vector_store = Mock()
                rag_engine.embedding_model = Mock()
                rag_engine.ai_client = Mock()
                
                # Mock embedding generation
                rag_engine.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] * 34]
                
                # Process the document
                doc_id = rag_engine.process_and_store_document(str(doc_path))
                
                assert doc_id is not None
                assert doc_id in rag_engine.processed_documents
                
                # Verify document was processed correctly
                doc_info = rag_engine.processed_documents[doc_id]
                assert len(doc_info['chunks']) > 0
                assert all(chunk.content for chunk in doc_info['chunks'])


class TestDataFlowAIProviders:
    """Test data flow through AI providers"""
    
    @pytest.fixture
    def mock_ai_responses(self):
        """Mock AI provider responses"""
        return {
            'openai': ModelResponse(
                content="OpenAI: AI enhances healthcare through diagnostic accuracy, personalized treatment, and drug discovery acceleration.",
                model="gpt-3.5-turbo",
                provider="openai",
                response_time=1.2,
                tokens_used=45,
                cost=0.001
            ),
            'anthropic': ModelResponse(
                content="Anthropic: AI transforms healthcare by improving diagnostic tools, enabling personalized medicine, and accelerating pharmaceutical research.",
                model="claude-3-haiku-20240307",
                provider="anthropic",
                response_time=0.8,
                tokens_used=42,
                cost=0.001
            )
        }
    
    def test_ai_provider_data_flow(self, mock_environment, mock_ai_responses):
        """Test complete data flow through AI providers"""
        with patch('src.askspark.core.ai_providers.openai'), \
             patch('src.askspark.core.ai_providers.anthropic'), \
             patch('src.askspark.core.ai_providers.genai'), \
             patch('src.askspark.core.ai_providers.groq'):
            
            # Create AI client
            client = UnifiedAIClient()
            
            # Mock the clients to return predefined responses
            mock_openai_client = Mock()
            mock_openai_client.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content=mock_ai_responses['openai'].content))],
                usage=Mock(prompt_tokens=20, completion_tokens=25)
            )
            
            mock_anthropic_client = Mock()
            mock_anthropic_client.messages.create.return_value = Mock(
                content=[Mock(text=mock_ai_responses['anthropic'].content)],
                usage=Mock(input_tokens=20, output_tokens=22)
            )
            
            client.clients['openai'] = mock_openai_client
            client.clients['anthropic'] = mock_anthropic_client
            
            # Test data flow: input -> AI provider -> response
            prompt = "How is AI transforming healthcare?"
            
            # Test OpenAI flow
            openai_response = client.generate_response(prompt, "openai", "gpt-3.5-turbo")
            assert openai_response is not None
            assert openai_response.provider == "openai"
            assert openai_response.content == mock_ai_responses['openai'].content
            assert openai_response.response_time > 0
            
            # Test Anthropic flow
            anthropic_response = client.generate_response(prompt, "anthropic", "claude-3-haiku-20240307")
            assert anthropic_response is not None
            assert anthropic_response.provider == "anthropic"
            assert anthropic_response.content == mock_ai_responses['anthropic'].content
            assert anthropic_response.response_time > 0


class TestDataFlowModelComparison:
    """Test data flow through model comparison engine"""
    
    def test_model_comparison_data_flow(self, mock_environment):
        """Test complete data flow: prompt -> multiple AI providers -> comparison results"""
        with patch('src.askspark.core.model_comparison.UnifiedAIClient') as mock_client_class:
            # Mock AI client
            mock_client = Mock()
            
            # Mock responses for different providers
            def mock_generate_response(prompt, provider, model):
                if provider == "openai":
                    return ModelResponse(
                        content="OpenAI response about AI benefits with detailed analysis.",
                        model="gpt-3.5-turbo",
                        provider="openai",
                        response_time=1.5,
                        tokens_used=60,
                        cost=0.002
                    )
                elif provider == "anthropic":
                    return ModelResponse(
                        content="Anthropic response about AI benefits with comprehensive overview.",
                        model="claude-3-haiku-20240307",
                        provider="anthropic",
                        response_time=1.0,
                        tokens_used=55,
                        cost=0.002
                    )
                return None
            
            mock_client.generate_response = mock_generate_response
            mock_client_class.return_value = mock_client
            
            # Create comparison engine
            engine = ModelComparisonEngine()
            
            # Test complete data flow
            prompt = "What are the main benefits of AI in business?"
            providers_models = [
                ("openai", "gpt-3.5-turbo"),
                ("anthropic", "claude-3-haiku-20240307")
            ]
            
            # Execute comparison
            results = engine.compare_models(prompt, providers_models)
            
            # Verify data flow
            assert len(results) == 2
            
            # Check OpenAI result
            openai_result = next(r for r in results if r.provider == "openai")
            assert openai_result.content == "OpenAI response about AI benefits with detailed analysis."
            assert openai_result.response_time == 1.5
            assert openai_result.quality_score is not None
            
            # Check Anthropic result
            anthropic_result = next(r for r in results if r.provider == "anthropic")
            assert anthropic_result.content == "Anthropic response about AI benefits with comprehensive overview."
            assert anthropic_result.response_time == 1.0
            assert anthropic_result.quality_score is not None
            
            # Test data flow to report generation
            report = engine.generate_comparison_report()
            assert report['total_comparisons'] == 2
            assert 'openai' in report['providers']
            assert 'anthropic' in report['providers']


class TestDataFlowWorkflows:
    """Test data flow through workflow automation"""
    
    def test_workflow_data_flow(self, mock_environment):
        """Test complete workflow data flow: trigger -> actions -> notifications"""
        with patch('src.askspark.workflows.engine.NotificationService') as mock_notification_service_class:
            # Mock notification service
            mock_notification_service = Mock()
            mock_notification_service_class.return_value = mock_notification_service
            
            # Mock AI client for model comparison action
            with patch('src.askspark.workflows.actions.ModelComparisonEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_result = Mock()
                mock_result.provider = "openai"
                mock_result.model = "gpt-3.5-turbo"
                mock_result.quality_score = 0.85
                mock_engine.compare_models.return_value = [mock_result]
                mock_engine_class.return_value = mock_engine
                
                # Create workflow engine
                workflow_engine = WorkflowEngine({
                    'PUSHOVER_USER_KEY': 'test_key',
                    'PUSHOVER_APP_TOKEN': 'test_token'
                })
                
                # Create test workflow
                triggers = [{"type": "manual", "config": {}}]
                actions = [
                    {
                        "type": "run_model_comparison",
                        "parameters": {
                            "prompt": "Test AI benefits",
                            "providers_models": [("openai", "gpt-3.5-turbo")]
                        }
                    },
                    {
                        "type": "send_notification",
                        "parameters": {
                            "message": "Comparison completed for {prompt}",
                            "title": "Workflow Results"
                        }
                    }
                ]
                
                workflow_engine.create_workflow(
                    "test_workflow",
                    "Test Workflow",
                    "Test data flow",
                    triggers,
                    actions
                )
                
                # Execute workflow (complete data flow)
                trigger_data = {"prompt": "AI benefits analysis"}
                result = workflow_engine.execute_workflow("test_workflow", trigger_data)
                
                # Verify data flow completion
                assert result is True
                
                # Verify model comparison was called
                mock_engine.compare_models.assert_called_once()
                
                # Verify notification was sent with template substitution
                assert mock_notification_service.send_notification.call_count >= 1
                
                # Check that template substitution worked
                call_args = mock_notification_service.send_notification.call_args_list
                notification_calls = [call for call in call_args if "Comparison completed" in str(call)]
                assert len(notification_calls) > 0


class TestDataFlowDocumentIntelligence:
    """Test data flow through document intelligence"""
    
    def test_document_intelligence_data_flow(self, mock_environment):
        """Test complete document processing data flow: file -> chunks -> embeddings -> query -> answer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test document
            doc_path = Path(tmpdir) / "ai_healthcare.txt"
            doc_content = """
            Artificial Intelligence in Healthcare
            
            AI is transforming healthcare through:
            1. Diagnostic Imaging: AI algorithms analyze medical images with high accuracy
            2. Drug Discovery: Machine learning accelerates pharmaceutical research
            3. Personalized Medicine: AI tailors treatments to individual patients
            4. Predictive Analytics: AI predicts disease outbreaks and patient risks
            
            These applications are revolutionizing patient care and medical research.
            """
            doc_path.write_text(doc_content)
            
            # Mock dependencies
            with patch('src.askspark.core.document_intelligence.chromadb'), \
                 patch('src.askspark.core.document_intelligence.SentenceTransformer'), \
                 patch('src.askspark.core.document_intelligence.UnifiedAIClient'):
                
                rag_engine = RAGEngine()
                rag_engine.vector_store = Mock()
                rag_engine.embedding_model = Mock()
                rag_engine.ai_client = Mock()
                
                # Mock embedding generation
                rag_engine.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] * 34]
                
                # Step 1: Process document (file -> chunks -> embeddings)
                doc_id = rag_engine.process_and_store_document(str(doc_path))
                assert doc_id is not None
                
                # Step 2: Mock similarity search results
                mock_search_results = [
                    {
                        "id": "chunk_1",
                        "metadata": {"source": str(doc_path), "chunk_id": "chunk_1"},
                        "document": "AI algorithms analyze medical images with high accuracy",
                        "distance": 0.1
                    },
                    {
                        "id": "chunk_2",
                        "metadata": {"source": str(doc_path), "chunk_id": "chunk_2"},
                        "document": "Machine learning accelerates pharmaceutical research",
                        "distance": 0.2
                    }
                ]
                rag_engine.vector_store.query.return_value = mock_search_results
                
                # Step 3: Mock AI response
                rag_engine.ai_client.generate_response.return_value = Mock(
                    content="Based on the document, AI transforms healthcare through diagnostic imaging, drug discovery, personalized medicine, and predictive analytics.",
                    model="gpt-3.5-turbo",
                    provider="openai",
                    response_time=1.0,
                    tokens_used=75,
                    cost=0.002
                )
                
                # Step 4: Query documents (query -> search -> AI -> answer)
                query = "How does AI transform healthcare?"
                result = rag_engine.query_documents(query)
                
                # Verify complete data flow
                assert result is not None
                assert 'answer' in result
                assert 'sources' in result
                assert len(result['sources']) == 2
                assert "diagnostic imaging" in result['answer'].lower()
                assert "drug discovery" in result['answer'].lower()


class TestEndToEndDataFlow:
    """Test complete end-to-end data flow"""
    
    def test_complete_application_data_flow(self, mock_environment):
        """Test complete data flow from environment setup to final output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Environment setup (already done via mock_environment fixture)
            
            # Step 2: Create test document
            doc_path = Path(tmpdir) / "business_ai.txt"
            doc_content = """
            AI in Business
            
            Artificial Intelligence provides significant business value through:
            - Automation of routine tasks
            - Data-driven decision making
            - Enhanced customer experiences
            - Predictive analytics for forecasting
            - Cost reduction through optimization
            """
            doc_path.write_text(doc_content)
            
            # Mock all external dependencies
            with patch('src.askspark.core.ai_providers.openai'), \
                 patch('src.askspark.core.ai_providers.anthropic'), \
                 patch('src.askspark.core.model_comparison.UnifiedAIClient'), \
                 patch('src.askspark.core.document_intelligence.chromadb'), \
                 patch('src.askspark.core.document_intelligence.SentenceTransformer'), \
                 patch('src.askspark.workflows.engine.NotificationService'):
                
                # Step 3: Initialize core components
                ai_client = UnifiedAIClient()
                model_engine = ModelComparisonEngine()
                rag_engine = RAGEngine()
                workflow_engine = WorkflowEngine({
                    'PUSHOVER_USER_KEY': 'test_key',
                    'PUSHOVER_APP_TOKEN': 'test_token'
                })
                
                # Mock AI responses
                mock_ai_response = ModelResponse(
                    content="AI provides business value through automation, data-driven decisions, customer experience enhancement, predictive analytics, and cost optimization.",
                    model="gpt-3.5-turbo",
                    provider="openai",
                    response_time=1.2,
                    tokens_used=65,
                    cost=0.002
                )
                
                ai_client.generate_response = Mock(return_value=mock_ai_response)
                model_engine.ai_client = ai_client
                
                # Mock document processing
                rag_engine.vector_store = Mock()
                rag_engine.embedding_model = Mock()
                rag_engine.ai_client = ai_client
                rag_engine.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] * 34]
                
                # Step 4: Process document
                doc_id = rag_engine.process_and_store_document(str(doc_path))
                assert doc_id is not None
                
                # Step 5: Query document
                mock_search_results = [{
                    "id": "chunk_1",
                    "metadata": {"source": str(doc_path), "chunk_id": "chunk_1"},
                    "document": "AI provides business value through automation and optimization",
                    "distance": 0.1
                }]
                rag_engine.vector_store.query.return_value = mock_search_results
                
                query_result = rag_engine.query_documents("What are the business benefits of AI?")
                assert query_result is not None
                assert 'answer' in query_result
                
                # Step 6: Model comparison
                comparison_results = model_engine.compare_models(
                    "Compare AI models for business analysis",
                    [("openai", "gpt-3.5-turbo")]
                )
                assert len(comparison_results) == 1
                
                # Step 7: Workflow automation
                workflow_engine.create_workflow(
                    "business_analysis",
                    "Business AI Analysis",
                    "Analyze AI business value",
                    [{"type": "manual", "config": {}}],
                    [{
                        "type": "send_notification",
                        "parameters": {
                            "message": "Business analysis completed successfully",
                            "channels": ["pushover"]
                        }
                    }]
                )
                
                workflow_result = workflow_engine.execute_workflow("business_analysis")
                assert workflow_result is True
                
                # Step 8: Verify final outputs
                assert query_result['answer'] is not None
                assert comparison_results[0].content is not None
                assert workflow_engine.notification_service.send_notification.called
                
                # Complete end-to-end data flow verified
                assert True  # All steps completed successfully
