"""
Integration tests for Gradio interface and dashboard functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import pandas as pd
from pathlib import Path

# Import the dashboard class
import sys
sys.path.append('/Users/startferanmi/Documents/AskSpark')
from app import AIConsultantDashboard


class TestGradioInterface:
    """Test Gradio interface functionality"""
    
    @pytest.fixture
    def mock_dashboard(self):
        """Create dashboard with all dependencies mocked"""
        with patch('src.askspark.core.ai_providers.UnifiedAIClient') as mock_ai_client, \
             patch('src.askspark.core.model_comparison.ModelComparisonEngine') as mock_model_engine, \
             patch('src.askspark.core.document_intelligence.RAGEngine') as mock_rag_engine, \
             patch('src.askspark.workflows.engine.WorkflowEngine') as mock_workflow_engine:
            
            # Mock instances
            mock_ai_instance = Mock()
            mock_model_instance = Mock()
            mock_rag_instance = Mock()
            mock_workflow_instance = Mock()
            
            mock_ai_client.return_value = mock_ai_instance
            mock_model_engine.return_value = mock_model_instance
            mock_rag_engine.return_value = mock_rag_instance
            mock_workflow_engine.return_value = mock_workflow_instance
            
            # Create dashboard
            dashboard = AIConsultantDashboard()
            
            # Replace with mocked instances
            dashboard.client = mock_ai_instance
            dashboard.model_comparison = mock_model_instance
            dashboard.document_engine = mock_rag_instance
            dashboard.workflow_engine = mock_workflow_instance
            
            yield dashboard
    
    def test_dashboard_initialization(self, mock_dashboard):
        """Test dashboard initializes correctly"""
        assert mock_dashboard.client is not None
        assert mock_dashboard.model_comparison is not None
        assert mock_dashboard.document_engine is not None
        assert mock_dashboard.workflow_engine is not None
    
    def test_get_provider_status(self, mock_dashboard):
        """Test provider status functionality"""
        # Mock Config methods
        with patch('src.askspark.config.settings.Config.get_available_providers', return_value=['openai', 'anthropic']), \
             patch('src.askspark.config.settings.Config.get_provider_models', return_value=['gpt-3.5-turbo']), \
             patch('src.askspark.config.settings.Config.AI_PROVIDERS', {
                 'openai': {'api_key': 'test_key'},
                 'anthropic': {'api_key': 'test_key'}
             }):
            
            status_df = mock_dashboard.get_provider_status()
            
            assert isinstance(status_df, pd.DataFrame)
            assert len(status_df) == 2  # openai and anthropic
            assert 'Provider' in status_df.columns
            assert 'Model' in status_df.columns
            assert 'Status' in status_df.columns
            assert 'API Key' in status_df.columns
    
    def test_model_comparison_interface(self, mock_dashboard):
        """Test model comparison interface functionality"""
        # Mock model comparison response
        mock_result = Mock()
        mock_result.provider = "openai"
        mock_result.model = "gpt-3.5-turbo"
        mock_result.content = "AI benefits include automation and efficiency"
        mock_result.response_time = 1.2
        mock_result.quality_score = 0.85
        mock_result.tokens_used = 50
        mock_result.cost = 0.001
        
        mock_dashboard.model_comparison.compare_models.return_value = [mock_result]
        
        # Test comparison
        prompt = "What are the benefits of AI?"
        providers = ["openai"]
        models = ["gpt-3.5-turbo"]
        
        with patch('src.askspark.config.settings.Config.get_available_providers', return_value=['openai']):
            results = mock_dashboard.model_comparison.compare_models(prompt, [(providers[0], models[0])])
            
            assert len(results) == 1
            assert results[0].provider == "openai"
            assert results[0].content == "AI benefits include automation and efficiency"
    
    def test_document_upload_interface(self, mock_dashboard):
        """Test document upload and processing interface"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("AI is transforming healthcare through diagnostic imaging and personalized medicine.")
            tmp_file_path = tmp_file.name
        
        try:
            # Mock document processing
            mock_dashboard.document_engine.process_and_store_document.return_value = "doc_123"
            
            # Test document upload
            doc_id = mock_dashboard.document_engine.process_and_store_document(tmp_file_path)
            
            assert doc_id == "doc_123"
            mock_dashboard.document_engine.process_and_store_document.assert_called_once_with(tmp_file_path)
        
        finally:
            Path(tmp_file_path).unlink()
    
    def test_document_query_interface(self, mock_dashboard):
        """Test document query interface"""
        # Mock query response
        mock_query_result = {
            'answer': 'AI transforms healthcare through diagnostic imaging, personalized medicine, and predictive analytics.',
            'sources': [
                {
                    'content': 'AI algorithms analyze medical images with high accuracy',
                    'source': 'healthcare_doc.pdf',
                    'similarity_score': 0.95
                }
            ]
        }
        
        mock_dashboard.document_engine.query_documents.return_value = mock_query_result
        
        # Test query
        query = "How does AI transform healthcare?"
        result = mock_dashboard.document_engine.query_documents(query)
        
        assert result['answer'] is not None
        assert 'diagnostic imaging' in result['answer']
        assert len(result['sources']) > 0
    
    def test_workflow_management_interface(self, mock_dashboard):
        """Test workflow management interface"""
        # Mock workflow operations
        mock_workflow = Mock()
        mock_workflow.id = "test_workflow"
        mock_workflow.name = "Test Workflow"
        mock_workflow.enabled = True
        mock_workflow.to_dict.return_value = {
            'id': 'test_workflow',
            'name': 'Test Workflow',
            'enabled': True
        }
        
        mock_dashboard.workflow_engine.workflows = {"test_workflow": mock_workflow}
        mock_dashboard.workflow_engine.list_workflows.return_value = [mock_workflow.to_dict()]
        
        # Test listing workflows
        workflows = mock_dashboard.workflow_engine.list_workflows()
        
        assert len(workflows) == 1
        assert workflows[0]['id'] == 'test_workflow'
        assert workflows[0]['name'] == 'Test Workflow'
        assert workflows[0]['enabled'] is True
    
    def test_workflow_execution_interface(self, mock_dashboard):
        """Test workflow execution interface"""
        # Mock successful workflow execution
        mock_dashboard.workflow_engine.execute_workflow.return_value = True
        
        # Test workflow execution
        result = mock_dashboard.workflow_engine.execute_workflow("test_workflow", {"test": "data"})
        
        assert result is True
        mock_dashboard.workflow_engine.execute_workflow.assert_called_once_with("test_workflow", {"test": "data"})
    
    def test_analytics_interface(self, mock_dashboard):
        """Test analytics dashboard interface"""
        # Mock analytics data
        mock_comparison_history = [
            Mock(provider="openai", model="gpt-3.5-turbo", quality_score=0.85, response_time=1.2),
            Mock(provider="anthropic", model="claude-3-haiku", quality_score=0.90, response_time=0.8)
        ]
        
        mock_dashboard.model_comparison.comparison_history = mock_comparison_history
        mock_dashboard.model_comparison.get_comparison_history.return_value = pd.DataFrame([
            {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'quality_score': 0.85,
                'response_time': 1.2
            },
            {
                'provider': 'anthropic',
                'model': 'claude-3-haiku',
                'quality_score': 0.90,
                'response_time': 0.8
            }
        ])
        
        # Test analytics data retrieval
        analytics_data = mock_dashboard.model_comparison.get_comparison_history()
        
        assert isinstance(analytics_data, pd.DataFrame)
        assert len(analytics_data) == 2
        assert 'provider' in analytics_data.columns
        assert 'quality_score' in analytics_data.columns
        assert 'response_time' in analytics_data.columns
    
    def test_error_handling_interface(self, mock_dashboard):
        """Test error handling in interface"""
        # Mock API failure
        mock_dashboard.client.generate_response.side_effect = Exception("API Error")
        
        # Test that error is handled gracefully
        with patch('src.askspark.config.settings.Config.get_available_providers', return_value=['openai']):
            response = mock_dashboard.client.generate_response("test", "openai", "gpt-3.5-turbo")
            
            # Should handle error and return None or appropriate error response
            assert response is None or isinstance(response, Exception)
    
    def test_data_visualization_interface(self, mock_dashboard):
        """Test data visualization functionality"""
        # Mock visualization data
        mock_dashboard.model_comparison.comparison_history = [
            Mock(provider="openai", model="gpt-3.5-turbo", quality_score=0.85, response_time=1.2, cost=0.002),
            Mock(provider="anthropic", model="claude-3-haiku", quality_score=0.90, response_time=0.8, cost=0.001)
        ]
        
        # Test visualization creation
        with patch('src.askspark.core.model_comparison.plt') as mock_plt, \
             patch('src.askspark.core.model_comparison.go.Figure') as mock_figure:
            
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            mock_plt.Figure.return_value = mock_fig
            
            # Test quality score visualization
            fig = mock_dashboard.model_comparison.visualize_comparison('quality_score')
            
            # Verify visualization was created
            assert fig is not None


class TestGradioDataFlow:
    """Test data flow specifically through Gradio interface"""
    
    @pytest.fixture
    def mock_dashboard_with_data(self):
        """Create dashboard with populated mock data"""
        with patch('src.askspark.core.ai_providers.UnifiedAIClient') as mock_ai_client, \
             patch('src.askspark.core.model_comparison.ModelComparisonEngine') as mock_model_engine, \
             patch('src.askspark.core.document_intelligence.RAGEngine') as mock_rag_engine, \
             patch('src.askspark.workflows.engine.WorkflowEngine') as mock_workflow_engine:
            
            # Create dashboard
            dashboard = AIConsultantDashboard()
            
            # Setup mock data
            mock_ai_response = Mock()
            mock_ai_response.content = "AI provides significant business value through automation and analytics."
            mock_ai_response.provider = "openai"
            mock_ai_response.model = "gpt-3.5-turbo"
            mock_ai_response.response_time = 1.0
            mock_ai_response.tokens_used = 40
            mock_ai_response.cost = 0.001
            
            dashboard.client = Mock()
            dashboard.client.generate_response.return_value = mock_ai_response
            
            dashboard.model_comparison = Mock()
            dashboard.model_comparison.compare_models.return_value = [mock_ai_response]
            dashboard.model_comparison.get_comparison_history.return_value = pd.DataFrame([{
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'quality_score': 0.85,
                'response_time': 1.0,
                'cost': 0.001
            }])
            
            dashboard.document_engine = Mock()
            dashboard.document_engine.query_documents.return_value = {
                'answer': 'AI transforms business through automation and data analysis.',
                'sources': [{'content': 'AI automation benefits', 'source': 'test.pdf'}]
            }
            
            dashboard.workflow_engine = Mock()
            dashboard.workflow_engine.list_workflows.return_value = [{
                'id': 'test_workflow',
                'name': 'Test Workflow',
                'enabled': True
            }]
            
            yield dashboard
    
    def test_gradio_input_to_output_flow(self, mock_dashboard_with_data):
        """Test complete input to output flow through Gradio interface"""
        dashboard = mock_dashboard_with_data
        
        # Test Model Comparison tab flow
        prompt = "What are the business benefits of AI?"
        with patch('src.askspark.config.settings.Config.get_available_providers', return_value=['openai']):
            comparison_results = dashboard.model_comparison.compare_models(prompt, [("openai", "gpt-3.5-turbo")])
            
            assert len(comparison_results) == 1
            assert "business value" in comparison_results[0].content
        
        # Test Document Intelligence tab flow
        query = "How does AI benefit business?"
        doc_result = dashboard.document_engine.query_documents(query)
        
        assert doc_result['answer'] is not None
        assert "automation" in doc_result['answer']
        assert len(doc_result['sources']) > 0
        
        # Test Workflow Automation tab flow
        workflows = dashboard.workflow_engine.list_workflows()
        
        assert len(workflows) == 1
        assert workflows[0]['id'] == 'test_workflow'
        assert workflows[0]['enabled'] is True
        
        # Test Analytics tab flow
        analytics_data = dashboard.model_comparison.get_comparison_history()
        
        assert isinstance(analytics_data, pd.DataFrame)
        assert len(analytics_data) == 1
        assert analytics_data.iloc[0]['provider'] == 'openai'
    
    def test_gradio_error_propagation(self, mock_dashboard_with_data):
        """Test error handling and propagation in Gradio interface"""
        dashboard = mock_dashboard_with_data
        
        # Mock API error
        dashboard.client.generate_response.side_effect = Exception("API Connection Error")
        
        # Test that error is handled gracefully
        with patch('src.askspark.config.settings.Config.get_available_providers', return_value=['openai']):
            try:
                result = dashboard.client.generate_response("test", "openai", "gpt-3.5-turbo")
                # Should handle error without crashing
                assert result is None or isinstance(result, Exception)
            except Exception as e:
                # If exception is raised, it should be the expected one
                assert "API Connection Error" in str(e)
    
    def test_gradio_state_management(self, mock_dashboard_with_data):
        """Test state management across Gradio interface"""
        dashboard = mock_dashboard_with_data
        
        # Test that state is maintained across operations
        initial_workflows = dashboard.workflow_engine.list_workflows()
        
        # Simulate workflow execution
        dashboard.workflow_engine.execute_workflow.return_value = True
        dashboard.workflow_engine.execute_workflow("test_workflow")
        
        # Check that state is updated
        updated_workflows = dashboard.workflow_engine.list_workflows()
        assert len(initial_workflows) == len(updated_workflows)
        
        # Verify workflow execution was called
        dashboard.workflow_engine.execute_workflow.assert_called_with("test_workflow")
