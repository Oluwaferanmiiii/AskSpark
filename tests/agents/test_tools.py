"""
Unit tests for AskSpark Agent Tools
Week 2 Lab 1 - Tool functionality testing
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
from pathlib import Path

# Add src to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from askspark.agents.tools import AskSparkTools, ToolResult


class TestToolResult:
    """Test ToolResult dataclass"""
    
    def test_tool_result_creation_success(self):
        """Test creating successful ToolResult"""
        result = ToolResult(
            success=True,
            data={"key": "value"},
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.execution_time == 1.5
        assert result.error_message is None
    
    def test_tool_result_creation_failure(self):
        """Test creating failed ToolResult"""
        result = ToolResult(
            success=False,
            data=None,
            execution_time=0.5,
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.data is None
        assert result.execution_time == 0.5
        assert result.error_message == "Test error"


class TestAskSparkToolsUnit:
    """Unit tests for AskSparkTools"""
    
    @pytest.fixture
    def tools(self):
        """Create AskSparkTools instance with mocked dependencies"""
        with patch('askspark.agents.tools.UnifiedAIClient') as mock_client:
            with patch('askspark.agents.tools.ModelComparisonEngine') as mock_comparison:
                mock_client.return_value = Mock()
                mock_comparison.return_value = Mock()
                return AskSparkTools()
    
    def test_initialization(self, tools):
        """Test tools initialization"""
        assert tools.unified_client is not None
        assert tools.model_comparison is not None
        assert tools.rag_engine is None  # Lazy initialization
    
    def test_get_rag_engine_lazy_initialization(self, tools):
        """Test RAG engine lazy initialization"""
        with patch('askspark.agents.tools.RAGEngine') as mock_rag:
            mock_rag_instance = Mock()
            mock_rag.return_value = mock_rag_instance
            
            # First call should initialize RAG engine
            engine1 = tools._get_rag_engine()
            assert engine1 is mock_rag_instance
            
            # Second call should return same instance
            engine2 = tools._get_rag_engine()
            assert engine1 is engine2
    
    def test_get_best_model_empty_results(self, tools):
        """Test _get_best_model with empty results"""
        result = tools._get_best_model([])
        
        assert result["model"] == "none"
        assert "No results available" in result["reason"]
    
    def test_get_best_model_with_results(self, tools):
        """Test _get_best_model with valid results"""
        mock_results = [
            {
                "model": "gpt-4o-mini",
                "provider": "openai",
                "quality_score": 0.8,
                "response_time": 1.0,
                "cost": 0.01
            },
            {
                "model": "claude-3-haiku",
                "provider": "anthropic", 
                "quality_score": 0.7,
                "response_time": 1.5,
                "cost": 0.02
            }
        ]
        
        result = tools._get_best_model(mock_results)
        
        assert result["model"] == "gpt-4o-mini"
        assert result["provider"] == "openai"
        assert "quality" in result["reason"].lower()
        assert "speed" in result["reason"].lower()
        assert "cost" in result["reason"].lower()


@pytest.mark.asyncio
class TestAskSparkToolsAsync:
    """Async tests for AskSparkTools"""
    
    @pytest.fixture
    def tools(self):
        """Create AskSparkTools instance with mocked dependencies"""
        with patch('askspark.agents.tools.UnifiedAIClient') as mock_client:
            with patch('askspark.agents.tools.ModelComparisonEngine') as mock_comparison:
                mock_client.return_value = Mock()
                mock_comparison.return_value = Mock()
                return AskSparkTools()
    
    async def test_compare_ai_models_success(self, tools):
        """Test successful model comparison"""
        # Mock the model comparison engine
        mock_results = [
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "quality_score": 0.8,
                "response_time": 1.0,
                "cost": 0.01
            }
        ]
        
        tools.model_comparison.compare_models_async = AsyncMock(return_value=mock_results)
        
        result = await tools.compare_ai_models("Test task", ["openai", "anthropic"])
        
        assert result["success"] is True
        assert result["task"] == "Test task"
        assert result["results"] == mock_results
        assert "recommendation" in result
        assert "execution_time" in result
        assert "timestamp" in result
        
        # Verify the async method was called correctly
        tools.model_comparison.compare_models_async.assert_called_once()
        call_args = tools.model_comparison.compare_models_async.call_args[0]
        assert call_args[0] == "Test task"
        # Check that providers were converted to model tuples
        providers = call_args[1]
        assert ("openai", "default") in providers
        assert ("anthropic", "default") in providers
    
    async def test_compare_ai_models_failure(self, tools):
        """Test model comparison failure"""
        tools.model_comparison.compare_models_async = AsyncMock(side_effect=Exception("Test error"))
        
        result = await tools.compare_ai_models("Test task")
        
        assert result["success"] is False
        assert result["error"] == "Test error"
        assert "execution_time" in result
    
    async def test_analyze_document_content_success(self, tools):
        """Test successful document analysis"""
        # Mock RAG engine
        mock_rag = Mock()
        mock_rag.process_and_store_document_content = Mock()
        mock_rag.query_documents.return_value = {
            "answer": "Document analysis result",
            "sources": ["source1", "source2"]
        }
        
        tools.rag_engine = mock_rag
        
        result = await tools.analyze_document_content("Test document", "Test question")
        
        assert result["success"] is True
        assert result["question"] == "Test question"
        assert result["answer"] == "Document analysis result"
        assert result["sources"] == ["source1", "source2"]
        assert "execution_time" in result
        assert "timestamp" in result
        
        # Verify RAG methods were called
        mock_rag.process_and_store_document_content.assert_called_once()
        mock_rag.query_documents.assert_called_once_with("Test question", n_results=3)
    
    async def test_analyze_document_content_without_question(self, tools):
        """Test document analysis without specific question"""
        mock_rag = Mock()
        mock_rag.process_and_store_document_content = Mock()
        mock_rag.query_documents.return_value = {
            "answer": "Document summary",
            "sources": ["source1"]
        }
        
        tools.rag_engine = mock_rag
        
        result = await tools.analyze_document_content("Test document")
        
        assert result["success"] is True
        assert result["question"] == ""
        assert result["answer"] == "Document summary"
        
        # Verify summary query was used
        mock_rag.query_documents.assert_called_once()
        call_args = mock_rag.query_documents.call_args[0][0]
        assert "summary" in call_args.lower()
    
    async def test_analyze_document_content_failure(self, tools):
        """Test document analysis failure"""
        with patch.object(tools, '_get_rag_engine') as mock_get_rag:
            mock_get_rag.side_effect = Exception("RAG engine error")
            
            result = await tools.analyze_document_content("Test document", "Test question")
            
            assert result["success"] is False
            assert result["error"] == "RAG engine error"
            assert "execution_time" in result
    
    async def test_get_provider_status_success(self, tools):
        """Test successful provider status check"""
        # Mock unified client get_client method
        def mock_get_client(provider):
            if provider in ["openai", "anthropic"]:
                return Mock()  # Available client
            else:
                raise Exception("Not configured")
        
        tools.unified_client.get_client = mock_get_client
        
        result = await tools.get_provider_status()
        
        assert result["success"] is True
        assert "providers" in result
        assert "total_providers" in result
        assert "available_providers" in result
        assert result["total_providers"] == 5  # All providers checked
        
        # Check specific provider statuses
        providers = result["providers"]
        assert providers["openai"]["available"] is True
        assert providers["openai"]["status"] == "online"
        assert providers["anthropic"]["available"] is True
        assert providers["anthropic"]["status"] == "online"
        assert providers["google"]["available"] is False
        assert providers["google"]["status"] == "error"
    
    async def test_get_provider_status_partial_failure(self, tools):
        """Test provider status with some failures"""
        tools.unified_client.get_client = Mock(side_effect=Exception("Client error"))
        
        result = await tools.get_provider_status()
        
        assert result["success"] is True
        assert result["available_providers"] == 0
        
        # All providers should show error status
        for provider_status in result["providers"].values():
            assert provider_status["available"] is False
            assert provider_status["status"] == "error"
    
    async def test_calculate_cost_estimate_success(self, tools):
        """Test successful cost calculation"""
        result = await tools.calculate_cost_estimate("gpt-4o-mini", 5000, "completion")
        
        assert result["success"] is True
        assert result["model"] == "gpt-4o-mini"
        assert result["tokens"] == 5000
        assert result["operation_type"] == "completion"
        
        # Check cost breakdown
        cost_breakdown = result["cost_breakdown"]
        assert "input_tokens" in cost_breakdown
        assert "output_tokens" in cost_breakdown
        assert "input_cost" in cost_breakdown
        assert "output_cost" in cost_breakdown
        assert "total_cost" in cost_breakdown
        
        # Verify token split (70/30)
        assert cost_breakdown["input_tokens"] == 3500  # 70% of 5000
        assert cost_breakdown["output_tokens"] == 1500  # 30% of 5000
        
        # Verify cost calculation
        assert cost_breakdown["total_cost"] > 0
        assert cost_breakdown["total_cost"] == cost_breakdown["input_cost"] + cost_breakdown["output_cost"]
    
    async def test_calculate_cost_estimate_unknown_model(self, tools):
        """Test cost calculation with unknown model"""
        result = await tools.calculate_cost_estimate("unknown-model", 1000)
        
        assert result["success"] is True
        assert result["model"] == "unknown-model"
        
        # Should use default cost rates
        cost_breakdown = result["cost_breakdown"]
        assert cost_breakdown["input_cost"] == 0.001  # Default rate
        assert cost_breakdown["output_cost"] == 0.002  # Default rate
    
    async def test_calculate_cost_estimate_optimization_suggestions(self, tools):
        """Test cost calculation optimization suggestions"""
        # Test with high cost
        result = await tools.calculate_cost_estimate("gpt-4", 100000)
        
        assert result["success"] is True
        suggestions = result["optimization_suggestions"]
        assert len(suggestions) > 0
        
        # Should suggest smaller model for high cost
        high_cost_suggestion = any("smaller model" in s.lower() for s in suggestions)
        assert high_cost_suggestion
        
        # Should suggest chunking for high token usage
        chunking_suggestion = any("chunking" in s.lower() for s in suggestions)
        assert chunking_suggestion


class TestToolIntegration:
    """Test tool integration and registry"""
    
    def test_tools_registry(self):
        """Test tools registry contains all expected tools"""
        from askspark.agents.tools import tools_registry
        
        expected_tools = [
            'compare_ai_models',
            'analyze_document_content', 
            'get_provider_status',
            'calculate_cost_estimate'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools_registry
            assert callable(tools_registry[tool_name])
    
    def test_get_all_tools(self):
        """Test get_all_tools function"""
        from askspark.agents.tools import get_all_tools
        
        tools = get_all_tools()
        assert len(tools) >= 4
        
        # Verify all tools are callable
        for tool in tools:
            assert callable(tool)


# Performance tests for tools
@pytest.mark.performance
class TestToolPerformance:
    """Performance tests for AskSparkTools"""
    
    @pytest.mark.asyncio
    async def test_cost_calculation_performance(self):
        """Test cost calculation performance"""
        with patch('askspark.agents.tools.UnifiedAIClient'):
            with patch('askspark.agents.tools.ModelComparisonEngine'):
                tools = AskSparkTools()
                
                start_time = datetime.now()
                result = await tools.calculate_cost_estimate("gpt-4o-mini", 10000)
                end_time = datetime.now()
                
                # Should complete very quickly (no external calls)
                execution_time = (end_time - start_time).total_seconds()
                assert execution_time < 0.1  # Should be under 100ms
                assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_provider_status_performance(self):
        """Test provider status check performance"""
        with patch('askspark.agents.tools.UnifiedAIClient') as mock_client:
            with patch('askspark.agents.tools.ModelComparisonEngine'):
                # Mock fast client responses
                mock_client.return_value.get_client.return_value = Mock()
                
                tools = AskSparkTools()
                
                start_time = datetime.now()
                result = await tools.get_provider_status()
                end_time = datetime.now()
                
                # Should complete quickly even with multiple providers
                execution_time = (end_time - start_time).total_seconds()
                assert execution_time < 0.5  # Should be under 500ms
                assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
