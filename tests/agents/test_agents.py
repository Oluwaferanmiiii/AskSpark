"""
Test suite for AskSpark Agents - Week 2 Lab 1
Comprehensive testing for OpenAI Agents SDK integration
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add src to path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv

# Load test environment
load_dotenv()

# Import modules to test
from askspark.agents.base_agent import (
    AskSparkAgentBase,
    ModelComparisonAgent,
    DocumentAnalysisAgent,
    WorkflowOrchestrationAgent,
    AgentManager,
    AgentResponse
)
from askspark.agents.tools import AskSparkTools, tools_registry
from askspark.agents.demo import AskSparkAgentDemo


class TestAgentResponse:
    """Test AgentResponse dataclass"""
    
    def test_agent_response_creation(self):
        """Test creating AgentResponse with all fields"""
        response = AgentResponse(
            content="Test response",
            agent_name="TestAgent",
            execution_time=1.5,
            trace_id="test-trace-123",
            model_used="gpt-4o-mini",
            tokens_used=100,
            cost=0.01
        )
        
        assert response.content == "Test response"
        assert response.agent_name == "TestAgent"
        assert response.execution_time == 1.5
        assert response.trace_id == "test-trace-123"
        assert response.model_used == "gpt-4o-mini"
        assert response.tokens_used == 100
        assert response.cost == 0.01
    
    def test_agent_response_optional_fields(self):
        """Test creating AgentResponse with optional fields"""
        response = AgentResponse(
            content="Test response",
            agent_name="TestAgent",
            execution_time=1.0,
            trace_id="test-trace-456",
            model_used="gpt-4o-mini"
        )
        
        assert response.tokens_used is None
        assert response.cost is None


class TestAskSparkAgentBase:
    """Test base agent functionality"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                return AskSparkAgentBase("TestAgent", "Test instructions")
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization"""
        assert mock_agent.name == "TestAgent"
        assert mock_agent.instructions == "Test instructions"
        assert mock_agent.model == "gpt-4o-mini"
        assert mock_agent.trace_id is None
    
    def test_calculate_cost_with_tokens(self, mock_agent):
        """Test cost calculation with token count"""
        cost = mock_agent._calculate_cost(1000)
        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_calculate_cost_no_tokens(self, mock_agent):
        """Test cost calculation without token count"""
        cost = mock_agent._calculate_cost(None)
        assert cost is None
    
    def test_calculate_cost_different_models(self, mock_agent):
        """Test cost calculation for different models"""
        mock_agent.model = "gpt-4o"
        cost_4o = mock_agent._calculate_cost(1000)
        
        mock_agent.model = "gpt-4o-mini"
        cost_4o_mini = mock_agent._calculate_cost(1000)
        
        # gpt-4o should be more expensive than gpt-4o-mini
        assert cost_4o > cost_4o_mini


@pytest.mark.asyncio
class TestModelComparisonAgent:
    """Test ModelComparisonAgent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create ModelComparisonAgent for testing"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                return ModelComparisonAgent()
    
    def test_agent_initialization(self, agent):
        """Test ModelComparisonAgent initialization"""
        assert agent.name == "ModelComparisonAgent"
        assert "model comparison expert" in agent.instructions.lower()
        assert agent.model == "gpt-4o-mini"
    
    async def test_compare_models_for_task(self, agent):
        """Test model comparison for specific task"""
        with patch.object(agent, 'run') as mock_run:
            mock_response = AgentResponse(
                content="I recommend using gpt-4o-mini for this task",
                agent_name="ModelComparisonAgent",
                execution_time=1.2,
                trace_id="test-123",
                model_used="gpt-4o-mini"
            )
            mock_run.return_value = mock_response
            
            result = await agent.compare_models_for_task("Create a sales email")
            
            assert result.content == "I recommend using gpt-4o-mini for this task"
            assert result.agent_name == "ModelComparisonAgent"
            mock_run.assert_called_once()


@pytest.mark.asyncio
class TestDocumentAnalysisAgent:
    """Test DocumentAnalysisAgent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create DocumentAnalysisAgent for testing"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                return DocumentAnalysisAgent()
    
    def test_agent_initialization(self, agent):
        """Test DocumentAnalysisAgent initialization"""
        assert agent.name == "DocumentAnalysisAgent"
        assert "document analyst" in agent.instructions.lower()
        assert agent.model == "gpt-4o-mini"
    
    async def test_analyze_document_with_question(self, agent):
        """Test document analysis with specific question"""
        with patch.object(agent, 'run') as mock_run:
            mock_response = AgentResponse(
                content="Based on the document, the key features are...",
                agent_name="DocumentAnalysisAgent",
                execution_time=2.1,
                trace_id="test-456",
                model_used="gpt-4o-mini"
            )
            mock_run.return_value = mock_response
            
            result = await agent.analyze_document("Test document content", "What are the key features?")
            
            assert "key features" in result.content
            assert result.agent_name == "DocumentAnalysisAgent"
            mock_run.assert_called_once()
    
    async def test_analyze_document_without_question(self, agent):
        """Test document analysis without specific question"""
        with patch.object(agent, 'run') as mock_run:
            mock_response = AgentResponse(
                content="Document analysis shows...",
                agent_name="DocumentAnalysisAgent",
                execution_time=1.8,
                trace_id="test-789",
                model_used="gpt-4o-mini"
            )
            mock_run.return_value = mock_response
            
            result = await agent.analyze_document("Test document content")
            
            assert "Document analysis shows" in result.content
            mock_run.assert_called_once()


@pytest.mark.asyncio
class TestWorkflowOrchestrationAgent:
    """Test WorkflowOrchestrationAgent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create WorkflowOrchestrationAgent for testing"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                return WorkflowOrchestrationAgent()
    
    def test_agent_initialization(self, agent):
        """Test WorkflowOrchestrationAgent initialization"""
        assert agent.name == "WorkflowOrchestrationAgent"
        assert "workflow orchestration expert" in agent.instructions.lower()
        assert agent.model == "gpt-4o-mini"
    
    async def test_plan_workflow(self, agent):
        """Test workflow planning"""
        with patch.object(agent, 'run') as mock_run:
            mock_response = AgentResponse(
                content="I recommend starting with document analysis, then model comparison",
                agent_name="WorkflowOrchestrationAgent",
                execution_time=1.5,
                trace_id="test-workflow",
                model_used="gpt-4o-mini"
            )
            mock_run.return_value = mock_response
            
            result = await agent.plan_workflow(
                "Analyze document and compare models",
                ["document_analysis", "model_comparison"]
            )
            
            assert "document analysis" in result.content.lower()
            assert result.agent_name == "WorkflowOrchestrationAgent"
            mock_run.assert_called_once()


class TestAgentManager:
    """Test AgentManager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create AgentManager for testing"""
        with patch('askspark.agents.base_agent.ModelComparisonAgent'):
            with patch('askspark.agents.base_agent.DocumentAnalysisAgent'):
                with patch('askspark.agents.base_agent.WorkflowOrchestrationAgent'):
                    return AgentManager()
    
    def test_manager_initialization(self, manager):
        """Test AgentManager initialization"""
        assert len(manager.agents) == 3
        assert 'model_comparison' in manager.agents
        assert 'document_analysis' in manager.agents
        assert 'workflow_orchestration' in manager.agents
    
    def test_list_agents(self, manager):
        """Test listing available agents"""
        agents = manager.list_agents()
        assert len(agents) == 3
        assert 'model_comparison' in agents
        assert 'document_analysis' in agents
        assert 'workflow_orchestration' in agents
    
    def test_get_agent(self, manager):
        """Test getting specific agent"""
        agent = manager.get_agent('model_comparison')
        assert agent is not None
        assert agent.name == 'ModelComparisonAgent'
    
    def test_get_nonexistent_agent(self, manager):
        """Test getting non-existent agent"""
        agent = manager.get_agent('nonexistent')
        assert agent is None
    
    def test_get_agent_stats(self, manager):
        """Test getting agent statistics"""
        stats = manager.get_agent_stats()
        assert stats['total_agents'] == 3
        assert len(stats['available_agents']) == 3
        assert len(stats['models_used']) == 3


@pytest.mark.asyncio
class TestAskSparkTools:
    """Test AskSparkTools functionality"""
    
    @pytest.fixture
    def tools(self):
        """Create AskSparkTools for testing"""
        with patch('askspark.agents.tools.UnifiedAIClient'):
            with patch('askspark.agents.tools.ModelComparisonEngine'):
                return AskSparkTools()
    
    def test_tools_initialization(self, tools):
        """Test tools initialization"""
        assert tools.unified_client is not None
        assert tools.model_comparison is not None
        assert tools.rag_engine is None  # Lazy initialization
    
    async def test_compare_ai_models(self, tools):
        """Test model comparison tool"""
        with patch.object(tools.model_comparison, 'compare_models_async') as mock_compare:
            mock_compare.return_value = [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "quality_score": 0.8,
                    "response_time": 1.2,
                    "cost": 0.01
                }
            ]
            
            result = await tools.compare_ai_models("Test task")
            
            assert result['success'] is True
            assert 'task' in result
            assert 'results' in result
            assert 'recommendation' in result
            assert 'execution_time' in result
    
    async def test_analyze_document_content(self, tools):
        """Test document analysis tool"""
        with patch.object(tools, '_get_rag_engine') as mock_rag:
            mock_rag_instance = Mock()
            mock_rag_instance.process_and_store_document_content = Mock()
            mock_rag_instance.query_documents.return_value = {
                "answer": "Document analysis result",
                "sources": ["source1", "source2"]
            }
            mock_rag.return_value = mock_rag_instance
            
            result = await tools.analyze_document_content("Test document", "Test question")
            
            assert result['success'] is True
            assert 'answer' in result
            assert 'sources' in result
            assert result['question'] == "Test question"
    
    async def test_get_provider_status(self, tools):
        """Test provider status tool"""
        with patch.object(tools.unified_client, 'get_client') as mock_get_client:
            mock_get_client.return_value = Mock()  # Simulate available client
            
            result = await tools.get_provider_status()
            
            assert result['success'] is True
            assert 'providers' in result
            assert 'total_providers' in result
            assert 'available_providers' in result
    
    async def test_calculate_cost_estimate(self, tools):
        """Test cost calculation tool"""
        result = await tools.calculate_cost_estimate("gpt-4o-mini", 1000)
        
        assert result['success'] is True
        assert 'model' in result
        assert 'tokens' in result
        assert 'cost_breakdown' in result
        assert 'total_cost' in result['cost_breakdown']
        assert result['cost_breakdown']['total_cost'] > 0


class TestAskSparkAgentDemo:
    """Test AskSparkAgentDemo functionality"""
    
    @pytest.fixture
    def demo(self):
        """Create AskSparkAgentDemo for testing"""
        with patch('askspark.agents.demo.AgentManager'):
            with patch('askspark.agents.demo.AskSparkTools'):
                with patch('askspark.agents.demo.UnifiedAIClient'):
                    return AskSparkAgentDemo()
    
    async def test_demo_initialization(self, demo):
        """Test demo initialization"""
        assert demo.agent_manager is not None
        assert demo.tools is not None
        assert demo.unified_client is not None
    
    async def test_demo_provider_status(self, demo):
        """Test provider status demo"""
        with patch.object(demo.tools, 'get_provider_status') as mock_status:
            mock_status.return_value = {
                'success': True,
                'total_providers': 3,
                'available_providers': 2,
                'providers': {
                    'openai': {'available': True, 'status': 'online'},
                    'anthropic': {'available': False, 'status': 'not_configured'}
                }
            }
            
            result = await demo.demo_provider_status()
            
            assert result['success'] is True
            assert result['total_providers'] == 3
            assert result['available_providers'] == 2
    
    async def test_demo_cost_calculation(self, demo):
        """Test cost calculation demo"""
        with patch.object(demo.tools, 'calculate_cost_estimate') as mock_cost:
            mock_cost.return_value = {
                'success': True,
                'model': 'gpt-4o-mini',
                'tokens': 5000,
                'cost_breakdown': {
                    'total_cost': 0.0015,
                    'input_cost': 0.0005,
                    'output_cost': 0.001
                }
            }
            
            result = await demo.demo_cost_calculation('gpt-4o-mini', 5000)
            
            assert result['success'] is True
            assert result['model'] == 'gpt-4o-mini'
            assert result['tokens'] == 5000


class TestIntegration:
    """Integration tests for the complete agent system"""
    
    @pytest.mark.integration
    async def test_full_agent_workflow(self):
        """Test complete agent workflow integration"""
        # This test requires actual API keys, so we'll mock the external calls
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                manager = AgentManager()
                
                # Test agent availability
                agents = manager.list_agents()
                assert len(agents) >= 3
                
                # Test agent stats
                stats = manager.get_agent_stats()
                assert stats['total_agents'] >= 3
    
    @pytest.mark.integration
    def test_tools_registry(self):
        """Test tools registry integration"""
        tools = list(tools_registry.keys())
        expected_tools = [
            'compare_ai_models',
            'analyze_document_content',
            'get_provider_status',
            'calculate_cost_estimate'
        ]
        
        for tool in expected_tools:
            assert tool in tools


# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance tests for agent system"""
    
    @pytest.mark.asyncio
    async def test_agent_response_time(self):
        """Test agent response time performance"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                agent = ModelComparisonAgent()
                
                # Mock the run method to simulate response
                with patch.object(agent, 'run') as mock_run:
                    mock_response = AgentResponse(
                        content="Test response",
                        agent_name="ModelComparisonAgent",
                        execution_time=0.5,  # Fast response
                        trace_id="test-perf",
                        model_used="gpt-4o-mini"
                    )
                    mock_run.return_value = mock_response
                    
                    start_time = datetime.now()
                    result = await agent.compare_models_for_task("Test task")
                    end_time = datetime.now()
                    
                    # The actual execution should be very fast since we're mocking
                    assert (end_time - start_time).total_seconds() < 1.0
                    assert result.execution_time == 0.5


# Error handling tests
@pytest.mark.error_handling
class TestErrorHandling:
    """Test error handling in agent system"""
    
    @pytest.mark.asyncio
    async def test_agent_execution_error(self):
        """Test agent execution error handling"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                agent = ModelComparisonAgent()
                
                # Mock run method to raise an exception
                with patch.object(agent, 'run') as mock_run:
                    mock_run.side_effect = Exception("Test error")
                    
                    with pytest.raises(Exception):
                        await agent.compare_models_for_task("Test task")
    
    async def test_tools_error_handling(self):
        """Test tools error handling"""
        with patch('askspark.agents.tools.UnifiedAIClient'):
            with patch('askspark.agents.tools.ModelComparisonEngine'):
                tools = AskSparkTools()
                
                # Mock model comparison to raise an error
                with patch.object(tools.model_comparison, 'compare_models_async') as mock_compare:
                    mock_compare.side_effect = Exception("Comparison failed")
                    
                    result = await tools.compare_ai_models("Test task")
                    
                    assert result['success'] is False
                    assert 'error' in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
