"""
Integration tests for AskSpark Agents
Week 2 Lab 1 - End-to-end testing
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add src to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from askspark.agents import (
    AgentManager,
    AskSparkAgentDemo,
    get_all_tools,
    get_package_info
)


@pytest.mark.integration
class TestAgentSystemIntegration:
    """Integration tests for the complete agent system"""
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment for integration testing"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel') as mock_model:
            with patch('askspark.agents.base_agent.UnifiedAIClient') as mock_client:
                with patch('askspark.agents.tools.UnifiedAIClient') as mock_tools_client:
                    with patch('askspark.agents.tools.ModelComparisonEngine') as mock_comparison:
                        # Configure mocks
                        mock_client.return_value = Mock()
                        mock_tools_client.return_value = Mock()
                        mock_comparison.return_value = Mock()
                        
                        yield {
                            'model': mock_model,
                            'client': mock_client,
                            'tools_client': mock_tools_client,
                            'comparison': mock_comparison
                        }
    
    def test_package_info(self):
        """Test package information retrieval"""
        info = get_package_info()
        
        assert info['name'] == 'AskSpark Agents'
        assert 'version' in info
        assert 'author' in info
        assert 'description' in info
        assert 'features' in info
        assert len(info['features']) > 0
    
    def test_get_all_tools_integration(self):
        """Test getting all tools for integration"""
        tools = get_all_tools()
        
        assert len(tools) >= 4
        for tool in tools:
            assert callable(tool)
    
    @pytest.mark.asyncio
    async def test_agent_manager_integration(self, mock_environment):
        """Test AgentManager integration"""
        manager = AgentManager()
        
        # Test agent availability
        agents = manager.list_agents()
        expected_agents = ['model_comparison', 'document_analysis', 'workflow_orchestration']
        
        for agent_name in expected_agents:
            assert agent_name in agents
        
        # Test agent stats
        stats = manager.get_agent_stats()
        assert stats['total_agents'] >= 3
        assert len(stats['available_agents']) >= 3
        assert len(stats['models_used']) >= 3
        
        # Test getting specific agents
        for agent_name in expected_agents:
            agent = manager.get_agent(agent_name)
            assert agent is not None
            assert hasattr(agent, 'run')
    
    @pytest.mark.asyncio
    async def test_demo_integration(self, mock_environment):
        """Test AskSparkAgentDemo integration"""
        demo = AskSparkAgentDemo()
        
        # Verify demo components
        assert demo.agent_manager is not None
        assert demo.tools is not None
        assert demo.unified_client is not None
        
        # Test provider status demo
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
    
    @pytest.mark.asyncio
    async def test_cross_agent_communication(self, mock_environment):
        """Test communication between different agents"""
        manager = AgentManager()
        
        # Mock agent responses
        model_agent = manager.get_agent('model_comparison')
        workflow_agent = manager.get_agent('workflow_orchestration')
        
        # Mock model agent response
        with patch.object(model_agent, 'run') as mock_model_run:
            mock_model_run.return_value = Mock(
                content="Use gpt-4o-mini for this task",
                agent_name="ModelComparisonAgent",
                execution_time=1.0,
                trace_id="test-123",
                model_used="gpt-4o-mini"
            )
            
            # Mock workflow agent response
            with patch.object(workflow_agent, 'run') as mock_workflow_run:
                mock_workflow_run.return_value = Mock(
                    content="I recommend using model_comparison agent first, then document_analysis",
                    agent_name="WorkflowOrchestrationAgent", 
                    execution_time=1.5,
                    trace_id="test-456",
                    model_used="gpt-4o-mini"
                )
                
                # Execute workflow planning
                task = "Analyze document and recommend best model"
                available_agents = manager.list_agents()
                
                workflow_result = await workflow_agent.plan_workflow(task, available_agents)
                
                # Verify workflow agent considered other agents
                assert "model_comparison" in workflow_result.content
                assert workflow_result.agent_name == "WorkflowOrchestrationAgent"
                
                # Verify model agent can be called independently
                model_result = await model_agent.compare_models_for_task("Create summary")
                assert model_result.agent_name == "ModelComparisonAgent"


@pytest.mark.integration
@pytest.mark.asyncio
class TestToolIntegration:
    """Integration tests for tool system"""
    
    @pytest.fixture
    def mock_tools_environment(self):
        """Create mock environment for tools testing"""
        with patch('askspark.agents.tools.UnifiedAIClient') as mock_client:
            with patch('askspark.agents.tools.ModelComparisonEngine') as mock_comparison:
                with patch('askspark.agents.tools.RAGEngine') as mock_rag:
                    # Configure mocks
                    mock_client.return_value = Mock()
                    mock_comparison.return_value = Mock()
                    mock_rag.return_value = Mock()
                    
                    yield {
                        'client': mock_client,
                        'comparison': mock_comparison,
                        'rag': mock_rag
                    }
    
    async def test_tools_workflow_integration(self, mock_tools_environment):
        """Test complete workflow using multiple tools"""
        from askspark.agents.tools import AskSparkTools
        
        tools = AskSparkTools()
        
        # Mock tool responses
        mock_tools_environment['comparison'].compare_models_async = AsyncMock(return_value=[
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "quality_score": 0.8,
                "response_time": 1.0,
                "cost": 0.01
            }
        ])
        
        mock_rag_instance = Mock()
        mock_rag_instance.process_and_store_document_content = Mock()
        mock_rag_instance.query_documents.return_value = {
            "answer": "Document contains technical specifications",
            "sources": ["doc_page_1", "doc_page_2"]
        }
        mock_tools_environment['rag'].return_value = mock_rag_instance
        
        # Mock provider status
        def mock_get_client(provider):
            if provider == "openai":
                return Mock()
            else:
                raise Exception("Not available")
        
        tools.unified_client.get_client = mock_get_client
        
        # Execute complete workflow
        # 1. Check provider status
        status_result = await tools.get_provider_status()
        assert status_result['success'] is True
        
        # 2. Compare models for task
        model_result = await tools.compare_ai_models("Analyze technical document")
        assert model_result['success'] is True
        assert len(model_result['results']) > 0
        
        # 3. Analyze document
        doc_result = await tools.analyze_document_content("Technical specs...", "What are the key features?")
        assert doc_result['success'] is True
        assert "technical specifications" in doc_result['answer']
        
        # 4. Calculate costs
        cost_result = await tools.calculate_cost_estimate("gpt-4o-mini", 5000)
        assert cost_result['success'] is True
        assert cost_result['cost_breakdown']['total_cost'] > 0
    
    async def test_tool_error_propagation(self, mock_tools_environment):
        """Test error handling across tool chain"""
        from askspark.agents.tools import AskSparkTools
        
        tools = AskSparkTools()
        
        # Mock comparison engine to fail
        mock_tools_environment['comparison'].compare_models_async = AsyncMock(
            side_effect=Exception("Comparison service unavailable")
        )
        
        # Test error propagation
        result = await tools.compare_ai_models("Test task")
        
        assert result['success'] is False
        assert "Comparison service unavailable" in result['error']
        assert 'execution_time' in result


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndScenarios:
    """End-to-end scenario tests"""
    
    @pytest.fixture
    def complete_mock_environment(self):
        """Create complete mock environment"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel') as mock_model:
            with patch('askspark.agents.base_agent.UnifiedAIClient') as mock_client:
                with patch('askspark.agents.tools.UnifiedAIClient') as mock_tools_client:
                    with patch('askspark.agents.tools.ModelComparisonEngine') as mock_comparison:
                        with patch('askspark.agents.tools.RAGEngine') as mock_rag:
                            # Configure all mocks
                            mock_client.return_value = Mock()
                            mock_tools_client.return_value = Mock()
                            mock_comparison.return_value = Mock()
                            mock_rag.return_value = Mock()
                            
                            yield {
                                'model': mock_model,
                                'client': mock_client,
                                'tools_client': mock_tools_client,
                                'comparison': mock_comparison,
                                'rag': mock_rag
                            }
    
    async def test_business_document_analysis_scenario(self, complete_mock_environment):
        """Test complete business document analysis scenario"""
        # Setup
        manager = AgentManager()
        model_agent = manager.get_agent('model_comparison')
        doc_agent = manager.get_agent('document_analysis')
        workflow_agent = manager.get_agent('workflow_orchestration')
        
        # Mock responses
        with patch.object(model_agent, 'run') as mock_model_run:
            with patch.object(doc_agent, 'run') as mock_doc_run:
                with patch.object(workflow_agent, 'run') as mock_workflow_run:
                    
                    # Configure mock responses
                    mock_model_run.return_value = Mock(
                        content="For business document analysis, I recommend gpt-4o-mini for cost efficiency",
                        agent_name="ModelComparisonAgent",
                        execution_time=1.2,
                        trace_id="model-123",
                        model_used="gpt-4o-mini"
                    )
                    
                    mock_doc_run.return_value = Mock(
                        content="The business plan outlines Q1 objectives, financial projections, and market analysis",
                        agent_name="DocumentAnalysisAgent",
                        execution_time=2.1,
                        trace_id="doc-456",
                        model_used="gpt-4o-mini"
                    )
                    
                    mock_workflow_run.return_value = Mock(
                        content="Workflow: 1) Use model_comparison to select best model, 2) Use document_analysis for content extraction",
                        agent_name="WorkflowOrchestrationAgent",
                        execution_time=1.0,
                        trace_id="workflow-789",
                        model_used="gpt-4o-mini"
                    )
                    
                    # Execute scenario
                    business_plan = "Executive Summary:\nOur company aims to..."
                    
                    # Step 1: Plan workflow
                    workflow_result = await workflow_agent.plan_workflow(
                        "Analyze business plan and extract key insights",
                        manager.list_agents()
                    )
                    
                    # Step 2: Get model recommendation
                    model_result = await model_agent.compare_models_for_task(
                        "Analyze business document for key insights"
                    )
                    
                    # Step 3: Analyze document
                    doc_result = await doc_agent.analyze_document(
                        business_plan,
                        "What are the key business objectives?"
                    )
                    
                    # Verify results
                    assert "model_comparison" in workflow_result.content
                    assert "document_analysis" in workflow_result.content
                    assert "gpt-4o-mini" in model_result.content
                    assert "business plan" in doc_result.content.lower()
                    assert "Q1 objectives" in doc_result.content
    
    async def test_cost_optimization_scenario(self, complete_mock_environment):
        """Test cost optimization scenario"""
        from askspark.agents.tools import AskSparkTools
        
        tools = AskSparkTools()
        
        # Mock comparison engine with different cost models
        mock_tools_environment['comparison'].compare_models_async = AsyncMock(return_value=[
            {
                "provider": "openai",
                "model": "gpt-4o",
                "quality_score": 0.9,
                "response_time": 2.0,
                "cost": 0.10
            },
            {
                "provider": "openai", 
                "model": "gpt-4o-mini",
                "quality_score": 0.8,
                "response_time": 1.0,
                "cost": 0.01
            },
            {
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "quality_score": 0.75,
                "response_time": 1.5,
                "cost": 0.02
            }
        ])
        
        # Execute cost optimization analysis
        task = "Generate customer support responses"
        
        # Get model comparisons
        comparison_result = await tools.compare_ai_models(task)
        assert comparison_result['success'] is True
        
        # Get cost estimates for high-usage scenario
        high_usage_cost = await tools.calculate_cost_estimate("gpt-4o", 50000)
        low_usage_cost = await tools.calculate_cost_estimate("gpt-4o-mini", 50000)
        
        # Verify cost optimization
        assert high_usage_cost['success'] is True
        assert low_usage_cost['success'] is True
        assert high_usage_cost['cost_breakdown']['total_cost'] > low_usage_cost['cost_breakdown']['total_cost']
        
        # Check optimization suggestions for high usage
        assert len(high_usage_cost['optimization_suggestions']) > 0
        
        # Verify recommendation points to cost-effective option
        recommendation = comparison_result['recommendation']
        assert recommendation['model'] in ['gpt-4o-mini', 'claude-3-haiku']  # Should recommend cheaper option


@pytest.mark.integration
class TestSystemPerformance:
    """System performance integration tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self):
        """Test concurrent execution of multiple agents"""
        with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel'):
            with patch('askspark.agents.base_agent.UnifiedAIClient'):
                manager = AgentManager()
                
                # Get multiple agents
                agents = [
                    manager.get_agent('model_comparison'),
                    manager.get_agent('document_analysis'),
                    manager.get_agent('workflow_orchestration')
                ]
                
                # Mock all agent runs
                for agent in agents:
                    with patch.object(agent, 'run') as mock_run:
                        mock_run.return_value = Mock(
                            content=f"Response from {agent.name}",
                            agent_name=agent.name,
                            execution_time=0.5,
                            trace_id=f"trace-{agent.name}",
                            model_used="gpt-4o-mini"
                        )
                
                # Execute agents concurrently
                tasks = [
                    agent.compare_models_for_task("Test task") if agent.name == "ModelComparisonAgent"
                    else agent.analyze_document("Test doc", "Test question") if agent.name == "DocumentAnalysisAgent"
                    else agent.plan_workflow("Test workflow", ["model_comparison"])
                    for agent in agents
                ]
                
                start_time = datetime.now()
                results = await asyncio.gather(*tasks)
                end_time = datetime.now()
                
                # Verify all completed successfully
                assert len(results) == 3
                for result in results:
                    assert result.agent_name in ["ModelComparisonAgent", "DocumentAnalysisAgent", "WorkflowOrchestrationAgent"]
                
                # Verify concurrent execution (should be faster than sequential)
                execution_time = (end_time - start_time).total_seconds()
                assert execution_time < 2.0  # Should complete in under 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
