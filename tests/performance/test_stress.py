"""
Performance and stress tests for AskSpark application
"""

import pytest
from unittest.mock import Mock, patch
import time
import threading
import concurrent.futures
from typing import List, Dict

from src.askspark.core.ai_providers import UnifiedAIClient, ModelResponse
from src.askspark.core.model_comparison import ModelComparisonEngine
from src.askspark.workflows.engine import WorkflowEngine


class TestPerformanceStress:
    """Performance and stress testing"""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Mock AI client with performance tracking"""
        client = Mock(spec=UnifiedAIClient)
        
        # Track response times
        response_times = []
        
        def mock_generate_response(prompt, provider, model):
            # Simulate realistic response times
            response_time = 0.5 + (hash(prompt) % 100) / 100  # 0.5-1.5 seconds
            time.sleep(response_time / 100)  # Simulate processing time
            
            response = ModelResponse(
                content=f"Response from {provider} about {prompt[:50]}...",
                model=model,
                provider=provider,
                response_time=response_time,
                tokens_used=len(prompt.split()) * 2,
                cost=0.001
            )
            response_times.append(response_time)
            return response
        
        client.generate_response = mock_generate_response
        client.response_times = response_times
        return client
    
    def test_concurrent_api_calls(self, mock_ai_client):
        """Test concurrent API calls performance"""
        num_requests = 10
        prompts = [f"Test prompt {i}" for i in range(num_requests)]
        
        start_time = time.time()
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    mock_ai_client.generate_response, 
                    prompt, "openai", "gpt-3.5-turbo"
                )
                for prompt in prompts
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all requests completed
        assert len(results) == num_requests
        assert all(result is not None for result in results)
        
        # Verify performance (should be faster than sequential)
        assert total_time < 2.0  # Should complete in under 2 seconds with concurrency
        
        # Verify response time tracking
        assert len(mock_ai_client.response_times) == num_requests
        avg_response_time = sum(mock_ai_client.response_times) / len(mock_ai_client.response_times)
        assert 0.5 <= avg_response_time <= 1.5
    
    def test_model_comparison_performance(self, mock_ai_client):
        """Test model comparison engine performance"""
        with patch('src.askspark.core.model_comparison.UnifiedAIClient', return_value=mock_ai_client):
            engine = ModelComparisonEngine()
            
            num_comparisons = 5
            prompt = "Compare AI model performance"
            providers_models = [
                ("openai", "gpt-3.5-turbo"),
                ("anthropic", "claude-3-haiku-20240307")
            ]
            
            start_time = time.time()
            
            # Run multiple comparisons
            for _ in range(num_comparisons):
                results = engine.compare_models(prompt, providers_models)
                assert len(results) == 2
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify performance
            assert total_time < 5.0  # Should complete in under 5 seconds
            assert len(engine.comparison_history) == num_comparisons * 2
    
    def test_workflow_engine_performance(self):
        """Test workflow engine performance under load"""
        with patch('src.askspark.workflows.engine.NotificationService') as mock_notification:
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            
            engine = WorkflowEngine({})
            
            # Create multiple workflows
            num_workflows = 10
            for i in range(num_workflows):
                engine.create_workflow(
                    f"workflow_{i}",
                    f"Workflow {i}",
                    f"Test workflow {i}",
                    [{"type": "manual", "config": {}}],
                    [{
                        "type": "send_notification",
                        "parameters": {"message": f"Workflow {i} completed"}
                    }]
                )
            
            start_time = time.time()
            
            # Execute all workflows concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(engine.execute_workflow, f"workflow_{i}")
                    for i in range(num_workflows)
                ]
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all workflows executed successfully
            assert len(results) == num_workflows
            assert all(results)  # All should return True
            
            # Verify performance
            assert total_time < 2.0  # Should complete quickly
            
            # Verify notifications were sent
            assert mock_notification_service.send_notification.call_count == num_workflows
    
    def test_memory_usage_stability(self, mock_ai_client):
        """Test memory usage stability over extended operation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('src.askspark.core.model_comparison.UnifiedAIClient', return_value=mock_ai_client):
            engine = ModelComparisonEngine()
            
            # Run many comparisons to test memory stability
            num_iterations = 50
            for i in range(num_iterations):
                prompt = f"Test prompt {i}"
                providers_models = [("openai", "gpt-3.5-turbo")]
                engine.compare_models(prompt, providers_models)
                
                # Check memory every 10 iterations
                if i % 10 == 0:
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable (< 50MB)
                    assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"
        
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_memory_increase < 100 * 1024 * 1024, f"Total memory increased by {total_memory_increase / 1024 / 1024:.2f}MB"
    
    def test_error_handling_under_load(self, mock_ai_client):
        """Test error handling under high load"""
        # Simulate intermittent failures
        failure_rate = 0.3  # 30% failure rate
        
        def mock_generate_response_with_failures(prompt, provider, model):
            import random
            if random.random() < failure_rate:
                raise Exception("Simulated API failure")
            return mock_ai_client.generate_response(prompt, provider, model)
        
        mock_ai_client.generate_response = mock_generate_response_with_failures
        
        with patch('src.askspark.core.model_comparison.UnifiedAIClient', return_value=mock_ai_client):
            engine = ModelComparisonEngine()
            
            num_requests = 20
            successful_requests = 0
            failed_requests = 0
            
            for i in range(num_requests):
                prompt = f"Test prompt {i}"
                providers_models = [("openai", "gpt-3.5-turbo")]
                
                try:
                    results = engine.compare_models(prompt, providers_models)
                    if results:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                except Exception:
                    failed_requests += 1
            
            # Verify error handling
            assert successful_requests + failed_requests == num_requests
            assert successful_requests > 0  # Some requests should succeed
            assert failed_requests > 0  # Some requests should fail
            
            # Success rate should be roughly expected (allowing for variance)
            actual_success_rate = successful_requests / num_requests
            expected_success_rate = 1 - failure_rate
            assert abs(actual_success_rate - expected_success_rate) < 0.2  # Allow 20% variance
    
    def test_response_time_consistency(self, mock_ai_client):
        """Test response time consistency under load"""
        response_times = []
        
        # Make multiple requests and track response times
        for i in range(20):
            start_time = time.time()
            result = mock_ai_client.generate_response(f"Test {i}", "openai", "gpt-3.5-turbo")
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert result is not None
        
        # Analyze response time statistics
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Response times should be consistent
        assert avg_response_time < 1.0  # Average should be under 1 second
        assert max_response_time < 2.0  # Maximum should be under 2 seconds
        assert max_response_time / min_response_time < 3.0  # Variance should be reasonable
    
    def test_concurrent_workflow_execution(self):
        """Test concurrent workflow execution"""
        with patch('src.askspark.workflows.engine.NotificationService') as mock_notification:
            mock_notification_service = Mock()
            mock_notification.return_value = mock_notification_service
            
            engine = WorkflowEngine({})
            
            # Create a workflow that simulates work
            engine.create_workflow(
                "stress_test_workflow",
                "Stress Test Workflow",
                "Workflow for stress testing",
                [{"type": "manual", "config": {}}],
                [{
                    "type": "send_notification",
                    "parameters": {"message": "Stress test completed"}
                }]
            )
            
            num_concurrent_executions = 15
            
            def execute_workflow_with_delay():
                # Simulate some processing time
                time.sleep(0.1)
                return engine.execute_workflow("stress_test_workflow")
            
            start_time = time.time()
            
            # Execute workflows concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(execute_workflow_with_delay)
                    for _ in range(num_concurrent_executions)
                ]
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all executions completed successfully
            assert len(results) == num_concurrent_executions
            assert all(results)  # All should return True
            
            # Verify concurrent execution was faster than sequential
            assert total_time < 1.0  # Should complete in under 1 second
            
            # Verify all notifications were sent
            assert mock_notification_service.send_notification.call_count == num_concurrent_executions
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        with patch('src.askspark.workflows.engine.NotificationService'):
            engine = WorkflowEngine({})
            
            # Start and stop scheduler multiple times
            for _ in range(5):
                engine.start_scheduler()
                assert engine.running is True
                assert engine.scheduler_thread is not None
                
                engine.stop_scheduler()
                assert engine.running is False
            
            # Verify no resource leaks
            assert engine.running is False
            assert engine.scheduler_thread is None or not engine.scheduler_thread.is_alive()


class TestLoadScenarios:
    """Real-world load scenarios"""
    
    def test_dashboard_load_scenario(self):
        """Test typical dashboard usage scenario"""
        with patch('src.askspark.core.ai_providers.UnifiedAIClient') as mock_ai_client, \
             patch('src.askspark.core.model_comparison.ModelComparisonEngine') as mock_model_engine, \
             patch('src.askspark.core.document_intelligence.RAGEngine') as mock_rag_engine, \
             patch('src.askspark.workflows.engine.WorkflowEngine') as mock_workflow_engine:
            
            # Mock responses
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = Mock(
                content="AI response",
                provider="openai",
                model="gpt-3.5-turbo",
                response_time=0.8,
                tokens_used=40,
                cost=0.001
            )
            
            mock_ai_client.return_value = mock_ai_instance
            mock_model_engine.return_value = Mock()
            mock_rag_engine.return_value = Mock()
            mock_workflow_engine.return_value = Mock()
            
            # Import dashboard
            import sys
            sys.path.append('/Users/startferanmi/Documents/AskSpark')
            from app import AIConsultantDashboard
            
            start_time = time.time()
            
            # Create dashboard (initialization)
            dashboard = AIConsultantDashboard()
            
            # Simulate typical user interactions
            for _ in range(5):
                # Get provider status
                with patch('src.askspark.config.settings.Config.get_available_providers', return_value=['openai']):
                    dashboard.get_provider_status()
                
                # Model comparison
                dashboard.model_comparison.compare_models("Test prompt", [("openai", "gpt-3.5-turbo")])
                
                # Document query
                dashboard.document_engine.query_documents("Test query")
                
                # List workflows
                dashboard.workflow_engine.list_workflows()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete typical interactions quickly
            assert total_time < 3.0  # Under 3 seconds for typical usage
    
    def test_peak_load_scenario(self):
        """Test peak load scenario"""
        with patch('src.askspark.core.ai_providers.UnifiedAIClient') as mock_ai_client:
            # Mock fast responses for peak load testing
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = Mock(
                content="Fast response",
                provider="openai",
                model="gpt-3.5-turbo",
                response_time=0.1,
                tokens_used=20,
                cost=0.0005
            )
            mock_ai_client.return_value = mock_ai_instance
            
            with patch('src.askspark.core.model_comparison.UnifiedAIClient', return_value=mock_ai_instance):
                engine = ModelComparisonEngine()
                
                # Simulate peak load: many concurrent comparisons
                num_concurrent_comparisons = 20
                
                def run_comparison():
                    return engine.compare_models("Peak load test", [("openai", "gpt-3.5-turbo")])
                
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(run_comparison) for _ in range(num_concurrent_comparisons)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Verify peak load handling
                assert len(results) == num_concurrent_comparisons
                assert all(len(result) == 1 for result in results)
                assert total_time < 2.0  # Should handle peak load efficiently
