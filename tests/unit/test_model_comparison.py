"""
Unit tests for Model Comparison Engine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from src.askspark.core.model_comparison import ModelComparisonEngine, ComparisonMetrics
from src.askspark.core.ai_providers import ModelResponse


class TestComparisonMetrics:
    """Test ComparisonMetrics dataclass"""
    
    def test_comparison_metrics_creation(self):
        """Test creating ComparisonMetrics"""
        metrics = ComparisonMetrics(
            model="gpt-3.5-turbo",
            provider="openai",
            response_time=1.5,
            tokens_used=100,
            cost=0.002,
            quality_score=0.85
        )
        
        assert metrics.model == "gpt-3.5-turbo"
        assert metrics.provider == "openai"
        assert metrics.response_time == 1.5
        assert metrics.tokens_used == 100
        assert metrics.cost == 0.002
        assert metrics.quality_score == 0.85


class TestModelComparisonEngine:
    """Test ModelComparisonEngine"""
    
    @pytest.fixture
    def mock_ai_client(self):
        """Mock AI client"""
        client = Mock()
        
        # Mock responses for different providers
        def mock_generate_response(prompt, provider, model):
            if provider == "openai" and model == "gpt-3.5-turbo":
                return ModelResponse(
                    content="OpenAI response about AI benefits",
                    model="gpt-3.5-turbo",
                    provider="openai",
                    response_time=1.2,
                    tokens_used=150,
                    cost=0.003
                )
            elif provider == "anthropic" and model == "claude-3-haiku-20240307":
                return ModelResponse(
                    content="Anthropic response about AI benefits",
                    model="claude-3-haiku-20240307",
                    provider="anthropic",
                    response_time=0.8,
                    tokens_used=120,
                    cost=0.002
                )
            return None
        
        client.generate_response = mock_generate_response
        client.get_available_providers.return_value = ["openai", "anthropic"]
        
        return client
    
    @pytest.fixture
    def comparison_engine(self, mock_ai_client):
        """Create ModelComparisonEngine with mocked AI client"""
        with patch('src.askspark.core.model_comparison.UnifiedAIClient', return_value=mock_ai_client):
            engine = ModelComparisonEngine()
            return engine
    
    def test_engine_initialization(self, comparison_engine):
        """Test engine initialization"""
        assert comparison_engine.ai_client is not None
        assert comparison_engine.comparison_history == []
    
    def test_compare_models_success(self, comparison_engine):
        """Test successful model comparison"""
        prompt = "What are the benefits of AI automation?"
        providers_models = [
            ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-haiku-20240307")
        ]
        
        results = comparison_engine.compare_models(prompt, providers_models)
        
        assert len(results) == 2
        assert all(isinstance(result, ComparisonMetrics) for result in results)
        
        # Check OpenAI result
        openai_result = next(r for r in results if r.provider == "openai")
        assert openai_result.model == "gpt-3.5-turbo"
        assert openai_result.content == "OpenAI response about AI benefits"
        assert openai_result.response_time == 1.2
        
        # Check Anthropic result
        anthropic_result = next(r for r in results if r.provider == "anthropic")
        assert anthropic_result.model == "claude-3-haiku-20240307"
        assert anthropic_result.content == "Anthropic response about AI benefits"
        assert anthropic_result.response_time == 0.8
    
    def test_compare_models_with_failure(self, comparison_engine):
        """Test model comparison with some failures"""
        # Mock one provider to fail
        def mock_generate_response_fail(prompt, provider, model):
            if provider == "openai":
                return None  # Simulate failure
            elif provider == "anthropic":
                return ModelResponse(
                    content="Anthropic response",
                    model="claude-3-haiku-20240307",
                    provider="anthropic",
                    response_time=0.8,
                    tokens_used=120,
                    cost=0.002
                )
            return None
        
        comparison_engine.ai_client.generate_response = mock_generate_response_fail
        
        prompt = "Test prompt"
        providers_models = [
            ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-haiku-20240307")
        ]
        
        results = comparison_engine.compare_models(prompt, providers_models)
        
        # Should only return successful results
        assert len(results) == 1
        assert results[0].provider == "anthropic"
    
    def test_calculate_quality_score(self, comparison_engine):
        """Test quality score calculation"""
        response = ModelResponse(
            content="This is a comprehensive response about AI benefits with multiple points and good structure.",
            model="gpt-3.5-turbo",
            provider="openai",
            response_time=1.0,
            tokens_used=100,
            cost=0.002
        )
        
        score = comparison_engine._calculate_quality_score(response, "What are the benefits of AI automation?")
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_get_comparison_history(self, comparison_engine):
        """Test retrieving comparison history"""
        # Add some mock history
        comparison_engine.comparison_history = [
            ComparisonMetrics(
                model="gpt-3.5-turbo",
                provider="openai",
                response_time=1.0,
                tokens_used=100,
                cost=0.002,
                quality_score=0.8
            ),
            ComparisonMetrics(
                model="claude-3-haiku-20240307",
                provider="anthropic",
                response_time=0.8,
                tokens_used=120,
                cost=0.002,
                quality_score=0.85
            )
        ]
        
        history = comparison_engine.get_comparison_history()
        
        assert len(history) == 2
        assert isinstance(history, pd.DataFrame)
        assert 'model' in history.columns
        assert 'provider' in history.columns
        assert 'response_time' in history.columns
        assert 'quality_score' in history.columns
    
    def test_get_best_model(self, comparison_engine):
        """Test getting the best model from comparison results"""
        # Mock comparison results
        comparison_engine.comparison_history = [
            ComparisonMetrics(
                model="gpt-3.5-turbo",
                provider="openai",
                response_time=1.5,
                tokens_used=100,
                cost=0.003,
                quality_score=0.75
            ),
            ComparisonMetrics(
                model="claude-3-haiku-20240307",
                provider="anthropic",
                response_time=0.8,
                tokens_used=120,
                cost=0.002,
                quality_score=0.90
            )
        ]
        
        best_model = comparison_engine.get_best_model()
        
        assert best_model.provider == "anthropic"
        assert best_model.model == "claude-3-haiku-20240307"
        assert best_model.quality_score == 0.90
    
    def test_generate_comparison_report(self, comparison_engine):
        """Test generating comparison report"""
        # Add mock comparison data
        comparison_engine.comparison_history = [
            ComparisonMetrics(
                model="gpt-3.5-turbo",
                provider="openai",
                response_time=1.2,
                tokens_used=150,
                cost=0.003,
                quality_score=0.82
            ),
            ComparisonMetrics(
                model="claude-3-haiku-20240307",
                provider="anthropic",
                response_time=0.8,
                tokens_used=120,
                cost=0.002,
                quality_score=0.88
            )
        ]
        
        report = comparison_engine.generate_comparison_report()
        
        assert isinstance(report, dict)
        assert 'total_comparisons' in report
        assert 'providers' in report
        assert 'avg_response_time' in report
        assert 'avg_quality_score' in report
        assert 'cost_efficiency' in report
        
        assert report['total_comparisons'] == 2
        assert len(report['providers']) == 2
        assert 'openai' in report['providers']
        assert 'anthropic' in report['providers']
    
    def test_export_results(self, comparison_engine):
        """Test exporting comparison results"""
        # Add mock data
        comparison_engine.comparison_history = [
            ComparisonMetrics(
                model="gpt-3.5-turbo",
                provider="openai",
                response_time=1.2,
                tokens_used=150,
                cost=0.003,
                quality_score=0.82
            )
        ]
        
        # Test CSV export
        csv_data = comparison_engine.export_results('csv')
        assert isinstance(csv_data, str)
        assert 'model,provider,response_time' in csv_data
        
        # Test JSON export
        json_data = comparison_engine.export_results('json')
        assert isinstance(json_data, str)
        assert '"model":' in json_data
        assert '"provider":' in json_data
    
    def test_visualize_comparison(self, comparison_engine):
        """Test creating comparison visualizations"""
        # Add mock data
        comparison_engine.comparison_history = [
            ComparisonMetrics(
                model="gpt-3.5-turbo",
                provider="openai",
                response_time=1.2,
                tokens_used=150,
                cost=0.003,
                quality_score=0.82
            ),
            ComparisonMetrics(
                model="claude-3-haiku-20240307",
                provider="anthropic",
                response_time=0.8,
                tokens_used=120,
                cost=0.002,
                quality_score=0.88
            )
        ]
        
        # Test visualization creation
        fig = comparison_engine.visualize_comparison('quality_score')
        assert fig is not None
        
        fig = comparison_engine.visualize_comparison('response_time')
        assert fig is not None
        
        fig = comparison_engine.visualize_comparison('cost')
        assert fig is not None
