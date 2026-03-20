"""
Unit tests for AI Providers with comprehensive mocking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, Any

from src.askspark.core.ai_providers import UnifiedAIClient, ModelResponse
from src.askspark.config.settings import Config


class TestModelResponse:
    """Test ModelResponse dataclass"""
    
    def test_model_response_creation(self):
        """Test creating a ModelResponse"""
        response = ModelResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            provider="openai",
            response_time=0.5,
            tokens_used=100,
            cost=0.001
        )
        
        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.response_time == 0.5
        assert response.tokens_used == 100
        assert response.cost == 0.001


class TestUnifiedAIClient:
    """Test UnifiedAIClient with mocked API calls"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test API keys"""
        return {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'GOOGLE_API_KEY': 'test-google-key',
            'GROQ_API_KEY': 'test-groq-key',
            'DEEPSEEK_API_KEY': 'test-deepseek-key'
        }
    
    @pytest.fixture
    def client_with_mocks(self, mock_config):
        """Create client with all APIs mocked"""
        with patch.multiple(
            Config,
            OPENAI_API_KEY=mock_config['OPENAI_API_KEY'],
            ANTHROPIC_API_KEY=mock_config['ANTHROPIC_API_KEY'],
            GOOGLE_API_KEY=mock_config['GOOGLE_API_KEY'],
            GROQ_API_KEY=mock_config['GROQ_API_KEY'],
            DEEPSEEK_API_KEY=mock_config['DEEPSEEK_API_KEY']
        ):
            with patch('src.askspark.core.ai_providers.openai'), \
                 patch('src.askspark.core.ai_providers.anthropic'), \
                 patch('src.askspark.core.ai_providers.genai'), \
                 patch('src.askspark.core.ai_providers.groq'):
                
                client = UnifiedAIClient()
                yield client
    
    def test_client_initialization_with_keys(self, mock_config):
        """Test client initializes properly when API keys are provided"""
        with patch.multiple(Config, **mock_config):
            with patch('src.askspark.core.ai_providers.openai') as mock_openai, \
                 patch('src.askspark.core.ai_providers.anthropic') as mock_anthropic, \
                 patch('src.askspark.core.ai_providers.genai') as mock_genai, \
                 patch('src.askspark.core.ai_providers.groq') as mock_groq:
                
                client = UnifiedAIClient()
                
                # Verify clients are created
                assert 'openai' in client.clients
                assert 'anthropic' in client.clients
                assert 'google' in client.clients
                assert 'groq' in client.clients
    
    def test_client_initialization_without_keys(self):
        """Test client handles missing API keys gracefully"""
        with patch.multiple(
            Config,
            OPENAI_API_KEY=None,
            ANTHROPIC_API_KEY=None,
            GOOGLE_API_KEY=None,
            GROQ_API_KEY=None,
            DEEPSEEK_API_KEY=None
        ):
            client = UnifiedAIClient()
            assert len(client.clients) == 0
    
    @patch('src.askspark.core.ai_providers.openai')
    def test_openai_api_call_success(self, mock_openai_module, client_with_mocks):
        """Test successful OpenAI API call"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "OpenAI test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response
        
        client_with_mocks.clients['openai'] = mock_client
        
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "openai", 
            "gpt-3.5-turbo"
        )
        
        assert response is not None
        assert response.content == "OpenAI test response"
        assert response.provider == "openai"
        assert response.model == "gpt-3.5-turbo"
        assert response.tokens_used == 30
        assert response.response_time > 0
    
    @patch('src.askspark.core.ai_providers.anthropic')
    def test_anthropic_api_call_success(self, mock_anthropic_module, client_with_mocks):
        """Test successful Anthropic API call"""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Anthropic test response"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25
        mock_client.messages.create.return_value = mock_response
        
        client_with_mocks.clients['anthropic'] = mock_client
        
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "anthropic", 
            "claude-3-haiku-20240307"
        )
        
        assert response is not None
        assert response.content == "Anthropic test response"
        assert response.provider == "anthropic"
        assert response.model == "claude-3-haiku-20240307"
        assert response.tokens_used == 40
    
    @patch('src.askspark.core.ai_providers.genai')
    def test_google_api_call_success(self, mock_genai_module, client_with_mocks):
        """Test successful Google API call"""
        # Mock the Google client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Google test response"
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 18
        mock_client.generate_content.return_value = mock_response
        
        client_with_mocks.clients['google'] = mock_client
        
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "google", 
            "gemini-pro"
        )
        
        assert response is not None
        assert response.content == "Google test response"
        assert response.provider == "google"
        assert response.model == "gemini-pro"
        assert response.tokens_used == 30
    
    @patch('src.askspark.core.ai_providers.groq')
    def test_groq_api_call_success(self, mock_groq_module, client_with_mocks):
        """Test successful Groq API call"""
        # Mock the Groq client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Groq test response"
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.completion_tokens = 12
        mock_client.chat.completions.create.return_value = mock_response
        
        client_with_mocks.clients['groq'] = mock_client
        
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "groq", 
            "llama2-70b-4096"
        )
        
        assert response is not None
        assert response.content == "Groq test response"
        assert response.provider == "groq"
        assert response.model == "llama2-70b-4096"
        assert response.tokens_used == 20
    
    def test_api_call_failure(self, client_with_mocks):
        """Test API call failure handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        client_with_mocks.clients['openai'] = mock_client
        
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "openai", 
            "gpt-3.5-turbo"
        )
        
        assert response is None
    
    def test_unsupported_provider(self, client_with_mocks):
        """Test handling of unsupported provider"""
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "unsupported_provider", 
            "test-model"
        )
        
        assert response is None
    
    def test_get_available_providers(self, client_with_mocks):
        """Test getting available providers"""
        providers = client_with_mocks.get_available_providers()
        
        # Should return all initialized providers
        expected_providers = ['openai', 'anthropic', 'google', 'groq']
        for provider in expected_providers:
            assert provider in providers
    
    def test_failover_mechanism(self, client_with_mocks):
        """Test failover when primary provider fails"""
        # Mock primary provider to fail
        mock_primary = Mock()
        mock_primary.chat.completions.create.side_effect = Exception("Primary failed")
        
        # Mock backup provider to succeed
        mock_backup = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Backup response"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_backup.chat.completions.create.return_value = mock_response
        
        client_with_mocks.clients['openai'] = mock_primary
        client_with_mocks.clients['anthropic'] = mock_backup
        
        response = client_with_mocks.generate_response_with_failover(
            "Test prompt",
            primary_provider="openai",
            backup_providers=["anthropic"]
        )
        
        assert response is not None
        assert response.content == "Backup response"
        assert response.provider == "anthropic"
    
    def test_cost_calculation(self, client_with_mocks):
        """Test cost calculation for different providers"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        
        client_with_mocks.clients['openai'] = mock_client
        
        response = client_with_mocks.generate_response(
            "Test prompt", 
            "openai", 
            "gpt-3.5-turbo"
        )
        
        # Check if cost is calculated (based on Config.MODEL_PRICING)
        assert response.cost is not None
        assert response.cost > 0
