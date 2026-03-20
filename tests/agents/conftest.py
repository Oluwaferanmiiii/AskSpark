"""
Test configuration and utilities for AskSpark Agents
Week 2 Lab 1 - Test framework setup
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for all tests
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Test configuration
pytest_plugins = []

# Mock environment variables for testing
TEST_ENV_VARS = {
    'OPENAI_API_KEY': 'test-openai-key',
    'ANTHROPIC_API_KEY': 'test-anthropic-key',
    'GOOGLE_API_KEY': 'test-google-key',
    'GROQ_API_KEY': 'test-groq-key',
    'DEEPSEEK_API_KEY': 'test-deepseek-key'
}


@pytest.fixture(scope='session')
def mock_env():
    """Set up mock environment variables for testing"""
    original_env = {}
    
    # Store original values and set test values
    for key, value in TEST_ENV_VARS.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield TEST_ENV_VARS
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_openai_model():
    """Mock OpenAI model for agent testing"""
    with patch('askspark.agents.base_agent.OpenAIChatCompletionsModel') as mock:
        mock_model = Mock()
        mock.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_unified_client():
    """Mock UnifiedAIClient for testing"""
    with patch('askspark.agents.base_agent.UnifiedAIClient') as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_model_comparison_engine():
    """Mock ModelComparisonEngine for testing"""
    with patch('askspark.agents.tools.ModelComparisonEngine') as mock:
        mock_engine = Mock()
        mock.return_value = mock_engine
        yield mock_engine


@pytest.fixture
def mock_rag_engine():
    """Mock RAGEngine for testing"""
    with patch('askspark.agents.tools.RAGEngine') as mock:
        mock_engine = Mock()
        mock.return_value = mock_engine
        yield mock_engine


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.error_handling = pytest.mark.error_handling


# Async test configuration
@pytest.fixture(scope='session')
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock data for testing
MOCK_AGENT_RESPONSE = {
    'content': 'Test agent response',
    'agent_name': 'TestAgent',
    'execution_time': 1.0,
    'trace_id': 'test-trace-123',
    'model_used': 'gpt-4o-mini',
    'tokens_used': 100,
    'cost': 0.01
}

MOCK_MODEL_COMPARISON_RESULTS = [
    {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'quality_score': 0.8,
        'response_time': 1.0,
        'cost': 0.01
    },
    {
        'provider': 'anthropic',
        'model': 'claude-3-haiku',
        'quality_score': 0.75,
        'response_time': 1.2,
        'cost': 0.02
    }
]

MOCK_DOCUMENT_ANALYSIS_RESULT = {
    'answer': 'Document analysis result',
    'sources': ['source1', 'source2'],
    'question': 'Test question'
}

MOCK_PROVIDER_STATUS = {
    'openai': {'available': True, 'status': 'online'},
    'anthropic': {'available': True, 'status': 'online'},
    'google': {'available': False, 'status': 'not_configured'},
    'groq': {'available': False, 'status': 'error'},
    'deepseek': {'available': False, 'status': 'not_configured'}
}

# Helper functions for testing
def create_mock_agent_response(**kwargs):
    """Create a mock agent response with customizable parameters"""
    response = MOCK_AGENT_RESPONSE.copy()
    response.update(kwargs)
    return Mock(**response)


def create_mock_model_comparison_results(**kwargs):
    """Create mock model comparison results"""
    return [result.copy() for result in MOCK_MODEL_COMPARISON_RESULTS]


def create_mock_document_analysis_result(**kwargs):
    """Create mock document analysis result"""
    result = MOCK_DOCUMENT_ANALYSIS_RESULT.copy()
    result.update(kwargs)
    return result


def create_mock_provider_status(**kwargs):
    """Create mock provider status"""
    status = MOCK_PROVIDER_STATUS.copy()
    for provider, provider_status in status.items():
        provider_status.update(kwargs.get(provider, {}))
    return status


# Test utilities
class AsyncMockContextManager:
    """Context manager for async mocks"""
    
    def __init__(self, mock_obj, method_name, return_value=None, side_effect=None):
        self.mock_obj = mock_obj
        self.method_name = method_name
        self.return_value = return_value
        self.side_effect = side_effect
        self.original_method = None
    
    def __enter__(self):
        self.original_method = getattr(self.mock_obj, self.method_name)
        mock_method = AsyncMock(return_value=self.return_value, side_effect=self.side_effect)
        setattr(self.mock_obj, self.method_name, mock_method)
        return mock_method
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self.mock_obj, self.method_name, self.original_method)


# Performance testing utilities
class PerformanceTracker:
    """Track performance metrics during tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start(self):
        """Start performance tracking"""
        self.start_time = asyncio.get_event_loop().time()
    
    def end(self, metric_name):
        """End tracking and record metric"""
        if self.start_time:
            self.end_time = asyncio.get_event_loop().time()
            duration = self.end_time - self.start_time
            self.metrics[metric_name] = duration
            self.start_time = None
            return duration
        return None
    
    def get_metric(self, metric_name):
        """Get recorded metric"""
        return self.metrics.get(metric_name)
    
    def get_all_metrics(self):
        """Get all recorded metrics"""
        return self.metrics.copy()


# Error testing utilities
class ErrorSimulator:
    """Simulate various error conditions for testing"""
    
    @staticmethod
    def create_timeout_error():
        """Create a timeout error"""
        return asyncio.TimeoutError("Operation timed out")
    
    @staticmethod
    def create_connection_error():
        """Create a connection error"""
        return ConnectionError("Failed to connect to service")
    
    @staticmethod
    def create_api_error(status_code, message):
        """Create an API error"""
        return Exception(f"API Error {status_code}: {message}")
    
    @staticmethod
    def create_validation_error(field, message):
        """Create a validation error"""
        return ValueError(f"Validation error for {field}: {message}")


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_agent_tasks(count=5):
        """Generate test agent tasks"""
        tasks = [
            "Analyze document and extract key insights",
            "Compare AI models for specific use case",
            "Generate cost optimization recommendations",
            "Create workflow automation plan",
            "Provide provider status analysis"
        ]
        return tasks[:count]
    
    @staticmethod
    def generate_documents(count=3):
        """Generate test document content"""
        documents = [
            """
            Business Plan 2024
            
            Executive Summary:
            Our company focuses on AI-driven solutions for enterprise clients.
            
            Key Objectives:
            1. Expand market presence by 25%
            2. Launch new product line
            3. Improve customer retention
            
            Financial Projections:
            - Q1 Revenue: $2.5M
            - Q2 Revenue: $3.0M
            - Q3 Revenue: $3.5M
            - Q4 Revenue: $4.0M
            """,
            """
            Technical Specifications
            
            System Architecture:
            - Microservices design
            - RESTful APIs
            - Real-time processing
            
            Technology Stack:
            - Backend: Python, FastAPI
            - Database: PostgreSQL, Redis
            - Frontend: React, TypeScript
            - Infrastructure: Docker, Kubernetes
            """,
            """
            Market Research Report
            
            Target Market Analysis:
            - Total Addressable Market: $50B
            - Serviceable Market: $15B
            - Obtainable Market: $3B
            
            Competitive Landscape:
            - 5 major competitors
            - Market leader: 40% share
            - Our positioning: Premium segment
            
            Growth Opportunities:
            - Geographic expansion
            - Product diversification
            - Strategic partnerships
            """
        ]
        return documents[:count]
    
    @staticmethod
    def generate_cost_scenarios():
        """Generate different cost calculation scenarios"""
        return [
            {"model": "gpt-4o-mini", "tokens": 1000, "expected_cost_range": (0.0001, 0.01)},
            {"model": "gpt-4o", "tokens": 5000, "expected_cost_range": (0.01, 0.1)},
            {"model": "claude-3-haiku", "tokens": 10000, "expected_cost_range": (0.001, 0.05)},
            {"model": "unknown-model", "tokens": 2000, "expected_cost_range": (0.001, 0.01)}
        ]


# Test configuration file
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: Mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "error_handling: Mark test as an error handling test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location"""
    for item in items:
        # Add unit marker to tests in test_agents.py and test_tools.py
        if "test_agents.py" in str(item.fspath) or "test_tools.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in test_integration.py
        elif "test_integration.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance-related tests
        if "performance" in item.name.lower() or "perf" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add error handling marker to error-related tests
        if "error" in item.name.lower() or "exception" in item.name.lower():
            item.add_marker(pytest.mark.error_handling)
