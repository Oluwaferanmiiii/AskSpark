"""
Pytest configuration and fixtures for AskSpark tests
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.askspark.config.settings import Config
from src.askspark.config.logging import setup_logging


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    setup_logging(level="DEBUG")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "GOOGLE_API_KEY": "test-key",
        "GROQ_API_KEY": "test-key",
        "DEEPSEEK_API_KEY": "test-key",
        "PUSHOVER_USER_KEY": "test-key",
        "PUSHOVER_APP_TOKEN": "test-token",
        "DEBUG": "true"
    }


@pytest.fixture
def mock_env_vars(test_config, monkeypatch):
    """Mock environment variables"""
    for key, value in test_config.items():
        monkeypatch.setenv(key, value)
    return test_config
