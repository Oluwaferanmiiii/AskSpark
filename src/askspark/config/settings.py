import os
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

class Config:
    """Configuration management for AI Consultant Assistant"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Notification Services
    PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY")
    PUSHOVER_APP_TOKEN = os.getenv("PUSHOVER_APP_TOKEN")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_consultant.db")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    
    # AI Provider Configuration
    AI_PROVIDERS = {
        "openai": {
            "api_key": OPENAI_API_KEY,
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            "base_url": "https://api.openai.com/v1"
        },
        "anthropic": {
            "api_key": ANTHROPIC_API_KEY,
            "models": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "base_url": "https://api.anthropic.com"
        },
        "google": {
            "api_key": GOOGLE_API_KEY,
            "models": ["gemini-pro", "gemini-pro-vision"],
            "base_url": None
        },
        "groq": {
            "api_key": GROQ_API_KEY,
            "models": ["llama2-70b-4096", "mixtral-8x7b-32768"],
            "base_url": "https://api.groq.com/openai/v1"
        },
        "deepseek": {
            "api_key": DEEPSEEK_API_KEY,
            "models": ["deepseek-chat", "deepseek-coder"],
            "base_url": "https://api.deepseek.com/v1"
        },
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "models": ["openrouter/meta-llama/llama-3.1-8b-instruct:free", "openrouter/meta-llama/llama-3.1-70b-instruct", "openrouter/mistralai/mistral-7b-instruct", "openrouter/openai/gpt-3.5-turbo"],
            "base_url": "https://openrouter.ai/api/v1"
        }
    }
    
    # Model Pricing (per 1K tokens)
    MODEL_PRICING = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
        "llama2-70b-4096": {"input": 0.0008, "output": 0.0008},
        "mixtral-8x7b-32768": {"input": 0.0005, "output": 0.0005},
        "deepseek-chat": {"input": 0.0001, "output": 0.0002},
        "deepseek-coder": {"input": 0.0001, "output": 0.0002}
    }
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available AI providers with API keys"""
        available = []
        for provider, config in cls.AI_PROVIDERS.items():
            if config["api_key"]:
                available.append(provider)
        return available
    
    @classmethod
    def get_provider_models(cls, provider: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider in cls.AI_PROVIDERS:
            return cls.AI_PROVIDERS[provider]["models"]
        return []
