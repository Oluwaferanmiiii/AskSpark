import openai
import anthropic
import google.generativeai as genai
import groq
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ModelResponse:
    """Standard response format across all providers"""
    content: str
    model: str
    provider: str
    response_time: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

class UnifiedAIClient:
    """Unified client for multiple AI providers with failover support"""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all available AI provider clients"""
        # OpenAI
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            self.clients['openai'] = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        
        # Anthropic
        if Config.ANTHROPIC_API_KEY:
            self.clients['anthropic'] = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            logger.info("Anthropic client initialized")
        
        # Google
        if Config.GOOGLE_API_KEY:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.clients['google'] = genai.GenerativeModel('gemini-pro')
            logger.info("Google client initialized")
        
        # Groq
        if Config.GROQ_API_KEY:
            self.clients['groq'] = groq.Groq(api_key=Config.GROQ_API_KEY)
            logger.info("Groq client initialized")
        
        # DeepSeek (using OpenAI-compatible API)
        if Config.DEEPSEEK_API_KEY:
            self.clients['deepseek'] = openai.OpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1"
            )
            logger.info("DeepSeek client initialized")
    
    def _calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        if model not in Config.MODEL_PRICING:
            return 0.0
        
        pricing = Config.MODEL_PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
    
    def _call_openai(self, model: str, messages: List[Dict]) -> ModelResponse:
        """Call OpenAI API"""
        start_time = time.time()
        client = self.clients['openai']
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        response_time = time.time() - start_time
        content = response.choices[0].message.content
        
        # Calculate tokens and cost
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = self._calculate_cost('openai', model, input_tokens, output_tokens)
        
        return ModelResponse(
            content=content,
            model=model,
            provider='openai',
            response_time=response_time,
            tokens_used=input_tokens + output_tokens,
            cost=cost
        )
    
    def _call_anthropic(self, model: str, messages: List[Dict]) -> ModelResponse:
        """Call Anthropic API"""
        start_time = time.time()
        client = self.clients['anthropic']
        
        # Convert messages to Anthropic format
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append(msg)
        
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            system=system_message,
            messages=user_messages
        )
        
        response_time = time.time() - start_time
        content = response.content[0].text
        
        # Calculate cost (Anthropic doesn't provide token count in basic response)
        estimated_tokens = len(content.split()) * 1.3  # Rough estimate
        cost = self._calculate_cost('anthropic', model, estimated_tokens * 0.3, estimated_tokens)
        
        return ModelResponse(
            content=content,
            model=model,
            provider='anthropic',
            response_time=response_time,
            tokens_used=int(estimated_tokens),
            cost=cost
        )
    
    def _call_google(self, model: str, messages: List[Dict]) -> ModelResponse:
        """Call Google Gemini API"""
        start_time = time.time()
        
        # Convert messages to Gemini format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        
        response_time = time.time() - start_time
        content = response.text
        
        # Estimate cost
        estimated_tokens = len(content.split()) * 1.3
        cost = self._calculate_cost('google', model, estimated_tokens * 0.3, estimated_tokens)
        
        return ModelResponse(
            content=content,
            model=model,
            provider='google',
            response_time=response_time,
            tokens_used=int(estimated_tokens),
            cost=cost
        )
    
    def _call_groq(self, model: str, messages: List[Dict]) -> ModelResponse:
        """Call Groq API"""
        start_time = time.time()
        client = self.clients['groq']
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        response_time = time.time() - start_time
        content = response.choices[0].message.content
        
        # Calculate tokens and cost
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = self._calculate_cost('groq', model, input_tokens, output_tokens)
        
        return ModelResponse(
            content=content,
            model=model,
            provider='groq',
            response_time=response_time,
            tokens_used=input_tokens + output_tokens,
            cost=cost
        )
    
    def _call_deepseek(self, model: str, messages: List[Dict]) -> ModelResponse:
        """Call DeepSeek API"""
        start_time = time.time()
        client = self.clients['deepseek']
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        response_time = time.time() - start_time
        content = response.choices[0].message.content
        
        # Calculate tokens and cost
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = self._calculate_cost('deepseek', model, input_tokens, output_tokens)
        
        return ModelResponse(
            content=content,
            model=model,
            provider='deepseek',
            response_time=response_time,
            tokens_used=input_tokens + output_tokens,
            cost=cost
        )
    
    def call_model(self, provider: str, model: str, messages: List[Dict]) -> ModelResponse:
        """Call a specific model from a provider"""
        if provider not in self.clients:
            raise ValueError(f"Provider {provider} not available")
        
        try:
            if provider == 'openai':
                return self._call_openai(model, messages)
            elif provider == 'anthropic':
                return self._call_anthropic(model, messages)
            elif provider == 'google':
                return self._call_google(model, messages)
            elif provider == 'groq':
                return self._call_groq(model, messages)
            elif provider == 'deepseek':
                return self._call_deepseek(model, messages)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        except Exception as e:
            logger.error(f"Error calling {provider}/{model}: {str(e)}")
            raise
    
    def call_with_fallback(self, providers_models: List[Tuple[str, str]], messages: List[Dict]) -> ModelResponse:
        """Try multiple providers/models in order until one succeeds"""
        last_error = None
        
        for provider, model in providers_models:
            if provider not in self.clients:
                logger.warning(f"Provider {provider} not available, skipping")
                continue
            
            try:
                logger.info(f"Trying {provider}/{model}")
                return self.call_model(provider, model, messages)
            except Exception as e:
                logger.error(f"Failed to call {provider}/{model}: {str(e)}")
                last_error = e
                continue
        
        raise Exception(f"All providers failed. Last error: {str(last_error)}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.clients.keys())
    
    def get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a provider"""
        return Config.get_provider_models(provider)
