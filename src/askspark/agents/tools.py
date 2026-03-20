"""
Agent Tools for AskSpark Platform
Week 2 Lab 1: Tools and function calling integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from agents import function_tool
import json
import os
from datetime import datetime

from ..core.ai_providers import UnifiedAIClient
from ..core.model_comparison import ModelComparisonEngine
from ..core.document_intelligence import RAGEngine
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ToolResult:
    """Standard result format for tool execution"""
    success: bool
    data: Any
    execution_time: float
    error_message: Optional[str] = None

class AskSparkTools:
    """Collection of tools for AskSpark agents"""
    
    def __init__(self):
        self.unified_client = UnifiedAIClient()
        self.model_comparison = ModelComparisonEngine()
        self.rag_engine = None  # Initialize on demand
        logger.info("Initialized AskSpark tools")
    
    def _get_rag_engine(self):
        """Lazy initialization of RAG engine"""
        if not self.rag_engine:
            self.rag_engine = RAGEngine()
        return self.rag_engine
    
    @function_tool
    async def compare_ai_models(self, task_description: str, 
                              providers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare AI models for a specific task and provide recommendations.
        
        Args:
            task_description: Description of the AI task
            providers: Optional list of providers to consider
            
        Returns:
            Dictionary with model comparison results and recommendations
        """
        start_time = datetime.now()
        
        try:
            if providers:
                provider_models = [(provider, "default") for provider in providers]
            else:
                provider_models = [
                    ("openai", "gpt-4o-mini"),
                    ("anthropic", "claude-3-haiku"),
                    ("google", "gemini-pro")
                ]
            
            results = await self.model_comparison.compare_models_async(
                task_description, provider_models
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "task": task_description,
                "results": results,
                "recommendation": self._get_best_model(results),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model comparison tool failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    @function_tool
    async def analyze_document_content(self, document_text: str, 
                                     question: str = "") -> Dict[str, Any]:
        """
        Analyze document content and answer questions using RAG.
        
        Args:
            document_text: The document content to analyze
            question: Optional specific question about the document
            
        Returns:
            Dictionary with analysis results and insights
        """
        start_time = datetime.now()
        
        try:
            rag = self._get_rag_engine()
            
            # Process document temporarily
            temp_doc_id = f"temp_doc_{start_time.timestamp()}"
            rag.process_and_store_document_content(document_text, temp_doc_id)
            
            # Analyze or answer question
            if question:
                result = rag.query_documents(question, n_results=3)
            else:
                # Generate summary
                result = rag.query_documents(
                    "Please provide a comprehensive summary of this document including key insights.",
                    n_results=5
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document analysis tool failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    @function_tool
    async def get_provider_status(self) -> Dict[str, Any]:
        """
        Check the status of all AI providers.
        
        Returns:
            Dictionary with provider availability and status information
        """
        start_time = datetime.now()
        
        try:
            providers = ["openai", "anthropic", "google", "groq", "deepseek"]
            status = {}
            
            for provider in providers:
                try:
                    # Simple connectivity check
                    client = self.unified_client.get_client(provider)
                    if client:
                        status[provider] = {
                            "available": True,
                            "status": "online",
                            "last_check": start_time.isoformat()
                        }
                    else:
                        status[provider] = {
                            "available": False,
                            "status": "not_configured",
                            "last_check": start_time.isoformat()
                        }
                except Exception as e:
                    status[provider] = {
                        "available": False,
                        "status": "error",
                        "error": str(e),
                        "last_check": start_time.isoformat()
                    }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "providers": status,
                "total_providers": len(providers),
                "available_providers": len([p for p, s in status.items() if s["available"]]),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Provider status check failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    @function_tool
    async def calculate_cost_estimate(self, model: str, tokens: int, 
                                    operation_type: str = "completion") -> Dict[str, Any]:
        """
        Calculate cost estimate for AI model usage.
        
        Args:
            model: The AI model name
            tokens: Number of tokens to process
            operation_type: Type of operation (completion, embedding, etc.)
            
        Returns:
            Dictionary with cost breakdown and optimization suggestions
        """
        start_time = datetime.now()
        
        try:
            # Cost per 1K tokens (approximate rates)
            costs = {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.002, "output": 0.002},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "gemini-pro": {"input": 0.0005, "output": 0.0015}
            }
            
            model_costs = costs.get(model, {"input": 0.001, "output": 0.002})
            
            # Estimate input/output split (70/30 for general tasks)
            input_tokens = int(tokens * 0.7)
            output_tokens = int(tokens * 0.3)
            
            input_cost = (input_tokens / 1000) * model_costs["input"]
            output_cost = (output_tokens / 1000) * model_costs["output"]
            total_cost = input_cost + output_cost
            
            # Optimization suggestions
            suggestions = []
            if total_cost > 1.0:
                suggestions.append("Consider using a smaller model for cost efficiency")
            if tokens > 100000:
                suggestions.append("Large token usage - consider chunking or batching")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "model": model,
                "tokens": tokens,
                "operation_type": operation_type,
                "cost_breakdown": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost
                },
                "optimization_suggestions": suggestions,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cost calculation tool failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _get_best_model(self, results: List[Dict]) -> Dict[str, Any]:
        """Get best model recommendation from comparison results"""
        if not results:
            return {"model": "none", "reason": "No results available"}
        
        # Simple scoring based on quality, speed, and cost
        best_score = -1
        best_model = None
        
        for result in results:
            # Calculate weighted score
            quality_score = result.get("quality_score", 0.5)
            speed_score = 1.0 / max(result.get("response_time", 1.0), 0.1)
            cost_score = 1.0 / max(result.get("cost", 0.001), 0.0001)
            
            # Weighted combination (quality 50%, speed 30%, cost 20%)
            total_score = (quality_score * 0.5) + (speed_score * 0.3) + (cost_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_model = result
        
        if best_model:
            return {
                "model": best_model.get("model", "unknown"),
                "provider": best_model.get("provider", "unknown"),
                "reason": f"Best combination of quality ({best_model.get('quality_score', 0):.2f}), "
                        f"speed ({best_model.get('response_time', 0):.2f}s), and cost (${best_model.get('cost', 0):.4f})"
            }
        
        return {"model": "none", "reason": "Unable to determine best model"}

# Tool registry for easy access
tools_registry = {
    "compare_ai_models": AskSparkTools().compare_ai_models,
    "analyze_document_content": AskSparkTools().analyze_document_content,
    "get_provider_status": AskSparkTools().get_provider_status,
    "calculate_cost_estimate": AskSparkTools().calculate_cost_estimate
}

def get_all_tools() -> List:
    """Get all available tools for agent configuration"""
    return list(tools_registry.values())
