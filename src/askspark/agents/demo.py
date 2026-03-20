"""
AskSpark Agent Integration Demo
Week 2 Lab 1: OpenAI Agents SDK implementation
"""

import asyncio
import logging
from typing import Dict, List, Any
import json
from datetime import datetime

from .base_agent import AgentManager, ModelComparisonAgent, DocumentAnalysisAgent
from .tools import AskSparkTools, get_all_tools
from ..core.ai_providers import UnifiedAIClient
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

class AskSparkAgentDemo:
    """Demonstration of AskSpark agent capabilities"""
    
    def __init__(self):
        self.agent_manager = AgentManager()
        self.tools = AskSparkTools()
        self.unified_client = UnifiedAIClient()
        
    async def demo_model_comparison(self, task_description: str) -> Dict[str, Any]:
        """Demo model comparison using agent"""
        print(f"\n=== Model Comparison Demo ===")
        print(f"Task: {task_description}")
        
        try:
            # Use agent for intelligent comparison
            agent = self.agent_manager.get_agent('model_comparison')
            result = await agent.compare_models_for_task(task_description)
            
            print(f"Agent Recommendation: {result.content[:200]}...")
            print(f"Execution Time: {result.execution_time:.2f}s")
            print(f"Model Used: {result.model_used}")
            
            # Also use tool for detailed comparison
            tool_result = await self.tools.compare_ai_models(task_description)
            
            return {
                "agent_result": {
                    "content": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used,
                    "tokens": result.tokens_used,
                    "cost": result.cost
                },
                "tool_result": tool_result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Model comparison demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_document_analysis(self, document_text: str, question: str) -> Dict[str, Any]:
        """Demo document analysis using agent"""
        print(f"\n=== Document Analysis Demo ===")
        print(f"Question: {question}")
        
        try:
            # Use agent for analysis
            agent = self.agent_manager.get_agent('document_analysis')
            result = await agent.analyze_document(document_text, question)
            
            print(f"Agent Analysis: {result.content[:200]}...")
            print(f"Execution Time: {result.execution_time:.2f}s")
            
            # Also use tool for detailed analysis
            tool_result = await self.tools.analyze_document_content(document_text, question)
            
            return {
                "agent_result": {
                    "content": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used,
                    "tokens": result.tokens_used
                },
                "tool_result": tool_result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Document analysis demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_provider_status(self) -> Dict[str, Any]:
        """Demo provider status checking"""
        print(f"\n=== Provider Status Demo ===")
        
        try:
            result = await self.tools.get_provider_status()
            
            print(f"Total Providers: {result['total_providers']}")
            print(f"Available: {result['available_providers']}")
            
            for provider, status in result['providers'].items():
                status_icon = "✓" if status['available'] else "✗"
                print(f"  {status_icon} {provider}: {status['status']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Provider status demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_cost_calculation(self, model: str, tokens: int) -> Dict[str, Any]:
        """Demo cost calculation"""
        print(f"\n=== Cost Calculation Demo ===")
        print(f"Model: {model}, Tokens: {tokens}")
        
        try:
            result = await self.tools.calculate_cost_estimate(model, tokens)
            
            if result['success']:
                cost_breakdown = result['cost_breakdown']
                print(f"Input Cost: ${cost_breakdown['input_cost']:.4f}")
                print(f"Output Cost: ${cost_breakdown['output_cost']:.4f}")
                print(f"Total Cost: ${cost_breakdown['total_cost']:.4f}")
                
                if result['optimization_suggestions']:
                    print("Suggestions:")
                    for suggestion in result['optimization_suggestions']:
                        print(f"  - {suggestion}")
            
            return result
            
        except Exception as e:
            logger.error(f"Cost calculation demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Run complete demonstration of all agent capabilities"""
        print("AskSpark Agent Platform Demo")
        print("=" * 50)
        
        demo_results = {}
        
        # Demo 1: Provider Status
        demo_results['provider_status'] = await self.demo_provider_status()
        
        # Demo 2: Model Comparison
        task = "Create a professional email for a sales outreach campaign"
        demo_results['model_comparison'] = await self.demo_model_comparison(task)
        
        # Demo 3: Document Analysis
        sample_doc = """
        AskSpark AI Platform Overview
        
        AskSpark is a comprehensive AI consultant assistant platform that integrates
        multiple AI providers to deliver intelligent model comparison, document analysis,
        and workflow automation capabilities. The platform supports OpenAI, Anthropic,
        Google, Groq, and DeepSeek providers with automatic failover and cost optimization.
        
        Key Features:
        - Multi-provider API management
        - Intelligent model comparison
        - Document intelligence with RAG
        - Workflow automation
        - Real-time analytics
        
        The platform demonstrates enterprise-ready AI engineering with production-grade
        architecture and comprehensive error handling.
        """
        
        question = "What are the key features of the AskSpark platform?"
        demo_results['document_analysis'] = await self.demo_document_analysis(sample_doc, question)
        
        # Demo 4: Cost Calculation
        demo_results['cost_calculation'] = await self.demo_cost_calculation("gpt-4o-mini", 5000)
        
        # Summary
        print(f"\n=== Demo Summary ===")
        successful_demos = sum(1 for result in demo_results.values() if result.get('success', True))
        print(f"Successful demos: {successful_demos}/{len(demo_results)}")
        
        return demo_results

# Standalone demo function
async def run_agent_demo():
    """Run standalone demonstration"""
    demo = AskSparkAgentDemo()
    return await demo.run_full_demo()

# Quick test function
def test_agent_integration():
    """Quick test of agent integration"""
    print("Testing AskSpark Agent Integration...")
    
    async def run_test():
        try:
            demo = AskSparkAgentDemo()
            
            # Quick provider status check
            result = await demo.demo_provider_status()
            return result.get('success', False)
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    # Run the full demo
    result = asyncio.run(run_agent_demo())
    print(f"\nDemo completed. Results: {json.dumps(result, indent=2)}")
