"""
OpenAI Agents SDK integration for AskSpark Platform
Week 2 Lab 1: Foundation for multi-agent system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from agents import Agent, Runner, trace, gen_trace_id
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
import time
import os

from ..core.ai_providers import UnifiedAIClient
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class AgentResponse:
    """Standard response format for agent interactions"""
    content: str
    agent_name: str
    execution_time: float
    trace_id: str
    model_used: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

class AskSparkAgentBase:
    """Base class for all AskSpark agents using OpenAI Agents SDK"""
    
    def __init__(self, name: str, instructions: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.unified_client = UnifiedAIClient()
        self.trace_id = None
        
        # Initialize OpenAI Agent
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=model,
                api_key=Config.OPENAI_API_KEY
            )
        )
        
        logger.info(f"Initialized agent: {name} with model: {model}")
    
    async def run(self, input_text: str, **kwargs) -> AgentResponse:
        """Execute agent with input and return standardized response"""
        start_time = time.time()
        self.trace_id = gen_trace_id()
        
        try:
            with trace("AskSpark Agent", trace_id=self.trace_id):
                result = await Runner.run(self.agent, input=input_text, **kwargs)
                
            execution_time = time.time() - start_time
            
            # Extract token usage if available
            tokens_used = getattr(result, 'usage', None)
            if tokens_used:
                tokens_used = tokens_used.total_tokens
            
            # Calculate cost (rough estimation)
            cost = self._calculate_cost(tokens_used)
            
            return AgentResponse(
                content=result.output,
                agent_name=self.name,
                execution_time=execution_time,
                trace_id=self.trace_id,
                model_used=self.model,
                tokens_used=tokens_used,
                cost=cost
            )
            
        except Exception as e:
            logger.error(f"Agent {self.name} execution failed: {str(e)}")
            raise
    
    def _calculate_cost(self, tokens: Optional[int]) -> Optional[float]:
        """Calculate cost based on token usage"""
        if not tokens:
            return None
        
        # Rough cost estimation per 1K tokens
        cost_per_1k = {
            "gpt-4o-mini": 0.00015,
            "gpt-4o": 0.005,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002
        }
        
        cost_rate = cost_per_1k.get(self.model, 0.001)
        return (tokens / 1000) * cost_rate

class ModelComparisonAgent(AskSparkAgentBase):
    """Agent for intelligent model comparison and recommendations"""
    
    def __init__(self):
        instructions = """
        You are an AI model comparison expert. Your task is to:
        1. Analyze user requirements for AI tasks
        2. Recommend the most suitable AI models
        3. Provide cost-benefit analysis
        4. Suggest optimization strategies
        
        Consider factors like:
        - Task complexity and requirements
        - Cost efficiency
        - Response speed
        - Output quality needs
        - Specific model capabilities
        
        Provide clear, actionable recommendations with reasoning.
        """
        super().__init__("ModelComparisonAgent", instructions)
    
    async def compare_models_for_task(self, task_description: str, 
                                    providers: Optional[List[str]] = None) -> AgentResponse:
        """Compare models for specific task"""
        if providers:
            input_text = f"Task: {task_description}\nAvailable providers: {', '.join(providers)}"
        else:
            input_text = f"Task: {task_description}"
        
        return await self.run(input_text)

class DocumentAnalysisAgent(AskSparkAgentBase):
    """Agent for intelligent document analysis using RAG"""
    
    def __init__(self):
        instructions = """
        You are an expert document analyst. Your task is to:
        1. Analyze document content and context
        2. Extract key insights and information
        3. Answer questions about documents accurately
        4. Provide structured summaries
        
        When analyzing documents:
        - Focus on accuracy and relevance
        - Cite specific parts when possible
        - Provide clear, structured responses
        - Ask clarifying questions if needed
        
        Always base your answers on the provided document context.
        """
        super().__init__("DocumentAnalysisAgent", instructions)
    
    async def analyze_document(self, document_content: str, 
                              question: str = "") -> AgentResponse:
        """Analyze document and answer questions"""
        if question:
            input_text = f"Document Content:\n{document_content}\n\nQuestion: {question}"
        else:
            input_text = f"Document Content:\n{document_content}\n\nPlease analyze and provide key insights."
        
        return await self.run(input_text)

class WorkflowOrchestrationAgent(AskSparkAgentBase):
    """Agent for orchestrating multi-agent workflows"""
    
    def __init__(self):
        instructions = """
        You are a workflow orchestration expert. Your task is to:
        1. Coordinate multiple AI agents
        2. Manage task distribution and handoffs
        3. Optimize workflow execution
        4. Handle errors and fallbacks
        
        Available agents:
        - ModelComparisonAgent: For model analysis and recommendations
        - DocumentAnalysisAgent: For document processing and analysis
        - ResearchAgent: For deep research and data gathering
        
        Plan workflows efficiently, considering:
        - Task dependencies
        - Agent specializations
        - Performance optimization
        - Error handling
        
        Provide clear workflow plans and coordinate execution.
        """
        super().__init__("WorkflowOrchestrationAgent", instructions)
    
    async def plan_workflow(self, task_description: str, 
                          available_agents: List[str]) -> AgentResponse:
        """Plan workflow for given task"""
        input_text = f"Task: {task_description}\nAvailable agents: {', '.join(available_agents)}"
        return await self.run(input_text)

class AgentManager:
    """Central manager for all AskSpark agents"""
    
    def __init__(self):
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents"""
        self.agents = {
            'model_comparison': ModelComparisonAgent(),
            'document_analysis': DocumentAnalysisAgent(),
            'workflow_orchestration': WorkflowOrchestrationAgent()
        }
        
        logger.info("Initialized all AskSpark agents")
    
    def get_agent(self, agent_name: str) -> Optional[AskSparkAgentBase]:
        """Get agent by name"""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all available agents"""
        return list(self.agents.keys())
    
    async def execute_agent(self, agent_name: str, input_text: str, **kwargs) -> AgentResponse:
        """Execute specific agent"""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        return await agent.run(input_text, **kwargs)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about all agents"""
        return {
            'total_agents': len(self.agents),
            'available_agents': list(self.agents.keys()),
            'models_used': [agent.model for agent in self.agents.values()]
        }

# Import OpenAI model for agent configuration
try:
    from agents.models import OpenAIChatCompletionsModel
except ImportError:
    # Fallback for environments without the specific import
    logger.warning("OpenAIChatCompletionsModel not available, using default configuration")
    OpenAIChatCompletionsModel = None
