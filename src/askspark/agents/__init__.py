"""
AskSpark Agents Package Initialization
Week 2 Lab 1: OpenAI Agents SDK integration
"""

from .base_agent import (
    AskSparkAgentBase,
    ModelComparisonAgent, 
    DocumentAnalysisAgent,
    WorkflowOrchestrationAgent,
    AgentManager,
    AgentResponse
)
from .tools import (
    AskSparkTools,
    get_all_tools,
    tools_registry
)
from .demo import (
    AskSparkAgentDemo,
    run_agent_demo,
    test_agent_integration
)
from .sales_outreach import (
    SalesOutreachManager,
    LeadResearchAgent,
    EmailPersonalizationAgent,
    EmailDeliveryAgent,
    SalesOutreachTools,
    Lead,
    LeadStatus,
    OutreachCampaign
)
from .structured_outputs import (
    StructuredOutputsManager,
    StructuredOutputAgent,
    GuardrailAgent,
    InputGuardrails,
    LeadQualificationResult,
    EmailContentAnalysis,
    ModelComparisonStructured,
    DocumentInsightStructured,
    ContentSafetyLevel
)
from .deep_research import (
    DeepResearchManager,
    DeepResearchAgent,
    ClarificationAgent,
    ResearchQualityAgent,
    DeepResearchTools,
    ResearchQuery,
    ResearchFinding,
    ResearchReport,
    ResearchSource,
    ResearchDepth
)
from .deep_research_demo import (
    DeepResearchDemo,
    run_deep_research_demo,
    test_deep_research_integration
)

__version__ = "0.1.0"
__author__ = "Oluwagbamila Oluwaferanmi"

# Package exports
__all__ = [
    # Base classes
    "AskSparkAgentBase",
    "AgentManager",
    "AgentResponse",
    
    # Agent implementations
    "ModelComparisonAgent",
    "DocumentAnalysisAgent", 
    "WorkflowOrchestrationAgent",
    
    # Tools
    "AskSparkTools",
    "get_all_tools",
    "tools_registry",
    
    # Demo and testing
    "AskSparkAgentDemo",
    "run_agent_demo",
    "test_agent_integration",
    
    # Sales outreach system
    "SalesOutreachManager",
    "LeadResearchAgent",
    "EmailPersonalizationAgent",
    "EmailDeliveryAgent",
    "SalesOutreachTools",
    "Lead",
    "LeadStatus",
    "OutreachCampaign",
    "SalesOutreachDemo",
    "run_sales_outreach_demo",
    "test_sales_outreach_integration",
    
    # Structured outputs and guardrails
    "StructuredOutputsManager",
    "StructuredOutputAgent",
    "GuardrailAgent",
    "InputGuardrails",
    "LeadQualificationResult",
    "EmailContentAnalysis",
    "ModelComparisonStructured",
    "DocumentInsightStructured",
    "ContentSafetyLevel",
    "StructuredOutputsDemo",
    "run_structured_outputs_demo",
    "test_structured_outputs_integration",
    
    # Deep research system
    "DeepResearchManager",
    "DeepResearchAgent",
    "ClarificationAgent",
    "ResearchQualityAgent",
    "DeepResearchTools",
    "ResearchQuery",
    "ResearchFinding",
    "ResearchReport",
    "ResearchSource",
    "ResearchDepth",
    "DeepResearchDemo",
    "run_deep_research_demo",
    "test_deep_research_integration"
]

def get_package_info():
    """Get package information"""
    return {
        "name": "AskSpark Agents",
        "version": __version__,
        "author": __author__,
        "description": "OpenAI Agents SDK integration for AskSpark platform",
        "features": [
            "Multi-agent orchestration",
            "Intelligent model comparison",
            "Document analysis with RAG",
            "Workflow automation",
            "Cost optimization",
            "Provider management",
            "Sales outreach automation",
            "Lead research and qualification",
            "Email personalization",
            "Campaign management",
            "Structured outputs and validation",
            "Safety guardrails and compliance",
            "Content filtering and spam detection",
            "Data quality assurance",
            "Deep research automation",
            "Multi-source intelligence gathering",
            "Research synthesis and analysis",
            "Quality evaluation and improvement"
        ]
    }

print(f"AskSpark Agents v{__version__} initialized")
