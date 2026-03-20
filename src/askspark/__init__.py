"""
AskSpark - AI Consultant Assistant

A professional multi-provider AI analysis platform that showcases advanced integration skills
and provides genuine business value.
"""

__version__ = "1.0.0"
__author__ = "AskSpark Team"
__description__ = "Professional multi-provider AI analysis platform"

from .core.ai_providers import UnifiedAIClient
from .core.model_comparison import ModelComparisonEngine
from .core.document_intelligence import RAGEngine
from .workflows.engine import WorkflowEngine

__all__ = [
    "UnifiedAIClient",
    "ModelComparisonEngine", 
    "RAGEngine",
    "WorkflowEngine"
]
