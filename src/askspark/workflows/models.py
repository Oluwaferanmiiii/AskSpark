"""
Workflow data models for AskSpark
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class TriggerType(Enum):
    """Workflow trigger types"""
    SCHEDULE = "schedule"
    MODEL_COMPARISON = "model_comparison"
    DOCUMENT_ANALYSIS = "document_analysis"
    MANUAL = "manual"


@dataclass
class WorkflowTrigger:
    """Workflow trigger configuration"""
    trigger_type: TriggerType
    config: Dict = None
    last_triggered: Optional[datetime] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class WorkflowAction:
    """Workflow action definition"""
    action_type: str
    parameters: Dict
    condition: Optional[str] = None


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    triggers: List[WorkflowTrigger]
    actions: List[WorkflowAction]
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert workflow to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "triggers": [asdict(trigger) for trigger in self.triggers],
            "actions": [asdict(action) for action in self.actions]
        }
