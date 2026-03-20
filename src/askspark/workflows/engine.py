"""
Workflow automation engine for AskSpark
"""

import logging
import schedule
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from .models import Workflow, WorkflowTrigger, WorkflowAction, TriggerType
from .actions import ActionRegistry
from ..notifications.service import NotificationService
from ..config.logging import get_logger

logger = get_logger(__name__)


class WorkflowEngine:
    """Main workflow automation engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.notification_service = NotificationService(config)
        self.action_registry = ActionRegistry()
        self.workflows: Dict[str, Workflow] = {}
        self.running = False
        self.scheduler_thread = None
        
    def create_workflow(self, 
                       workflow_id: str, 
                       name: str, 
                       description: str,
                       triggers: List[Dict], 
                       actions: List[Dict]) -> Workflow:
        """
        Create a new workflow
        
        Args:
            workflow_id: Unique workflow identifier
            name: Workflow name
            description: Workflow description
            triggers: List of trigger configurations
            actions: List of action configurations
            
        Returns:
            Created workflow instance
        """
        # Convert triggers
        workflow_triggers = []
        for trigger in triggers:
            workflow_triggers.append(WorkflowTrigger(
                trigger_type=TriggerType(trigger['type']),
                config=trigger.get('config', {})
            ))
        
        # Convert actions
        workflow_actions = []
        for action in actions:
            workflow_actions.append(WorkflowAction(
                action_type=action['type'],
                parameters=action.get('parameters', {}),
                condition=action.get('condition')
            ))
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            triggers=workflow_triggers,
            actions=workflow_actions
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name}")
        
        return workflow
    
    def execute_workflow(self, workflow_id: str, trigger_data: Dict = None) -> bool:
        """
        Execute a workflow
        
        Args:
            workflow_id: ID of workflow to execute
            trigger_data: Data from the trigger
            
        Returns:
            True if successful, False otherwise
        """
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False
        
        workflow = self.workflows[workflow_id]
        
        if not workflow.enabled:
            logger.info(f"Workflow {workflow_id} is disabled")
            return False
        
        logger.info(f"Executing workflow: {workflow.name}")
        
        try:
            for action in workflow.actions:
                if self._evaluate_condition(action.condition, trigger_data):
                    success = self._execute_action(action, trigger_data)
                    if not success:
                        logger.error(f"Action {action.action_type} failed in workflow {workflow_id}")
                        return False
            
            logger.info(f"Workflow {workflow.name} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, data: Dict) -> bool:
        """
        Evaluate action condition
        
        Args:
            condition: Condition string to evaluate
            data: Data to use in evaluation
            
        Returns:
            True if condition passes, False otherwise
        """
        if not condition:
            return True
        
        try:
            # Simple condition evaluation (can be enhanced)
            return eval(condition, {"__builtins__": {}}, data or {})
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return True
    
    def _execute_action(self, action: WorkflowAction, trigger_data: Dict) -> bool:
        """
        Execute a single workflow action
        
        Args:
            action: Action to execute
            trigger_data: Data from trigger
            
        Returns:
            True if successful, False otherwise
        """
        try:
            action_instance = self.action_registry.get_action(
                action.action_type, 
                self.notification_service
            )
            return action_instance.execute(action.parameters, trigger_data)
                
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {e}")
            return False
    
    def start_scheduler(self):
        """Start the workflow scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Workflow scheduler started")
    
    def stop_scheduler(self):
        """Stop the workflow scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Workflow scheduler stopped")
    
    def get_workflow_status(self, workflow_id: str) -> Dict:
        """
        Get workflow status
        
        Args:
            workflow_id: ID of workflow
            
        Returns:
            Workflow status dictionary
        """
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        return workflow.to_dict()
    
    def list_workflows(self) -> List[Dict]:
        """
        List all workflows
        
        Returns:
            List of workflow dictionaries
        """
        return [self.get_workflow_status(wf_id) for wf_id in self.workflows.keys()]
    
    def enable_workflow(self, workflow_id: str) -> bool:
        """
        Enable a workflow
        
        Args:
            workflow_id: ID of workflow to enable
            
        Returns:
            True if successful, False otherwise
        """
        if workflow_id in self.workflows:
            self.workflows[workflow_id].enabled = True
            logger.info(f"Enabled workflow {workflow_id}")
            return True
        return False
    
    def disable_workflow(self, workflow_id: str) -> bool:
        """
        Disable a workflow
        
        Args:
            workflow_id: ID of workflow to disable
            
        Returns:
            True if successful, False otherwise
        """
        if workflow_id in self.workflows:
            self.workflows[workflow_id].enabled = False
            logger.info(f"Disabled workflow {workflow_id}")
            return True
        return False
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow
        
        Args:
            workflow_id: ID of workflow to delete
            
        Returns:
            True if successful, False otherwise
        """
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        return False
