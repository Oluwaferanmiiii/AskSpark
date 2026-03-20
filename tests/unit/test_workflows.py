"""
Unit tests for workflow engine
"""

import pytest
from unittest.mock import Mock, patch

from src.askspark.workflows.engine import WorkflowEngine
from src.askspark.workflows.models import TriggerType
from src.askspark.notifications.service import NotificationService


class TestWorkflowEngine:
    """Test cases for WorkflowEngine"""
    
    @pytest.fixture
    def workflow_engine(self, test_config):
        """Create workflow engine fixture"""
        return WorkflowEngine(test_config)
    
    def test_create_workflow(self, workflow_engine):
        """Test workflow creation"""
        triggers = [{"type": "manual", "config": {}}]
        actions = [{"type": "send_notification", "parameters": {"message": "test"}}]
        
        workflow = workflow_engine.create_workflow(
            "test-workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        assert workflow.id == "test-workflow"
        assert workflow.name == "Test Workflow"
        assert workflow.enabled is True
        assert len(workflow.triggers) == 1
        assert len(workflow.actions) == 1
    
    def test_execute_workflow_success(self, workflow_engine):
        """Test successful workflow execution"""
        # Create workflow
        triggers = [{"type": "manual", "config": {}}]
        actions = [{"type": "send_notification", "parameters": {"message": "test"}}]
        
        workflow_engine.create_workflow(
            "test-workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        # Mock the notification service
        with patch.object(workflow_engine.notification_service, 'send_notification', return_value=True):
            result = workflow_engine.execute_workflow("test-workflow")
            assert result is True
    
    def test_execute_workflow_not_found(self, workflow_engine):
        """Test executing non-existent workflow"""
        result = workflow_engine.execute_workflow("non-existent")
        assert result is False
    
    def test_disable_enable_workflow(self, workflow_engine):
        """Test enabling and disabling workflows"""
        triggers = [{"type": "manual", "config": {}}]
        actions = [{"type": "send_notification", "parameters": {"message": "test"}}]
        
        workflow_engine.create_workflow(
            "test-workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        # Test disable
        result = workflow_engine.disable_workflow("test-workflow")
        assert result is True
        assert workflow_engine.workflows["test-workflow"].enabled is False
        
        # Test enable
        result = workflow_engine.enable_workflow("test-workflow")
        assert result is True
        assert workflow_engine.workflows["test-workflow"].enabled is True
    
    def test_delete_workflow(self, workflow_engine):
        """Test workflow deletion"""
        triggers = [{"type": "manual", "config": {}}]
        actions = [{"type": "send_notification", "parameters": {"message": "test"}}]
        
        workflow_engine.create_workflow(
            "test-workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        # Test deletion
        result = workflow_engine.delete_workflow("test-workflow")
        assert result is True
        assert "test-workflow" not in workflow_engine.workflows
