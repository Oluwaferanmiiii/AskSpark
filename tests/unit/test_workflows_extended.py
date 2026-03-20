"""
Extended unit tests for Workflow Engine and related components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from src.askspark.workflows.engine import WorkflowEngine
from src.askspark.workflows.models import Workflow, WorkflowTrigger, WorkflowAction, TriggerType
from src.askspark.workflows.actions import ActionRegistry, SendNotificationAction, RunModelComparisonAction
from src.askspark.notifications.service import NotificationService
from src.askspark.notifications.channels import NotificationChannel


class TestActionRegistry:
    """Test ActionRegistry"""
    
    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service"""
        return Mock(spec=NotificationService)
    
    def test_registry_initialization(self, mock_notification_service):
        """Test registry initializes with default actions"""
        registry = ActionRegistry()
        
        actions = registry.list_actions()
        assert "send_notification" in actions
        assert "run_model_comparison" in actions
        assert "analyze_document" in actions
        assert "generate_report" in actions
    
    def test_register_custom_action(self, mock_notification_service):
        """Test registering a custom action"""
        registry = ActionRegistry()
        
        class CustomAction:
            def __init__(self, notification_service):
                self.notification_service = notification_service
            
            def execute(self, parameters, trigger_data=None):
                return True
        
        registry.register("custom_action", CustomAction)
        
        action = registry.get_action("custom_action", mock_notification_service)
        assert isinstance(action, CustomAction)
    
    def test_get_unknown_action(self, mock_notification_service):
        """Test getting an unknown action raises error"""
        registry = ActionRegistry()
        
        with pytest.raises(ValueError, match="Unknown action type"):
            registry.get_action("unknown_action", mock_notification_service)


class TestSendNotificationAction:
    """Test SendNotificationAction"""
    
    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service"""
        service = Mock(spec=NotificationService)
        service.send_notification.return_value = True
        return service
    
    @pytest.fixture
    def notification_action(self, mock_notification_service):
        """Create SendNotificationAction instance"""
        return SendNotificationAction(mock_notification_service)
    
    def test_send_notification_success(self, notification_action, mock_notification_service):
        """Test successful notification sending"""
        parameters = {
            "message": "Test message",
            "title": "Test Title",
            "channels": ["pushover"]
        }
        trigger_data = {"user": "test_user"}
        
        result = notification_action.execute(parameters, trigger_data)
        
        assert result is True
        mock_notification_service.send_notification.assert_called_once()
    
    def test_send_notification_with_template_substitution(self, notification_action, mock_notification_service):
        """Test notification with template substitution"""
        parameters = {
            "message": "Hello {user}, your workflow {workflow_id} completed.",
            "title": "Workflow {workflow_id} Complete",
            "channels": ["pushover"]
        }
        trigger_data = {"user": "John", "workflow_id": "test_123"}
        
        result = notification_action.execute(parameters, trigger_data)
        
        assert result is True
        
        # Check that template substitution occurred
        call_args = mock_notification_service.send_notification.call_args
        assert "John" in call_args[0][0]  # message
        assert "test_123" in call_args[0][0]  # message
        assert "test_123" in call_args[0][1]  # title
    
    def test_send_notification_failure(self, notification_action, mock_notification_service):
        """Test notification sending failure"""
        mock_notification_service.send_notification.return_value = False
        
        parameters = {"message": "Test message"}
        result = notification_action.execute(parameters)
        
        assert result is False


class TestRunModelComparisonAction:
    """Test RunModelComparisonAction"""
    
    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service"""
        return Mock(spec=NotificationService)
    
    @pytest.fixture
    def model_comparison_action(self, mock_notification_service):
        """Create RunModelComparisonAction with mocked dependencies"""
        with patch('src.askspark.workflows.actions.ModelComparisonEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            action = RunModelComparisonAction(mock_notification_service)
            action.model_comparison = mock_engine
            yield action
    
    def test_model_comparison_success(self, model_comparison_action, mock_notification_service):
        """Test successful model comparison"""
        # Mock comparison results
        mock_result = Mock()
        mock_result.provider = "openai"
        mock_result.model = "gpt-3.5-turbo"
        mock_result.quality_score = 0.85
        
        model_comparison_action.model_comparison.compare_models.return_value = [mock_result]
        
        parameters = {
            "prompt": "Compare AI models",
            "providers_models": [("openai", "gpt-3.5-turbo")]
        }
        
        result = model_comparison_action.execute(parameters)
        
        assert result is True
        mock_notification_service.send_notification.assert_called_once()
        
        # Check notification content
        call_args = mock_notification_service.send_notification.call_args
        assert "Model comparison completed" in call_args[0][1]  # title
    
    def test_model_comparison_failure(self, model_comparison_action, mock_notification_service):
        """Test model comparison failure"""
        model_comparison_action.model_comparison.compare_models.side_effect = Exception("API Error")
        
        parameters = {"prompt": "Compare AI models"}
        result = model_comparison_action.execute(parameters)
        
        assert result is False


class TestWorkflowEngineExtended:
    """Test WorkflowEngine extended functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            "PUSHOVER_USER_KEY": "test_user_key",
            "PUSHOVER_APP_TOKEN": "test_app_token"
        }
    
    @pytest.fixture
    def workflow_engine(self, mock_config):
        """Create WorkflowEngine with mocked dependencies"""
        with patch('src.askspark.workflows.engine.NotificationService') as mock_notification_service:
            engine = WorkflowEngine(mock_config)
            engine.notification_service = Mock(spec=NotificationService)
            yield engine
    
    def test_create_workflow_with_conditions(self, workflow_engine):
        """Test workflow creation with conditional actions"""
        triggers = [{"type": "manual", "config": {}}]
        actions = [
            {
                "type": "send_notification",
                "parameters": {"message": "success"},
                "condition": "data.get('status') == 'success'"
            }
        ]
        
        workflow = workflow_engine.create_workflow(
            "test_workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        assert len(workflow.actions) == 1
        assert workflow.actions[0].condition == "data.get('status') == 'success'"
    
    def test_execute_workflow_with_conditions(self, workflow_engine):
        """Test workflow execution with conditions"""
        # Create workflow with conditional action
        triggers = [{"type": "manual", "config": {}}]
        actions = [
            {
                "type": "send_notification",
                "parameters": {"message": "success"},
                "condition": "data.get('status') == 'success'"
            }
        ]
        
        workflow_engine.create_workflow(
            "test_workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        # Mock action execution
        with patch.object(workflow_engine, '_execute_action', return_value=True) as mock_execute:
            # Test with matching condition
            workflow_engine.execute_workflow("test_workflow", {"status": "success"})
            mock_execute.assert_called_once()
            
            # Reset mock
            mock_execute.reset_mock()
            
            # Test with non-matching condition
            workflow_engine.execute_workflow("test_workflow", {"status": "failed"})
            mock_execute.assert_not_called()
    
    def test_workflow_error_handling(self, workflow_engine):
        """Test workflow error handling"""
        triggers = [{"type": "manual", "config": {}}]
        actions = [{"type": "send_notification", "parameters": {"message": "test"}}]
        
        workflow_engine.create_workflow(
            "test_workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        # Mock action to raise exception
        with patch.object(workflow_engine, '_execute_action', side_effect=Exception("Test error")):
            result = workflow_engine.execute_workflow("test_workflow")
            
            assert result is False
    
    def test_get_workflow_status(self, workflow_engine):
        """Test getting workflow status"""
        triggers = [{"type": "manual", "config": {}}]
        actions = [{"type": "send_notification", "parameters": {"message": "test"}}]
        
        workflow_engine.create_workflow(
            "test_workflow",
            "Test Workflow",
            "Test Description",
            triggers,
            actions
        )
        
        status = workflow_engine.get_workflow_status("test_workflow")
        
        assert status["id"] == "test_workflow"
        assert status["name"] == "Test Workflow"
        assert status["enabled"] is True
        assert "created_at" in status
        assert "triggers" in status
        assert "actions" in status
    
    def test_get_workflow_status_not_found(self, workflow_engine):
        """Test getting status of non-existent workflow"""
        status = workflow_engine.get_workflow_status("non_existent")
        
        assert "error" in status
        assert status["error"] == "Workflow not found"
