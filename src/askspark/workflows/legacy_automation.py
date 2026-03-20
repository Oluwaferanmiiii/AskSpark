import json
import time
import schedule
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from pushover import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ai_providers import UnifiedAIClient
from model_comparison import ModelComparisonEngine
from document_intelligence import RAGEngine

logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    PUSHOVER = "pushover"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"

class TriggerType(Enum):
    SCHEDULE = "schedule"
    MODEL_COMPARISON = "model_comparison"
    DOCUMENT_ANALYSIS = "document_analysis"
    MANUAL = "manual"

@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

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

class NotificationService:
    """Multi-channel notification service"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pushover_client = None
        
        # Initialize Pushover if configured
        if config.get('PUSHOVER_USER_KEY') and config.get('PUSHOVER_APP_TOKEN'):
            self.pushover_client = Client(
                config['PUSHOVER_USER_KEY'],
                api_token=config['PUSHOVER_APP_TOKEN']
            )
    
    def send_notification(self, message: str, title: str = "AI Consultant Alert", 
                         channels: List[NotificationChannel] = None,
                         channel_config: Dict = None) -> bool:
        """Send notification through multiple channels"""
        if channels is None:
            channels = [NotificationChannel.PUSHOVER]
        
        if channel_config is None:
            channel_config = {}
        
        success = False
        
        for channel in channels:
            try:
                if channel == NotificationChannel.PUSHOVER:
                    success |= self._send_pushover(message, title, channel_config.get('pushover', {}))
                elif channel == NotificationChannel.EMAIL:
                    success |= self._send_email(message, title, channel_config.get('email', {}))
                elif channel == NotificationChannel.SLACK:
                    success |= self._send_slack(message, title, channel_config.get('slack', {}))
                elif channel == NotificationChannel.WEBHOOK:
                    success |= self._send_webhook(message, title, channel_config.get('webhook', {}))
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {str(e)}")
        
        return success
    
    def _send_pushover(self, message: str, title: str, config: Dict) -> bool:
        """Send Pushover notification"""
        if not self.pushover_client:
            logger.warning("Pushover not configured")
            return False
        
        try:
            self.pushover_client.send_message(message, title=title)
            logger.info("Pushover notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Pushover notification failed: {str(e)}")
            return False
    
    def _send_email(self, message: str, title: str, config: Dict) -> bool:
        """Send email notification"""
        required_fields = ['smtp_server', 'smtp_port', 'sender_email', 'sender_password', 'recipient']
        
        for field in required_fields:
            if field not in config:
                logger.warning(f"Email configuration missing {field}")
                return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = config['recipient']
            msg['Subject'] = title
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {str(e)}")
            return False
    
    def _send_slack(self, message: str, title: str, config: Dict) -> bool:
        """Send Slack notification"""
        if 'webhook_url' not in config:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            payload = {
                "text": f"*{title}*\n{message}",
                "username": "AI Consultant Assistant"
            }
            
            response = requests.post(config['webhook_url'], json=payload)
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
            return False
    
    def _send_webhook(self, message: str, title: str, config: Dict) -> bool:
        """Send generic webhook notification"""
        if 'url' not in config:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            payload = {
                "title": title,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "AI Consultant Assistant"
            }
            
            # Add custom headers if provided
            headers = config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            response = requests.post(config['url'], json=payload, headers=headers)
            response.raise_for_status()
            
            logger.info("Webhook notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Webhook notification failed: {str(e)}")
            return False

class WorkflowEngine:
    """Main workflow automation engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.notification_service = NotificationService(config)
        self.ai_client = UnifiedAIClient()
        self.model_comparison = ModelComparisonEngine()
        self.document_engine = RAGEngine()
        self.workflows: Dict[str, Workflow] = {}
        self.running = False
        self.scheduler_thread = None
        
    def create_workflow(self, workflow_id: str, name: str, description: str,
                       triggers: List[Dict], actions: List[Dict]) -> Workflow:
        """Create a new workflow"""
        
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
        """Execute a workflow"""
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
            logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            return False
    
    def _evaluate_condition(self, condition: str, data: Dict) -> bool:
        """Evaluate action condition"""
        if not condition:
            return True
        
        try:
            # Simple condition evaluation (can be enhanced)
            return eval(condition, {"__builtins__": {}}, data or {})
        except Exception as e:
            logger.error(f"Error evaluating condition: {str(e)}")
            return True
    
    def _execute_action(self, action: WorkflowAction, trigger_data: Dict) -> bool:
        """Execute a single workflow action"""
        try:
            if action.action_type == "send_notification":
                return self._action_send_notification(action.parameters, trigger_data)
            elif action.action_type == "run_model_comparison":
                return self._action_run_model_comparison(action.parameters, trigger_data)
            elif action.action_type == "analyze_document":
                return self._action_analyze_document(action.parameters, trigger_data)
            elif action.action_type == "generate_report":
                return self._action_generate_report(action.parameters, trigger_data)
            elif action.action_type == "schedule_next":
                return self._action_schedule_next(action.parameters, trigger_data)
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {str(e)}")
            return False
    
    def _action_send_notification(self, params: Dict, trigger_data: Dict) -> bool:
        """Send notification action"""
        message = params.get('message', 'Workflow executed successfully')
        title = params.get('title', 'Workflow Notification')
        
        # Template substitution
        if trigger_data:
            for key, value in trigger_data.items():
                message = message.replace(f"{{{key}}}", str(value))
                title = title.replace(f"{{{key}}}", str(value))
        
        channels = [NotificationChannel(ch) for ch in params.get('channels', ['pushover'])]
        
        return self.notification_service.send_notification(message, title, channels)
    
    def _action_run_model_comparison(self, params: Dict, trigger_data: Dict) -> bool:
        """Run model comparison action"""
        prompt = params.get('prompt', 'Compare AI models on this task')
        providers_models = params.get('providers_models', [('openai', 'gpt-3.5-turbo')])
        
        try:
            results = self.model_comparison.compare_models(prompt, providers_models)
            
            # Send notification with results
            if results:
                best_model = max(results, key=lambda x: x.quality_score)
                message = f"Model comparison completed. Best model: {best_model.provider}/{best_model.model} (Quality: {best_model.quality_score:.3f})"
                self.notification_service.send_notification(message, "Model Comparison Results")
            
            return True
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return False
    
    def _action_analyze_document(self, params: Dict, trigger_data: Dict) -> bool:
        """Analyze document action"""
        document_path = params.get('document_path')
        query = params.get('query', 'Summarize this document')
        
        if not document_path:
            logger.error("Document path not provided")
            return False
        
        try:
            # Process document
            doc_id = self.document_engine.process_and_store_document(document_path)
            
            # Query document
            result = self.document_engine.query_documents(query)
            
            # Send notification
            message = f"Document analysis completed for {document_path}. Answer: {result['answer'][:200]}..."
            self.notification_service.send_notification(message, "Document Analysis Results")
            
            return True
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            return False
    
    def _action_generate_report(self, params: Dict, trigger_data: Dict) -> bool:
        """Generate report action"""
        report_type = params.get('type', 'summary')
        
        try:
            if report_type == 'model_performance':
                # Generate model performance report
                if self.model_comparison.comparison_history:
                    report_data = {
                        'total_comparisons': len(self.model_comparison.comparison_history),
                        'providers': list(set(r.provider for r in self.model_comparison.comparison_history)),
                        'avg_quality': sum(r.quality_score for r in self.model_comparison.comparison_history) / len(self.model_comparison.comparison_history)
                    }
                    
                    message = f"Performance Report: {report_data['total_comparisons']} comparisons, {len(report_data['providers'])} providers, Avg Quality: {report_data['avg_quality']:.3f}"
                    self.notification_service.send_notification(message, "Performance Report")
            
            return True
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return False
    
    def _action_schedule_next(self, params: Dict, trigger_data: Dict) -> bool:
        """Schedule next workflow execution"""
        delay_minutes = params.get('delay_minutes', 60)
        workflow_id = params.get('workflow_id')
        
        if not workflow_id:
            logger.error("Workflow ID not provided for scheduling")
            return False
        
        # Schedule next execution
        schedule.every(delay_minutes).minutes.do(self.execute_workflow, workflow_id)
        
        logger.info(f"Scheduled workflow {workflow_id} to run in {delay_minutes} minutes")
        return True
    
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
        """Get workflow status"""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "enabled": workflow.enabled,
            "created_at": workflow.created_at.isoformat(),
            "triggers": [asdict(trigger) for trigger in workflow.triggers],
            "actions": [asdict(action) for action in workflow.actions]
        }
    
    def list_workflows(self) -> List[Dict]:
        """List all workflows"""
        return [self.get_workflow_status(wf_id) for wf_id in self.workflows.keys()]
    
    def enable_workflow(self, workflow_id: str) -> bool:
        """Enable a workflow"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].enabled = True
            logger.info(f"Enabled workflow {workflow_id}")
            return True
        return False
    
    def disable_workflow(self, workflow_id: str) -> bool:
        """Disable a workflow"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].enabled = False
            logger.info(f"Disabled workflow {workflow_id}")
            return True
        return False
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        return False
