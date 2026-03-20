"""
Workflow action handlers for AskSpark
"""

import logging
from typing import Dict, Type, Callable, Any
from abc import ABC, abstractmethod

from ..core.model_comparison import ModelComparisonEngine
from ..core.document_intelligence import RAGEngine
from ..notifications.service import NotificationService
from ..config.logging import get_logger

logger = get_logger(__name__)


class BaseAction(ABC):
    """Base class for workflow actions"""
    
    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service
    
    @abstractmethod
    def execute(self, parameters: Dict, trigger_data: Dict = None) -> bool:
        """
        Execute the action
        
        Args:
            parameters: Action parameters
            trigger_data: Data from the trigger
            
        Returns:
            True if successful, False otherwise
        """
        pass


class SendNotificationAction(BaseAction):
    """Send notification action"""
    
    def execute(self, parameters: Dict, trigger_data: Dict = None) -> bool:
        """Send notification action"""
        message = parameters.get('message', 'Workflow executed successfully')
        title = parameters.get('title', 'Workflow Notification')
        
        # Template substitution
        if trigger_data:
            for key, value in trigger_data.items():
                message = message.replace(f"{{{key}}}", str(value))
                title = title.replace(f"{{{key}}}", str(value))
        
        from ..notifications.channels import NotificationChannel
        channels = [NotificationChannel(ch) for ch in parameters.get('channels', ['pushover'])]
        
        return self.notification_service.send_notification(message, title, channels)


class RunModelComparisonAction(BaseAction):
    """Run model comparison action"""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__(notification_service)
        self.model_comparison = ModelComparisonEngine()
    
    def execute(self, parameters: Dict, trigger_data: Dict = None) -> bool:
        """Run model comparison action"""
        prompt = parameters.get('prompt', 'Compare AI models on this task')
        providers_models = parameters.get('providers_models', [('openai', 'gpt-3.5-turbo')])
        
        try:
            results = self.model_comparison.compare_models(prompt, providers_models)
            
            # Send notification with results
            if results:
                best_model = max(results, key=lambda x: x.quality_score)
                message = f"Model comparison completed. Best model: {best_model.provider}/{best_model.model} (Quality: {best_model.quality_score:.3f})"
                self.notification_service.send_notification(message, "Model Comparison Results")
            
            return True
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return False


class AnalyzeDocumentAction(BaseAction):
    """Analyze document action"""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__(notification_service)
        self.document_engine = RAGEngine()
    
    def execute(self, parameters: Dict, trigger_data: Dict = None) -> bool:
        """Analyze document action"""
        document_path = parameters.get('document_path')
        query = parameters.get('query', 'Summarize this document')
        
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
            logger.error(f"Document analysis failed: {e}")
            return False


class GenerateReportAction(BaseAction):
    """Generate report action"""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__(notification_service)
        self.model_comparison = ModelComparisonEngine()
    
    def execute(self, parameters: Dict, trigger_data: Dict = None) -> bool:
        """Generate report action"""
        report_type = parameters.get('type', 'summary')
        
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
            logger.error(f"Report generation failed: {e}")
            return False


class ActionRegistry:
    """Registry for workflow actions"""
    
    def __init__(self):
        self._actions: Dict[str, Type[BaseAction]] = {}
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register default actions"""
        self.register("send_notification", SendNotificationAction)
        self.register("run_model_comparison", RunModelComparisonAction)
        self.register("analyze_document", AnalyzeDocumentAction)
        self.register("generate_report", GenerateReportAction)
    
    def register(self, action_type: str, action_class: Type[BaseAction]):
        """Register an action type"""
        self._actions[action_type] = action_class
    
    def get_action(self, action_type: str, notification_service: NotificationService) -> BaseAction:
        """Get an action instance"""
        if action_type not in self._actions:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return self._actions[action_type](notification_service)
    
    def list_actions(self) -> list:
        """List all registered actions"""
        return list(self._actions.keys())
