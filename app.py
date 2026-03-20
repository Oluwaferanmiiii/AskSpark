import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import json
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Import our modules
from src.askspark.config.settings import Config
from src.askspark.config.logging import setup_logging, get_logger
from src.askspark.core.ai_providers import UnifiedAIClient
from src.askspark.core.model_comparison import ModelComparisonEngine
from src.askspark.core.document_intelligence import RAGEngine
from src.askspark.workflows.engine import WorkflowEngine
from src.askspark.workflows.models import TriggerType
from src.askspark.notifications.channels import NotificationChannel

setup_logging()
logger = get_logger(__name__)

class AIConsultantDashboard:
    """Main dashboard for AI Consultant Assistant"""
    
    def __init__(self):
        self.client = UnifiedAIClient()
        self.model_comparison = ModelComparisonEngine()
        self.document_engine = RAGEngine()
        self.workflow_engine = WorkflowEngine({
            'PUSHOVER_USER_KEY': Config.PUSHOVER_USER_KEY,
            'PUSHOVER_APP_TOKEN': Config.PUSHOVER_APP_TOKEN
        })
        
        # Start workflow scheduler
        self.workflow_engine.start_scheduler()
        
        # Initialize some demo workflows
        self._create_demo_workflows()
    
    def _create_demo_workflows(self):
        """Create demo workflows for showcase"""
        # Model comparison workflow
        self.workflow_engine.create_workflow(
            workflow_id="daily_model_check",
            name="Daily Model Performance Check",
            description="Automated daily comparison of AI models",
            triggers=[{"type": "schedule", "config": {"time": "09:00"}}],
            actions=[
                {
                    "type": "run_model_comparison",
                    "parameters": {
                        "prompt": "Analyze the business benefits of AI automation",
                        "providers_models": [("openai", "gpt-3.5-turbo"), ("anthropic", "claude-3-haiku-20240307")]
                    }
                },
                {
                    "type": "send_notification",
                    "parameters": {
                        "message": "Daily model comparison completed. Check dashboard for results.",
                        "channels": ["pushover"]
                    }
                }
            ]
        )
        
        # Document analysis workflow
        self.workflow_engine.create_workflow(
            workflow_id="document_alert",
            name="Document Analysis Alert",
            description="Alert when new documents are processed",
            triggers=[{"type": "document_analysis"}],
            actions=[
                {
                    "type": "send_notification",
                    "parameters": {
                        "message": "New document analyzed: {document_name}",
                        "channels": ["pushover"]
                    }
                }
            ]
        )
    
    def get_provider_status(self) -> pd.DataFrame:
        """Get status of all AI providers"""
        providers = []
        
        for provider in Config.get_available_providers():
            models = Config.get_provider_models(provider)
            for model in models:
                providers.append({
                    "Provider": provider.capitalize(),
                    "Model": model,
                    "Status": "Available",
                    "API Key": "Configured" if Config.AI_PROVIDERS[provider]["api_key"] else "Missing"
                })
        
        return pd.DataFrame(providers)
    
    def compare_models_interface(self, prompt: str, providers: List[str]) -> Tuple[pd.DataFrame, str, go.Figure]:
        """Interface for model comparison"""
        if not prompt.strip():
            return pd.DataFrame(), "Please enter a prompt", None
        
        # Build provider-model pairs
        providers_models = []
        for provider in providers:
            models = Config.get_provider_models(provider)
            if models:
                providers_models.append((provider, models[0]))  # Use first model from each provider
        
        if not providers_models:
            return pd.DataFrame(), "No providers available", None
        
        try:
            # Run comparison
            results = self.model_comparison.compare_models(prompt, providers_models)
            
            if not results:
                return pd.DataFrame(), "No results from comparison", None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                "Provider": r.provider.capitalize(),
                "Model": r.model,
                "Response Time (s)": round(r.response_time, 2),
                "Tokens Used": r.tokens_used,
                "Cost ($)": round(r.cost, 6),
                "Quality Score": round(r.quality_score, 3),
                "Relevance": round(r.relevance_score, 3),
                "Completeness": round(r.completeness_score, 3)
            } for r in results])
            
            # Get recommendations
            recommendations = self.model_comparison.get_recommendations(results)
            
            # Format recommendations
            rec_text = "\n".join([
                f"**{rec['category']}**: {rec['provider']}/{rec['model']} - {rec['reasoning']}"
                for rec in recommendations
            ])
            
            # Create visualization
            fig = self.model_comparison.create_comparison_visualization(results)
            
            return df, rec_text, fig
            
        except Exception as e:
            logger.error(f"Comparison error: {str(e)}")
            return pd.DataFrame(), f"Error: {str(e)}", None
    
    def upload_document_interface(self, file_obj) -> str:
        """Interface for document upload"""
        if file_obj is None:
            return "Please upload a document"
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
                tmp_file.write(file_obj.read())
                tmp_file_path = tmp_file.name
            
            # Process document
            doc_id = self.document_engine.process_and_store_document(tmp_file_path, file_obj.name)
            
            # Clean up
            os.unlink(tmp_file_path)
            
            # Trigger workflow
            self.workflow_engine.execute_workflow("document_alert", {"document_name": file_obj.name})
            
            return f"Document '{file_obj.name}' processed successfully. Document ID: {doc_id}"
            
        except Exception as e:
            logger.error(f"Document upload error: {str(e)}")
            return f"Error processing document: {str(e)}"
    
    def query_documents_interface(self, query: str) -> Tuple[str, pd.DataFrame]:
        """Interface for document Q&A"""
        if not query.strip():
            return "Please enter a query", pd.DataFrame()
        
        try:
            # Query documents
            result = self.document_engine.query_documents(query)
            
            # Format sources
            if result['sources']:
                sources_df = pd.DataFrame(result['sources'])
                sources_df = sources_df[['source', 'page', 'score', 'snippet']]
                sources_df.columns = ['Document', 'Page', 'Score', 'Snippet']
            else:
                sources_df = pd.DataFrame([{"Document": "No sources found", "Page": "-", "Score": "-", "Snippet": "-"}])
            
            return result['answer'], sources_df
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return f"Error: {str(e)}", pd.DataFrame()
    
    def get_workflows_status(self) -> pd.DataFrame:
        """Get status of all workflows"""
        workflows = self.workflow_engine.list_workflows()
        
        if not workflows:
            return pd.DataFrame([{"Workflow": "No workflows", "Status": "-", "Created": "-"}])
        
        df_data = []
        for wf in workflows:
            df_data.append({
                "Workflow": wf['name'],
                "Status": "Enabled" if wf['enabled'] else "Disabled",
                "Created": wf['created_at'][:10],
                "Triggers": len(wf['triggers']),
                "Actions": len(wf['actions'])
            })
        
        return pd.DataFrame(df_data)
    
    def execute_workflow_interface(self, workflow_id: str) -> str:
        """Interface for manual workflow execution"""
        if not workflow_id:
            return "Please select a workflow"
        
        try:
            success = self.workflow_engine.execute_workflow(workflow_id)
            if success:
                return f"Workflow '{workflow_id}' executed successfully"
            else:
                return f"Workflow '{workflow_id}' execution failed"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def send_test_notification(self, message: str, channels: List[str]) -> str:
        """Interface for sending test notifications"""
        if not message.strip():
            return "Please enter a message"
        
        try:
            channel_objs = [NotificationChannel(ch) for ch in channels]
            success = self.workflow_engine.notification_service.send_notification(
                message, "Test Notification", channel_objs
            )
            
            if success:
                return "Test notification sent successfully"
            else:
                return "Failed to send notification"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_dashboard(self):
        """Create the main Gradio dashboard"""
        with gr.Blocks(title="AI Consultant Assistant", theme=gr.themes.Soft()) as dashboard:
            gr.Markdown("# AI Consultant Assistant")
            gr.Markdown("Professional multi-provider AI analysis platform for business intelligence and automation")
            
            with gr.Tabs():
                # Tab 1: Provider Status
                with gr.Tab("Provider Status"):
                    gr.Markdown("## AI Provider Configuration")
                    provider_df = gr.Dataframe(
                        value=self.get_provider_status(),
                        label="Available AI Providers",
                        interactive=False
                    )
                    refresh_btn = gr.Button("Refresh Status", variant="secondary")
                    refresh_btn.click(self.get_provider_status, outputs=provider_df)
                
                # Tab 2: Model Comparison
                with gr.Tab("Model Comparison"):
                    gr.Markdown("## Compare AI Models Performance")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Test Prompt",
                                placeholder="Enter a prompt to test across models...",
                                lines=3
                            )
                            provider_checkboxes = gr.CheckboxGroup(
                                choices=[p.capitalize() for p in Config.get_available_providers()],
                                label="Select Providers",
                                value=[p.capitalize() for p in Config.get_available_providers()[:2]] if Config.get_available_providers() else []
                            )
                            compare_btn = gr.Button("Run Comparison", variant="primary")
                        
                        with gr.Column(scale=1):
                            recommendations_output = gr.Markdown("### Recommendations\nRun a comparison to see recommendations")
                    
                    with gr.Row():
                        comparison_df = gr.Dataframe(label="Comparison Results", interactive=False)
                    
                    with gr.Row():
                        comparison_plot = gr.Plot(label="Performance Visualization")
                    
                    compare_btn.click(
                        self.compare_models_interface,
                        inputs=[prompt_input, provider_checkboxes],
                        outputs=[comparison_df, recommendations_output, comparison_plot]
                    )
                
                # Tab 3: Document Intelligence
                with gr.Tab("Document Intelligence"):
                    gr.Markdown("## Upload and Analyze Documents")
                    
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Upload Document",
                                file_types=[".pdf", ".docx", ".txt"]
                            )
                            upload_btn = gr.Button("Process Document", variant="primary")
                            upload_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Column():
                            doc_list = gr.Textbox(
                                label="Processed Documents",
                                value="\n".join(self.document_engine.list_documents()) if self.document_engine.list_documents() else "No documents processed",
                                interactive=False,
                                lines=5
                            )
                    
                    gr.Markdown("## Query Documents")
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question about your documents..."
                        )
                        query_btn = gr.Button("Query", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)
                        with gr.Column():
                            sources_df = gr.Dataframe(label="Sources", interactive=False)
                    
                    upload_btn.click(self.upload_document_interface, inputs=file_upload, outputs=upload_status)
                    query_btn.click(self.query_documents_interface, inputs=query_input, outputs=[answer_output, sources_df])
                
                # Tab 4: Workflow Automation
                with gr.Tab("Workflow Automation"):
                    gr.Markdown("## Automated Workflows")
                    
                    with gr.Row():
                        workflows_df = gr.Dataframe(
                            value=self.get_workflows_status(),
                            label="Active Workflows",
                            interactive=False
                        )
                    
                    gr.Markdown("## Manual Workflow Execution")
                    with gr.Row():
                        workflow_dropdown = gr.Dropdown(
                            choices=[wf['id'] for wf in self.workflow_engine.list_workflows()],
                            label="Select Workflow"
                        )
                        execute_btn = gr.Button("Execute Workflow", variant="primary")
                        execution_status = gr.Textbox(label="Execution Status", interactive=False)
                    
                    gr.Markdown("## Test Notifications")
                    with gr.Row():
                        with gr.Column():
                            notification_input = gr.Textbox(
                                label="Test Message",
                                placeholder="Enter test notification message..."
                            )
                            notification_channels = gr.CheckboxGroup(
                                choices=["pushover", "email", "slack", "webhook"],
                                label="Notification Channels",
                                value=["pushover"]
                            )
                            notify_btn = gr.Button("Send Test Notification", variant="secondary")
                        
                        with gr.Column():
                            notification_status = gr.Textbox(label="Notification Status", interactive=False)
                    
                    execute_btn.click(self.execute_workflow_interface, inputs=workflow_dropdown, outputs=execution_status)
                    notify_btn.click(self.send_test_notification, inputs=[notification_input, notification_channels], outputs=notification_status)
                
                # Tab 5: Analytics Dashboard
                with gr.Tab("Analytics"):
                    gr.Markdown("## Performance Analytics")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Overview")
                            gr.Markdown(f"- **Available Providers**: {len(Config.get_available_providers())}")
                            gr.Markdown(f"- **Total Models**: {sum(len(Config.get_provider_models(p)) for p in Config.get_available_providers())}")
                            gr.Markdown(f"- **Processed Documents**: {len(self.document_engine.list_documents())}")
                            gr.Markdown(f"- **Active Workflows**: {len([w for w in self.workflow_engine.list_workflows() if w['enabled']])}")
                            gr.Markdown(f"- **Comparison History**: {len(self.model_comparison.comparison_history)}")
                        
                        with gr.Column():
                            gr.Markdown("### Quick Actions")
                            quick_compare_btn = gr.Button("Quick Model Comparison", variant="primary")
                            quick_notify_btn = gr.Button("Test System Notification", variant="secondary")
                            
                            quick_status = gr.Textbox(label="Quick Action Status", interactive=False)
                    
                    def quick_comparison():
                        try:
                            results = self.model_comparison.compare_models(
                                "What are the benefits of AI automation in business?",
                                [("openai", "gpt-3.5-turbo"), ("anthropic", "claude-3-haiku-20240307")]
                            )
                            if results:
                                best = max(results, key=lambda x: x.quality_score)
                                return f"Quick comparison completed. Best model: {best.provider}/{best.model}"
                            return "No results from quick comparison"
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    def quick_notification():
                        try:
                            success = self.workflow_engine.notification_service.send_notification(
                                "System test notification from AI Consultant Assistant",
                                "System Test"
                            )
                            return "Test notification sent" if success else "Notification failed"
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    quick_compare_btn.click(quick_comparison, outputs=quick_status)
                    quick_notify_btn.click(quick_notification, outputs=quick_status)
            
            gr.Markdown("---")
            gr.Markdown("### AI Consultant Assistant v1.0 | Professional AI Analysis Platform")
        
        return dashboard

def main():
    """Main function to run the dashboard"""
    dashboard_app = AIConsultantDashboard()
    dashboard = dashboard_app.create_dashboard()
    
    # Launch the dashboard
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=Config.DEBUG
    )

if __name__ == "__main__":
    main()
