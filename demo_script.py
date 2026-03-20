#!/usr/bin/env python3
"""
Demo Script for AI Consultant Assistant
Showcases key features and capabilities
"""

import os
import time
import json
from datetime import datetime

# Import our modules
from src.askspark.config.settings import Config
from src.askspark.config.logging import setup_logging, get_logger
from src.askspark.core.ai_providers import UnifiedAIClient
from src.askspark.core.model_comparison import ModelComparisonEngine
from src.askspark.core.document_intelligence import RAGEngine
from src.askspark.workflows.engine import WorkflowEngine

setup_logging()
logger = get_logger(__name__)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

def demo_provider_status():
    """Demo: Show available AI providers"""
    print_section("AI Provider Status")
    
    available_providers = Config.get_available_providers()
    print(f"Available Providers: {len(available_providers)}")
    
    for provider in available_providers:
        models = Config.get_provider_models(provider)
        print(f"  • {provider.capitalize()}: {len(models)} models")
        for model in models:
            pricing = Config.MODEL_PRICING.get(model, {})
            print(f"    - {model} (${pricing.get('input', 0):.4f}/1K input, ${pricing.get('output', 0):.4f}/1K output)")

def demo_model_comparison():
    """Demo: Compare AI models"""
    print_section("Model Comparison Demo")
    
    client = UnifiedAIClient()
    comparison_engine = ModelComparisonEngine()
    
    # Test prompt
    test_prompt = "What are the top 3 benefits of AI automation for small businesses?"
    print(f"Test Prompt: {test_prompt}")
    
    # Select available providers for demo
    available_providers = Config.get_available_providers()[:2]  # Use first 2 providers
    providers_models = []
    
    for provider in available_providers:
        models = Config.get_provider_models(provider)
        if models:
            providers_models.append((provider, models[0]))
    
    if not providers_models:
        print("No providers available for comparison")
        return
    
    print(f"Comparing {len(providers_models)} model combinations...")
    
    try:
        results = comparison_engine.compare_models(
            test_prompt, 
            providers_models,
            expected_keywords=["automation", "business", "efficiency", "cost"]
        )
        
        if results:
            print_subsection("Results")
            for result in results:
                print(f"  {result.provider}/{result.model}:")
                print(f"    • Response Time: {result.response_time:.2f}s")
                print(f"    • Quality Score: {result.quality_score:.3f}")
                print(f"    • Cost: ${result.cost:.6f}")
                print(f"    • Tokens Used: {result.tokens_used}")
            
            # Get recommendations
            recommendations = comparison_engine.get_recommendations(results)
            print_subsection("Recommendations")
            for rec in recommendations:
                print(f"  {rec['category']}: {rec['provider']}/{rec['model']}")
                print(f"    {rec['reasoning']}")
        else:
            print("No results from comparison")
            
    except Exception as e:
        print(f"Comparison failed: {str(e)}")

def demo_document_intelligence():
    """Demo: Document processing and Q&A"""
    print_section("Document Intelligence Demo")
    
    # Create a sample document for demo
    sample_text = """
    AI Consultant Assistant - Business Proposal
    
    Executive Summary:
    Our AI Consultant Assistant platform provides businesses with comprehensive AI analysis capabilities,
    enabling data-driven decision making and process automation.
    
    Key Features:
    1. Multi-Provider Integration: Support for OpenAI, Anthropic, Google, Groq, and DeepSeek
    2. Model Performance Comparison: Real-time benchmarking and optimization
    3. Document Intelligence: RAG-powered document analysis and Q&A
    4. Workflow Automation: Automated business processes and notifications
    
    Business Benefits:
    - Cost Reduction: 30-50% reduction in AI implementation costs
    - Risk Mitigation: Model comparison prevents costly mistakes
    - Efficiency Gains: Automated workflows save 20+ hours/week
    - Quality Assurance: Continuous performance monitoring
    
    Implementation Timeline:
    Phase 1: Setup and Configuration (1 week)
    Phase 2: Model Testing and Optimization (1 week)
    Phase 3: Workflow Integration (1 week)
    Phase 4: Full Deployment (1 week)
    
    ROI Projection:
    Initial Investment: $10,000
    Monthly Savings: $5,000
    Break-even Point: 2 months
    Annual ROI: 500%
    
    Contact: ai-consultant@example.com
    Phone: (555) 123-4567
    """
    
    # Create temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        rag_engine = RAGEngine()
        
        print_subsection("Document Processing")
        print(f"Processing sample document...")
        
        # Process document
        doc_id = rag_engine.process_and_store_document(temp_file, "business_proposal.txt")
        print(f"Document processed: {doc_id}")
        
        # Test queries
        test_queries = [
            "What are the key features of the AI Consultant Assistant?",
            "What is the projected ROI?",
            "How long does implementation take?",
            "What are the business benefits?"
        ]
        
        print_subsection("Document Q&A")
        for query in test_queries:
            print(f"\n❓ Question: {query}")
            try:
                result = rag_engine.query_documents(query)
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Sources: {len(result['sources'])} found")
            except Exception as e:
                print(f"Query failed: {str(e)}")
        
        # Generate summary
        print_subsection("Document Summary")
        try:
            summary = rag_engine.summarize_document(doc_id)
            print(f"Summary: {summary['summary'][:300]}...")
        except Exception as e:
            print(f"Summary failed: {str(e)}")
        
    finally:
        # Clean up
        os.unlink(temp_file)

def demo_workflow_automation():
    """Demo: Workflow automation"""
    print_section("Workflow Automation Demo")
    
    # Initialize workflow engine
    workflow_engine = WorkflowEngine({
        'PUSHOVER_USER_KEY': Config.PUSHOVER_USER_KEY,
        'PUSHOVER_APP_TOKEN': Config.PUSHOVER_APP_TOKEN
    })
    
    print_subsection("Creating Demo Workflow")
    
    # Create a demo workflow
    demo_workflow = workflow_engine.create_workflow(
        workflow_id="demo_analysis",
        name="Demo Analysis Workflow",
        description="Demonstrates workflow automation capabilities",
        triggers=[
            {"type": "manual", "config": {}}
        ],
        actions=[
            {
                "type": "send_notification",
                "parameters": {
                    "message": "Demo workflow started at {timestamp}",
                    "title": "Workflow Demo"
                }
            }
        ]
    )
    
    print(f"Created workflow: {demo_workflow.name}")
    
    # List workflows
    print_subsection("Active Workflows")
    workflows = workflow_engine.list_workflows()
    for wf in workflows:
        status = "Enabled" if wf['enabled'] else "Disabled"
        print(f"  • {wf['name']} ({wf['id']}) - {status}")
        print(f"    Triggers: {len(wf['triggers'])}, Actions: {len(wf['actions'])}")
    
    # Execute demo workflow
    print_subsection("Executing Demo Workflow")
    try:
        success = workflow_engine.execute_workflow("demo_analysis", {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"Workflow execution: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(f"Workflow execution failed: {str(e)}")
    
    # Test notification service
    print_subsection("Notification Service Test")
    try:
        notification_success = workflow_engine.notification_service.send_notification(
            "This is a test notification from AI Consultant Assistant Demo",
            "Demo Notification"
        )
        print(f"Test notification: {'Sent' if notification_success else 'Failed'}")
    except Exception as e:
        print(f"Notification test failed: {str(e)}")

def demo_system_metrics():
    """Demo: Show system metrics and statistics"""
    print_section("System Metrics Demo")
    
    # Initialize engines
    client = UnifiedAIClient()
    comparison_engine = ModelComparisonEngine()
    rag_engine = RAGEngine()
    workflow_engine = WorkflowEngine({})
    
    print_subsection("System Overview")
    print(f"Available AI Providers: {len(Config.get_available_providers())}")
    print(f"Total AI Models: {sum(len(Config.get_provider_models(p)) for p in Config.get_available_providers())}")
    print(f"Processed Documents: {len(rag_engine.list_documents())}")
    print(f"Active Workflows: {len([w for w in workflow_engine.list_workflows() if w['enabled']])}")
    print(f"Comparison History: {len(comparison_engine.comparison_history)}")
    
    print_subsection("Configuration Status")
    config_items = [
        ("OpenAI API", bool(Config.OPENAI_API_KEY)),
        ("Anthropic API", bool(Config.ANTHROPIC_API_KEY)),
        ("Google API", bool(Config.GOOGLE_API_KEY)),
        ("Groq API", bool(Config.GROQ_API_KEY)),
        ("DeepSeek API", bool(Config.DEEPSEEK_API_KEY)),
        ("Pushover Notifications", bool(Config.PUSHOVER_USER_KEY and Config.PUSHOVER_APP_TOKEN))
    ]
    
    for name, configured in config_items:
        status = "Configured" if configured else "Not Configured"
        print(f"  • {name}: {status}")

def main():
    """Run the complete demo"""
    print("AI Consultant Assistant - Demo Showcase")
    print("=" * 60)
    print("This demo showcases the key features and capabilities")
    print("of the AI Consultant Assistant platform.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_provider_status()
        demo_model_comparison()
        demo_document_intelligence()
        demo_workflow_automation()
        demo_system_metrics()
        
        print_section("Demo Complete")
        print("All demos completed successfully!")
        print("\nNext Steps:")
        print("1. Configure your API keys in .env file")
        print("2. Run 'python app.py' to start the dashboard")
        print("3. Open http://localhost:7860 in your browser")
        print("4. Explore the interactive features")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {str(e)}")
        print("Please check your configuration and try again")

if __name__ == "__main__":
    main()
