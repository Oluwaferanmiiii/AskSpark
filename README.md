# AI Consultant Assistant

Professional multi-provider AI analysis platform that showcases advanced integration skills and provides genuine business value.

## Features

### Multi-Provider API Management
- **Unified API Client**: Single interface for OpenAI, Anthropic, Google, Groq, and DeepSeek
- **Provider Failover**: Automatic fallback between providers for reliability
- **Cost Tracking**: Real-time cost calculation and optimization
- **Error Handling**: Comprehensive error management and logging

### Intelligent Model Comparison Engine
- **Performance Metrics**: Response time, quality score, relevance, completeness
- **Visual Analytics**: Interactive charts and performance dashboards
- **Smart Recommendations**: AI-powered model selection based on use case
- **Benchmark Testing**: Automated evaluation across multiple criteria

### Document Intelligence System
- **Multi-Format Support**: PDF, DOCX, TXT processing
- **RAG Implementation**: Semantic search with vector embeddings
- **Q&A Interface**: Context-aware document questioning
- **Automated Insights**: Document summarization and analysis

### Workflow Automation Hub
- **Multi-Channel Notifications**: Pushover, Email, Slack, Webhooks
- **Scheduled Tasks**: Automated workflow execution
- **Trigger-Based Actions**: Model comparison, document analysis, reporting
- **Progress Tracking**: Real-time workflow monitoring

## Installation

### Prerequisites
- Python 3.8+
- API keys for AI providers (see configuration)

### Setup Steps

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd AI-Consultant-Assistant
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run the Application**
```bash
python app.py
```

4. **Access Dashboard**
Open http://localhost:7860 in your browser

## Configuration

### API Keys Required

Add these to your `.env` file:

```env
# AI Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Notification Services (Optional)
PUSHOVER_USER_KEY=your_pushover_user_key_here
PUSHOVER_APP_TOKEN=your_pushover_app_token_here
```

### Supported Providers

| Provider | Models | Pricing |
|----------|--------|---------|
| OpenAI | gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview | Standard |
| Anthropic | claude-3-sonnet, claude-3-haiku | Standard |
| Google | gemini-pro, gemini-pro-vision | Standard |
| Groq | llama2-70b-4096, mixtral-8x7b-32768 | Low cost |
| DeepSeek | deepseek-chat, deepseek-coder | Very low cost |

## Usage Guide

### Model Comparison

1. Navigate to the **Model Comparison** tab
2. Enter a test prompt
3. Select providers to compare
4. View performance metrics and recommendations

### Document Analysis

1. Go to **Document Intelligence** tab
2. Upload a PDF, DOCX, or TXT file
3. Ask questions about the document
4. Review AI-generated answers with sources

### Workflow Automation

1. Access **Workflow Automation** tab
2. View pre-configured workflows
3. Execute workflows manually or let them run automatically
4. Configure notifications for workflow results

### Analytics Dashboard

1. Check **Analytics** tab for system overview
2. Monitor provider status and performance
3. Run quick system tests
4. View usage statistics

## Architecture

```
AI Consultant Assistant
├── Core Services
│   ├── ai_providers.py      # Multi-provider API client
│   ├── model_comparison.py  # Performance comparison engine
│   ├── document_intelligence.py  # RAG document processing
│   └── workflow_automation.py    # Automation engine
├── Configuration
│   ├── config.py            # Central configuration management
│   └── .env                 # Environment variables
├── Interface
│   └── app.py              # Gradio dashboard
└── Data
    ├── SQLite database     # Metrics and history
    └── ChromaDB           # Vector embeddings
```

## Business Value

### For Clients
- **Risk Reduction**: Model comparison prevents costly AI implementation mistakes
- **Cost Optimization**: Identify the most cost-effective models for specific tasks
- **Quality Assurance**: Automated quality metrics ensure consistent performance
- **Document Intelligence**: Extract insights from business documents automatically

### For Developers
- **Enterprise Architecture**: Multi-provider integration with fallback mechanisms
- **Performance Monitoring**: Real-time metrics and visualization
- **Scalable Design**: Modular architecture for easy extension
- **Professional Codebase**: Clean, documented, production-ready code

## Testing

### Run Quick Tests
```bash
# Test model comparison
python -c "
from model_comparison import ModelComparisonEngine
engine = ModelComparisonEngine()
results = engine.compare_models('Test prompt', [('openai', 'gpt-3.5-turbo')])
print(results)
"

# Test document processing
python -c "
from document_intelligence import RAGEngine
rag = RAGEngine()
print('Document engine initialized')
"

# Test workflow engine
python -c "
from workflow_automation import WorkflowEngine
engine = WorkflowEngine({})
print('Workflow engine initialized')
"
```

### Integration Tests
The dashboard includes built-in test functions in the Analytics tab for quick system validation.

## Performance Metrics

The system tracks:
- **Response Time**: Latency for each model/provider
- **Quality Score**: AI-generated quality metrics
- **Cost Efficiency**: Price/performance ratios
- **Reliability**: Success rates and error tracking
- **Usage Statistics**: Token consumption and API calls

## Notifications

Configure multiple notification channels:
- **Pushover**: Mobile push notifications
- **Email**: SMTP-based email alerts
- **Slack**: Workspace notifications
- **Webhooks**: Custom HTTP endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the configuration guide
2. Review API key setup
3. Consult the troubleshooting section
4. Create an issue with detailed error logs

## Demo Showcase

This MVP demonstrates professional AI engineering capabilities:

### Technical Skills
- Multi-provider API integration
- Vector database operations
- Performance benchmarking
- Workflow automation
- Interactive dashboard design

### Business Applications
- AI model selection and optimization
- Document analysis and insights
- Automated reporting
- Cost management
- Quality assurance

### Portfolio Value
- **Enterprise-Ready**: Production-grade architecture
- **Comprehensive**: End-to-end AI solution
- **Scalable**: Modular and extensible design
- **Professional**: Clean code and documentation
- **Innovative**: Advanced RAG and automation features

---

**AI Consultant Assistant** - Transforming AI capabilities into business value
