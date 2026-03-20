# AskSpark Test Suite

Comprehensive test suite for the AskSpark AI Consultant Assistant application.

## Test Coverage

### Unit Tests
- **AI Providers**: Mock API calls for OpenAI, Anthropic, Google, Groq
- **Model Comparison**: Comparison engine logic and metrics calculation
- **Document Intelligence**: RAG engine, document processing, embeddings
- **Workflows**: Workflow engine, actions, triggers, and execution
- **Notifications**: Multi-channel notification system

### Integration Tests
- **Data Flow**: End-to-end data flow from OS level to final output
- **API Integration**: Real API call patterns with mocked responses
- **Component Integration**: Cross-module interaction testing
- **Gradio Interface**: Dashboard functionality and user interactions

### Performance Tests
- **Stress Testing**: High-load scenarios and concurrent operations
- **Memory Management**: Resource usage and cleanup
- **Response Time**: Performance under various load conditions
- **Error Handling**: System stability under failure conditions

## Running Tests

### Quick Test Run
```bash
# Run all tests with comprehensive reporting
python tests/test_runner.py

# Or use pytest directly
pytest tests/ -v
```

### Individual Test Suites
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests only
pytest tests/performance/ -v

# Data flow validation only
pytest tests/integration/test_data_flow.py -v
```

### Coverage Analysis
```bash
# Generate coverage report
pytest tests/ --cov=src/askspark --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

## Test Categories

### 1. OS Level Tests
- Environment variable loading
- Configuration management
- File system operations
- Resource initialization

### 2. API Provider Tests
- Mock API responses for all providers
- Failover mechanism testing
- Rate limiting and error handling
- Cost calculation verification

### 3. Data Processing Tests
- Document upload and processing
- Text chunking and embedding
- Vector database operations
- Query and retrieval accuracy

### 4. Workflow Tests
- Workflow creation and execution
- Trigger-based automation
- Action registry and execution
- Notification delivery

### 5. Interface Tests
- Gradio component functionality
- User interaction flows
- Data visualization
- Error propagation

## Test Configuration

### Environment Setup
Tests use mocked API keys and responses:
```python
# Mock environment variables
OPENAI_API_KEY=test_openai_key
ANTHROPIC_API_KEY=test_anthropic_key
GOOGLE_API_KEY=test_google_key
GROQ_API_KEY=test_groq_key
```

### Mock Data
- Predefined AI responses for consistent testing
- Sample documents for processing tests
- Workflow configurations for automation tests
- Performance metrics for load testing

## Performance Benchmarks

### Response Time Targets
- API calls: < 2 seconds
- Document processing: < 5 seconds
- Workflow execution: < 3 seconds
- Dashboard loading: < 1 second

### Concurrent Load Testing
- 10+ concurrent API calls
- 5+ simultaneous workflows
- 20+ concurrent users simulation
- Memory usage < 100MB increase

## Debugging Failed Tests

### Common Issues
1. **Missing Dependencies**: Install test requirements
   ```bash
   pip install -r requirements.txt
   ```

2. **Import Errors**: Check Python path and virtual environment
   ```bash
   source ai_consultant_env/bin/activate
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. **Mock Configuration**: Verify mock objects and return values
4. **Async Issues**: Check threading and concurrent execution

### Test Output Analysis
- Use `-v` flag for verbose output
- Check `test_report.md` for detailed analysis
- Review coverage reports for untested code
- Monitor performance metrics during stress tests

## Test Development

### Adding New Tests
1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use descriptive test method names
4. Mock external dependencies
5. Include edge cases and error conditions

### Test Structure Example
```python
class TestNewFeature:
    @pytest.fixture
    def mock_dependency(self):
        return Mock()
    
    def test_functionality(self, mock_dependency):
        # Arrange
        setup_test_data()
        
        # Act
        result = test_function()
        
        # Assert
        assert result is not None
        assert result.success is True
```

## Test Goals

### Primary Objectives
1. **Reliability**: Ensure application works under all conditions
2. **Performance**: Verify performance targets are met
3. **Data Integrity**: Validate end-to-end data flow
4. **Error Handling**: Test failure scenarios and recovery
5. **User Experience**: Verify interface functionality

### Success Criteria
- All unit tests pass
- Integration tests validate data flow
- Performance tests meet benchmarks
- Coverage > 80% for critical modules
- No memory leaks or resource issues

## Test Checklist

Before deployment, ensure:
- [ ] All unit tests pass
- [ ] Integration tests validate workflows
- [ ] Performance tests meet benchmarks
- [ ] Coverage report is acceptable
- [ ] No critical security vulnerabilities
- [ ] Documentation is updated
- [ ] Test report is generated

## Continuous Integration

### Automated Testing
- Run tests on every commit
- Generate coverage reports
- Performance regression detection
- Automated test reporting

### Test Environments
- Development: Local testing with mocks
- Staging: Real API endpoints (rate-limited)
- Production: Monitoring and alerting

## 📞 Support

For test-related issues:
1. Check test logs and output
2. Review this documentation
3. Examine test reports
4. Verify environment setup
5. Check for known issues in GitHub

---

**Note**: This test suite is designed to work with mocked API responses, ensuring tests run reliably without requiring actual API keys or incurring costs.
