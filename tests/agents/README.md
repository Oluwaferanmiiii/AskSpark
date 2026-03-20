# AskSpark Agents Test Suite

## Overview
Comprehensive test suite for Week 2 Lab 1 OpenAI Agents SDK integration.

## Test Structure

### Test Files
- **`test_agents.py`** - Core agent functionality tests
- **`test_tools.py`** - Tool integration and functionality tests  
- **`test_integration.py`** - End-to-end integration tests
- **`conftest.py`** - Test configuration and utilities
- **`run_tests.py`** - Test runner script

### Test Categories

#### Unit Tests (`@pytest.mark.unit`)
- Agent initialization and configuration
- Tool functionality and error handling
- Data model validation
- Cost calculation accuracy
- Provider status checking

#### Integration Tests (`@pytest.mark.integration`)
- Multi-agent orchestration
- Tool workflow integration
- Cross-agent communication
- End-to-end scenarios

#### Performance Tests (`@pytest.mark.performance`)
- Agent response times
- Concurrent execution
- Memory usage
- Scalability testing

#### Error Handling Tests (`@pytest.mark.error_handling`)
- API failure scenarios
- Network connectivity issues
- Invalid input handling
- Timeout scenarios

## Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run unit tests only
python run_tests.py --type unit

# Run with coverage
python run_tests.py --coverage

# Verbose output
python run_tests.py --verbose
```

### Advanced Usage
```bash
# Run specific test file
pytest test_agents.py -v

# Run performance tests
pytest -m performance

# Run with coverage report
pytest --cov=askspark.agents --cov-report=html

# Run quick tests only
python run_tests.py --quick
```

## Test Coverage

### Agents Module Coverage
- ✅ Agent initialization and configuration
- ✅ Model comparison agent functionality
- ✅ Document analysis with RAG
- ✅ Workflow orchestration
- ✅ Agent manager operations
- ✅ Error handling and recovery

### Tools Module Coverage
- ✅ Model comparison tool
- ✅ Document analysis tool
- ✅ Provider status checking
- ✅ Cost calculation utilities
- ✅ Tool registry and integration

### Integration Coverage
- ✅ Multi-agent workflows
- ✅ Tool chain integration
- ✅ Business scenario testing
- ✅ Performance under load
- ✅ Error propagation

## Mock Strategy

### External Dependencies
- OpenAI API calls are mocked
- Database operations are simulated
- Network requests are faked
- File system operations are virtualized

### Test Data
- Pre-configured mock responses
- Realistic test scenarios
- Edge case data sets
- Performance benchmarks

## Test Utilities

### Mock Factories
```python
# Create mock agent response
response = create_mock_agent_response(
    content="Test response",
    execution_time=1.0
)

# Create mock model comparison results
results = create_mock_model_comparison_results()

# Create mock document analysis
analysis = create_mock_document_analysis_result()
```

### Performance Tracking
```python
tracker = PerformanceTracker()
tracker.start()
# ... run test code ...
duration = tracker.end("test_operation")
```

### Error Simulation
```python
# Simulate API timeout
error = ErrorSimulator.create_timeout_error()

# Simulate connection error
error = ErrorSimulator.create_connection_error()
```

## Test Scenarios

### Business Document Analysis
1. Workflow planning for document analysis
2. Model selection for document processing
3. Document content extraction and analysis
4. Cost optimization for large documents

### Cost Optimization
1. Model comparison for cost efficiency
2. High-volume processing scenarios
3. Budget constraint handling
4. Performance-cost tradeoffs

### Error Recovery
1. API service unavailability
2. Network connectivity issues
3. Invalid input handling
4. Timeout and retry logic

## Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Tests
  run: |
    cd tests/agents
    python run_tests.py --coverage --type all
```

### Quality Gates
- Minimum 90% test coverage required
- All tests must pass
- Performance benchmarks must be met
- No critical security vulnerabilities

## Test Data Management

### Test Documents
- Sample business plans
- Technical specifications
- Market research reports
- Legal documents

### Mock API Responses
- OpenAI API responses
- Anthropic API responses
- Google API responses
- Error scenarios

## Performance Benchmarks

### Response Time Targets
- Agent initialization: < 100ms
- Model comparison: < 2s
- Document analysis: < 5s
- Workflow planning: < 1s

### Concurrent Execution
- Support for 10+ concurrent agents
- Memory usage < 512MB
- CPU usage < 50%

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure src/ is in PYTHONPATH
2. **Mock failures**: Check mock configuration in conftest.py
3. **Async test errors**: Verify async/await usage
4. **Coverage gaps**: Add tests for uncovered code paths

### Debug Mode
```bash
# Run with debug output
pytest -s -v test_agents.py::TestModelComparisonAgent::test_agent_initialization

# Run single test with coverage
pytest --cov=askspark.agents --cov-report=term-missing test_agents.py -k "test_agent_initialization"
```

## Contributing

### Adding New Tests
1. Follow existing naming conventions
2. Use appropriate test markers
3. Include mock configuration
4. Add documentation for complex scenarios
5. Update coverage reports

### Test Standards
- All tests must be independent
- Use descriptive test names
- Include assertions for all expected behaviors
- Handle both success and failure scenarios
- Document complex test logic
