# Testing Suite for Stone Agent Core

This directory contains comprehensive unit tests for the Stone Agent Core framework.

## Test Structure

### Core Framework Tests (`test_core_framework.py`)
Tests the modular agent framework components:
- **Module Registration**: Verifies modules can be registered and ordered correctly
- **Dependency Resolution**: Tests topological sorting of module dependencies
- **Circular Dependency Detection**: Ensures circular dependencies are caught
- **Graph Creation**: Validates LangGraph integration and node/edge creation
- **State Management**: Tests state passing between modules
- **Error Handling**: Verifies graceful error handling

### LLM Component Tests (`test_llm_components.py`)
Tests the LLM integration layer:
- **LLM Client Factory**: Tests client creation for different providers
- **LiteLLM Client**: Tests the concrete implementation
- **Tool Conversion**: Validates tool schema conversion
- **Message Formatting**: Tests message preprocessing
- **Error Handling**: Tests API error scenarios
- **Streaming**: Tests streaming generation functionality

### Example Tests (`test_examples.py`)
Tests the example applications:
- **Tool LLM Example**: Tests tool calling functionality
- **Framework with LLM**: Tests integration examples
- **Simple LLM**: Tests basic usage patterns
- **Import Verification**: Ensures all examples are importable

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install -e ".[test]"
```

### Quick Start
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/stone_agent_core --cov-report=term-missing

# Run specific test files
python -m pytest tests/test_core_framework.py -v
python -m pytest tests/test_llm_components.py -v
python -m pytest tests/test_examples.py -v
```

### Alternative: Using the Test Runner
```bash
# Run the comprehensive test suite
python run_tests.py

# Quick verification without pytest
python verify_tests.py
```

## Test Configuration

The tests are configured in `pyproject.toml`:

- **Coverage**: 80% minimum coverage requirement
- **Markers**: `unit`, `integration`, `slow` test categories
- **Output**: Verbose with short tracebacks
- **Reports**: Terminal and HTML coverage reports

## Mock Strategy

The tests use comprehensive mocking to avoid external dependencies:

### LangGraph Mocking
- `DummyStateGraph`: Mocks LangGraph's StateGraph
- `DummyCompiledGraph`: Mocks compiled graph execution
- Captures invocation history for verification

### LiteLLM Mocking
- `MockLiteLLMResponse`: Simulates API responses
- `MockLiteLLMToolResponse`: Simulates tool call responses
- Patches `litellm.completion` to avoid real API calls

### Environment Mocking
- Uses `patch.dict` to control environment variables
- Tests API key detection and error handling

## Test Categories

### Unit Tests
- Fast, isolated tests
- No external dependencies
- Mock all external services

### Integration Tests
- Test component interactions
- Verify framework workflows
- Still use mocks for external APIs

### Example Tests
- Verify example applications work
- Test end-to-end scenarios
- Ensure documentation examples are functional

## Writing New Tests

### Adding Framework Tests
1. Create test classes inheriting from appropriate base classes
2. Use the existing mock infrastructure
3. Follow the naming convention: `test_<functionality>`

### Adding LLM Tests
1. Mock the `litellm` module
2. Use `MockLiteLLMResponse` for response simulation
3. Test both success and error scenarios

### Adding Example Tests
1. Import the example module
2. Mock external dependencies
3. Test the main execution flows

## Best Practices

### Test Structure
```python
class TestComponent:
    def test_success_scenario(self):
        # Test the happy path
        pass
    
    def test_error_handling(self):
        # Test error scenarios
        pass
    
    def test_edge_cases(self):
        # Test boundary conditions
        pass
```

### Mock Usage
```python
@patch('module.to.mock')
def test_with_mock(self, mock_function):
    mock_function.return_value = expected_value
    # Test implementation
```

### Assertions
- Use specific assertions (`assertEqual`, `assertIn`, etc.)
- Include descriptive error messages
- Test both positive and negative cases

## Continuous Integration

The tests are designed to run in CI/CD environments:
- No external dependencies required
- Fast execution (< 30 seconds)
- Clear pass/fail indicators
- Coverage reporting

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Mock Failures**: Check patch targets are correct
3. **Async Issues**: Use `pytest-asyncio` for async tests
4. **Coverage Issues**: Exclude test files from coverage

### Debug Mode
```bash
# Run with verbose output
python -m pytest tests/ -v -s

# Stop on first failure
python -m pytest tests/ -x

# Run specific test
python -m pytest tests/test_file.py::TestClass::test_method
```

## Contributing

When adding new features:
1. Write tests before implementation (TDD)
2. Ensure 80%+ coverage
3. Mock all external dependencies
4. Test error scenarios
5. Update this README if adding new test categories

## Test Data

The tests use minimal, deterministic test data:
- No real API keys or secrets
- Predictable mock responses
- Small, focused test cases
- Clear input/output relationships
