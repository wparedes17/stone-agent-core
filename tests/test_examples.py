import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

# Ensure the project src directory is on sys.path for local test execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class MockLiteLLMResponse:
    """Mock response object for litellm"""
    def __init__(self, content: str = "Test response", prompt_tokens: int = 10, completion_tokens: int = 5):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content
        self.choices[0].finish_reason = "stop"
        self.usage = Mock()
        self.usage.prompt_tokens = prompt_tokens
        self.usage.completion_tokens = completion_tokens
        self.usage.total_tokens = prompt_tokens + completion_tokens


class MockLiteLLMToolResponse(MockLiteLLMResponse):
    """Mock response for tool calls"""
    def __init__(self, tool_calls: List[Dict] = None):
        super().__init__()
        self.choices[0].message.tool_calls = tool_calls or []
        self.choices[0].finish_reason = "tool_calls"


class TestToolLLMExample:
    """Test cases for tool_llm.py example"""
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment variables"""
        env_vars = {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
        }
        with patch.dict(os.environ, env_vars):
            yield env_vars
    
    @pytest.fixture
    def mock_litellm(self):
        """Mock litellm module"""
        with patch('stone_agent_core.llm.llm_litellm.litellm') as mock:
            yield mock
    
    def test_tool_llm_example_initialization(self, mock_env):
        """Test that the tool example can be initialized"""
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from tool_llm import ToolLLMExample
        
        example = ToolLLMExample()
        assert example.available_providers == ["openai", "anthropic"]
    
    def test_detect_available_providers(self):
        """Test provider detection based on API keys"""
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        
        # Remove the module from cache before testing
        if 'tool_llm' in sys.modules:
            del sys.modules['tool_llm']
        
        # Test with no API keys
        with patch.dict(os.environ, {}, clear=True):
            # Import after environment is patched
            import tool_llm
            # Mock load_dotenv to prevent loading from .env files
            with patch.object(tool_llm, 'load_dotenv', return_value=None):
                example = tool_llm.ToolLLMExample()
                assert example.available_providers == []
        
        # Clean up module cache again
        if 'tool_llm' in sys.modules:
            del sys.modules['tool_llm']
        
        # Test with only OPENAI_API_KEY
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}, clear=True):
            # Import after environment is patched
            import tool_llm
            # Mock load_dotenv to prevent loading from .env files
            with patch.object(tool_llm, 'load_dotenv', return_value=None):
                example = tool_llm.ToolLLMExample()
                assert "openai" in example.available_providers
                assert "anthropic" not in example.available_providers
    
    def test_process_tool_calls_success(self, mock_env):
        """Test successful tool call processing"""
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from tool_llm import ToolLLMExample
        
        example = ToolLLMExample()
        
        tool_calls = [
            {
                "id": "call_123",
                "name": "get_weather",
                "arguments": {"location": "Paris, France", "unit": "celsius"}
            }
        ]
        
        results = example.process_tool_calls(tool_calls)
        
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_123"
        assert results[0]["name"] == "get_weather"
        assert "Paris, France" in results[0]["content"]
    
    def test_process_tool_calls_error(self, mock_env):
        """Test tool call processing with errors"""
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from tool_llm import ToolLLMExample
        
        example = ToolLLMExample()
        
        tool_calls = [
            {
                "id": "call_123",
                "name": "unknown_tool",
                "arguments": {"location": "Paris"}
            }
        ]
        
        results = example.process_tool_calls(tool_calls)
        
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_123"
        assert "Error: Unknown tool unknown_tool" in results[0]["content"]
    
    @patch('stone_agent_core.llm.llm_litellm.litellm.completion')
    def test_run_tool_example_openai_success(self, mock_completion, mock_env):
        """Test successful OpenAI tool example run"""
        # Mock the tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Paris, France"}'
        
        mock_completion.side_effect = [
            MockLiteLLMToolResponse([mock_tool_call]),  # First call with tool calls
            MockLiteLLMResponse("The weather in Paris is 22°C", 20, 10)  # Second call with tool results
        ]
        
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from tool_llm import ToolLLMExample
        
        example = ToolLLMExample()
        
        # Should not raise an exception
        example.run_tool_example("openai")
        
        # Verify litellm was called twice
        assert mock_completion.call_count == 2
    
    @patch('stone_agent_core.llm.llm_litellm.litellm.completion')
    def test_run_tool_example_anthropic_success(self, mock_completion, mock_env):
        """Test successful Anthropic tool example run"""
        # Mock the tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Paris, France"}'
        
        mock_completion.side_effect = [
            MockLiteLLMToolResponse([mock_tool_call]),  # First call with tool calls
            MockLiteLLMResponse("The weather in Paris is 22°C", 20, 10)  # Second call with tool results
        ]
        
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from tool_llm import ToolLLMExample
        
        example = ToolLLMExample()
        
        # Should not raise an exception
        example.run_tool_example("anthropic")
        
        # Verify litellm was called twice
        assert mock_completion.call_count == 2
    
    def test_run_tool_example_missing_api_key(self):
        """Test tool example with missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            sys.path.insert(0, str(PROJECT_ROOT / "examples"))
            from tool_llm import ToolLLMExample
            
            example = ToolLLMExample()
            
            # Should handle missing API key gracefully
            example.run_tool_example("openai")  # Should not crash


class TestFrameworkWithLLMExample:
    """Test cases for framework_with_llm.py example"""
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment variables"""
        env_vars = {
            'OPENAI_API_KEY': 'test-openai-key',
        }
        with patch.dict(os.environ, env_vars):
            yield env_vars
    
    @patch('stone_agent_core.llm.llm_litellm.litellm.completion')
    def test_create_conversational_agent(self, mock_completion, mock_env):
        """Test creating a conversational agent"""
        mock_completion.return_value = MockLiteLLMResponse("Hello! How can I help you?", 15, 8)
        
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from framework_with_llm import create_conversational_agent, ConversationalState
        
        agent = create_conversational_agent(provider="openai", model="gpt-4")
        
        assert agent is not None
        assert agent.state_class == ConversationalState
        assert len(agent.modules) == 1
        assert "llm_responder" in agent.modules
    
    @patch('stone_agent_core.llm.llm_litellm.litellm.completion')
    def test_run_conversational_example_success(self, mock_completion, mock_env):
        """Test successful conversational example execution"""
        mock_completion.return_value = MockLiteLLMResponse("Why do programmers prefer dark mode? Because light attracts bugs!", 20, 15)
        
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from framework_with_llm import run_conversational_example
        
        # Should not raise an exception
        run_conversational_example()
        
        # Verify litellm was called
        mock_completion.assert_called()
    
    def test_conversational_state_structure(self):
        """Test that ConversationalState has the correct structure"""
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from framework_with_llm import ConversationalState
        
        # Test creating state with all required fields
        state = ConversationalState(
            module_results={},
            errors=[],
            current_module="",
            execution_complete=False,
            messages=[{"role": "user", "content": "Hello"}],
            execution_path=[],
            llm_provider="openai",
            llm_model="gpt-4"
        )
        
        assert state["messages"] == [{"role": "user", "content": "Hello"}]
        assert state["llm_provider"] == "openai"
        assert state["llm_model"] == "gpt-4"
        assert state["execution_complete"] is False
    
    @patch('stone_agent_core.llm.llm_litellm.litellm.completion')
    def test_llm_response_module_error_handling(self, mock_completion, mock_env):
        """Test LLM response module error handling"""
        mock_completion.side_effect = Exception("API Error")
        
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from framework_with_llm import create_conversational_agent, ConversationalState
        
        agent = create_conversational_agent(provider="openai", model="gpt-4")
        
        state = ConversationalState(
            module_results={},
            errors=[],
            current_module="",
            execution_complete=False,
            messages=[{"role": "user", "content": "Hello"}],
            execution_path=[],
            llm_provider="openai",
            llm_model="gpt-4"
        )
        
        # Should handle API errors gracefully
        result_state = agent.run_agent(state)
        
        assert len(result_state["errors"]) > 0
        assert "API Error" in result_state["errors"][0]


class TestSimpleLLMExample:
    """Test cases for simple_llm.py example"""
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment variables"""
        env_vars = {
            'OPENAI_API_KEY': 'test-openai-key',
        }
        with patch.dict(os.environ, env_vars):
            yield env_vars
    
    @patch('stone_agent_core.llm.llm_litellm.litellm.completion')
    def test_simple_llm_example_success(self, mock_completion, mock_env):
        """Test successful simple LLM example execution"""
        mock_completion.return_value = MockLiteLLMResponse("This is a simple response", 10, 5)
        
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        from simple_llm import run_simple_example
        
        # Should not raise an exception
        run_simple_example()
        
        # Verify litellm was called
        mock_completion.assert_called()


class TestExampleIntegration:
    """Integration tests for examples"""
    
    def test_all_examples_importable(self):
        """Test that all example modules can be imported"""
        examples_path = PROJECT_ROOT / "examples"
        sys.path.insert(0, str(examples_path))
        
        try:
            import tool_llm
            import framework_with_llm
            import simple_llm
        except ImportError as e:
            pytest.fail(f"Failed to import example module: {e}")
    
    def test_examples_have_main_blocks(self):
        """Test that all examples have main execution blocks"""
        examples_path = PROJECT_ROOT / "examples"
        
        for example_file in ["tool_llm.py", "framework_with_llm.py", "simple_llm.py"]:
            file_path = examples_path / example_file
            with open(file_path, 'r') as f:
                content = f.read()
                assert 'if __name__ == "__main__":' in content, f"{example_file} missing main block"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
