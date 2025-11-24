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

from stone_agent_core.llm.llm_factory import LLMClientFactory
from stone_agent_core.llm.llm_litellm import LiteLLMClient
from stone_agent_core.llm.llm_base import BaseLLMClient


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
        super().__init__(content="")  # Empty content for tool calls
        self.choices[0].message.tool_calls = tool_calls or []
        self.choices[0].finish_reason = "tool_calls"


class TestLLMClientFactory:
    """Test cases for LLMClientFactory"""
    
    def test_create_openai_client(self):
        """Test creating OpenAI client"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClientFactory.create(provider="openai", model="gpt-4")
            assert isinstance(client, LiteLLMClient)
            assert client.provider == "openai"
            assert client.model == "gpt-4"
            assert client.api_key == "test-key"
    
    def test_create_anthropic_client(self):
        """Test creating Anthropic client"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            client = LLMClientFactory.create(provider="anthropic", model="claude-3")
            assert isinstance(client, LiteLLMClient)
            assert client.provider == "anthropic"
            assert client.model == "claude-3"
    
    def test_create_client_with_default_model(self):
        """Test creating client with default model"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClientFactory.create(provider="openai")
            assert client.model == "gpt-4o"  # Default model for OpenAI
    
    def test_create_client_missing_api_key(self):
        """Test error when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):
                LLMClientFactory.create(provider="openai")
    
    def test_create_unsupported_provider(self):
        """Test creating client with unsupported provider"""
        # Pass API key directly to avoid environment lookup issues
        client = LLMClientFactory.create(provider="unsupported", api_key="test-key")
        assert client.provider == "unsupported"


class TestLiteLLMClient:
    """Test cases for LiteLLMClient"""
    
    @pytest.fixture
    def mock_litellm(self):
        """Mock litellm module"""
        with patch('stone_agent_core.llm.llm_litellm.litellm') as mock:
            yield mock
    
    @pytest.fixture
    def openai_client(self):
        """Create a test OpenAI client"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return LiteLLMClient(provider="openai", model="gpt-4")
    
    def test_initialization(self, openai_client):
        """Test client initialization"""
        assert openai_client.provider == "openai"
        assert openai_client.model == "gpt-4"
        assert openai_client.api_key == "test-key"
    
    def test_format_model_openai(self, openai_client):
        """Test model formatting for OpenAI"""
        assert openai_client._format_model() == "openai/gpt-4"
    
    def test_format_model_anthropic(self):
        """Test model formatting for Anthropic"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            client = LiteLLMClient(provider="anthropic", model="claude-3")
            assert client._format_model() == "anthropic/claude-3"
    
    def test_get_default_model(self):
        """Test getting default models for different providers"""
        defaults = {
            'openai': 'gpt-4o',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'cohere': 'command-r-plus',
            'google': 'gemini-1.5-pro',
            'azure': 'gpt-4o',
        }
        
        for provider, expected_model in defaults.items():
            with patch.dict(os.environ, {f'{provider.upper()}_API_KEY': 'test-key'}):
                client = LiteLLMClient(provider=provider)
                assert client._get_default_model() == expected_model
    
    def test_get_api_key_env(self):
        """Test getting API key environment variable names"""
        env_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'azure': 'AZURE_API_KEY',
        }
        
        # Set up mock API keys for all providers
        mock_env_vars = {key: 'test-key' for key in env_keys.values()}
        
        with patch.dict(os.environ, mock_env_vars):
            for provider, expected_env in env_keys.items():
                client = LiteLLMClient(provider=provider)
                assert client._get_api_key_env() == expected_env
    
    def test_format_messages(self, openai_client):
        """Test message formatting"""
        system_prompt = "You are a helpful assistant"
        messages = [{"role": "user", "content": "Hello"}]
        
        formatted = openai_client._format_messages(system_prompt, messages)
        
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
        assert formatted[1]["role"] == "user"
        assert formatted[1]["content"] == "Hello"
    
    def test_convert_tools_to_litellm(self, openai_client):
        """Test tool conversion to litellm format"""
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}}
                }
            }
        ]
        
        converted = openai_client._convert_tools_to_litellm(tools)
        
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "test_tool"
        assert converted[0]["function"]["description"] == "A test tool"
        assert converted[0]["function"]["parameters"]["type"] == "object"
    
    def test_generate_success(self, openai_client, mock_litellm):
        """Test successful generation"""
        mock_response = MockLiteLLMResponse("Test response", 10, 5)
        mock_litellm.completion.return_value = mock_response
        
        result = openai_client.generate("System prompt", [{"role": "user", "content": "Hello"}])
        
        assert len(result) == 2
        response, usage = result
        assert response == "Test response"
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15
    
    def test_generate_with_tools(self, openai_client, mock_litellm):
        """Test generation with tools"""
        tools = [{"name": "test_tool", "description": "Test", "input_schema": {}}]
        mock_response = MockLiteLLMResponse("Test response", 10, 5)
        mock_litellm.completion.return_value = mock_response
        
        result = openai_client.generate("System prompt", [{"role": "user", "content": "Hello"}], tools=tools)
        
        assert result[0] == "Test response"
        mock_litellm.completion.assert_called_once()
        call_args = mock_litellm.completion.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == 1
    
    def test_generate_with_tools_tool_calls(self, openai_client, mock_litellm):
        """Test generation with tools that returns tool calls"""
        tools = [{"name": "test_tool", "description": "Test", "input_schema": {}}]
        
        # Mock tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'
        
        mock_response = MockLiteLLMToolResponse([mock_tool_call])
        mock_litellm.completion.return_value = mock_response
        
        result = openai_client.generate_with_tools("System prompt", [{"role": "user", "content": "Hello"}], tools)
        
        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["name"] == "test_tool"
        assert result["tool_calls"][0]["arguments"] == {"param": "value"}
        assert result["stop_reason"] == "tool_calls"
    
    def test_generate_stream(self, openai_client, mock_litellm):
        """Test streaming generation"""
        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),
        ]
        mock_litellm.completion.return_value = mock_chunks
        
        result = list(openai_client.generate_stream("System prompt", [{"role": "user", "content": "Hello"}]))
        
        assert result == ["Hello", " world"]
    
    def test_generate_error_handling(self, openai_client, mock_litellm):
        """Test error handling in generation"""
        mock_litellm.completion.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            openai_client.generate("System prompt", [{"role": "user", "content": "Hello"}])
    
    def test_generate_with_callback(self, openai_client, mock_litellm):
        """Test generation with callback"""
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
        ]
        mock_litellm.completion.return_value = mock_chunks
        
        callback_calls = []
        def test_callback(chunk):
            callback_calls.append(chunk)
        
        result = openai_client.generate_with_callback("System prompt", [{"role": "user", "content": "Hello"}], test_callback)
        
        assert result == "Hello world"
        assert callback_calls == ["Hello", " world"]


class TestBaseLLMClient:
    """Test cases for BaseLLMClient interface"""
    
    def test_base_client_is_abstract(self):
        """Test that BaseLLMClient cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseLLMClient()
    
    def test_concrete_client_implements_interface(self):
        """Test that concrete client implements required methods"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LiteLLMClient(provider="openai")
            
            # Check that all required methods are present
            assert hasattr(client, 'generate')
            assert hasattr(client, 'generate_stream')
            assert hasattr(client, 'generate_with_callback')
            assert hasattr(client, 'generate_with_tools')
            assert callable(getattr(client, 'generate'))
            assert callable(getattr(client, 'generate_stream'))
            assert callable(getattr(client, 'generate_with_callback'))
            assert callable(getattr(client, 'generate_with_tools'))


class TestLLMIntegration:
    """Integration tests for LLM components"""
    
    def test_factory_creates_working_client(self):
        """Test that factory creates a client that can be used"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClientFactory.create(provider="openai")
            
            # Should be able to call methods without immediate errors
            assert hasattr(client, 'generate')
            assert hasattr(client, 'generate_with_tools')
            assert client.provider == "openai"
    
    def test_multiple_providers_different_configs(self):
        """Test creating clients for multiple providers"""
        providers_models = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("cohere", "command-r-plus"),
        ]
        
        for provider, model in providers_models:
            with patch.dict(os.environ, {f'{provider.upper()}_API_KEY': 'test-key'}):
                client = LLMClientFactory.create(provider=provider, model=model)
                assert client.provider == provider
                assert client.model == model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
