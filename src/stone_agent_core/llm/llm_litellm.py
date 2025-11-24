import os
import litellm
from typing import Iterator, List, Dict, Optional, Any
from stone_agent_core.llm.llm_base import BaseLLMClient

class LiteLLMClient(BaseLLMClient):
    """LiteLLM implementation supporting multiple providers"""
    
    def __init__(self, provider: str = "openai", api_key: str = None, model: str = None, **kwargs):
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self.api_key = api_key or self._get_api_key()
        
        # Validate API key
        if not self.api_key:
            env_key = self._get_api_key_env()
            raise ValueError(f"API key not found for provider '{self.provider}'. Set {env_key} environment variable or pass api_key parameter.")
        
        # Set environment variables for litellm
        if self.api_key:
            os.environ[self._get_api_key_env()] = self.api_key
            
        # Configure litellm
        litellm.set_verbose = kwargs.get('verbose', False)

    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            'openai': 'gpt-4o',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'cohere': 'command-r-plus',
            'google': 'gemini-1.5-pro',
            'azure': 'gpt-4o',
        }
        return defaults.get(self.provider, 'gpt-4o')

    def _get_api_key(self) -> Optional[str]:
        """Get API key for provider"""
        env_key = self._get_api_key_env()
        return os.getenv(env_key)

    def _get_api_key_env(self) -> str:
        """Get environment variable name for API key"""
        api_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'azure': 'AZURE_API_KEY',
        }
        return api_keys.get(self.provider, 'OPENAI_API_KEY')

    def _format_model(self) -> str:
        """Format model name for litellm"""
        if self.provider == 'openai':
            return f"openai/{self.model}"
        elif self.provider == 'anthropic':
            return f"anthropic/{self.model}"
        elif self.provider == 'cohere':
            return f"cohere/{self.model}"
        elif self.provider == 'google':
            return f"google/{self.model}"
        elif self.provider == 'azure':
            return f"azure/{self.model}"
        else:
            return self.model

    def _format_messages(self, system_prompt: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages with system prompt"""
        formatted = [{"role": "system", "content": system_prompt}]
        formatted.extend(messages)
        return formatted

    def _convert_tools_to_litellm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert unified tool format to litellm format"""
        litellm_tools = []
        for tool in tools:
            litellm_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        return litellm_tools

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], 
                 tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        try:
            formatted_messages = self._format_messages(system_prompt, messages)
            model_name = self._format_model()
            
            create_params = {
                "model": model_name,
                "messages": formatted_messages,
                "max_tokens": kwargs.get('max_tokens', 2048),
                "temperature": kwargs.get('temperature', 0.1),
            }
            
            if tools:
                create_params["tools"] = self._convert_tools_to_litellm(tools)
            
            response = litellm.completion(**create_params)

            full_response = response.choices[0].message.content or ""
            usage_info = {}
            
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }

            return full_response, usage_info

        except Exception as e:
            print(f"LiteLLM generation error: {str(e)}")
            raise

    def generate_stream(self, system_prompt: str, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Iterator[str]:
        try:
            formatted_messages = self._format_messages(system_prompt, messages)
            model_name = self._format_model()
            
            create_params = {
                "model": model_name,
                "messages": formatted_messages,
                "max_tokens": kwargs.get('max_tokens', 4096),
                "temperature": kwargs.get('temperature', 0.1),
                "stream": True
            }
            
            if tools:
                create_params["tools"] = self._convert_tools_to_litellm(tools)
            
            stream = litellm.completion(**create_params)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"LiteLLM streaming error: {str(e)}")
            raise

    def generate_with_callback(self, system_prompt: str, messages: List[Dict[str, str]],
                               callback: callable, tools: Optional[List[Dict[str, Any]]] = None, 
                               **kwargs) -> str:
        full_response = ""
        try:
            for chunk in self.generate_stream(system_prompt, messages, tools=tools, **kwargs):
                full_response += chunk
                callback(chunk)
            return full_response
        except Exception as e:
            print(f"LiteLLM generation error: {str(e)}")
            raise

    def generate_with_tools(self, system_prompt: str, messages: List[Dict[str, str]], 
                           tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate response that may include tool calls
        
        Returns:
            Dict with 'content' (text), 'tool_calls' (list), and 'usage' (dict)
        """
        try:
            formatted_messages = self._format_messages(system_prompt, messages)
            litellm_tools = self._convert_tools_to_litellm(tools)
            model_name = self._format_model()
            
            response = litellm.completion(
                model=model_name,
                messages=formatted_messages,
                tools=litellm_tools,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.1)
            )

            message = response.choices[0].message
            
            result = {
                'content': message.content or '',
                'tool_calls': [],
                'usage': {},
                'stop_reason': response.choices[0].finish_reason
            }
            
            if hasattr(response, 'usage') and response.usage:
                result['usage'] = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }

            if hasattr(message, 'tool_calls') and message.tool_calls:
                import json
                for tool_call in message.tool_calls:
                    result['tool_calls'].append({
                        'id': tool_call.id,
                        'name': tool_call.function.name,
                        'arguments': json.loads(tool_call.function.arguments)
                    })

            return result

        except Exception as e:
            print(f"LiteLLM tool generation error: {str(e)}")
            raise
