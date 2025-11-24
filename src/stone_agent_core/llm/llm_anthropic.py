import os
import anthropic
from typing import Iterator, List, Dict, Optional, Any
from stone_agent_core.llm.llm_base import BaseLLMClient

class AnthropicLLMClient(BaseLLMClient):
    """Anthropic Claude implementation"""
    
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], 
                 tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        try:
            create_params = {
                "model": self.model,
                "max_tokens": kwargs.get('max_tokens', 2048),
                "temperature": kwargs.get('temperature', 0.1),
                "system": system_prompt,
                "messages": messages,
                "stream": True
            }
            
            if tools:
                create_params["tools"] = tools
            
            stream = self.client.messages.create(**create_params)

            full_response = ""
            usage_info = {}
            
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        full_response += chunk.delta.text
                elif chunk.type == "message_stop":
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                        usage = chunk.message.usage
                        usage_info = {
                            'input_tokens': usage.input_tokens,
                            'output_tokens': usage.output_tokens,
                            'total_tokens': usage.input_tokens + usage.output_tokens
                        }

            return full_response, usage_info

        except Exception as e:
            print(f"Anthropic generation error: {str(e)}")
            raise

    def generate_stream(self, system_prompt: str, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Iterator[str]:
        try:
            create_params = {
                "model": self.model,
                "max_tokens": kwargs.get('max_tokens', 32000),
                "temperature": kwargs.get('temperature', 0.1),
                "system": system_prompt,
                "messages": messages,
                "stream": True
            }
            
            if tools:
                create_params["tools"] = tools
            
            stream = self.client.messages.create(**create_params)

            for chunk in stream:
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text

        except Exception as e:
            print(f"Anthropic streaming error: {str(e)}")
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
            print(f"Anthropic generation error: {str(e)}")
            raise

    def generate_with_tools(self, system_prompt: str, messages: List[Dict[str, str]], 
                           tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate response that may include tool calls
        
        Returns:
            Dict with 'content' (text), 'tool_calls' (list), and 'usage' (dict)
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.1),
                system=system_prompt,
                messages=messages,
                tools=tools
            )

            result = {
                'content': '',
                'tool_calls': [],
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                'stop_reason': response.stop_reason
            }

            for block in response.content:
                if block.type == 'text':
                    result['content'] += block.text
                elif block.type == 'tool_use':
                    result['tool_calls'].append({
                        'id': block.id,
                        'name': block.name,
                        'arguments': block.input
                    })

            return result

        except Exception as e:
            print(f"Anthropic tool generation error: {str(e)}")
            raise