import os
import openai
from typing import Iterator, List, Dict, Optional, Any
from stone_agent_core.llm.llm_base import BaseLLMClient

class OpenAILLMClient(BaseLLMClient):
    """OpenAI GPT implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")

        self.client = openai.OpenAI(api_key=self.api_key)

    def _format_messages(self, system_prompt: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages with system prompt for OpenAI"""
        formatted = [{"role": "system", "content": system_prompt}]
        formatted.extend(messages)
        return formatted

    def _convert_tools_to_openai(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert unified tool format to OpenAI format"""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        return openai_tools

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], 
                 tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        try:
            formatted_messages = self._format_messages(system_prompt, messages)
            
            create_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get('max_tokens', 2048),
                "temperature": kwargs.get('temperature', 0.1),
                "stream": False
            }
            
            if tools:
                create_params["tools"] = self._convert_tools_to_openai(tools)
            
            response = self.client.chat.completions.create(**create_params)

            full_response = response.choices[0].message.content or ""
            usage_info = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            return full_response, usage_info

        except Exception as e:
            print(f"OpenAI generation error: {str(e)}")
            raise

    def generate_stream(self, system_prompt: str, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Iterator[str]:
        try:
            formatted_messages = self._format_messages(system_prompt, messages)
            
            create_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get('max_tokens', 4096),
                "temperature": kwargs.get('temperature', 0.1),
                "stream": True
            }
            
            if tools:
                create_params["tools"] = self._convert_tools_to_openai(tools)
            
            stream = self.client.chat.completions.create(**create_params)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"OpenAI streaming error: {str(e)}")
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
            print(f"OpenAI generation error: {str(e)}")
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
            openai_tools = self._convert_tools_to_openai(tools)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=openai_tools,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.1)
            )

            message = response.choices[0].message
            
            result = {
                'content': message.content or '',
                'tool_calls': [],
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'stop_reason': response.choices[0].finish_reason
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    import json
                    result['tool_calls'].append({
                        'id': tool_call.id,
                        'name': tool_call.function.name,
                        'arguments': json.loads(tool_call.function.arguments)
                    })

            return result

        except Exception as e:
            print(f"OpenAI tool generation error: {str(e)}")
            raise