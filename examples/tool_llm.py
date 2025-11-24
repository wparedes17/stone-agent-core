"""
Example demonstrating tool usage with LLM clients.

This example shows how to use the tool calling functionality with both
OpenAI and Anthropic LLM clients.
"""
import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import the LLM factory
from stone_agent_core.llm.llm_factory import LLMClientFactory

# Define the weather tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Mock function to get weather information"""
    # In a real implementation, this would call a weather API
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "forecast": ["sunny", "windy"],
        "humidity": 60
    }
    return json.dumps(weather_data)

# Define the tools schema
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

class ToolLLMExample:
    def __init__(self):
        self.setup_environment()
        self.available_providers = self.detect_available_providers()
    
    def setup_environment(self):
        """Load environment variables"""
        load_dotenv()
    
    def detect_available_providers(self) -> List[str]:
        """Check which LLM providers have API keys available"""
        providers = []
        supported_providers = ["openai", "anthropic", "cohere", "google", "azure"]
        
        for provider in supported_providers:
            if os.getenv(f"{provider.upper()}_API_KEY"):
                providers.append(provider)
        
        return providers
    
    def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tool calls and return results"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            args = tool_call.get('arguments', {})
            
            print(f"\n🔧 Calling tool: {tool_name}")
            print(f"   Arguments: {json.dumps(args, indent=2)}")
            
            try:
                if tool_name == "get_weather":
                    result = get_weather(**args)
                else:
                    result = f"Error: Unknown tool {tool_name}"
                
                print(f"   Result: {result}")
                results.append({
                    "tool_call_id": tool_call.get('id', ''),
                    "name": tool_name,
                    "content": result
                })
            except Exception as e:
                print(f"   Error executing tool: {e}")
                results.append({
                    "tool_call_id": tool_call.get('id', ''),
                    "name": tool_name,
                    "content": f"Error: {str(e)}"
                })
        
        return results
    
    def run_tool_example(self, provider: str, model: str = None):
        """Run a tool usage example with the specified provider"""
        print(f"\n{'='*50}")
        print(f"Running {provider.upper()} Tool Example")
        print(f"{'='*50}")
        
        try:
            # Create the LLM client (LiteLLM will handle API key detection)
            client = LLMClientFactory.create(
                provider=provider,
                model=model
            )
            
            # System prompt
            system_prompt = """
            You are a helpful assistant that can get weather information.
            When the user asks about the weather, use the get_weather tool.
            The location should be in the format "City, Country" or "City, State, Country".
            """
            
            # Example conversation
            messages = [
                {"role": "user", "content": "What's the weather like in Paris?"}
            ]
            
            print("\n🧠 Processing user request:", messages[0]['content'])
            
            # First, get the model's response which may include tool calls
            response = client.generate_with_tools(
                system_prompt=system_prompt,
                messages=messages,
                tools=TOOLS
            )
            
            # Process tool calls if any
            if response.get('tool_calls'):
                print("\n🛠️  Tool calls detected!")
                
                # Process the tool calls
                tool_results = self.process_tool_calls(response['tool_calls'])
                
                # Add tool results to the messages
                for result in tool_results:
                    # Add the assistant's tool call message
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": result['tool_call_id'],
                            "type": "function",
                            "function": {
                                "name": result['name'],
                                "arguments": json.dumps(response['tool_calls'][0].get('arguments', {}))
                            }
                        }]
                    })
                    
                    # Add the tool's response message
                    messages.append({
                        "role": "tool",
                        "content": result['content'],
                        "tool_call_id": result['tool_call_id']
                    })
                
                # Get the final response with tool results
                final_response = client.generate_with_tools(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=TOOLS
                )
                
                print("\n🤖 Final response:")
                print(final_response.get('content', 'No content in response'))
            else:
                print("\n🤖 Response:")
                print(response.get('content', 'No content in response'))
            
            print("\n📊 Usage:")
            print(f"Input tokens: {response.get('usage', {}).get('input_tokens', 'N/A')}")
            print(f"Output tokens: {response.get('usage', {}).get('output_tokens', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ Error running {provider} example: {e}")
    
    def run_all_examples(self):
        """Run examples for all available providers"""
        if not self.available_providers:
            print("No API keys found. Please copy .env.example to .env and fill in your API keys.")
            print("Supported providers: OpenAI, Anthropic, Cohere, Google, Azure")
            return
        
        # Models to use for each provider
        provider_models = {
            "anthropic": "claude-sonnet-4-20250514",
        }
        
        for provider in self.available_providers:
            self.run_tool_example(provider, provider_models.get(provider))

if __name__ == "__main__":
    example = ToolLLMExample()
    example.run_all_examples()
