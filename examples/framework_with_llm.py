"""
Example demonstrating LLM integration with the modular agent framework.

This example shows how to use LiteLLM clients within agent modules
to create a conversational AI agent with multiple providers.
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph

from stone_agent_core.core.framework import ModularAgentFramework
from stone_agent_core.core.module import BaseAgentModule
from stone_agent_core.core.state import BaseAgentState
from stone_agent_core.llm.llm_factory import LLMClientFactory
from stone_agent_core.utils.visualization import visualize_execution

# Define our custom state that extends BaseAgentState
class ConversationalState(BaseAgentState):
    """Extended state for our conversational agent"""
    messages: List[Dict[str, str]]
    current_response: str = ""
    execution_path: List[Dict[str, Any]]
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"

# LLM-powered module for generating responses
class LLMResponseModule(BaseAgentModule[ConversationalState]):
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        super().__init__(name="llm_responder", dependencies=[])
        self.provider = provider
        self.model = model
        self.client = None
    
    def add_nodes_to_graph(self, graph: StateGraph, config: Dict[str, Any] = None):
        def generate_llm_response(state: ConversationalState) -> Dict[str, Any]:
            # Initialize LLM client if needed
            if self.client is None:
                try:
                    self.client = LLMClientFactory.create(
                        provider=self.provider,
                        model=self.model
                    )
                except Exception as e:
                    error_msg = f"Failed to create LLM client: {e}"
                    return {
                        "errors": state.get("errors", []) + [error_msg],
                        "current_module": self.module_name,
                        "execution_path": state.get('execution_path', []) + [{"name": self.module_name, "error": error_msg}]
                    }
            
            # Get the last user message
            if not state.get("messages"):
                return {
                    "errors": state.get("errors", []) + ["No messages found"],
                    "current_module": self.module_name,
                    "execution_path": state.get('execution_path', []) + [{"name": self.module_name, "error": "No messages found"}]
                }
            
            last_message = state["messages"][-1]
            if last_message.get("role") != "user":
                return {
                    "errors": state.get("errors", []) + ["Last message is not from user"],
                    "current_module": self.module_name,
                    "execution_path": state.get('execution_path', []) + [{"name": self.module_name, "error": "Last message is not from user"}]
                }
            
            # Generate response using LLM
            try:
                system_prompt = "You are a helpful assistant. Be concise and friendly."
                user_prompt = last_message["content"]
                
                response, usage = self.client.generate_single(system_prompt, user_prompt)
                
                # Add the assistant's response to messages
                new_messages = state["messages"] + [
                    {"role": "assistant", "content": response}
                ]
                
                return {
                    "messages": new_messages,
                    "current_response": response,
                    "current_module": self.module_name,
                    "execution_path": state.get('execution_path', []) + [
                        {
                            "name": self.module_name, 
                            "response": response,
                            "provider": self.provider,
                            "usage": usage
                        }
                    ]
                }
                
            except Exception as e:
                error_msg = f"LLM generation failed: {e}"
                return {
                    "errors": state.get("errors", []) + [error_msg],
                    "current_module": self.module_name,
                    "execution_path": state.get('execution_path', []) + [{"name": self.module_name, "error": error_msg}]
                }
        
        graph.add_node(self.module_name, generate_llm_response)
        return self.module_name, self.module_name

def create_conversational_agent(provider: str = "openai", model: str = "gpt-4o") -> ModularAgentFramework[ConversationalState]:
    """Create and configure a conversational agent with LLM"""
    agent = ModularAgentFramework(state_class=ConversationalState)
    
    # Register LLM module
    agent.register_module(LLMResponseModule(provider=provider, model=model))
    
    # Build the execution graph
    agent.create_main_graph()
    
    return agent

def run_conversational_example():
    """Run a conversational example with different providers"""
    # Set up environment
    load_dotenv()
    
    # Available providers and models
    providers = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-20250514"),
    ]
    
    for provider, model in providers:
        if os.getenv(f"{provider.upper()}_API_KEY"):
            print(f"\n{'='*60}")
            print(f"Running Conversational Agent with {provider.upper()}")
            print(f"{'='*60}")
            
            try:
                # Create agent with specified provider
                agent = create_conversational_agent(provider=provider, model=model)
                
                # Initialize state with a user message
                state = ConversationalState(
                    module_results={},
                    errors=[],
                    current_module="",
                    execution_complete=False,
                    messages=[{"role": "user", "content": "Hello! Tell me a short joke about programming."}],
                    execution_path=[],
                    llm_provider=provider,
                    llm_model=model
                )
                
                print(f"\n🧠 User: {state['messages'][-1]['content']}")
                
                # Run the agent
                final_state = agent.run_agent(state)
                
                # Display results
                if final_state.get("errors"):
                    print(f"\n❌ Errors: {final_state['errors']}")
                else:
                    print(f"\n🤖 Assistant: {final_state.get('current_response', 'No response')}")
                    
                    # Show execution path
                    print(f"\n📊 Execution Path:")
                    for step in final_state.get('execution_path', []):
                        if 'response' in step:
                            print(f"  ✅ {step['name']}: Generated response")
                            if 'provider' in step:
                                print(f"     Provider: {step['provider']}")
                            if 'usage' in step['usage']:
                                usage = step['usage']
                                print(f"     Usage: {usage.get('input_tokens', 0)} input, {usage.get('output_tokens', 0)} output tokens")
                        elif 'error' in step:
                            print(f"  ❌ {step['name']}: {step['error']}")
                
                print(f"\n✨ {provider.upper()} example completed successfully!")
                
            except Exception as e:
                print(f"\n❌ Error running {provider} example: {e}")
        else:
            print(f"\n=== {provider.upper()} Skipped ===")
            print(f"No {provider.upper()}_API_KEY found in environment variables.")
    
    # Check if no providers were available
    available_keys = [key for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY", "GOOGLE_API_KEY"] if os.getenv(key)]
    if not available_keys:
        print("\nNo API keys found in environment variables.")
        print("Please copy .env.example to .env and fill in your API keys.")
        print("Supported providers: OpenAI, Anthropic, Cohere, Google, Azure")

if __name__ == "__main__":
    run_conversational_example()
