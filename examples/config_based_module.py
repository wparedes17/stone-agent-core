"""
Example demonstrating the config-based approach for adding nodes to modules.

This example shows how to use the new config parameter in add_nodes_to_graph
to define module structure declaratively instead of implementing the method.
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

# Example 1: Single node module using internal node definition
class InternalSingleNodeModule(BaseAgentModule[ConversationalState]):
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        super().__init__(name="internal_single_responder", dependencies=[])
        self.provider = provider
        self.model = model
        self.client = None
        
        # Define internal node
        self.add_node("responder", self.generate_response)
    
    def generate_response(self, state: ConversationalState) -> Dict[str, Any]:
        """Generate LLM response - this will be added via internal node definition"""
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

# Example 2: Multi-node module using internal node and connection definition
class InternalMultiNodeModule(BaseAgentModule[ConversationalState]):
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        super().__init__(name="internal_multi_responder", dependencies=[])
        self.provider = provider
        self.model = model
        self.client = None
        
        # Define internal nodes
        self.add_node("prepare_input", self.prepare_input)
        self.add_node("generate_response", self.generate_response)
        self.add_node("finalize_output", self.finalize_output)
        
        # Define internal connections within the module
        self.add_connection("prepare_input", "generate_response")
        self.add_connection("generate_response", "finalize_output")
    
    def prepare_input(self, state: ConversationalState) -> Dict[str, Any]:
        """Prepare input for LLM generation"""
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
        
        return {
            "messages": state["messages"],
            "prepared_input": last_message["content"],
            "current_module": self.module_name,
            "execution_path": state.get('execution_path', []) + [{"name": self.module_name, "step": "prepare_input"}]
        }
    
    def generate_response(self, state: ConversationalState) -> Dict[str, Any]:
        """Generate LLM response"""
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
        
        # Generate response using LLM
        try:
            system_prompt = "You are a helpful assistant. Be concise and friendly."
            user_prompt = state.get("prepared_input", "Hello!")
            
            response, usage = self.client.generate_single(system_prompt, user_prompt)
            
            return {
                "messages": state["messages"],
                "prepared_input": state.get("prepared_input", ""),
                "generated_response": response,
                "current_module": self.module_name,
                "execution_path": state.get('execution_path', []) + [
                    {
                        "name": self.module_name, 
                        "step": "generate_response",
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
    
    def finalize_output(self, state: ConversationalState) -> Dict[str, Any]:
        """Finalize the output by adding response to messages"""
        generated_response = state.get("generated_response", "")
        
        # Add the assistant's response to messages
        new_messages = state["messages"] + [
            {"role": "assistant", "content": generated_response}
        ]
        
        return {
            "messages": new_messages,
            "current_response": generated_response,
            "current_module": self.module_name,
            "execution_path": state.get('execution_path', []) + [{"name": self.module_name, "step": "finalize_output"}]
        }

def create_internal_based_agent(provider: str = "openai", model: str = "gpt-4o", use_multi_node: bool = False) -> ModularAgentFramework[ConversationalState]:
    """Create and configure an agent with internal node and connection definition"""
    agent = ModularAgentFramework(state_class=ConversationalState)
    
    if use_multi_node:
        # Register multi-node module (no config needed - nodes and connections are internal)
        multi_node_module = InternalMultiNodeModule(provider=provider, model=model)
        agent.register_module(multi_node_module)
    else:
        # Register single-node module (no config needed - node is internal)
        single_node_module = InternalSingleNodeModule(provider=provider, model=model)
        agent.register_module(single_node_module)
    
    # Build the execution graph
    agent.create_main_graph()
    
    return agent

def run_internal_based_example():
    """Run internal-based examples"""
    # Set up environment
    load_dotenv()
    
    # Test with OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        print(f"\n{'='*60}")
        print("Running Internal Single Node Agent")
        print(f"{'='*60}")
        
        try:
            # Create single-node agent
            agent = create_internal_based_agent(use_multi_node=False)
            
            # Initialize state with a user message
            state = ConversationalState(
                module_results={},
                errors=[],
                current_module="",
                execution_complete=False,
                messages=[{"role": "user", "content": "Hello! Tell me a short joke about programming."}],
                execution_path=[],
                llm_provider="openai",
                llm_model="gpt-4o"
            )
            
            print(f"\n🧠 User: {state['messages'][-1]['content']}")
            
            # Run the agent
            final_state = agent.run_agent(state)
            
            # Display results
            if final_state.get("errors"):
                print(f"\n❌ Errors: {final_state['errors']}")
            else:
                print(f"\n🤖 Assistant: {final_state.get('current_response', 'No response')}")
            
            print(f"\n✨ Single node internal example completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error running single node example: {e}")
        
        print(f"\n{'='*60}")
        print("Running Internal Multi Node Agent")
        print(f"{'='*60}")
        
        try:
            # Create multi-node agent
            agent = create_internal_based_agent(use_multi_node=True)
            
            # Initialize state with a user message
            state = ConversationalState(
                module_results={},
                errors=[],
                current_module="",
                execution_complete=False,
                messages=[{"role": "user", "content": "Hello! Tell me a short joke about programming."}],
                execution_path=[],
                llm_provider="openai",
                llm_model="gpt-4o"
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
                    if 'step' in step:
                        print(f"  ✅ {step['name']}: {step['step']}")
                    elif 'response' in step:
                        print(f"  ✅ {step['name']}: Generated response")
            
            print(f"\n✨ Multi node internal example completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error running multi node example: {e}")
    else:
        print("\nNo OPENAI_API_KEY found in environment variables.")
        print("Please copy .env.example to .env and fill in your API keys.")

if __name__ == "__main__":
    run_internal_based_example()
