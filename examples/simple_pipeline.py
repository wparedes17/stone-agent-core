from typing import Dict, Any
from langgraph.graph import StateGraph

from stone_agent_core.core.framework import ModularAgentFramework
from stone_agent_core.core.module import BaseAgentModule
from stone_agent_core.core.state import BaseAgentState
from stone_agent_core.utils.visualization import visualize_execution

# Define our custom state that extends BaseAgentState
class ChatState(BaseAgentState):
    """Extended state for our chat application"""
    messages: list[Dict[str, str]]
    current_response: str = "",
    execution_path: list[Dict[str, Any]]


# First module: Generates a greeting
class GreetingModule(BaseAgentModule[ChatState]):    
    def __init__(self):
        super().__init__(name="greeter", dependencies=[])
    
    def add_nodes_to_graph(self, graph: StateGraph, config: Dict[str, Any] = None):
        def generate_greeting(state: ChatState) -> Dict[str, Any]:
            return {
                "messages": [{"role": "assistant", "content": "Hello! How can I help you today?"}],
                "current_response": "Greeting generated",
                "current_module": self.module_name,
                "execution_path": state['execution_path'] + [{"name": self.module_name, "response": "Greeting generated"}]
            }
            
        graph.add_node(self.module_name, generate_greeting)
        return self.module_name, self.module_name

# Second module: Processes user input and generates a response
class ResponseModule(BaseAgentModule[ChatState]):
    def __init__(self):
        super().__init__(name="responder", dependencies=["greeter"])
    
    def add_nodes_to_graph(self, graph: StateGraph, config: Dict[str, Any] = None):
        def generate_response(state: ChatState) -> Dict[str, Any]:
            # Simple echo response - in a real app, this would use an LLM
            last_message = state["messages"][-1]["content"]
            response = f"You said: {last_message}"
            
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": response}],
                "current_response": response,
                "current_module": self.module_name,
                "execution_path": state['execution_path'] + [{"name": self.module_name, "response": response}]
            }

        graph.add_node(self.module_name, generate_response)
        return self.module_name, self.module_name

def create_chat_agent() -> ModularAgentFramework[ChatState]:
    """Create and configure a chat agent with our modules"""
    agent = ModularAgentFramework(state_class=ChatState)
    
    # Register modules
    agent.register_module(GreetingModule())
    agent.register_module(ResponseModule())
    
    # Build the execution graph
    agent.create_main_graph()
    
    return agent

def run_chat_example():
    """Run a simple chat example"""
    print("Creating chat agent...")
    agent = create_chat_agent()
    
    # Initialize state
    state = ChatState(
        module_results={},
        errors=[],
        current_module="",
        execution_complete=False,
        messages=[{"role": "user", "content": "Hi there!"}],
        execution_path=[],
    )
    
    print("\nRunning chat pipeline...")
    
    final_state = agent.run_agent(state)

    print("\nVisualizing execution...")
    print(final_state.get('execution_path'))
    visualize_execution(
        final_state.get('execution_path'),
        output_path="execution_flow.png"  
    )
    
    print("\nChat completed!")
    return final_state

if __name__ == "__main__":
    final_state = run_chat_example()
    print(final_state.get('messages')[-1])
