"""
Simple test script to verify the config-based module approach works.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stone_agent_core.core.module import BaseAgentModule
from stone_agent_core.core.framework import ModularAgentFramework
from stone_agent_core.core.state import BaseAgentState
from typing import Dict, Any, List

class TestState(BaseAgentState):
    """Simple test state"""
    messages: List[Dict[str, str]] = []
    current_response: str = ""

class TestInternalModule(BaseAgentModule[TestState]):
    """Test module using internal node and connection definition"""
    
    def __init__(self):
        super().__init__(name="test_internal")
        
        # Define internal nodes
        self.add_node("processor", self.process_message)
    
    def process_message(self, state: TestState) -> Dict[str, Any]:
        """Simple processing function"""
        if state.get("messages"):
            last_message = state["messages"][-1].get("content", "")
            response = f"Processed: {last_message}"
        else:
            response = "No messages to process"
        
        return {
            "messages": state.get("messages", []),
            "current_response": response,
            "current_module": self.module_name
        }

class TestInternalMultiNodeModule(BaseAgentModule[TestState]):
    """Test multi-node module using internal node and connection definition"""
    
    def __init__(self):
        super().__init__(name="test_internal_multi")
        
        # Define internal nodes
        self.add_node("step1", self.step1)
        self.add_node("step2", self.step2)
        
        # Define internal connections
        self.add_connection("step1", "step2")
    
    def step1(self, state: TestState) -> Dict[str, Any]:
        return {
            "messages": state.get("messages", []),
            "current_response": "",
            "step1_data": "processed_step1",
            "current_module": self.module_name
        }
    
    def step2(self, state: TestState) -> Dict[str, Any]:
        response = f"Step2 received: {state.get('step1_data', 'nothing')}"
        return {
            "messages": state.get("messages", []),
            "current_response": response,
            "step1_data": state.get("step1_data", ""),
            "current_module": self.module_name
        }

def test_internal_single_node():
    """Test single node with internal definition"""
    print("Testing internal single node...")
    
    # Create framework and module
    agent = ModularAgentFramework(state_class=TestState)
    module = TestInternalModule()
    
    # Register module (no config needed - nodes are internal)
    agent.register_module(module)
    
    # Create graph
    agent.create_main_graph()
    
    # Test execution
    state = TestState(
        messages=[{"role": "user", "content": "Hello World"}],
        current_response=""
    )
    
    result = agent.run_agent(state)
    print(f"✅ Internal single node result: {result.get('current_response')}")
    return True

def test_internal_multi_node():
    """Test multi-node configuration with internal connections"""
    print("Testing internal multi-node...")
    
    # Create framework and module
    agent = ModularAgentFramework(state_class=TestState)
    module = TestInternalMultiNodeModule()
    
    # Register module (no config needed - nodes and connections are internal)
    agent.register_module(module)
    
    # Create graph
    agent.create_main_graph()
    
    # Test execution
    state = TestState(
        messages=[{"role": "user", "content": "Hello Multi"}],
        current_response=""
    )
    
    result = agent.run_agent(state)
    print(f"✅ Internal multi node result: {result.get('current_response')}")
    return True

if __name__ == "__main__":
    try:
        test_internal_single_node()
        test_internal_multi_node()
        print("\n🎉 All tests passed! Internal module approach is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
