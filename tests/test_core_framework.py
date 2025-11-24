import sys
import types
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

import pytest


class DummyVisualization:
    def draw_mermaid_png(self) -> bytes:
        return b""


class DummyCompiledGraph:
    def __init__(self, state_graph: "DummyStateGraph") -> None:
        self.state_graph = state_graph
        self.invocations: List[Dict[str, Dict]] = []

    def invoke(self, state: Dict, config: Dict) -> Dict:
        # Store copies so later mutations don't affect recorded values
        self.invocations.append({
            "state": dict(state),
            "config": dict(config),
        })
        
        # Debug output
        print(f"Debug - Graph entry point: {self.state_graph.entry_point}")
        print(f"Debug - Graph nodes: {list(self.state_graph.nodes.keys())}")
        print(f"Debug - Graph edges: {self.state_graph.edges}")
        
        # Simulate graph execution by running nodes in order
        current_state = dict(state)
        
        # Execute nodes based on the graph structure
        if self.state_graph.entry_point and self.state_graph.entry_point in self.state_graph.nodes:
            current_state = self._execute_node(self.state_graph.entry_point, current_state)
            
            # Follow edges to execute remaining nodes
            executed_nodes = {self.state_graph.entry_point}
            current_node = self.state_graph.entry_point
            
            while True:
                # Find next node via edges from current node
                next_node = None
                for edge_start, edge_end in self.state_graph.edges:
                    if edge_start == current_node:
                        if edge_end == "END":
                            print(f"Debug - Found END edge from {current_node}")
                            # Reached the end, stop execution
                            next_node = None
                            break
                        elif edge_end not in executed_nodes and edge_end in self.state_graph.nodes:
                            next_node = edge_end
                            break
                
                # If we found a next node, execute it
                if next_node is not None:
                    print(f"Debug - Executing node: {next_node}")
                    current_state = self._execute_node(next_node, current_state)
                    executed_nodes.add(next_node)
                    current_node = next_node
                else:
                    # No more nodes to execute or reached END
                    break
        
        print(f"Debug - Final state: {current_state}")
        return current_state
    
    def _execute_node(self, node_name: str, state: Dict) -> Dict:
        """Execute a single node function"""
        if node_name in self.state_graph.nodes:
            node_func = self.state_graph.nodes[node_name]
            if callable(node_func):
                return node_func(state)
        return state

    def get_graph(self, xray: int = 1) -> DummyVisualization:  # pragma: no cover - simple stub
        return DummyVisualization()


class DummyStateGraph:
    instances: List["DummyStateGraph"] = []

    def __init__(self, state_cls):
        self.state_cls = state_cls
        self.entry_point = None
        self.edges: List[tuple] = []
        self.nodes: Dict[str, Any] = {}
        self.last_compiled: DummyCompiledGraph | None = None
        DummyStateGraph.instances.append(self)

    def set_entry_point(self, node: str) -> None:
        self.entry_point = node

    def add_edge(self, start: str, end: str) -> None:
        self.edges.append((start, end))
    
    def add_node(self, name: str, func: Any) -> None:
        self.nodes[name] = func

    def compile(self) -> DummyCompiledGraph:
        self.last_compiled = DummyCompiledGraph(self)
        return self.last_compiled


# Ensure the project src directory is on sys.path for local test execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Register the dummy langgraph implementation before importing framework modules
stub_module = types.ModuleType("langgraph.graph")
stub_module.StateGraph = DummyStateGraph
stub_module.END = "END"
sys.modules["langgraph.graph"] = stub_module

from stone_agent_core.core.config import FrameworkConfig
from stone_agent_core.core.framework import ModularAgentFramework
from stone_agent_core.core.module import BaseAgentModule
from stone_agent_core.core.state import BaseAgentState


class DummyModule(BaseAgentModule[BaseAgentState]):
    def __init__(self, name: str, dependencies: List[str] | None = None):
        super().__init__(name, dependencies)

    def add_nodes_to_graph(self, graph: DummyStateGraph, config: Dict | None = None) -> tuple[str, str]:
        entry_node = f"{self.module_name}_entry"
        exit_node = f"{self.module_name}_exit"
        graph.add_node(entry_node, lambda state: state)
        graph.add_node(exit_node, lambda state: state)
        # Add edge from entry to exit within the module
        graph.add_edge(entry_node, exit_node)
        return entry_node, exit_node


class DummyModuleWithExecution(DummyModule):
    """A dummy module that actually modifies state during execution"""
    
    def __init__(self, name: str, dependencies: List[str] | None = None, result_value: str = "test_result"):
        super().__init__(name, dependencies)
        self.result_value = result_value
    
    def add_nodes_to_graph(self, graph: DummyStateGraph, config: Dict | None = None) -> tuple[str, str]:
        entry_node = f"{self.module_name}_entry"
        exit_node = f"{self.module_name}_exit"
        
        def execute_module(state: BaseAgentState) -> BaseAgentState:
            new_state = dict(state)
            new_state["module_results"][self.module_name] = self.result_value
            new_state["current_module"] = self.module_name
            return new_state
        
        def pass_through(state: BaseAgentState) -> BaseAgentState:
            return state
        
        graph.add_node(entry_node, execute_module)
        graph.add_node(exit_node, pass_through)
        # Add edge from entry to exit within the module
        graph.add_edge(entry_node, exit_node)
        return entry_node, exit_node


@pytest.fixture(autouse=True)
def reset_dummy_state_graph():
    DummyStateGraph.instances.clear()
    yield
    DummyStateGraph.instances.clear()

def test_register_module_orders_by_dependencies():
    framework = ModularAgentFramework(dict)

    module_a = DummyModule("module_a")
    module_b = DummyModule("module_b", dependencies=["module_a"])

    framework.register_module(module_a)
    framework.register_module(module_b)

    assert framework.execution_order == ["module_a", "module_b"]


def test_create_main_graph_sets_entry_point_and_edges():
    framework = ModularAgentFramework(dict)

    module_a = DummyModule("module_a")
    module_b = DummyModule("module_b", dependencies=["module_a"])

    framework.register_module(module_a)
    framework.register_module(module_b)
    framework.create_main_graph()

    compiled_graph = framework.graph
    assert isinstance(compiled_graph, DummyCompiledGraph)

    graph_instance = DummyStateGraph.instances[-1]
    assert graph_instance.entry_point == "module_a_entry"
    # Edges include internal module edges (entry->exit) and inter-module edges
    assert graph_instance.edges == [
        ("module_a_entry", "module_a_exit"),  # Internal edge for module_a
        ("module_b_entry", "module_b_exit"),  # Internal edge for module_b
        ("module_a_exit", "module_b_entry"),  # Inter-module edge
        ("module_b_exit", "END"),              # Final edge to END
    ]


def test_run_agent_invokes_graph_and_marks_completion():
    framework = ModularAgentFramework(dict)

    module = DummyModule("module_a")
    framework.register_module(module)

    initial_state = {
        "module_results": {},
        "errors": [],
        "current_module": "",
        "execution_complete": False,
    }

    framework.create_main_graph()
    result_state = framework.run_agent(initial_state)

    assert result_state["execution_complete"] is True

    graph_instance = DummyStateGraph.instances[-1]
    assert graph_instance.last_compiled is not None
    invocation = graph_instance.last_compiled.invocations[0]
    assert invocation["config"] == FrameworkConfig().graph_config


# Test Circular Dependency Detection
def test_circular_dependency_detection():
    framework = ModularAgentFramework(dict)

    module_a = DummyModule("module_a", dependencies=["module_b"])
    module_b = DummyModule("module_b", dependencies=["module_a"])

    framework.register_module(module_a)
    framework.register_module(module_b)

    with pytest.raises(ValueError, match="Circular dependency detected in modules"):
        framework._update_execution_order()


# Test Unknown Dependency Detection
def test_unknown_dependency_detection():
    framework = ModularAgentFramework(dict)

    module_a = DummyModule("module_a", dependencies=["nonexistent_module"])

    framework.register_module(module_a)

    with pytest.raises(ValueError, match="Module 'module_a' depends on unknown module 'nonexistent_module'"):
        framework._update_execution_order()


# Test Empty Framework
def test_empty_framework():
    framework = ModularAgentFramework(dict)
    
    # Should not raise error with no modules
    framework.create_main_graph()
    
    # Should handle empty execution gracefully
    initial_state = {
        "module_results": {},
        "errors": [],
        "current_module": "",
        "execution_complete": False,
    }
    
    result_state = framework.run_agent(initial_state)
    assert result_state["execution_complete"] is True
    assert result_state["module_results"] == {}


# Test Module State Execution
def test_module_state_execution():
    framework = ModularAgentFramework(dict)

    module_a = DummyModuleWithExecution("module_a", result_value="result_a")
    module_b = DummyModuleWithExecution("module_b", dependencies=["module_a"], result_value="result_b")

    framework.register_module(module_a)
    framework.register_module(module_b)
    framework.create_main_graph()

    initial_state = {
        "module_results": {},
        "errors": [],
        "current_module": "",
        "execution_complete": False,
    }

    result_state = framework.run_agent(initial_state)
    
    # Debug output
    print(f"Debug - result_state: {result_state}")
    print(f"Debug - module_results: {result_state.get('module_results', 'NOT FOUND')}")
    
    assert result_state["execution_complete"] is True
    assert result_state["module_results"]["module_a"] == "result_a"
    assert result_state["module_results"]["module_b"] == "result_b"


# Test Module Properties
def test_module_properties():
    module = DummyModule("test_module", ["dep1", "dep2"])
    
    assert module.module_name == "test_module"
    assert module.dependencies == ["dep1", "dep2"]
    assert module.module_name == "test_module"  # Test property again


# Test Framework Configuration
def test_framework_configuration():
    config = FrameworkConfig()
    framework = ModularAgentFramework(dict, config)
    
    assert framework.config == config
    assert framework.state_class == dict


# Test Multiple Independent Modules
def test_multiple_independent_modules():
    framework = ModularAgentFramework(dict)

    module_a = DummyModuleWithExecution("module_a", result_value="result_a")
    module_b = DummyModuleWithExecution("module_b", result_value="result_b")
    module_c = DummyModuleWithExecution("module_c", result_value="result_c")

    framework.register_module(module_a)
    framework.register_module(module_b)
    framework.register_module(module_c)
    
    # All modules should be included in execution order
    assert len(framework.execution_order) == 3
    assert "module_a" in framework.execution_order
    assert "module_b" in framework.execution_order
    assert "module_c" in framework.execution_order


# Test Error Handling in Module Registration
def test_module_registration_logging():
    framework = ModularAgentFramework(dict)
    module = DummyModule("test_module")
    
    # Should log loading when registering
    with patch('builtins.print') as mock_print:
        framework.register_module(module)
        mock_print.assert_called_with("[INFO] Loading module: test_module")


# Test Complex Dependency Chain
def test_complex_dependency_chain():
    framework = ModularAgentFramework(dict)

    # Create a complex dependency chain: A -> B -> C -> D
    module_d = DummyModule("module_d")
    module_c = DummyModule("module_c", dependencies=["module_d"])
    module_b = DummyModule("module_b", dependencies=["module_c"])
    module_a = DummyModule("module_a", dependencies=["module_b"])

    # Register in reverse order
    framework.register_module(module_a)
    framework.register_module(module_b)
    framework.register_module(module_c)
    framework.register_module(module_d)

    assert framework.execution_order == ["module_d", "module_c", "module_b", "module_a"]
