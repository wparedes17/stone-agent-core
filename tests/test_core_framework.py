import sys
import types
from pathlib import Path
from typing import Dict, List

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
        return dict(state)

    def get_graph(self, xray: int = 1) -> DummyVisualization:  # pragma: no cover - simple stub
        return DummyVisualization()


class DummyStateGraph:
    instances: List["DummyStateGraph"] = []

    def __init__(self, state_cls):
        self.state_cls = state_cls
        self.entry_point = None
        self.edges: List[tuple] = []
        self.last_compiled: DummyCompiledGraph | None = None
        DummyStateGraph.instances.append(self)

    def set_entry_point(self, node: str) -> None:
        self.entry_point = node

    def add_edge(self, start: str, end: str) -> None:
        self.edges.append((start, end))

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
    assert graph_instance.edges == [
        ("module_a_exit", "module_b_entry"),
        ("module_b_exit", "END"),
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
