from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from typing import Any, Dict, Generic, List, Optional, Tuple, TYPE_CHECKING, TypeVar

from .state import BaseAgentState

if TYPE_CHECKING:
    from ..llm.llm_base import BaseLLMClient as LLMClient

StateT = TypeVar("StateT", bound=BaseAgentState)

class BaseAgentModule(ABC, Generic[StateT]):
    """Base class for creating agent modules"""
    
    def __init__(self, name: str, dependencies: List[str] = None, llm_client: Optional['BaseLLMClient'] = None):
        self._module_name = name
        self._dependencies = dependencies or []
        self._llm_client = llm_client
        self._connections: List[Tuple[str, str]] = []
        self._nodes: List[Dict[str, Any]] = []
    
    @property
    def module_name(self) -> str:
        return self._module_name
    
    @property
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    @property
    def connections(self) -> List[Tuple[str, str]]:
        """Internal connections between nodes within this module.
        
        Returns:
            List of tuples where each tuple is (from_node_name, to_node_name)
            using the relative node names (without module prefix).
        """
        return self._connections
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """Internal nodes defined within this module.
        
        Returns:
            List of dictionaries where each dict contains:
            - "name": str - relative node name (without module prefix)
            - "function": callable - the function to execute
        """
        return self._nodes
    
    def add_connection(self, from_node: str, to_node: str) -> None:
        """Add an internal connection between two nodes.
        
        Args:
            from_node: Source node name (relative, without module prefix)
            to_node: Target node name (relative, without module prefix)
        """
        self._connections.append((from_node, to_node))
    
    def add_node(self, name: str, function: Any) -> None:
        """Add an internal node to this module.
        
        Args:
            name: Node name (relative, without module prefix)
            function: The function to execute for this node
        """
        self._nodes.append({"name": name, "function": function})
    
    def add_nodes_to_graph(
        self, 
        graph: StateGraph,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Default implementation that adds nodes based on internal definition.
        
        If the module has internal nodes defined, those are used.
        Otherwise, falls back to config-based approach for backward compatibility.
        """
        # Use internal nodes if defined, otherwise fall back to config
        nodes_config = self.nodes if self.nodes else None
        
        if nodes_config is None:
            if config is None:
                raise NotImplementedError("Either define internal nodes using add_node() or provide config")
            nodes_config = config.get("nodes", [])
        
        if not nodes_config:
            raise ValueError("Module must contain at least one node")
        
        # Add nodes to graph
        for node_config in nodes_config:
            node_name = f"{self.module_name}_{node_config['name']}"
            node_function = node_config['function']
            graph.add_node(node_name, node_function)
        
        # Add internal connections (edges) defined within the module
        for connection in self.connections:
            from_node = f"{self.module_name}_{connection[0]}"
            to_node = f"{self.module_name}_{connection[1]}"
            graph.add_edge(from_node, to_node)
        
        # Return entry and exit points
        if len(nodes_config) == 1:
            # Single node case
            node_name = f"{self.module_name}_{nodes_config[0]['name']}"
            return (node_name, node_name)
        else:
            # Multiple nodes case - first is entry, last is exit
            entry_node = f"{self.module_name}_{nodes_config[0]['name']}"
            exit_node = f"{self.module_name}_{nodes_config[-1]['name']}"
            return (entry_node, exit_node)
    
    def log_loading(self) -> None:
        print(f"[INFO] Loading module: {self.module_name}")