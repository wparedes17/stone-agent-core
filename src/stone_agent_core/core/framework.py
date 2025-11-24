from langgraph.graph import StateGraph, END
from typing import Dict, List, Generic, TypeVar, Optional

# Core state that all subgraphs share
from .module import BaseAgentModule
from .config import FrameworkConfig
from .state import BaseAgentState

StateT = TypeVar('StateT', bound=BaseAgentState)

class ModularAgentFramework(Generic[StateT]):
    """Core framework that orchestrates agent modules"""
    
    def __init__(self, state_class: type[StateT], config: Optional[FrameworkConfig] = None):
        self.state_class = state_class
        self.modules: Dict[str, BaseAgentModule[StateT]] = {}
        self.execution_order: List[str] = []
        self.config = config or FrameworkConfig()
        self._graph = None 
    
    @property
    def graph(self) -> StateGraph:
        return self._graph
    
    def register_module(self, module: type[StateT]):
        """Register a migration module"""

        module.log_loading()

        self.modules[module.module_name] = module
        self._update_execution_order()

    def _update_execution_order(self):
        """Update execution order based on dependencies"""

        ordered = []
        remaining = set(self.modules.keys())

        while remaining:
            ready = []
            for module_name in remaining:
                module = self.modules[module_name]
                if all(dep in ordered for dep in module.dependencies):
                    ready.append(module_name)

            if not ready:
                raise ValueError("Circular dependency detected in modules")

            # Add ready modules to order
            for module_name in ready:
                ordered.append(module_name)
                remaining.remove(module_name)

        self.execution_order = ordered

    def create_main_graph(self):
        """Create a single, flat migration graph with all module nodes."""

        main_graph = StateGraph(self.state_class)

        module_endpoints = {}

        for module_name in self.execution_order:
            module = self.modules[module_name]
            entry_node, exit_node = module.add_nodes_to_graph(main_graph)
            module_endpoints[module_name] = {"entry": entry_node, "exit": exit_node}

        if self.execution_order:
            first_module_name = self.execution_order[0]
            main_graph.set_entry_point(module_endpoints[first_module_name]["entry"])

            for i in range(len(self.execution_order) - 1):
                current_module_name = self.execution_order[i]
                next_module_name = self.execution_order[i + 1]

                exit_of_current = module_endpoints[current_module_name]["exit"]
                entry_of_next = module_endpoints[next_module_name]["entry"]
                main_graph.add_edge(exit_of_current, entry_of_next)

            last_module_name = self.execution_order[-1]
            main_graph.add_edge(module_endpoints[last_module_name]["exit"], END)

        self._graph = main_graph.compile()

    def run_agent(self, state: StateT) -> StateT:
        """Run the complete execution with all registered modules
        
        Args:
            state: Initial state conforming to the configured state class
            
        Returns:
            Final state after all modules have executed
        """

        print(f"Starting execution with modules: {self.execution_order}")
        
        invoke_config = self.config.graph_config.copy()
        
        final_state = self._graph.invoke(state, invoke_config)
        
        final_state["execution_complete"] = True
        
        return final_state

    def visualize_graph(self, xray: int = 1):
        """
        Visualize the migration graph using mermaid diagram

        Args:
            xray (int): Level of detail for the graph visualization (0-3)
                       0: Basic structure
                       1: Standard detail (default)
                       2: More detail
                       3: Maximum detail

        Returns:
            IPython.display.Image: The rendered graph image
        """

        try:
            from IPython.display import Image, display

            # Create the main graph
            main_graph = self.create_main_graph()

            # Generate and display the mermaid diagram
            graph_image = main_graph.get_graph(xray=xray).draw_mermaid_png()

            # Display the graph
            display(Image(graph_image))
            # return Image(graph_image)

        except ImportError as e:
            print("IPython not available. Install with: pip install ipython")
            return None
        except Exception as e:
            print(f"Error generating graph visualization: {e}")
            return None
