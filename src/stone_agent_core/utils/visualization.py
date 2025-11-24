from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_VIS_DEPS = True
except ImportError:
    HAS_VIS_DEPS = False

class NodeType(Enum):
    MODULE = "module"
    START = "start"
    END = "end"

@dataclass
class NodeData:
    """Data structure for graph node visualization."""
    node_id: str
    label: str
    node_type: NodeType = NodeType.MODULE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeData:
    """Data structure for graph edge visualization."""
    source: str
    target: str
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class GraphVisualizer:
    """Utility class for visualizing agent execution graphs."""
    
    def __init__(self):
        if not HAS_VIS_DEPS:
            raise ImportError(
                "Visualization dependencies not found. Install with: "
                "pip install matplotlib networkx"
            )
        self._graph = nx.DiGraph()
        self._execution_times: Dict[str, float] = {}
        self._start_time: Optional[float] = None
    
    def add_node(self, node_data: NodeData) -> None:
        """Add a node to the visualization."""
        self._graph.add_node(
            node_data.node_id,
            label=node_data.label,
            node_type=node_data.node_type.value,
            **node_data.metadata
        )
    
    def add_edge(self, edge_data: EdgeData) -> None:
        """Add an edge to the visualization."""
        self._graph.add_edge(
            edge_data.source,
            edge_data.target,
            label=edge_data.label,
            **edge_data.metadata
        )
    
    def start_timer(self, node_id: str) -> None:
        """Start timing a node's execution."""
        self._start_time = time.time()
    
    def stop_timer(self, node_id: str) -> None:
        """Stop timing a node's execution and record the duration."""
        if self._start_time is not None:
            duration = time.time() - self._start_time
            self._execution_times[node_id] = duration
            self._start_time = None
    
    def render(self, output_path: Optional[str] = None) -> None:
        """
        Render the graph visualization.
        
        Args:
            output_path: Optional path to save the visualization. If None, displays the plot.
        """
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self._graph)
        
        # Draw nodes
        node_colors = []
        for node in self._graph.nodes():
            node_type = self._graph.nodes[node].get('node_type', NodeType.MODULE.value)
            if node_type == NodeType.START.value:
                node_colors.append('lightgreen')
            elif node_type == NodeType.END.value:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(
            self._graph, pos, 
            node_color=node_colors,
            node_size=2000,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self._graph, pos,
            width=1.5,
            alpha=0.7,
            edge_color='gray',
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        node_labels = nx.get_node_attributes(self._graph, 'label')
        nx.draw_networkx_labels(self._graph, pos, labels=node_labels)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self._graph, 'label')
        nx.draw_networkx_edge_labels(
            self._graph, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_execution_trace(self, output_path: str) -> None:
        """Save execution timing information to a JSON file."""
        trace_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'execution_times': self._execution_times,
            'total_duration': sum(self._execution_times.values()) if self._execution_times else 0
        }
        
        with open(output_path, 'w') as f:
            json.dump(trace_data, f, indent=2)

def visualize_execution(execution_path: List[Dict[str, Any]], output_path: Optional[str] = None) -> None:
    """
    Visualize an execution path through the agent's modules.
    
    Args:
        execution_path: List of execution steps, each containing module info
        output_path: Optional path to save the visualization
    """
    if not HAS_VIS_DEPS:
        print("Visualization dependencies not available. Install with: pip install matplotlib networkx")
        return
    
    viz = GraphVisualizer()
    
    # Add start node
    viz.add_node(NodeData(
        node_id="start",
        label="Start",
        node_type=NodeType.START
    ))
    
    # Add module nodes and edges
    for i, step in enumerate(execution_path):
        module_id = step.get('module_id', f'module_{i}')
        module_name = step.get('name', f'Module {i}')
        
        viz.add_node(NodeData(
            node_id=module_id,
            label=module_name,
            metadata=step.get('metadata', {})
        ))
        
        # Connect to previous node
        if i == 0:
            viz.add_edge(EdgeData(
                source="start",
                target=module_id,
                label=""
            ))
        else:
            prev_module_id = execution_path[i-1].get('module_id', f'module_{i-1}')
            viz.add_edge(EdgeData(
                source=prev_module_id,
                target=module_id,
                label=f"{execution_path[i-1].get('result', '')}"
            ))
    
    # Add end node
    viz.add_node(NodeData(
        node_id="end",
        label="End",
        node_type=NodeType.END
    ))
    
    # Connect last module to end
    if execution_path:
        last_module_id = execution_path[-1].get('module_id', f'module_{len(execution_path)-1}')
        viz.add_edge(EdgeData(
            source=last_module_id,
            target="end",
            label=execution_path[-1].get('result', '')
        ))
    
    # Render the visualization
    viz.render(output_path)