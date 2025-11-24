from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class FrameworkConfig:
    """Configuration for the agent framework"""
    enable_visualization: bool = True
    log_level: str = "INFO"
    parallel_execution: bool = False
    graph_config: Dict[str, Any] = field(default_factory=lambda: {
        "recursion_limit": 400
    })