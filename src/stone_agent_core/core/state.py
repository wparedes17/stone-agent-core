
from typing import TypedDict, Dict, List, Any

class BaseAgentState(TypedDict):
    """Minimal state all agents must have"""
    module_results: Dict[str, Any]
    errors: List[str]
    current_module: str
    execution_complete: bool