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
    
    @property
    def module_name(self) -> str:
        return self._module_name
    
    @property
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    @abstractmethod
    def add_nodes_to_graph(
        self, 
        graph: StateGraph,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Implement this to add your module's nodes"""
        pass
    
    def log_loading(self) -> None:
        print(f"[INFO] Loading module: {self.module_name}")