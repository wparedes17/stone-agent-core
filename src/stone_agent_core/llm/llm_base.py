from typing import Iterator, List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, system_prompt: str, messages: List[Dict[str, str]], 
                 tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Tuple[str, Dict]:
        """Generate a complete response"""
        pass
    
    @abstractmethod
    def generate_stream(self, system_prompt: str, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Iterator[str]:
        """Generate a streaming response"""
        pass
    
    @abstractmethod
    def generate_with_callback(self, system_prompt: str, messages: List[Dict[str, str]], 
                               callback: callable, tools: Optional[List[Dict[str, Any]]] = None, 
                               **kwargs) -> str:
        """Generate response with callback for each chunk"""
        pass
    
    @abstractmethod
    def generate_with_tools(self, system_prompt: str, messages: List[Dict[str, str]], 
                           tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate response that may include tool calls"""
        pass
    
    def generate_single(self, system_prompt: str, user_prompt: str, 
                       tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Convenience method for single-turn conversations"""
        messages = [{"role": "user", "content": user_prompt}]
        return self.generate(system_prompt, messages, tools=tools, **kwargs)
    
    def generate_stream_single(self, system_prompt: str, user_prompt: str, 
                              tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Iterator[str]:
        """Convenience method for single-turn streaming"""
        messages = [{"role": "user", "content": user_prompt}]
        return self.generate_stream(system_prompt, messages, tools=tools, **kwargs)
