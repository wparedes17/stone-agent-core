from stone_agent_core.llm.llm_base import BaseLLMClient
from stone_agent_core.llm.llm_litellm import LiteLLMClient

class LLMClientFactory:
    """Factory for creating LLM clients using LiteLLM"""
    
    @classmethod
    def create(cls, provider: str, api_key: str = None, model: str = None, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client using LiteLLM
        
        Args:
            provider: 'anthropic', 'openai', 'cohere', 'google', 'azure', etc.
            api_key: API key (optional, will use environment variable if not provided)
            model: Model name (optional, will use default for provider)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            BaseLLMClient instance
        
        Example:
            client = LLMClientFactory.create('anthropic')
            client = LLMClientFactory.create('openai', model='gpt-4o-mini')
            client = LLMClientFactory.create('cohere', model='command-r-plus')
        """
        return LiteLLMClient(provider=provider, api_key=api_key, model=model, **kwargs)
    
    @classmethod
    def register_client(cls, provider: str, client_class: type):
        """Register a new LLM client type (deprecated with LiteLLM)"""
        raise NotImplementedError("Custom client registration not supported with LiteLLM. Use provider parameter directly.")
