from stone_agent_core.llm.llm_base import BaseLLMClient
from stone_agent_core.llm.llm_anthropic import AnthropicLLMClient
from stone_agent_core.llm.llm_openai import OpenAILLMClient

class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients = {
        'anthropic': AnthropicLLMClient,
        'openai': OpenAILLMClient,
    }
    
    @classmethod
    def create(cls, provider: str, api_key: str = None, model: str = None, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client
        
        Args:
            provider: 'anthropic' or 'openai'
            api_key: API key (optional, will use environment variable if not provided)
            model: Model name (optional, will use default for provider)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            BaseLLMClient instance
        
        Example:
            client = LLMClientFactory.create('anthropic')
            client = LLMClientFactory.create('openai', model='gpt-4o-mini')
        """
        provider = provider.lower()
        
        if provider not in cls._clients:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._clients.keys())}")
        
        client_class = cls._clients[provider]
        
        # Build kwargs for client initialization
        init_kwargs = {'api_key': api_key}
        if model:
            init_kwargs['model'] = model
        init_kwargs.update(kwargs)
        
        return client_class(**init_kwargs)
    
    @classmethod
    def register_client(cls, provider: str, client_class: type):
        """Register a new LLM client type"""
        cls._clients[provider] = client_class
