"""
Simple example demonstrating the usage of LLM clients.

This example shows how to use the LLM factory to create and use different LLM providers.
It includes examples for both synchronous and streaming generation.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Import the LLM factory
from stone_agent_core.llm.llm_factory import LLMClientFactory

def setup_environment():
    """Load environment variables from .env file if it exists"""
    load_dotenv()

def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment variables"""
    return os.getenv(f"{provider.upper()}_API_KEY")

def run_basic_example(provider: str, model: str = None):
    """Run a basic example with the specified provider"""
    print(f"\n=== Running {provider.upper()} Example ===")
    
    # Create the LLM client (LiteLLM will handle API key detection)
    try:
        client = LLMClientFactory.create(
            provider=provider,
            model=model
        )
    except Exception as e:
        print(f"Error creating {provider} client: {e}")
        print(f"Please ensure {provider.upper()}_API_KEY is set in your environment or .env file")
        return
    
    # System prompt and user message
    system_prompt = "You are a helpful assistant."
    user_prompt = "Tell me a short joke about programming."
    
    # Generate a simple response
    print("\n--- Basic Generation ---")
    try:
        response, _ = client.generate_single(system_prompt, user_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error during generation: {e}")
    
    # Generate a streaming response
    print("\n--- Streaming Generation ---")
    try:
        print("Response: ", end="", flush=True)
        for chunk in client.generate_stream_single(system_prompt, user_prompt):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nError during streaming: {e}")

def run_simple_example():
    """Simple example function for testing"""
    setup_environment()
    
    # Try with Anthropic if available
    if os.getenv("ANTHROPIC_API_KEY"):
        return run_basic_example("anthropic", "claude-sonnet-4-20250514")
    elif os.getenv("OPENAI_API_KEY"):
        return run_basic_example("openai", "gpt-4")
    else:
        print("No API keys found")
        return None

def main():
    # Set up environment
    setup_environment()
    
    # Available providers and models
    providers = [
        ("anthropic", "claude-sonnet-4-20250514"),
    ]
    
    # Run examples for available providers
    for provider, model in providers:
        if os.getenv(f"{provider.upper()}_API_KEY"):
            run_basic_example(provider, model)
        else:
            print(f"\n=== {provider.upper()} Skipped ===")
            print(f"No {provider.upper()}_API_KEY found in environment variables.")
    
    # Check if no providers were available
    available_keys = [key for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY", "GOOGLE_API_KEY"] if os.getenv(key)]
    if not available_keys:
        print("\nNo API keys found in environment variables.")
        print("Please copy .env.example to .env and fill in your API keys.")
        print("Supported providers: OpenAI, Anthropic, Cohere, Google, Azure")

if __name__ == "__main__":
    main()
