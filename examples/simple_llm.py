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
    
    # Get API key from environment
    api_key = get_api_key(provider)
    if not api_key and provider != 'openai':  # OpenAI might have key in OPENAI_API_KEY
        print(f"Warning: {provider.upper()}_API_KEY not found in environment")
        return
    
    # Create the LLM client
    try:
        client = LLMClientFactory.create(
            provider=provider,
            api_key=api_key,
            model=model
        )
    except Exception as e:
        print(f"Error creating {provider} client: {e}")
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

def main():
    # Set up environment
    setup_environment()
    
    # Example with OpenAI (if API key is available)
    if os.getenv("OPENAI_API_KEY"):
        run_basic_example("openai", model="gpt-3.5-turbo")
    
    # Example with Anthropic (if API key is available)
    if os.getenv("ANTHROPIC_API_KEY"):
        run_basic_example("anthropic", model="claude-3-haiku-20240307")
    
    # Check if no providers were available
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("No API keys found in environment variables.")
        print("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment or .env file.")
        print("Example .env file:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("ANTHROPIC_API_KEY=your_anthropic_api_key_here")

if __name__ == "__main__":
    main()
