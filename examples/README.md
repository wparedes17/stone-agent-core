# Stone Agent Core Examples

This directory contains examples demonstrating how to use the Stone Agent Core framework with LiteLLM integration.

## Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys for the providers you want to use:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Examples

### 1. Simple LLM (`simple_llm.py`)

Demonstrates basic LLM usage with multiple providers:
- Synchronous text generation
- Streaming text generation
- Support for OpenAI, Anthropic, Cohere, and Google

**Run:**
```bash
python examples/simple_llm.py
```

### 2. Tool Usage (`tool_llm.py`)

Shows how to use function calling with LLMs:
- Tool schema definition
- Tool call processing
- Multi-turn conversations with tools

**Run:**
```bash
python examples/tool_llm.py
```

### 3. Simple Pipeline (`simple_pipeline.py`)

Basic framework example without LLM:
- Module registration and dependencies
- State management
- Execution flow visualization

**Run:**
```bash
python examples/simple_pipeline.py
```

### 4. Framework with LLM (`framework_with_llm.py`)

Integrates LiteLLM within the modular framework:
- LLM-powered agent modules
- Multi-provider support
- Execution tracking and usage metrics

**Run:**
```bash
python examples/framework_with_llm.py
```

## Supported Providers

The examples work with any of these providers (add API keys to `.env`):

- **OpenAI**: GPT models (gpt-4o, gpt-4o-mini, etc.)
- **Anthropic**: Claude models (claude-3-5-sonnet, etc.)
- **Cohere**: Command models (command-r-plus, etc.)
- **Google**: Gemini models (gemini-1.5-pro, etc.)
- **Azure**: Azure OpenAI models

## Features Demonstrated

- ✅ Multiple provider support via LiteLLM
- ✅ Streaming and non-streaming generation
- ✅ Function calling and tool usage
- ✅ Modular agent framework integration
- ✅ Error handling and fallbacks
- ✅ Usage tracking and metrics
- ✅ Environment-based configuration

## Troubleshooting

1. **Import errors**: Make sure you installed the package with `pip install -e .`
2. **API key errors**: Check that your `.env` file contains the correct keys
3. **Provider not available**: Ensure you have the required API key for the provider
4. **Model not found**: Verify the model name is correct for your provider
