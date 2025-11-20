
## `gais`: Generative AI Suite for Multi-Provider Access

[![PyPI version](https://img.shields.io/pypi/v/gais)](https://pypi.org/project/gais/)

A unified and consistent interface for working with multiple LLM providers.

`gais` provides:

* __Parallel multi-prompt generation:__ run several prompts at once using a single provider.
* __Multi-provider generation:__ send one prompt to multiple providers and collect all responses together.
* __Structured outputs with Pydantic:__  clean, validated responses using Pydantic models.
* __Universal parsing support:__ uses native parsing when available (e.g., OpenAI .parse), and an automatic fallback parser when it isn’t.

Currently supported providers: __OpenAI, DeepSeek, Groq, Anthropic, Fireworks, Gemini, and Ollama.__

> NOTE: For detailed usage, see the `gais` <a href="https://hanifsajid.github.io/gais" target="_blank">documentation</a>


## Installation

Install via pip:

    ```bash
    pip install gais
    ```

Or from source:

    ```bash

    git https://github.com/hanifsajid/gais.git
    cd gais
    pip install -e .
    ```

## Usage


```Python

    from pydantic import BaseModel, Field
    from gais import Generator, MultiPromptGenerator, MultiProviderGenerator, DeepSeek, OpenAI


    class Capital(BaseModel):
    """Structured Pydantic model representing a capital city."""
    capital: str = Field(..., description="The name of the capital city.")

    class Joke(BaseModel):
        """Structured Pydantic model representing a joke response."""
        joke: str = Field(..., description="A one- or two-sentence joke.")

    # Initialize provider

    api_key = "YOUR_OPENAI_API_KEY"
    openai_provider = OpenAI(model="gpt-4o-mini", api_key=api_key)
```


### 1. SINGLE-PROMPT GENERATION

```python

    single_gen = Generator(openai_provider)

    # ---- Parsed with Pydantic model ----
    parsed_response = single_gen.generate(
        "What is the capital of France?",
        parse=True,
        response_format=Capital,
    )
    print("Parsed Response:", parsed_response.parsed)

    # ---- Standard unparsed output ----
    std_response = single_gen.generate(
        "What is the capital of France?",
        parse=False,
    )
    print("Raw Response:", std_response)

    # ---- With custom generation kwargs ----
    custom_response = single_gen.generate(
        "What is the capital of France?",
        parse=True,
        response_format=Capital,
        temperature=0.2,
        max_tokens=50,
    )
    print("Custom Params Parsed Response:", custom_response.parsed)
```

#### 2. MULTI-PROMPT GENERATION

```python

    multi_prompts = [
        "What is the capital of France?",
        "Tell me a joke.",
        "What is the capital of USA?",
    ]

    # Per-prompt response formats
    multi_formats = [Capital, Joke, Capital]

    multi_gen = MultiPromptGenerator(provider=openai_provider, max_workers=3)

    # ---- Parsed multi-prompt output ----
    parsed_multi = multi_gen.run(
        multi_prompts,
        parse=True,
        response_format=multi_formats,
        temperature=0.3,
    )

    for i, r in enumerate(parsed_multi):
        print(f"Parsed Multi-Prompt {i+1}: {r.parsed}")

    # ---- Standared multi-prompt output ----
    raw_multi = multi_gen.run(multi_prompts, parse=False)
    for i, r in enumerate(raw_multi):
        print(f"Raw Multi-Prompt {i+1}:", r)
```

### 3. MULTI-PROVIDER GENERATION

```python

    from gais import DeepSeek

    providers = [
        openai_provider,
        DeepSeek(model="deepseek-chat", api_key=DEEPSEEK_API_KEY),
    ]

    multi_provider_gen = MultiProviderGenerator(providers, max_workers=2)

    # Per-provider response formats
    provider_response_formats = {
        providers[0].name: Capital,
        providers[1].name: Capital,
    }

    # Provider-specific overrides
    provider_specific_params = {
        providers[1].name: {
            "temperature": 0.1,   # Override for 2nd provider
            "max_tokens": 30,
        }
    }

    # ---- Parsed multi-provider output ----
    provider_results = multi_provider_gen.run(
        prompt="What is the capital of Germany?",
        parse=True,
        response_format=provider_response_formats,
        provider_params=provider_specific_params,
        temperature=0.5,        # Common parameter (all providers)
        max_tokens=80,
    )

    for r in provider_results:
        if r["success"]:
            print(f"Provider {r['provider']} Parsed:", r["response"].parsed)
        else:
            print(f"Provider {r['provider']} Error:", r["error"])

    # ---- Raw multi-provider output ----
    provider_results_raw = multi_provider_gen.run(
        prompt="Tell me a joke.",
        parse=False,
        provider_params=provider_specific_params,
        temperature=0.8,
    )

    for r in provider_results_raw:
        if r["success"]:
            print(f"Provider {r['provider']} Raw:", r["response"])
        else:
            print(f"Provider {r['provider']} Error:", r["error"])

```


## Cite this Project
-----------------

If you use ``gais`` in your work, please cite it as:

```
  @misc{gais,
    title        = {gais},
    author       = {Hanif Sajid},
    year         = {2025},
    month        = November,
    version      = {0.1.0},
    howpublished = {https://github.com/hanifsajid/gais},
    note         = {MIT License}
     }
```
