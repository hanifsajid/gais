Usage
=====

.. code-block:: python

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


1. SINGLE-PROMPT GENERATION
---------------------------

.. code-block:: python

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

2. MULTI-PROMPT GENERATION
--------------------------
.. code-block:: python

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


3. MULTI-PROVIDER GENERATION
-----------------------------

.. code-block:: python

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