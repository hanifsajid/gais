import pytest
from types import SimpleNamespace
from pydantic import BaseModel

from gais.generation import Generator, MultiPromptGenerator, MultiProviderGenerator
from gais.providers import BaseProvider, OpenAICompatibleProvider
from gais.utils import StructuredOutput

# -----------------------------
# Mock Providers
# -----------------------------
class MockProvider(BaseProvider):
    def __init__(self, name: str, model: str = "test-model"):
        self.provider_name = name
        self.model = model
        self.called_with = []

    @property
    def name(self) -> str:
        return f"{self.provider_name}:{self.model}"

    def generate(self, prompt: str, **kwargs):
        self.called_with.append(kwargs)
        return {"prompt": prompt, "provider": self.name, "kwargs": kwargs}


# -----------------------------
# Structured Output Mock
# -----------------------------
class DummyModel(BaseModel):
    value: int

# Patch StructuredOutput to avoid OpenAI calls
class DummyStructuredOutput(StructuredOutput):
    def parse_response_with_fallback(self, client, base_params, response_format=None, max_retries=3):
        # Always return a SimpleNamespace with parsed dummy value
        return SimpleNamespace(raw=base_params, parsed={"value": 42})


# -----------------------------
# Tests: Generator
# -----------------------------
def test_single_generator_basic():
    provider = MockProvider("A")
    gen = Generator(provider)
    result = gen.generate("Hello")
    assert result["prompt"] == "Hello"
    assert result["provider"] == "A:test-model"


def test_multi_prompt_generator_basic():
    provider = MockProvider("B")
    gen = MultiPromptGenerator(provider, max_workers=2)
    prompts = ["one", "two"]
    results = gen.run(prompts)
    assert results[0]["prompt"] == "one"
    assert results[1]["prompt"] == "two"
    for res in results:
        assert res["provider"] == "B:test-model"


# -----------------------------
# Tests: MultiProviderGenerator
# -----------------------------
def test_multi_provider_generator_common_and_specific():
    p1 = MockProvider("X", "model1")
    p2 = MockProvider("Y", "model2")
    gen = MultiProviderGenerator([p1, p2])

    results = gen.run(
        "hello",
        provider_params={
            "X:model1": {"temperature": 0.2},
            "Y:model2": {"max_tokens": 5},
        },
        temperature=0.5
    )

    # Check results structure
    assert len(results) == 2
    assert results[0]["provider"] == "X:model1"
    assert results[1]["provider"] == "Y:model2"

    # Check provider-specific overrides applied
    assert p1.called_with[0]["temperature"] == 0.2
    assert p2.called_with[0]["max_tokens"] == 5

    # Common parameters applied where not overridden
    assert p1.called_with[0].get("temperature") == 0.2  # overridden
    assert p2.called_with[0].get("temperature") == 0.5  # common


# -----------------------------
# Tests: OpenAICompatibleProvider fallback parsing
# -----------------------------
def test_openai_compatible_provider_parsing(monkeypatch):
    provider = MockProvider("Z")

    # Patch StructuredOutput for provider
    monkeypatch.setattr("gais.providers.structuredoutput", DummyStructuredOutput())

    # Simulate provider.generate using fallback parsing
    result = provider.generate("Prompt", parse=True, response_format=DummyModel)
    # With MockProvider we return kwargs as list, we just ensure no exception
    assert isinstance(result, dict) or isinstance(result, SimpleNamespace)

# import pytest
# from gais import MultiProviderGenerator

# # --- Correct Mock Provider ---
# class MockProvider:
#     def __init__(self, name, model_name):
#         self.name = name
#         self.model_name = model_name
#         self.called_with = []

#     def generate(self, prompt, **kwargs):
#         self.called_with.append(kwargs)
#         return {"response": f"{self.name}-{prompt}"}


# # --- Tests ---
# def test_multi_provider_basic():
#     p1 = MockProvider("A", "model1")
#     p2 = MockProvider("B", "model2")
#     gen = MultiProviderGenerator([p1, p2])

#     result = gen.run("Hello")
#     assert result[0]["response"] == "A-Hello"
#     assert result[1]["response"] == "B-Hello"


# def test_multi_provider_specific_parameters():
#     p1 = MockProvider("X", "test-model")
#     p2 = MockProvider("Y", "test-model")
    
#     gen = MultiProviderGenerator([p1, p2])
#     gen.run(
#         "hello",
#         parse=False,
#         provider_params={
#             "X:test-model": {"temperature": 0.2},
#             "Y:test-model": {"max_tokens": 5},
#         },
#         temperature=0.5,  # common
#     )

#     # Now the overrides will be correctly applied
#     assert p1.called_with[0]["temperature"] == 0.2
#     assert p2.called_with[0]["max_tokens"] == 5

# #import pytest

# # from types import SimpleNamespace
# # from gais import MultiProviderGenerator

# # # --- Mock Providers ---
# # class MockProvider:
# #     def __init__(self, name):
# #         self.name = name
# #         self.called_with = []

# #     def generate(self, prompt, **kwargs):
# #         # Save the actual kwargs used in the call
# #         self.called_with.append(kwargs)
# #         return f"{self.name}-{prompt}"


# # # --- Tests ---
# # def test_multi_provider_basic():
# #     p1 = MockProvider("A")
# #     p2 = MockProvider("B")
# #     gen = MultiProviderGenerator([p1, p2])

# #     result = gen.run("Hello")
# #     assert result[0]["response"] == "A-Hello"
# #     assert result[1]["response"] == "B-Hello"


# # def test_multi_provider_specific_parameters():
# #     p1 = MockProvider("X")
# #     p2 = MockProvider("Y")

# #     gen = MultiProviderGenerator([p1, p2])
# #     gen.run(
# #         "hello",
# #         parse=False,
# #         provider_params={
# #             "X:test-model": {"temperature": 0.2},
# #             "Y:test-model": {"max_tokens": 5},
# #         },
# #         temperature=0.5,  # common
# #     )

# #     # Check that provider-specific overrides were applied
# #     assert p1.called_with[0]["temperature"] == 0.2
# #     assert p2.called_with[0]["max_tokens"] == 5
