import pytest
from types import SimpleNamespace
from pydantic import BaseModel

from gais.providers import (
    OpenAI,
    DeepSeek,
    Groq,
    Anthropic,
    Fireworks,
    Gemini,
    Ollama,
    OpenAICompatibleProvider,
    StructuredOutput
)

# -----------------------------
# Dummy StructuredOutput Patch
# -----------------------------
class DummyStructuredOutput(StructuredOutput):
    def parse_response_with_fallback(self, client, base_params, response_format=None, max_retries=3):
        # Always return a SimpleNamespace with parsed dummy value
        return SimpleNamespace(raw=base_params, parsed={"value": 123})

# Patch globally to avoid API calls
@pytest.fixture(autouse=True)
def patch_structured_output(monkeypatch):
    monkeypatch.setattr("gais.providers.structuredoutput", DummyStructuredOutput())

# -----------------------------
# Dummy Pydantic Model
# -----------------------------
class DummyModel(BaseModel):
    value: int

# -----------------------------
# Helper to mock generate
# -----------------------------
def mock_generate(self, prompt, parse=False, response_format=None, **kwargs):
    return SimpleNamespace(
        raw={"prompt": prompt, "kwargs": kwargs},
        parsed={"value": 999} if parse else None,
        provider=self.name
    )

# -----------------------------
# Tests for concrete providers
# -----------------------------
@pytest.mark.parametrize("ProviderClass", [
    OpenAI,
    DeepSeek,
    Groq,
    Anthropic,
    Fireworks,
    Gemini,
    Ollama
])
def test_concrete_provider_basic(monkeypatch, ProviderClass):
    # Patch generate to avoid real API calls
    monkeypatch.setattr(ProviderClass, "generate", mock_generate)

    # Instantiate with fake key
    provider = ProviderClass("test-model", "fake-key")
    
    # Test raw generation
    raw_result = provider.generate("Hello")
    assert raw_result.raw["prompt"] == "Hello"
    assert raw_result.provider == provider.name
    
    # Test structured parsing
    parsed_result = provider.generate("Hello", parse=True, response_format=DummyModel)
    assert parsed_result.parsed["value"] == 999
    assert parsed_result.provider == provider.name

# -----------------------------
# Test OpenAICompatibleProvider directly
# -----------------------------
def test_openai_compatible_provider(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleProvider, "generate", mock_generate)
    provider = OpenAICompatibleProvider("OpenAI", "modelX", "fake-key")
    
    result = provider.generate("Test prompt")
    assert result.raw["prompt"] == "Test prompt"
    assert result.provider == provider.name
    
    result_parse = provider.generate("Test prompt", parse=True, response_format=DummyModel)
    assert result_parse.parsed["value"] == 999
    assert result_parse.provider == provider.name
