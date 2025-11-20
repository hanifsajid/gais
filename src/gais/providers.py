"""
providers.py

This module defines abstract and concrete classes for LLM providers.
It includes a base interface `BaseProvider`, an OpenAI-compatible wrapper
with optional structured output support, and concrete provider classes.
"""
from abc import ABC, abstractmethod
from typing import Any
from types import SimpleNamespace
from openai import OpenAI as OpenAIClient
from .utils import StructuredOutput

structuredoutput = StructuredOutput()


# ----------------------------
# Base Provider Classes
# ----------------------------
class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Any provider should implement the `generate` method and `name` property.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): Input prompt string.
            **kwargs: Provider-specific parameters.

        Returns:
            Any: Response from the LLM provider.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the provider.

        Returns:
            str: Provider name.
        """
        raise NotImplementedError()


class OpenAICompatibleProvider(BaseProvider):
    """
    Base wrapper for OpenAI-compatible LLM providers with optional structured parsing.

    Supports both OpenAI's native parsing (if available) or a fallback
    structured parser using `StructuredOutput`.
    """

    def __init__(self, provider_name: str, model: str, api_key: str, base_url: str | None = None):
        """
        Initialize the provider.

        Args:
            provider_name (str): Name of the provider (e.g., "OpenAI").
            model (str): Model name to use.
            api_key (str): API key for authentication.
            base_url (str | None, optional): Optional API base URL.
        """
        self.provider_name = provider_name
        self.model = model
        self.client = OpenAIClient(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        """
        Return provider name with model.

        Returns:
            str: Example: "OpenAI:gpt-4".
        """
        return f"{self.provider_name}:{self.model}"

    def generate(self, prompt: str, parse: bool = False, response_format=None, **kwargs) -> Any:
        """
        Generate a response for a prompt, optionally structured.

        Args:
            prompt (str): Prompt string.
            parse (bool, optional): Enable structured parsing. Defaults to False.
            response_format (Any, optional): Structured output format (Pydantic class or JSON).
            **kwargs: Provider-specific parameters.

        Returns:
            Any: Either raw response or SimpleNamespace with parsed result.
        """
        base_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }

        supports_native_parse = (
            self.provider_name.lower() == "openai"
            and getattr(self.client, "beta", None)
            and hasattr(self.client.beta.chat.completions, "parse")
        )

        if parse and supports_native_parse:
            response = self.client.beta.chat.completions.parse(
                **base_params,
                response_format=response_format
            )
            parsed_value = response.choices[0].message.parsed
            return SimpleNamespace(raw=response, parsed=parsed_value, provider=self.name)

        if parse:
            parsed_ns = structuredoutput.parse_response_with_fallback(
                self.client, base_params, response_format
            )
            parsed_ns.provider = self.name
            return parsed_ns

        return self.client.chat.completions.create(**base_params)


# ----------------------------
# Concrete Providers
# ----------------------------
class OpenAI(OpenAICompatibleProvider):
    """OpenAI provider wrapper."""
    def __init__(self, model: str, api_key: str):
        super().__init__("OpenAI", model, api_key, base_url=None)


class DeepSeek(OpenAICompatibleProvider):
    """DeepSeek provider wrapper."""
    def __init__(self, model: str, api_key: str):
        super().__init__("DeepSeek", model, api_key, base_url="https://api.deepseek.com/v1")


class Groq(OpenAICompatibleProvider):
    """Groq provider wrapper."""
    def __init__(self, model: str, api_key: str):
        super().__init__("Groq", model, api_key, base_url="https://api.groq.com/openai/v1")


class Anthropic(OpenAICompatibleProvider):
    """Anthropic provider wrapper."""
    def __init__(self, model: str, api_key: str):
        super().__init__("Anthropic", model, api_key, base_url="https://api.anthropic.com/v1/")


class Fireworks(OpenAICompatibleProvider):
    """Fireworks provider wrapper."""
    def __init__(self, model: str, api_key: str):
        super().__init__("Fireworks", model, api_key, base_url="https://api.fireworks.ai/inference/v1")


class Gemini(OpenAICompatibleProvider):
    """Gemini provider wrapper."""
    def __init__(self, model: str, api_key: str):
        super().__init__("Gemini", model, api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


class Ollama(OpenAICompatibleProvider):
    """Ollama provider wrapper (local server)."""
    def __init__(self, model: str, api_key: str):
        super().__init__("Ollama", model, api_key, base_url="http://localhost:11434/v1")




# from abc import ABC, abstractmethod
# from typing import Any
# from openai import OpenAI as OpenAIClient
# from .utils import StructuredOutput
# from types import SimpleNamespace

# structuredoutput = StructuredOutput()


# class BaseProvider(ABC):
#     """
#     Abstract base class for all LLM providers.

#     Any custom LLM provider should inherit from this class and implement
#     the `generate` method and `name` property.
#     """

#     @abstractmethod
#     def generate(self, prompt: str, **kwargs) -> Any:
#         """
#         Generate a response for the given prompt.

#         Args:
#             prompt (str): The input prompt to the LLM.
#             **kwargs: Provider-specific parameters.

#         Returns:
#             Any: The response object from the LLM provider.
#         """
#         raise NotImplementedError()

#     @property
#     @abstractmethod
#     def name(self) -> str:
#         """
#         Human-readable name of the provider.

#         Returns:
#             str: The provider name.
#         """
#         raise NotImplementedError()


# class OpenAICompatibleProvider(BaseProvider):
#     """
#     Base wrapper for OpenAI-compatible providers, supporting optional structured parsing.

#     This class wraps the OpenAIClient (or compatible API client) and provides
#     a unified interface for generating responses, with optional structured parsing
#     using either the native OpenAI parse feature or a fallback parser.
#     """

#     def __init__(self, provider_name: str, model: str, api_key: str, base_url: str | None = None):
#         """
#         Initialize the provider.

#         Args:
#             provider_name (str): Name of the provider (e.g., "OpenAI", "DeepSeek").
#             model (str): Model name to use (e.g., "gpt-4").
#             api_key (str): API key for authentication.
#             base_url (str | None, optional): Optional base URL for the provider API.
#         """
#         self.provider_name = provider_name
#         self.model = model
#         self.client = OpenAIClient(api_key=api_key, base_url=base_url)

#     @property
#     def name(self) -> str:
#         """
#         Provider name combined with the model name.

#         Returns:
#             str: A string like "OpenAI:gpt-4".
#         """
#         return f"{self.provider_name}:{self.model}"

#     def generate(self, prompt: str, parse: bool = False, response_format=None, **kwargs) -> Any:
#         """
#         Generate a response for a given prompt, optionally parsing the output.

#         Args:
#             prompt (str): The prompt to send to the model.
#             parse (bool, optional): Whether to parse the response into structured output. Defaults to False.
#             response_format (Any, optional): Format for structured output (e.g., "json").
#             **kwargs: Additional provider-specific parameters.

#         Returns:
#             Any: Either the raw response from the API or a `SimpleNamespace` containing:
#                 - `raw`: Original API response.
#                 - `parsed`: Parsed structured output (if parse=True).
#                 - `provider`: Name of the provider.
#         """
#         base_params = {
#             "model": self.model,
#             "messages": [{"role": "user", "content": prompt}],
#             **kwargs
#         }

#         supports_native_parse = (
#             self.provider_name.lower() == "openai"
#             and getattr(self.client, "beta", None)
#             and hasattr(self.client.beta.chat.completions, "parse")
#         )

#         if parse and supports_native_parse:
#             response = self.client.beta.chat.completions.parse(
#                 **base_params,
#                 response_format=response_format
#             )
#             parsed_value = response.choices[0].message.parsed
#             return SimpleNamespace(raw=response, parsed=parsed_value, provider=self.name)

#         if parse:
#             parsed_ns = structuredoutput.parse_response_with_fallback(
#                 self.client, base_params, response_format
#             )
#             parsed_ns.provider = self.name
#             return parsed_ns

#         return self.client.chat.completions.create(**base_params)


# # ----------------------------
# # Concrete provider classes
# # ----------------------------

# class OpenAI(OpenAICompatibleProvider):
#     """OpenAI provider wrapper."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("OpenAI", model, api_key, base_url=None)


# class DeepSeek(OpenAICompatibleProvider):
#     """DeepSeek provider wrapper."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("DeepSeek", model, api_key, base_url="https://api.deepseek.com/v1")


# class Groq(OpenAICompatibleProvider):
#     """Groq provider wrapper."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("Groq", model, api_key, base_url="https://api.groq.com/openai/v1")


# class Anthropic(OpenAICompatibleProvider):
#     """Anthropic provider wrapper."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("Anthropic", model, api_key, base_url="https://api.anthropic.com/v1/")


# class Fireworks(OpenAICompatibleProvider):
#     """Fireworks provider wrapper."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("Fireworks", model, api_key, base_url="https://api.fireworks.ai/inference/v1")


# class Gemini(OpenAICompatibleProvider):
#     """Gemini provider wrapper."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("Gemini", model, api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


# class Ollama(OpenAICompatibleProvider):
#     """Ollama provider wrapper (local server)."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__("Ollama", model, api_key, base_url="http://localhost:11434/v1")



