"""
generation.py

This module provides classes for generating LLM outputs from single or multiple prompts,
and supports multiple providers in parallel. Structured parsing is supported via
`parse` and `response_format` parameters.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Union
from .providers import BaseProvider



# ----------------------------
# Generator Classes
# ----------------------------
class Generator:
    """
    Single-prompt generator using a specific LLM provider.
    """

    def __init__(self, provider: BaseProvider):
        """
        Initialize the generator.

        Args:
            provider (BaseProvider): An LLM provider instance.
        """
        self.provider = provider

    def generate(self, prompt: str, parse: bool = False, response_format=None, **kwargs) -> Any:
        """
        Generate a response for a single prompt.

        Args:
            prompt (str): Prompt string.
            parse (bool, optional): Enable structured parsing. Defaults to False.
            response_format (Any, optional): Structured output format (Pydantic class).
            **kwargs: Provider-specific parameters.

        Returns:
            Any: Parsed or raw response from the provider.
        """
        return self.provider.generate(prompt, parse=parse, response_format=response_format, **kwargs)


class MultiPromptGenerator:
    """
    Generate multiple prompts in parallel with optional per-prompt structured output.
    """

    def __init__(self, provider: BaseProvider, max_workers: int = 5):
        """
        Initialize multi-prompt generator.

        Args:
            provider (BaseProvider): LLM provider instance.
            max_workers (int, optional): Max parallel threads. Defaults to 5.
        """
        self.provider = provider
        self.max_workers = max_workers

    def _run_single(self, prompt: str, parse: bool, response_format: Any, kwargs: Dict[str, Any]) -> Any:
        return self.provider.generate(prompt, parse=parse, response_format=response_format, **kwargs)

    def run(
        self,
        prompts: List[str],
        parse: bool = False,
        response_format: Union[Any, List[Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Run multiple prompts in parallel with optional structured parsing.

        Args:
            prompts (List[str]): List of prompt strings.
            parse (bool, optional): Enable structured parsing. Defaults to False.
            response_format (Any or List[Any], optional): Single format or list of formats per prompt.
            **kwargs: Provider-specific parameters for all prompts.

        Returns:
            List[Any]: List of parsed or raw responses.
        """
        if not isinstance(response_format, list):
            response_formats = [response_format] * len(prompts)
        else:
            if len(response_format) != len(prompts):
                raise ValueError("Length of response_format list must match number of prompts")
            response_formats = response_format

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(
                    lambda p_rf: self._run_single(p_rf[0], parse, p_rf[1], kwargs),
                    zip(prompts, response_formats)
                )
            )

        return results


class MultiProviderGenerator:
    """
    Run a single prompt across multiple providers in parallel.

    Supports per-provider structured output via `provider_params`.
    """

    def __init__(self, providers: List[BaseProvider], max_workers: int = 5):
        """
        Initialize multi-provider generator.

        Args:
            providers (List[BaseProvider]): List of LLM provider instances.
            max_workers (int, optional): Max parallel threads. Defaults to 5.
        """
        self.providers = providers
        self.max_workers = max_workers

    def _run_single(
        self,
        provider: BaseProvider,
        prompt: str,
        parse: bool,
        response_format: Any,
        common_kwargs: Dict[str, Any],
        provider_specific: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run a single prompt on a single provider with merged parameters.
        """
        params = {**common_kwargs}
        if provider.name in provider_specific:
            params.update(provider_specific[provider.name])

        try:
            response = provider.generate(prompt, parse=parse, response_format=response_format, **params)
            return {"provider": provider.name, "success": True, "response": response}
        except Exception as e:
            return {"provider": provider.name, "success": False, "error": str(e)}

    def run(
        self,
        prompt: str,
        parse: bool = False,
        response_format: Union[Any, Dict[str, Any]] = None,
        provider_params: Dict[str, Dict[str, Any]] = None,
        **common_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run the same prompt across multiple providers in parallel.

        Args:
            prompt (str): Prompt string.
            parse (bool, optional): Enable structured parsing. Defaults to False.
            response_format (Any or Dict[str, Any], optional): Single format for all providers
                or per-provider dict {provider_name: response_format}.
            provider_params (Dict[str, Dict[str, Any]], optional): Per-provider parameters.
            **common_kwargs: Common parameters for all providers.

        Returns:
            List[Dict[str, Any]]: List of results per provider.
        """
        provider_params = provider_params or {}

        results = [None] * len(self.providers)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for provider in self.providers:
                rf = response_format
                if isinstance(response_format, dict):
                    rf = response_format.get(provider.name)
                futures.append(
                    executor.submit(
                        self._run_single,
                        provider,
                        prompt,
                        parse,
                        rf,
                        common_kwargs,
                        provider_params
                    )
                )
            for i, future in enumerate(futures):
                results[i] = future.result()

        return results

 