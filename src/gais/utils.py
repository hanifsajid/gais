import json
import random
import time
from types import SimpleNamespace
from typing import Type
from pydantic import BaseModel, ValidationError


class StructuredOutput:
    """
    Extracts structured JSON from LLM responses.
    Provides fallback behavior for providers that do not support
    beta.chat.completions.parse().
    """

    def _build_prompt(self, schema: Type[BaseModel]) -> str:
        """
        Builds an instruction prompt requiring JSON-only output.
        """
        return (
            "You are an API that returns structured JSON only.\n"
            "Return exactly one JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
            "IMPORTANT:\n"
            "- Return ONLY the JSON.\n"
            "- Do NOT include explanation, markdown, or extra text.\n"
            "- Do NOT wrap the JSON in code blocks.\n"
        )

    def extract_json(self, text: str) -> dict:
        """
        Attempts to extract the first valid JSON object from the text.
        """
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response.")

        open_braces = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                open_braces += 1
            elif text[i] == "}":
                open_braces -= 1
                if open_braces == 0:
                    json_str = text[start:i + 1]
                    return json.loads(json_str)

        raise ValueError("Malformed JSON: unmatched braces.")

    def parse_response_with_fallback(
        self,
        client,
        base_params: dict,
        response_format: Type[BaseModel] = None,
        max_retries: int = 3
    ):
        """
        When provider does NOT support chat.completions.parse,
        use LLM instructions + post-parse validation.
        """
        response = None

        if response_format:
            schema_prompt = self._build_prompt(response_format)
            user_prompt = base_params["messages"][0]["content"]

            base_params = {
                **base_params,
                "messages": [
                    {"role": "system", "content": schema_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(**base_params)
                content = response.choices[0].message.content.strip()

                json_content = self.extract_json(content)
                parsed = (
                    response_format.model_validate(json_content)
                    if response_format
                    else json_content
                )

                return SimpleNamespace(raw=response, parsed=parsed)

            except (ValueError, ValidationError) as e:

                if attempt == max_retries:
                    return SimpleNamespace(
                        raw=response,
                        parsed=None,
                        error=f"JSON parsing/validation failed: {str(e)}"
                    )
                time.sleep(random.uniform(0.5, 1.5))