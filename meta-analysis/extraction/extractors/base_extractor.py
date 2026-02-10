"""
Base classes for AI-powered data extraction.

This module provides the LLMProvider wrapper and BaseExtractor
abstract class that all extractors inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import os


class LLMProvider:
    """
    Wrapper for LLM API calls (Anthropic Claude or OpenAI).

    Provides a unified interface for extracting structured data
    from text using large language models.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM provider.

        Args:
            provider: "anthropic" or "openai"
            model: Model name (defaults based on provider)
            api_key: API key (or uses environment variable)
        """
        self.provider = provider

        if provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        elif provider == "openai":
            self.model = model or "gpt-4o"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._client = None

    def _get_client(self):
        """Lazy initialization of API client."""
        if self._client is None:
            if self.provider == "anthropic":
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    raise ImportError("Install anthropic: pip install anthropic")
            elif self.provider == "openai":
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=self.api_key)
                except ImportError:
                    raise ImportError("Install openai: pip install openai")
        return self._client

    def extract(
        self,
        text: str,
        prompt_template: str,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using LLM.

        Args:
            text: The text to extract from
            prompt_template: Prompt with {text} placeholder
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Parsed JSON response from LLM
        """
        prompt = prompt_template.format(text=text)

        if self.provider == "anthropic":
            return self._extract_anthropic(prompt, max_tokens, temperature)
        else:
            return self._extract_openai(prompt, max_tokens, temperature)

    def _extract_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Extract using Anthropic Claude."""
        client = self._get_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Parse JSON from response
        response_text = response.content[0].text
        return self._parse_json_response(response_text)

    def _extract_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Extract using OpenAI."""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction assistant. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = response.choices[0].message.content
        return self._parse_json_response(response_text)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to find JSON object in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            # Try to find JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return {"items": json.loads(text[start:end])}
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Failed to parse JSON from LLM response: {e}")


class BaseExtractor(ABC):
    """
    Abstract base class for data extractors.

    All extractors (coordinate, effect size, metadata) inherit from this
    class and implement the extract() and validate() methods.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the extractor.

        Args:
            llm_provider: LLM provider for extraction (creates default if None)
        """
        self.llm = llm_provider or LLMProvider()

    @abstractmethod
    def extract(self, text: str, **kwargs) -> Any:
        """
        Extract data from paper text.

        Args:
            text: Full text or relevant section of paper
            **kwargs: Additional extraction parameters

        Returns:
            Extracted data (type depends on extractor)
        """
        pass

    @abstractmethod
    def validate(self, extracted: Any) -> tuple:
        """
        Validate extracted data.

        Args:
            extracted: Data returned by extract()

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        pass

    def extract_and_validate(self, text: str, **kwargs) -> tuple:
        """
        Extract data and validate in one call.

        Args:
            text: Text to extract from
            **kwargs: Extraction parameters

        Returns:
            Tuple of (extracted_data, is_valid, errors)
        """
        extracted = self.extract(text, **kwargs)
        is_valid, errors = self.validate(extracted)
        return extracted, is_valid, errors
