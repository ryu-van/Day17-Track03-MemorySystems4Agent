from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL, e.g. LiteLLM proxy)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_PROVIDER_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "open_ai": "openai",
    "open-ai": "openai",
    "gpt": "openai",
    "google": "gemini",
    "google-gemini": "gemini",
    "ollama_chat": "ollama",
    "open_router": "openrouter",
    "open-router": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map common aliases to canonical provider names."""
    cleaned = value.strip().lower()
    return _PROVIDER_ALIASES.get(cleaned, cleaned)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Returns a LangChain chat model instance.
    """
    provider = config.provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        # OpenRouter is OpenAI-compatible
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
            base_url="https://openrouter.ai/api/v1",
        )
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            "Supported: openai, custom, gemini, anthropic, ollama, openrouter"
        )
