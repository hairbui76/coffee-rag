"""LLM client — supports OpenAI API and Ollama (OpenAI-compatible)."""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config from .env ──────────────────────────────────────────
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
DEBUG = os.getenv("DEBUG", "false").lower() == "true" == True

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.ERROR, format="%(levelname)s | %(name)s | %(message)s")

DEFAULT_MODEL = OPENAI_MODEL if PROVIDER == "openai" else OLLAMA_MODEL


def get_client() -> OpenAI:
    if PROVIDER == "openai":
        if not OPENAI_API_KEY:
            log.error("LLM_PROVIDER=openai nhưng OPENAI_API_KEY trống! Hãy set trong .env")
            raise ValueError("OPENAI_API_KEY is not set. Copy .env.example → .env và điền key.")
        log.debug("OpenAI client  | model=%s | key=%s...%s", OPENAI_MODEL, OPENAI_API_KEY[:6], OPENAI_API_KEY[-4:])
        return OpenAI(api_key=OPENAI_API_KEY)

    log.debug("Ollama client  | base_url=%s | model=%s | key=%s", OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY)
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def generate(
    messages: list[dict],
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    if client is None:
        client = get_client()

    log.debug("generate() → model=%s, temperature=%s, max_tokens=%s", model, temperature, max_tokens)
    log.debug("messages (%d): %s", len(messages), [m["role"] for m in messages])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        log.debug("response OK — %d chars", len(content or ""))
        return content
    except Exception as e:
        log.error("LLM request FAILED: %s", e, exc_info=True)
        raise


def generate_structured(
    messages: list[dict],
    response_model: type,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """Call LLM with Pydantic structured output (OpenAI only)."""
    if client is None:
        client = get_client()

    log.debug("generate_structured() → model=%s, schema=%s", model, response_model.__name__)
    log.debug("messages (%d): %s", len(messages), [m["role"] for m in messages])

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = response.choices[0].message.parsed
        log.debug("structured response OK — %s", type(parsed).__name__)
        return parsed
    except Exception as e:
        log.error("Structured LLM request FAILED: %s", e, exc_info=True)
        raise


def generate_stream(
    messages: list[dict],
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """Yield response tokens for streaming UI."""
    if client is None:
        client = get_client()

    log.debug("generate_stream() → model=%s, temperature=%s", model, temperature)
    log.debug("messages (%d): %s", len(messages), [m["role"] for m in messages])

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        token_count = 0
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                token_count += 1
                yield delta.content
        log.debug("stream done — %d chunks yielded", token_count)
    except Exception as e:
        log.error("LLM stream FAILED: %s", e, exc_info=True)
        raise
