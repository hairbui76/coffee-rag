"""LLM client — supports OpenAI API, Ollama (OpenAI-compatible) and Anthropic.

Every caller in this project talks to the client through the OpenAI chat
interface (``client.chat.completions.create`` and
``client.beta.chat.completions.parse``). For Anthropic a thin adapter exposes
that same surface on top of ``anthropic.Anthropic``, so the pipeline, the entity
extractor and the Streamlit app need no changes.

.env:
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_MODEL=claude-haiku-5-5      # cheapest; any Claude model id works
"""

import json
import logging
import os
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config from .env ──────────────────────────────────────────
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.ERROR, format="%(levelname)s | %(name)s | %(message)s")

DEFAULT_MODEL = {
    "openai": OPENAI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
}.get(PROVIDER, OLLAMA_MODEL)


# ── Anthropic adapter (OpenAI-shaped) ─────────────────────────

class AnthropicChatClient:
    """Minimal OpenAI-compatible facade over the Anthropic Messages API.

    Supports exactly what this project calls:
      client.chat.completions.create(model, messages, temperature, max_tokens, stream=False)
      client.beta.chat.completions.parse(model, messages, response_format=PydanticModel, ...)
    Structured output is obtained with a forced tool call whose input schema is
    the Pydantic model's JSON schema, then validated with the model itself.
    """

    def __init__(self, api_key: str):
        from anthropic import Anthropic  # imported lazily so OpenAI/Ollama users need no SDK

        self._client = Anthropic(api_key=api_key)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse)))

    @staticmethod
    def _split(messages: list[dict]) -> tuple[str, list[dict]]:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        rest = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
        if not rest:
            rest = [{"role": "user", "content": system or "Hello"}]
            system = ""
        return system, rest

    def _create(self, model: str, messages: list[dict], temperature: float = 0.3,
                max_tokens: int = 1024, stream: bool = False, **_):
        system, rest = self._split(messages)
        kwargs = dict(model=model, messages=rest, temperature=temperature, max_tokens=max_tokens)
        if system:
            kwargs["system"] = system
        if stream:
            return self._stream(kwargs)
        resp = self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def _stream(self, kwargs: dict):
        with self._client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    def _parse(self, model: str, messages: list[dict], response_format: type, temperature: float = 0.3,
               max_tokens: int = 1024, **_):
        system, rest = self._split(messages)
        schema = response_format.model_json_schema()
        tool = {
            "name": "emit_response",
            "description": f"Return the final answer as a {response_format.__name__} object.",
            "input_schema": schema,
        }
        kwargs = dict(model=model, messages=rest, temperature=temperature, max_tokens=max_tokens,
                      tools=[tool], tool_choice={"type": "tool", "name": "emit_response"})
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        block = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
        if block is None:
            raise ValueError("Anthropic returned no tool_use block for structured output")
        parsed = response_format.model_validate(block.input)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=parsed, content=json.dumps(block.input, ensure_ascii=False)))])


def get_client():
    if PROVIDER == "openai":
        if not OPENAI_API_KEY:
            log.error("LLM_PROVIDER=openai nhưng OPENAI_API_KEY trống! Hãy set trong .env")
            raise ValueError("OPENAI_API_KEY is not set. Copy .env.example → .env và điền key.")
        log.debug("OpenAI client  | model=%s | key=%s...%s", OPENAI_MODEL, OPENAI_API_KEY[:6], OPENAI_API_KEY[-4:])
        return OpenAI(api_key=OPENAI_API_KEY)

    if PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            log.error("LLM_PROVIDER=anthropic nhưng ANTHROPIC_API_KEY trống! Hãy set trong .env")
            raise ValueError("ANTHROPIC_API_KEY is not set. Copy .env.example → .env và điền key.")
        log.debug("Anthropic client | model=%s | key=%s...%s", ANTHROPIC_MODEL, ANTHROPIC_API_KEY[:10], ANTHROPIC_API_KEY[-4:])
        return AnthropicChatClient(api_key=ANTHROPIC_API_KEY)

    log.debug("Ollama client  | base_url=%s | model=%s | key=%s", OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY)
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def generate(
    messages: list[dict],
    client=None,
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
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """Call LLM with Pydantic structured output (OpenAI and Anthropic)."""
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
    client=None,
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
