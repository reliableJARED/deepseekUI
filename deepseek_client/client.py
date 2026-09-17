"""The client: one object that speaks OpenAI ``chat-completions``.

Designed for a local backend that relays to a hosted endpoint, so the parts that
matter here are streaming, tool calls, robust error mapping, and never leaking a
raw traceback to the browser.

    import asyncio
    from deepseek_client import DeepSeekClient, user

    async def main():
        async with DeepSeekClient.from_providers() as client:
            async with client.stream([user("hello")]) as stream:
                async for event in stream:
                    if event.type == "content":
                        print(event.text, end="", flush=True)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Mapping, Sequence

import httpx

from .config import (
    DEFAULT_API_FLAVOR,
    DEFAULT_API_PATH,
    DEPRECATED_PARAMS,
    REASONING_EFFORTS,
    THINKING_TOP_P_FLOOR,
    ModelSpec,
    Provider,
    ProvidersConfig,
)
from .errors import (
    AuthenticationError,
    BadRequestError,
    ConfigError,
    ConnectionFailure,
    ContextLengthError,
    DeepSeekError,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    RetryExhausted,
    ServerError,
)
from .messages import estimate_tokens, sanitize_messages
from .streaming import ChatStream
from .tools import ToolRegistry, run_tool_loop
from .types import ChatResponse

__all__ = [
    "DeepSeekClient",
    "raise_for_status",
    "raise_for_response",
    "DEFAULT_TIMEOUT",
    "DEFAULT_STREAM_TIMEOUT",
]

logger = logging.getLogger("deepseek_client.client")

DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
#: Streaming keeps a generous *read* timeout: it resets on every chunk, so a long
#: generation is fine as long as the server keeps talking.
DEFAULT_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})

_CONTEXT_MARKERS = (
    "maximum context length",
    "context length",
    "too many tokens",
    "context window",
    "reduce the length",
)

_USER_ID_OK = re.compile(r"[^a-zA-Z0-9\-_]")


def _normalise_reasoning_effort(value: Any) -> str:
    """Map the server's effort vocabulary onto ``none``/``low``/``high``/``max``."""
    text = str(value or "").strip().lower()
    if text in ("", "none", "off", "disabled", "false"):
        return "none"
    return {
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "high",
        "max": "max",
        "ultra": "max",
    }.get(text, "high")


def _clean_user_id(value: str) -> str:
    """Force a user id into the documented ``[a-zA-Z0-9\\-_]`` charset, max 512."""
    cleaned = _USER_ID_OK.sub("-", str(value))[:512]
    return cleaned or "anonymous"


def _has_reasoning(message: Mapping[str, Any]) -> bool:
    value = message.get("reasoning_content") or message.get("reasoning")
    return bool(isinstance(value, str) and value.strip())


# ── error mapping ─────────────────────────────────────────────────────────────

def _extract_message(body: bytes | str | None) -> str:
    """Pull a human-readable message out of an error response body."""
    if not body:
        return "empty response body"
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", errors="replace")
    else:
        text = body
    text = text.strip()
    if not text:
        return "empty response body"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or json.dumps(error)[:500])
        if isinstance(error, str):
            return error
        for key in ("message", "detail", "msg", "error_description"):
            if data.get(key):
                return str(data[key])
    return text[:500]


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def raise_for_status(
    status: int,
    body: bytes | str | None,
    *,
    url: str = "",
    retry_after: float | None = None,
) -> None:
    """Translate a non-2xx response into the matching exception.

    Shared with :class:`~deepseek_client.streaming.ChatStream`, which has to check
    the status of a response it has only just opened.

    ``retry_after`` carries the parsed ``Retry-After`` header, so the hint reaches
    the caller whether or not this particular call was retried internally.
    """
    if status < 400:
        return

    message = _extract_message(body)
    where = f" ({url})" if url else ""

    if status in (401, 403):
        raise AuthenticationError(
            f"{message}{where} — check that your API key is valid and has access to this model.",
            status=status, body=body,
        )
    if status == 402:
        raise PaymentRequiredError(f"{message}{where} — account balance exhausted.", status=status, body=body)
    if status == 404:
        raise NotFoundError(
            f"{message}{where} — check the base URL and model id.", status=status, body=body
        )
    if status == 429:
        raise RateLimitError(
            message, status=status, body=body, retry_after=retry_after
        )
    if status == 400:
        lowered = message.lower()
        if any(marker in lowered for marker in _CONTEXT_MARKERS):
            raise ContextLengthError(message, status=status, body=body)
        raise BadRequestError(message, status=status, body=body)
    if status == 422:
        # "Invalid Parameters" — same failure class as 400, distinct code.
        raise BadRequestError(message, status=status, body=body)
    if status >= 500:
        raise ServerError(f"{message}{where}", status=status, body=body)

    raise DeepSeekError(f"{message}{where}", status=status, body=body)


def raise_for_response(response: httpx.Response, *, url: str = "") -> None:
    """``raise_for_status`` for a response whose body has already been read.

    Unlike the raw helper this also lifts ``Retry-After`` onto the exception, so
    a throttled caller learns how long to wait even when the client itself did
    not retry (``max_retries=0``) or gave up.
    """
    raise_for_status(
        response.status_code,
        response.content,
        url=url or str(response.url),
        retry_after=_retry_after(response),
    )


# ── the client ────────────────────────────────────────────────────────────────

class DeepSeekClient:
    """Async client for an OpenAI ``chat-completions`` endpoint.

    Construct it in whichever way matches how you configured things::

        DeepSeekClient.from_env()                          # DEEPSEEK_API_KEY etc.
        DeepSeekClient.from_providers("providers.json")    # the JSON schema
        DeepSeekClient(api_key="sk-...", base_url="https://api.deepseek.com",
                       model="deepseek-flash")
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        model: str = "",
        api_path: str = DEFAULT_API_PATH,
        tool_calling: bool = True,
        vision: bool = False,
        max_input_tokens: int = 1_000_000,
        max_output_tokens: int = 393_216,
        provider: Provider | None = None,
        timeout: httpx.Timeout | float | None = None,
        stream_timeout: httpx.Timeout | float | None = None,
        max_retries: int = 2,
        retry_backoff: float = 0.75,
        include_usage: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        defaults: Mapping[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError("An API key is required to construct DeepSeekClient")

        self.api_key = api_key
        self.base_url = (base_url or (provider.url if provider else "")).rstrip("/")
        self.api_path = api_path or DEFAULT_API_PATH
        self.model = model or (provider.default_model if provider else "")
        self.tool_calling = tool_calling
        self.vision = vision
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.provider = provider
        self.max_retries = max(0, max_retries)
        self.retry_backoff = retry_backoff
        self.include_usage = include_usage
        self.defaults: dict[str, Any] = dict(provider.defaults) if provider else {}
        self.defaults.update(defaults or {})

        self._timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._stream_timeout = stream_timeout if stream_timeout is not None else DEFAULT_STREAM_TIMEOUT
        self._extra_headers = dict(extra_headers or {})

        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            },
        )

    # ── constructors ──

    @classmethod
    def from_provider(
        cls,
        provider: Provider,
        model: str | None = None,
        **kwargs: Any,
    ) -> "DeepSeekClient":
        if not provider.is_configured:
            # Catching it here rather than in `__init__` gives a message that says
            # *what to do*; by the time the key reaches the constructor it is just
            # an empty string with no context.
            raise ConfigError(
                f"Provider {provider.name!r} has no API key. Set "
                f"DEEPSEEK_API_KEY in .env, or paste one into the app's "
                f"Settings panel."
            )
        spec = provider.model(model)
        kwargs.setdefault("tool_calling", spec.tool_calling)
        kwargs.setdefault("vision", spec.vision)
        kwargs.setdefault("max_input_tokens", spec.max_input_tokens)
        kwargs.setdefault("max_output_tokens", spec.max_output_tokens)
        kwargs.setdefault("api_path", provider.api_path)
        return cls(
            api_key=provider.api_key,
            base_url=provider.url or spec.url,
            model=spec.id,
            provider=provider,
            **kwargs,
        )

    @classmethod
    def from_providers(
        cls,
        path: str | Path | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path | None = ".env",
        prefix: str = "DEEPSEEK",
        **kwargs: Any,
    ) -> "DeepSeekClient":
        """Load ``providers.json`` (or fall back to ``.env``) and pick a target.

        ``environ`` supplies the values that ``${VAR}`` placeholders resolve
        against, which keeps resolution deterministic in tests and lets the
        server inject secrets it has already read.
        """
        config = ProvidersConfig.load_or_env(
            path, prefix=prefix, env_file=env_file, environ=environ
        )
        return cls.from_provider(config.get(provider), model, **kwargs)

    @classmethod
    def from_env(
        cls,
        prefix: str = "DEEPSEEK",
        *,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path = ".env",
        **kwargs: Any,
    ) -> "DeepSeekClient":
        config = ProvidersConfig.from_env(prefix, environ=environ, env_file=env_file)
        return cls.from_provider(config.get(None), None, **kwargs)

    @classmethod
    def from_config(cls, config: ProvidersConfig, *, provider: str | None = None, model: str | None = None, **kw: Any):
        return cls.from_provider(config.get(provider), model, **kw)

    # ── lifecycle ──

    @property
    def http(self) -> httpx.AsyncClient:
        return self._http

    async def aclose(self) -> None:
        """Close the underlying connection pool, if this client owns it."""
        if self._owns_http and not self._http.is_closed:
            await self._http.aclose()

    async def __aenter__(self) -> "DeepSeekClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"<DeepSeekClient model={self.model!r} base_url={self.base_url!r} "
            f"tools={self.tool_calling} vision={self.vision}>"
        )

    # ── request flavour ──

    @property
    def api_flavor(self) -> str:
        """Which request shape to send: the provider's, else ``deepseek``.

        :meth:`spec_for` can synthesise a :class:`ModelSpec` when no provider is
        configured, so ``self.provider is None`` is a normal state and falls back
        to the historical (DeepSeek) behaviour.
        """
        return self.provider.api_flavor if self.provider is not None else DEFAULT_API_FLAVOR

    @property
    def sends_reasoning_fields(self) -> bool:
        """True when ``thinking``/``reasoning_effort`` may be sent upstream."""
        if self.provider is not None:
            return self.provider.sends_reasoning_fields
        return self.api_flavor == DEFAULT_API_FLAVOR

    @property
    def sends_user_id(self) -> bool:
        """True when the DeepSeek-only ``user_id`` field may be sent upstream."""
        if self.provider is not None:
            return self.provider.sends_user_id
        return self.api_flavor == DEFAULT_API_FLAVOR

    @property
    def replays_reasoning(self) -> bool:
        """True when prior turns' ``reasoning_content`` must be echoed back."""
        if self.provider is not None:
            return self.provider.replays_reasoning
        return self.api_flavor == DEFAULT_API_FLAVOR

    # ── model resolution ──

    def spec_for(self, model: str | None = None) -> ModelSpec:
        """Resolve a model id to its configured spec, synthesising one if unknown."""
        wanted = model or self.model

        if self.provider is not None:
            if wanted and not self.provider.has_model(wanted):
                # Unknown ids are allowed — a caller may target a model that is not
                # in providers.json — but they inherit the client's own settings.
                if not wanted:
                    return self.provider.model(None)
                logger.info("model %r is not in the provider config; using client defaults", wanted)
            else:
                return self.provider.model(wanted)

        return ModelSpec(
            id=wanted or "default",
            name=wanted or "default",
            url=self.base_url,
            tool_calling=self.tool_calling,
            vision=self.vision,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
        )

    def endpoint(self, model: str | None = None) -> str:
        """Full request URL for a model, tolerating a ``/v1`` on the base URL."""
        return self.spec_for(model).endpoint(self.api_path, self.base_url)

    def supports(self, feature: str, model: str | None = None) -> bool:
        spec = self.spec_for(model)
        if feature == "tools":
            return spec.tool_calling
        if feature == "vision":
            return spec.vision
        raise ValueError(f"unknown feature {feature!r}")

    # ── payload ──

    def build_payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        stream: bool = False,
        tools: Sequence[Mapping[str, Any]] | ToolRegistry | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        response_format: Mapping[str, Any] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        user_id: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        sanitize: bool = True,
    ) -> dict[str, Any]:
        """Assemble the request body. Any ``None`` argument is omitted entirely.

        ``thinking`` / ``reasoning_effort`` control the chain of thought.  It is
        **on by default server-side**, so ``thinking=None`` deliberately sends
        nothing and lets the server's default stand.  Pass ``thinking=False`` or
        ``reasoning_effort="none"`` to switch it off.

        Note the consequences of thinking mode, which the API documents:

        * ``temperature`` is accepted but has **no effect**;
        * ``top_p`` works but is raised to ``THINKING_TOP_P_FLOOR`` (0.95) if lower;
        * when ``tools`` is present, ``reasoning_content`` from every prior turn
          **must** be echoed back or the request 400s — which is why
          :func:`sanitize_messages` keeps it by default.

        ``user_id`` isolates content-safety, KV-cache, and scheduling for a given
        end user.  Charset ``[a-zA-Z0-9\\-_]``, max 512 chars.
        Which of those fields actually go on the wire depends on the provider's
        ``apiFlavor``: ``deepseek`` (the default) sends them all, ``openai`` drops
        the DeepSeek-only ones entirely, whatever the caller asked for.
        """
        spec = self.spec_for(model)

        # ``reasoning_content`` is replayed to DeepSeek (mandatory with tools) and
        # stripped for everyone else — see ``self.replays_reasoning``.
        clean_messages = (
            sanitize_messages(
                messages, context=f"model={spec.id}",
                keep_reasoning=self.replays_reasoning,
            )
            if sanitize
            else [dict(m) for m in messages]
        )

        # Provider-wide defaults are the *base* layer, so an explicit argument
        # always wins.  (Merging them last, as this once did, silently discarded
        # a caller's `reasoning_effort="none"` whenever the provider config also
        # set an effort.)
        payload: dict[str, Any] = {**self.defaults, "model": spec.id, "messages": clean_messages}

        # The flavour is authoritative, not the config: a provider whose
        # ``defaults`` block still carries DeepSeek fields (as providers.json
        # does) must not leak them to an endpoint that cannot parse them.
        # ``extra_body`` below stays the deliberate escape hatch.
        if not self.sends_reasoning_fields:
            payload.pop("thinking", None)
            payload.pop("reasoning_effort", None)
        if not self.sends_user_id:
            payload.pop("user_id", None)

        if stream:
            payload["stream"] = True
            if self.include_usage:
                # Without this the final chunk carries no `usage` object.
                payload["stream_options"] = {"include_usage": True}

        tools_attached = False
        if tools is not None:
            schemas = tools.schemas() if isinstance(tools, ToolRegistry) else list(tools)
            if schemas:
                payload["tools"] = schemas
                payload["tool_choice"] = tool_choice or "auto"
                tools_attached = True

        # ── thinking mode ──
        # DeepSeek-only.  With ``apiFlavor: "openai"`` neither key may appear in
        # the payload, even if the caller passed ``thinking=True`` or an effort.
        effort = reasoning_effort
        if effort is not None:
            effort = _normalise_reasoning_effort(effort)
        if thinking is False and effort is None:
            effort = "none"
        if self.sends_reasoning_fields:
            if effort is not None:
                payload["reasoning_effort"] = effort
                if effort == "none":
                    payload["thinking"] = {"type": "disabled"}
                else:
                    payload["thinking"] = {"type": "enabled"}
            elif thinking is True:
                payload["thinking"] = {"type": "enabled"}

        # Reasoning content is mandatory once tools are in play. Re-sanitise with
        # knowledge of that so a caller who pre-sanitised cannot trip the 400.
        # Other endpoints do not want it replayed, so the flavour decides.
        if tools_attached:
            payload["messages"] = sanitize_messages(
                payload["messages"] or [], context=f"model={spec.id}",
                keep_reasoning=self.replays_reasoning,
            )
            if not any(_has_reasoning(m) for m in payload["messages"] if m.get("role") == "assistant"):
                logger.debug(
                    "tool call without any assistant reasoning_content; the API only "
                    "requires it on turns that produced one, so this is fine if none did"
                )

        # Only DeepSeek has an explicit thinking mode; without the reasoning
        # fields the endpoint is a plain completion server, so a top_p floor (and
        # the deprecated-parameter warnings) do not apply.
        thinking_on = self.sends_reasoning_fields and (
            payload.get("reasoning_effort", spec.default_reasoning_effort) != "none"
        )
        if thinking_on:
            for name in DEPRECATED_PARAMS:
                if name in payload:
                    # Accepted for OpenAI compatibility but silently ignored.
                    logger.debug("%s has no effect while thinking mode is enabled", name)

        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            # The server floors top_p at 0.95 in thinking mode; doing it here too
            # keeps the echoed request honest about what will actually be used.
            payload["top_p"] = max(top_p, THINKING_TOP_P_FLOOR) if thinking_on else top_p
        if max_tokens is not None:
            payload["max_tokens"] = max(1, min(int(max_tokens), spec.max_output_ceiling))
        if stop is not None:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if seed is not None:
            payload["seed"] = seed
        if user_id is not None and self.sends_user_id:
            payload["user_id"] = _clean_user_id(user_id)

        # `extra_body` is the deliberate escape hatch: it may override anything,
        # including the fields above.  `self.defaults` was already merged as the
        # base layer at the top of this method, so it loses to explicit arguments.
        if extra_body:
            payload.update(extra_body)

        # Guard against a caller overriding the message list through extra_body.
        payload["model"] = spec.id
        payload["messages"] = clean_messages if not tools_attached else payload["messages"]
        return payload

    # ── transport ──

    def _headers(self, *, stream: bool) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        if stream:
            headers["Accept"] = "text/event-stream"
            # Disable intermediary buffering so deltas are not held back.
            headers["X-Accel-Buffering"] = "no"
        else:
            headers["Accept"] = "application/json"
        return headers

    async def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            delay = min(retry_after, 30.0)
        else:
            delay = min(self.retry_backoff * (2 ** attempt), 20.0)
            delay *= 0.5 + random.random()  # jitter, to avoid synchronised retries
        logger.debug("retrying in %.2fs (attempt %d)", delay, attempt + 1)
        await asyncio.sleep(delay)

    def _wrap_connection_error(self, exc: Exception, url: str) -> ConnectionFailure:
        if isinstance(exc, httpx.ConnectTimeout):
            reason = "connection timed out"
        elif isinstance(exc, httpx.ReadTimeout):
            reason = "the endpoint stopped sending data"
        elif isinstance(exc, httpx.ConnectError):
            reason = "could not connect (DNS, TLS, proxy, or offline)"
        else:
            reason = f"{type(exc).__name__}: {exc}"
        return ConnectionFailure(f"Request to {url} failed — {reason}")

    # ── chat (non-streaming) ──

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        timeout: httpx.Timeout | float | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a completion request and return the whole response.

        Accepts every keyword of :meth:`build_payload` plus ``extra_body``.
        """
        payload = self.build_payload(messages, model=model, stream=False, **kwargs)
        url = self.endpoint(payload["model"])
        headers = self._headers(stream=False)

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            try:
                response = await self._http.post(
                    url, json=payload, headers=headers, timeout=timeout or self._timeout
                )
            except httpx.HTTPError as exc:
                last_error = self._wrap_connection_error(exc, url)
                if attempt >= self.max_retries:
                    break
                await self._sleep_before_retry(attempt, None)
                attempt += 1
                continue

            if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                logger.debug("HTTP %d from %s; retrying", response.status_code, url)
                await self._sleep_before_retry(attempt, _retry_after(response))
                attempt += 1
                continue

            raise_for_response(response, url=url)

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise DeepSeekError(
                    f"Endpoint returned non-JSON body: {response.text[:300]!r}",
                    status=response.status_code,
                ) from exc

            return ChatResponse.from_dict(data)

        raise RetryExhausted(
            f"Giving up after {attempt} attempt(s): {last_error}",
            cause=last_error, attempts=attempt,
        )

    # ── chat (streaming) ──

    def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        timeout: httpx.Timeout | float | None = None,
        on_raw_chunk: Any = None,
        **kwargs: Any,
    ) -> ChatStream:
        """Return a :class:`ChatStream` context manager.

        The request is *not* sent until you enter the context manager, so no
        retry wrapper is needed here — the caller sees a clean exception instead.

        ::

            async with client.stream(msgs) as stream:
                async for event in stream:
                    ...
                print(stream.response.text)
        """
        payload = self.build_payload(messages, model=model, stream=True, **kwargs)
        return ChatStream(
            self._http,
            self.endpoint(payload["model"]),
            payload,
            self._headers(stream=True),
            timeout=timeout or self._stream_timeout,
            on_raw_chunk=on_raw_chunk,
        )

    async def iter_text(
        self, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Yield only the visible answer deltas — the simplest possible stream."""
        async with self.stream(messages, **kwargs) as stream:
            async for event in stream:
                if event.type == "content":
                    yield event.text

    # ── convenience ──

    async def complete(
        self,
        prompt: str | Sequence[Mapping[str, Any]],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """One-shot helper: send a prompt, get the text back."""
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = list(prompt)

        response = await self.chat(messages, model=model, **kwargs)
        return response.text

    async def run_tools(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
        *,
        model: str | None = None,
        max_steps: int = 8,
        **chat_kwargs: Any,
    ) -> ChatResponse:
        """Agent loop: call, execute tools, re-call until the model answers."""
        return await run_tool_loop(
            self, messages, registry, model=model, max_steps=max_steps, **chat_kwargs
        )

    # ── introspection ──

    def models(self) -> list[ModelSpec]:
        """Models declared in the provider config."""
        return list(self.provider.models) if self.provider else [self.spec_for(None)]

    def count_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        """Rough token estimate for a message list (see :func:`estimate_tokens`)."""
        return estimate_tokens(messages)

    def context_usage(self, messages: Sequence[Mapping[str, Any]], model: str | None = None) -> float:
        """Fraction of the input window used, clamped to ``[0, 1]``."""
        spec = self.spec_for(model)
        if spec.max_input_tokens <= 0:
            return 0.0
        return min(1.0, self.count_tokens(messages) / spec.max_input_tokens)

    def check_context(self, messages: Sequence[Mapping[str, Any]], model: str | None = None) -> None:
        """Raise :class:`ContextLengthError` before wasting a round trip."""
        spec = self.spec_for(model)
        used = self.count_tokens(messages)
        budget = int(spec.max_input_tokens * 0.95)
        if spec.max_input_tokens and used > budget:
            raise ContextLengthError(
                f"Estimated {used} tokens exceeds the {spec.max_input_tokens}-token input "
                f"window of {spec.id!r}. Drop or summarise older messages.",
            )

    async def list_remote_models(self) -> list[dict[str, Any]]:
        """GET ``/models`` from the endpoint. Not all compatible servers implement it."""
        base = self.base_url or ""
        url = f"{base.rstrip('/')}/models"
        try:
            response = await self._http.get(url, headers=self._headers(stream=False), timeout=30.0)
        except httpx.HTTPError as exc:
            raise self._wrap_connection_error(exc, url) from exc
        raise_for_response(response, url=url)
        data = response.json()
        return list(data.get("data") or [])

    async def balance(self) -> dict[str, Any]:
        """DeepSeek's ``/user/balance`` probe. Returns the raw payload."""
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/user/balance"
        try:
            response = await self._http.get(url, headers=self._headers(stream=False), timeout=30.0)
        except httpx.HTTPError as exc:
            raise self._wrap_connection_error(exc, url) from exc
        raise_for_response(response, url=url)
        return response.json()

    async def health(self) -> dict[str, Any]:
        """Cheap liveness probe: does the endpoint answer an authenticated request?"""
        try:
            models = await self.list_remote_models()
            return {"ok": True, "models": [m.get("id") for m in models]}
        except DeepSeekError as exc:
            # /models is optional on some gateways; fall back to a tiny completion.
            try:
                response = await self.chat(
                    [{"role": "user", "content": "ping"}], max_tokens=1, sanitize=False
                )
                return {"ok": True, "models": [], "probe": response.finish_reason}
            except DeepSeekError as inner:
                return {"ok": False, "error": str(inner), "models_error": str(exc)}
