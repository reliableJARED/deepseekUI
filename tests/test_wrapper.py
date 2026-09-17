"""Offline tests for :mod:`deepseek_client`.

No network access: streaming is driven through ``httpx.MockTransport`` with a
byte stream we control, so chunk boundaries can be split mid-event on purpose.
Run with ``py -m pytest tests -q`` from the project root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek_client import (  # noqa: E402
    BadRequestError,
    ConfigError,
    ContextLengthError,
    DeepSeekClient,
    ProvidersConfig,
    RateLimitError,
    SSEDecoder,
    ToolRegistry,
    ToolResult,
    assistant,
    image_data,
    image_file,
    load_env_file,
    merge_text,
    sanitize_messages,
    system,
    text,
    tool,
    user,
)
from deepseek_client.streaming import DeltaAccumulator  # noqa: E402
from deepseek_client.config import Provider  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def sse(*chunks: dict) -> bytes:
    """Encode dicts as an SSE byte stream, terminated by ``[DONE]``."""
    body = b""
    for chunk in chunks:
        body += f"data: {json.dumps(chunk)}\n\n".encode()
    return body + b"data: [DONE]\n\n"


def chunk(content: str = "", **extra) -> dict:
    delta: dict = {"content": content} if content else {}
    delta.update(extra.pop("delta", {}))
    return {"id": "chatcmpl-1", "model": "deepseek-flash", "created": 1,
            "choices": [{"index": 0, "delta": delta, "finish_reason": extra.pop("finish_reason", None)}],
            **extra}


class ChunkedStream(httpx.AsyncByteStream):
    """A response body delivered in caller-controlled pieces."""

    def __init__(self, pieces: list[bytes]) -> None:
        self._pieces = pieces

    async def __aiter__(self):
        for piece in self._pieces:
            yield piece


def streaming_client(pieces: list[bytes]) -> DeepSeekClient:
    """A client whose transport replays ``pieces`` as an SSE response."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedStream(pieces),
            request=request,
        )

    return DeepSeekClient(
        api_key="test-key",
        base_url="https://api.test",
        model="deepseek-flash",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


# ── SSE decoding ──────────────────────────────────────────────────────────────

def test_sse_decoder_handles_split_chunks():
    """An event split across three network reads must decode exactly once."""
    decoder = SSEDecoder()
    assert decoder.feed('data: {"a"') == []
    assert decoder.feed(': 1}') == []
    events = decoder.feed("\n\n")
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'


def test_sse_decoder_multiline_data_and_comments():
    decoder = SSEDecoder()
    events = decoder.feed(": keep-alive\n\ndata: line1\ndata: line2\n\n")
    assert len(events) == 1
    assert events[0].data == "line1\nline2"   # joined with newline, per spec


def test_sse_decoder_crlf():
    decoder = SSEDecoder()
    events = decoder.feed("data: hello\r\n\r\n")
    assert [e.data for e in events] == ["hello"]


def test_sse_decoder_close_flushes_unterminated_event():
    decoder = SSEDecoder()
    assert decoder.feed("data: trailing") == []
    assert [e.data for e in decoder.close()] == ["trailing"]


# ── delta accumulation ────────────────────────────────────────────────────────

def test_accumulator_concatenates_content_and_usage():
    acc = DeltaAccumulator()
    acc.feed(chunk("Hello"))
    acc.feed(chunk(" world"))
    acc.feed({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                       "total_tokens": 12}})

    response = acc.build()
    assert response.text == "Hello world"
    assert response.usage.total_tokens == 12
    assert response.model == "deepseek-flash"


def test_accumulator_reassembles_fragmented_tool_call():
    """Tool arguments arrive as JSON split across chunks — the classic bug source."""
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [
        {"index": 0, "id": "call_abc", "function": {"name": "resize", "arguments": '{"pa'}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"index": 0, "function": {"arguments": 'th": "a.png", "w'}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"index": 0, "function": {"arguments": 'idth": 64}'}}
    ]}))

    calls = acc.build().tool_calls
    assert len(calls) == 1
    assert calls[0].id == "call_abc"
    assert calls[0].name == "resize"
    assert calls[0].parsed_arguments == {"path": "a.png", "width": 64}
    assert calls[0].arguments_valid


def test_accumulator_separates_parallel_tool_calls():
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [
        {"index": 0, "id": "c0", "function": {"name": "a", "arguments": "{}"}},
        {"index": 1, "id": "c1", "function": {"name": "b", "arguments": "{}"}},
    ]}))
    calls = acc.build().tool_calls
    assert [c.name for c in calls] == ["a", "b"]


def test_accumulator_keeps_reasoning_separate():
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"reasoning_content": "think..."}))
    acc.feed(chunk(delta={"content": "answer"}))
    response = acc.build()
    assert response.reasoning == "think..."
    assert response.text == "answer"


def test_truncated_arguments_report_invalid():
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [
        {"index": 0, "id": "c", "function": {"name": "x", "arguments": '{"a": '}}
    ]}))
    call = acc.build().tool_calls[0]
    assert not call.arguments_valid
    assert call.parsed_arguments == {}      # must not raise


# ── streaming over the transport ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_yields_events_and_assembles_response():
    pieces = [
        b'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    client = streaming_client(pieces)
    seen: list[str] = []

    async with client.stream([user("hi")]) as stream:
        async for event in stream:
            if event.type == "content":
                seen.append(event.text)
        assert stream.response is not None
        assert stream.response.text == "Hello"
        assert stream.response.finish_reason == "stop"

    assert seen == ["Hel", "lo"]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_split_mid_event_across_reads():
    """The same response chopped at awkward byte offsets must still parse."""
    full = sse(
        chunk("A"),
        chunk("B"),
    )
    # Slice every 7 bytes to guarantee events straddle read boundaries.
    pieces = [full[i:i + 7] for i in range(0, len(full), 7)]

    client = streaming_client(pieces)
    async with client.stream([user("hi")]) as stream:
        async for _ in stream:
            pass
        assert stream.response is not None
        assert stream.response.text == "AB"
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_survives_early_break():
    """Breaking out of the loop must still leave a usable partial response."""
    pieces = [
        b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"never seen"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    client = streaming_client(pieces)

    async with client.stream([user("hi")]) as stream:
        async for event in stream:
            if event.type == "content":
                break  # cancel generation here
        # Still mid-stream, so nothing is frozen yet.
        assert stream.response is None
    # Leaving the context manager finalises whatever was received.
    assert stream.response is not None
    assert stream.response.text == "partial"
    assert not stream.response.tool_calls
    await client.aclose()


# ── non-streaming + error mapping ─────────────────────────────────────────────

def response_client(status: int, payload: dict | str) -> DeepSeekClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return httpx.Response(status, content=body.encode(), request=request)

    return DeepSeekClient(
        api_key="k", base_url="https://api.test", model="deepseek-flash",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_chat_parses_response_and_usage():
    client = response_client(200, {
        "id": "x", "model": "deepseek-flash", "created": 5,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                  "prompt_cache_hit_tokens": 2},
    })
    response = await client.chat([user("hi")])
    assert response.text == "ok"
    assert response.usage.total_tokens == 4
    assert response.usage.prompt_cache_hit_tokens == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_auth_error_is_typed():
    client = response_client(401, {"error": {"message": "Invalid API key"}})
    with pytest.raises(Exception) as info:
        await client.chat([user("hi")])
    assert type(info.value).__name__ == "AuthenticationError"
    assert "Invalid API key" in str(info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_context_overflow_maps_to_context_length_error():
    client = response_client(400, {"error": {
        "message": "This model's maximum context length is 65536 tokens"
    }})
    with pytest.raises(ContextLengthError):
        await client.chat([user("hi")])
    await client.aclose()


@pytest.mark.asyncio
async def test_generic_400_is_bad_request():
    client = response_client(400, {"error": {"message": "messages[1] is invalid"}})
    with pytest.raises(BadRequestError):
        await client.chat([user("hi")])
    await client.aclose()


@pytest.mark.asyncio
async def test_429_retry_after_is_captured():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b'{"error":{"message":"slow down"}}',
                              headers={"retry-after": "2"}, request=request)

    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RateLimitError) as info:
        await client.chat([user("hi")])
    assert info.value.retry_after == 2.0
    assert info.value.retryable
    await client.aclose()


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, content=b"upstream down", request=request)
        return httpx.Response(200, content=json.dumps({
            "choices": [{"index": 0, "message": {"content": "recovered"},
                         "finish_reason": "stop"}]
        }).encode(), request=request)

    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        max_retries=2, retry_backoff=0.0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert (await client.chat([user("hi")])).text == "recovered"
    assert calls["n"] == 2
    await client.aclose()


# ── message sanitisation ──────────────────────────────────────────────────────

def test_sanitize_drops_orphaned_tool_result():
    messages = [
        user("hi"),
        {"role": "tool", "tool_call_id": "gone", "content": "stale"},
    ]
    cleaned = sanitize_messages(messages)
    assert [m["role"] for m in cleaned] == ["user"]


def test_sanitize_keeps_matched_tool_result():
    messages = [
        user("hi"),
        assistant(tool_calls=[{"id": "c1", "type": "function",
                               "function": {"name": "f", "arguments": "{}"}}]),
        tool("c1", "result"),
    ]
    cleaned = sanitize_messages(messages)
    assert [m["role"] for m in cleaned] == ["user", "assistant", "tool"]


def test_sanitize_drops_malformed_tool_call_and_its_result():
    messages = [
        assistant(tool_calls=[{"id": "c1", "type": "function",
                               "function": {"name": "f", "arguments": '{"broken'}}]),
        tool("c1", "never happened"),
    ]
    assert sanitize_messages(messages) == []


def test_sanitize_keeps_prose_when_tool_calls_are_malformed():
    messages = [
        assistant("here is my answer",
                  tool_calls=[{"id": "c1", "type": "function",
                               "function": {"name": "f", "arguments": "{oops"}}]),
    ]
    cleaned = sanitize_messages(messages)
    assert len(cleaned) == 1
    assert cleaned[0]["content"] == "here is my answer"
    assert "tool_calls" not in cleaned[0]


def test_sanitize_drops_empty_assistant_turn():
    assert sanitize_messages([assistant()]) == []


def test_sanitize_strips_internal_keys_but_keeps_reasoning_content():
    """``reasoning_content`` is a real API field and MUST survive sanitisation.

    When a request carries ``tools``, the API requires the chain of thought from
    every prior assistant turn to be echoed back, and returns 400 otherwise.  It
    is ignored (harmlessly) when ``tools`` is absent — so keeping it is correct in
    both cases, and the old behaviour of stripping it was a latent 400.
    """
    cleaned = sanitize_messages([
        {"role": "assistant", "content": "x", "reasoning": "secret",
         "_usage": {"a": 1}, "_timings": {}, "_is_summary": True}
    ])
    # The `reasoning` alias is normalised to the real field name; `_`-keys are gone.
    assert cleaned[0] == {
        "role": "assistant", "content": "x", "reasoning_content": "secret"
    }


def test_sanitize_can_drop_reasoning_when_asked():
    cleaned = sanitize_messages(
        [{"role": "assistant", "content": "x", "reasoning_content": "secret"}],
        keep_reasoning=False,
    )
    assert cleaned[0] == {"role": "assistant", "content": "x"}


def test_reasoning_only_belongs_on_assistant_turns():
    cleaned = sanitize_messages([
        {"role": "user", "content": "hi", "reasoning_content": "leaked"},
    ])
    assert cleaned[0] == {"role": "user", "content": "hi"}


def test_sanitize_drops_images_from_non_user_roles():
    """Images outside a user message are a hard 400, so drop the block and keep text."""
    from deepseek_client.messages import image_url, text
    cleaned = sanitize_messages([
        {"role": "system", "content": [text("be helpful"), image_url("data:image/png;base64,AAA")]},
        {"role": "assistant", "content": [image_url("data:image/png;base64,AAA"), text("look")]},
        {"role": "user", "content": [text("hi"), image_url("data:image/png;base64,AAA")]},
    ])
    assert [b["type"] for b in cleaned[0]["content"]] == ["text"]
    assert [b["type"] for b in cleaned[1]["content"]] == ["text"]
    # The user turn is untouched.
    assert [b["type"] for b in cleaned[2]["content"]] == ["text", "image_url"]


# ── content builders ──────────────────────────────────────────────────────────

def test_single_text_block_collapses_to_string():
    from deepseek_client.messages import content
    assert content(text("hello")) == "hello"


def test_mixed_blocks_stay_a_list():
    from deepseek_client.messages import content
    result = content(text("look"), image_data("AAAA", "image/png"))
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_data_passes_through_existing_data_uri():
    block = image_data("data:image/png;base64,AAAA")
    assert block["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_image_file_reads_and_encodes(tmp_path: Path):
    path = tmp_path / "dot.png"
    # 1x1 transparent PNG
    path.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    ))
    block = image_file(path)
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_merge_text_flattens_blocks():
    assert merge_text([text("a"), image_data("AA"), text("b")]) == "a [image] b"


def test_assistant_omits_empty_content():
    msg = assistant(tool_calls=[{"id": "c", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}])
    assert "content" not in msg          # servers reject content:"" here
    assert msg["role"] == "assistant"


# ── config ────────────────────────────────────────────────────────────────────

def test_env_file_parsing(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "export PLAIN=value\n"
        'QUOTED="has spaces"\n'
        "SINGLE='single'\n"
        "TRAILING=abc  # trailing comment\n"
        "EMPTY=\n"
        "BASE_URL=https://api.test\n"
        "DERIVED=${BASE_URL}/v1\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    parsed = load_env_file(env_file, environ=env)

    assert parsed["PLAIN"] == "value"
    assert parsed["QUOTED"] == "has spaces"
    assert parsed["SINGLE"] == "single"
    assert parsed["TRAILING"] == "abc"
    assert parsed["EMPTY"] == ""
    assert parsed["DERIVED"] == "https://api.test/v1"   # ${VAR} resolved


def test_env_file_does_not_mutate_real_environ(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("SHOULD_NOT_LEAK=1\n", encoding="utf-8")
    load_env_file(env_file, environ={})
    assert "SHOULD_NOT_LEAK" not in __import__("os").environ


def test_providers_config_loads_and_interpolates(tmp_path: Path):
    (tmp_path / "providers.json").write_text(json.dumps({
        "providers": [{
            "name": "DeepSeek",
            "vendor": "customendpoint",
            "apiKey": "${TEST_DS_KEY}",
            "apiType": "chat-completions",
            "models": [{
                "id": "deepseek-flash", "name": "deepseek-flash",
                "url": "https://api.deepseek.com",
                "toolCalling": True, "vision": True,
                "maxInputTokens": 128000, "maxOutputTokens": 16000,
            }],
        }]
    }), encoding="utf-8")
    (tmp_path / ".env").write_text("TEST_DS_KEY=sk-secret\n", encoding="utf-8")

    config = ProvidersConfig.load(tmp_path / "providers.json")

    provider = config.get("DeepSeek")
    assert provider.api_key == "sk-secret"           # placeholder resolved
    assert provider.is_configured
    assert provider.vendor == "customendpoint"

    spec = provider.model("deepseek-flash")
    assert spec.tool_calling and spec.vision
    assert spec.max_input_tokens == 128_000
    assert spec.max_output_tokens == 16_000
    # No apiPath was configured, so the DeepSeek default applies.  `/v1` is also
    # accepted by the API for OpenAI-SDK compatibility, but is not the default.
    assert provider.endpoint("deepseek-flash") == "https://api.deepseek.com/chat/completions"
    # Thinking mode is on by default server-side; the spec mirrors that.
    assert spec.thinking is True
    assert spec.default_reasoning_effort == "high"


def test_missing_env_var_raises_config_error(tmp_path: Path):
    (tmp_path / "providers.json").write_text(
        '{"providers":[{"name":"X","apiKey":"${DEFINITELY_NOT_SET_VAR}"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        ProvidersConfig.load(tmp_path / "providers.json", env_file=None, environ={})
    assert "DEFINITELY_NOT_SET_VAR" in str(info.value)


def test_unfilled_api_key_placeholder_is_not_configured(tmp_path: Path):
    """The exact 'PULL FROM ENV FILE' string from the pasted config."""
    (tmp_path / "providers.json").write_text(json.dumps({
        "providers": [{"name": "DeepSeek", "apiKey": "PULL FROM ENV FILE "}]
    }), encoding="utf-8")
    config = ProvidersConfig.load(tmp_path / "providers.json", env_file=None, environ={})
    assert not config.get(None).is_configured


def test_unsupported_api_type_rejected(tmp_path: Path):
    (tmp_path / "providers.json").write_text(
        '{"providers":[{"name":"X","apiKey":"k","apiType":"responses"}]}', encoding="utf-8"
    )
    with pytest.raises(ConfigError) as info:
        ProvidersConfig.load(tmp_path / "providers.json", env_file=None, environ={})
    assert "responses" in str(info.value)


def test_bare_provider_object_is_accepted(tmp_path: Path):
    """A single provider object, not wrapped in {"providers": [...]}."""
    (tmp_path / "providers.json").write_text(json.dumps({
        "name": "Solo", "apiKey": "k", "apiType": "chat-completions",
        "models": [{"id": "m1"}],
    }), encoding="utf-8")
    config = ProvidersConfig.load(tmp_path / "providers.json", env_file=None, environ={})
    assert len(config) == 1 and config.get(None).name == "Solo"


def test_from_env_builds_provider():
    config = ProvidersConfig.from_env(
        "DS", environ={"DS_API_KEY": "sk-x", "DS_MODEL": "deepseek-flash",
                       "DS_BASE_URL": "https://api.deepseek.com"},
        load_dotenv=False,
    )
    provider = config.get(None)
    assert provider.api_key == "sk-x"
    assert provider.model(None).id == "deepseek-flash"


def test_from_env_without_key_raises():
    with pytest.raises(ConfigError):
        ProvidersConfig.from_env("NOPE", environ={}, load_dotenv=False)


def test_endpoint_dedupes_double_v1():
    """base_url ending in /v1 plus a /v1/... path must not produce /v1/v1/."""
    client = DeepSeekClient(
        api_key="k", base_url="https://api.test/v1", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    assert client.endpoint("m") == "https://api.test/v1/chat/completions"


def test_client_requires_api_key():
    with pytest.raises(ConfigError):
        DeepSeekClient(api_key="", base_url="https://api.test")


def test_client_from_providers_file(tmp_path: Path):
    (tmp_path / "providers.json").write_text(json.dumps({
        "providers": [{
            "name": "DeepSeek", "apiKey": "${K}", "apiType": "chat-completions",
            "models": [{"id": "deepseek-flash", "url": "https://api.deepseek.com",
                        "toolCalling": True, "vision": True,
                        "maxInputTokens": 128000, "maxOutputTokens": 16000}],
        }]
    }), encoding="utf-8")

    client = DeepSeekClient.from_providers(
        tmp_path / "providers.json", environ={"K": "sk-secret"}, env_file=None
    )
    assert client.api_key == "sk-secret"
    assert client.model == "deepseek-flash"
    assert client.supports("tools")
    assert client.supports("vision")
    assert client.spec_for().max_output_tokens == 16_000


# ── payload building ──────────────────────────────────────────────────────────

def test_payload_includes_tools_and_stream_options():
    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    registry = ToolRegistry()
    registry.register(lambda **kw: "", name="noop", description="does nothing",
                      properties={"x": {"type": "string"}}, required=["x"])

    payload = client.build_payload([user("hi")], stream=True, tools=registry, temperature=0.2)

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["tools"][0]["function"]["name"] == "noop"
    assert payload["tool_choice"] == "auto"
    assert payload["temperature"] == 0.2


def test_payload_sanitizes_history():
    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    payload = client.build_payload([
        user("hi"),
        assistant(),                                          # empty — must go
        {"role": "tool", "tool_call_id": "ghost", "content": "x"},  # orphaned — must go
    ])
    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_context_usage_and_guard():
    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m", max_input_tokens=100,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    big = [system("x" * 1000)]
    assert client.context_usage(big) == 1.0          # clamped
    with pytest.raises(ContextLengthError):
        client.check_context(big)


# ── thinking mode ─────────────────────────────────────────────────────────────

def _payload_client(**kwargs):
    return DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        **kwargs,
    )


def test_thinking_mode_is_opt_in_at_payload_level():
    """Sending nothing must not disable thinking — the server defaults it ON."""
    payload = _payload_client().build_payload([user("hi")])
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


def test_thinking_can_be_disabled_two_equivalent_ways():
    by_flag = _payload_client().build_payload([user("hi")], thinking=False)
    by_effort = _payload_client().build_payload([user("hi")], reasoning_effort="none")
    for payload in (by_flag, by_effort):
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["reasoning_effort"] == "none"


def test_reasoning_effort_aliases_normalise():
    for alias, canonical in (("minimal", "low"), ("medium", "high"),
                             ("xhigh", "high"), ("ultra", "max")):
        payload = _payload_client().build_payload([user("hi")], reasoning_effort=alias)
        assert payload["reasoning_effort"] == canonical, alias


def test_top_p_is_floored_while_thinking():
    """The API raises sub-0.95 top_p in thinking mode; mirror that in the payload."""
    thinking = _payload_client().build_payload([user("hi")], top_p=0.1)
    assert thinking["top_p"] == 0.95

    off = _payload_client().build_payload([user("hi")], top_p=0.1, thinking=False)
    assert off["top_p"] == 0.1


def test_max_tokens_is_clamped_to_the_documented_ceiling():
    payload = _payload_client().build_payload([user("hi")], max_tokens=10_000_000)
    assert payload["max_tokens"] == 393_216


def test_user_id_is_sanitised_to_the_documented_charset():
    payload = _payload_client().build_payload([user("hi")], user_id="a b/c!d")
    assert payload["user_id"] == "a-b-c-d"
    assert len(_payload_client().build_payload([user("hi")], user_id="z" * 900)["user_id"]) == 512


def test_provider_defaults_do_not_override_explicit_arguments():
    """Regression: provider-level `defaults` used to be merged *last*.

    With `reasoning_effort: "high"` in the provider config, a caller asking for
    ``reasoning_effort="none"`` had it silently clobbered and thinking stayed on.
    Explicit arguments must win; `extra_body` remains the deliberate escape hatch.
    """
    client = _payload_client(defaults={"reasoning_effort": "high",
                                       "thinking": {"type": "enabled"}})

    # Nothing passed -> the provider default stands.
    assert client.build_payload([user("hi")])["reasoning_effort"] == "high"

    # Explicitly disabling wins over the provider default.
    off = client.build_payload([user("hi")], reasoning_effort="none")
    assert off["reasoning_effort"] == "none"
    assert off["thinking"] == {"type": "disabled"}

    # extra_body can still override everything, deliberately.
    forced = client.build_payload([user("hi")], reasoning_effort="none",
                                 extra_body={"reasoning_effort": "max"})
    assert forced["reasoning_effort"] == "max"


def test_provider_defaults_apply_as_a_base_layer():
    client = _payload_client(defaults={"top_k": 40})
    assert client.build_payload([user("hi")])["top_k"] == 40


def test_tool_calls_force_reasoning_content_to_be_kept():
    """The whole point of keeping reasoning: a tool loop 400s without it."""
    client = _payload_client()
    registry = ToolRegistry()
    registry.register(lambda **kw: "", name="noop", description="d")

    call = {"id": "c1", "type": "function",
            "function": {"name": "noop", "arguments": "{}"}}
    payload = client.build_payload(
        [user("hi"), assistant(None, tool_calls=[call], reasoning="I should call noop")],
        tools=registry,
    )
    assistant_turn = payload["messages"][1]
    assert assistant_turn["reasoning_content"] == "I should call noop"


# ── image token accounting ────────────────────────────────────────────────────

def test_image_tokens_capped_and_scaled_by_area():
    from deepseek_client.messages import IMAGE_TOKEN_CEILING, image_tokens

    # Unknown dimensions assume the worst case rather than under-counting.
    assert image_tokens() == IMAGE_TOKEN_CEILING
    # A huge image costs no more than the ceiling — the API resizes it down.
    assert image_tokens(8000, 8000) == IMAGE_TOKEN_CEILING
    # A small image costs proportionally less.
    assert image_tokens(512, 512) < IMAGE_TOKEN_CEILING
    assert image_tokens(512, 512) > 0


def test_estimate_tokens_does_not_count_base64_payload():
    """The bug this guards: charging the raw base64 length overstates cost ~1e4x."""
    from deepseek_client.messages import image_data, IMAGE_TOKEN_CEILING, estimate_tokens

    blob = image_data("A" * 400_000, mime="image/jpeg")   # ~400 KB of base64
    total = estimate_tokens([user([text("describe"), blob])])
    assert total < 20_000, f"base64 leaked into the estimate: {total}"
    assert total >= IMAGE_TOKEN_CEILING


def test_estimate_tokens_charges_reasoning_text():
    from deepseek_client.messages import estimate_tokens

    without = estimate_tokens([assistant("x" * 350)])
    with_reasoning = estimate_tokens(
        [{"role": "assistant", "content": "x" * 350, "reasoning_content": "y" * 350}]
    )
    assert with_reasoning > without


def test_assistant_builder_emits_reasoning_content_not_reasoning():
    """The wire field is what matters — `reasoning` is only an input alias."""
    turn = assistant("hi", reasoning="because")
    assert turn["reasoning_content"] == "because"
    assert "reasoning" not in turn


def test_merge_reasoning_collects_only_assistant_turns():
    from deepseek_client.messages import merge_reasoning

    assert merge_reasoning([
        user("q"),
        {"role": "assistant", "content": "a", "reasoning_content": "one"},
        {"role": "assistant", "content": "b", "reasoning_content": "two"},
    ]) == "one\n\ntwo"


# ── vision block builders ─────────────────────────────────────────────────────

def test_files_api_blocks_are_mutually_exclusive():
    from deepseek_client.messages import image_file_data, image_file_id

    by_id = image_file_id("file-api-abc")
    assert by_id["file_id"] == "file-api-abc"
    assert "file_data" not in by_id

    by_data = image_file_data(b"\xff\xd8\xff", filename="x.jpg")
    assert by_data["file_data"].startswith("data:image/jpeg;base64,")
    assert "file_id" not in by_data

    # A data URI passed in is not double-wrapped.
    assert image_file_data("data:image/png;base64,AAA")["file_data"] == "data:image/png;base64,AAA"
    # An http URL is not silently turned into a bogus data URI.
    assert image_file_data("https://x.test/a.png")["type"] == "image_url"


# ── tools ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_dispatches_and_captures_errors():
    registry = ToolRegistry()

    @registry.register(name="echo", description="echo back")
    def echo(value: str = "") -> str:
        return f"got {value}"

    assert "echo" in registry
    assert registry.schemas()[0]["function"]["name"] == "echo"

    assert (await registry.call("echo", {"value": "hi"})).content == "got hi"
    # Unknown tool becomes an error result, not an exception.
    assert (await registry.call("nope", {})).is_error

    @registry.register(name="boom", description="raises")
    def boom() -> str:
        raise RuntimeError("kaboom")

    result = await registry.call("boom", {})
    assert result.is_error and "kaboom" in result.content


@pytest.mark.asyncio
async def test_registry_async_handler_and_block_results():
    registry = ToolRegistry()

    @registry.register(name="pic", description="returns an image")
    async def pic() -> ToolResult:
        return ToolResult(content=[text("here"), image_data("AAAA", "image/png")])

    result = await registry.call("pic", {})
    assert isinstance(result.content, list)
    assert result.content[1]["type"] == "image_url"


def test_tool_message_json_encodes_block_lists():
    msg = tool("call_1", [text("a"), image_data("AA")])
    assert msg["role"] == "tool"
    decoded = json.loads(msg["content"])
    assert decoded[0]["type"] == "text"


@pytest.mark.asyncio
async def test_tool_loop_drives_to_final_answer():
    """Two round trips: model asks for a tool, then answers."""
    turns = [
        json.dumps({  # first call: a tool call
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "c1", "type": "function",
                                                     "function": {"name": "add",
                                                                  "arguments": '{"a":2,"b":3}'}}]}}],
        }).encode(),
        json.dumps({  # second call: the answer
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "The answer is 5."}}],
        }).encode(),
    ]
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = turns[state["n"]]
        state["n"] += 1
        return httpx.Response(200, content=body, request=request)

    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    registry = ToolRegistry()

    @registry.register(name="add", description="add two numbers",
                       properties={"a": {"type": "integer"}, "b": {"type": "integer"}},
                       required=["a", "b"])
    def add(a: int, b: int) -> str:
        return str(a + b)

    messages: list[dict] = [user("what is 2+3?")]
    seen_tools: list[tuple[str, str]] = []

    response = await client.run_tools(
        messages, registry,
        on_tool=lambda call, result: seen_tools.append((call.name, result.content)),
    )

    assert response.text == "The answer is 5."
    assert seen_tools == [("add", "5")]
    # The transcript must record the tool call and its result for replay.
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    await client.aclose()


@pytest.mark.asyncio
async def test_tool_loop_echoes_reasoning_content_back():
    """Regression: without this the live API returns 400 on the second step.

    When a request carries `tools`, DeepSeek requires `reasoning_content` from
    every prior assistant turn to be sent back.  The chain that has to work is
    ``ChatMessage.reasoning`` -> ``to_dict()`` -> the next request body.
    """
    turns = [
        json.dumps({
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "reasoning_content": "I need to add two numbers.",
                                     "tool_calls": [{"id": "c1", "type": "function",
                                                     "function": {"name": "add",
                                                                  "arguments": '{"a":2,"b":3}'}}]}}],
        }).encode(),
        json.dumps({
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "5"}}],
        }).encode(),
    ]
    state = {"n": 0}
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = turns[state["n"]]
        state["n"] += 1
        return httpx.Response(200, content=body, request=request)

    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    registry = ToolRegistry()
    registry.register(lambda **kw: "5", name="add", description="add")

    messages: list[dict] = [user("what is 2+3?")]
    await client.run_tools(messages, registry)

    # The second request is the one that repeats the assistant turn.
    second = sent[1]
    assistant_turns = [m for m in second["messages"] if m.get("role") == "assistant"]
    assert assistant_turns, "no assistant turn was replayed"
    assert assistant_turns[0].get("reasoning_content") == "I need to add two numbers."
    # And the in-memory transcript agrees, so a caller can persist it verbatim.
    assert messages[1]["reasoning_content"] == "I need to add two numbers."
    await client.aclose()


@pytest.mark.asyncio
async def test_tool_loop_reports_invalid_arguments_to_model():
    """A truncated tool call must become an error result, not a crash."""
    turns = [
        json.dumps({"choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant",
                                             "tool_calls": [{"id": "c1", "type": "function",
                                                             "function": {"name": "f",
                                                                          "arguments": "{bad"}}]}}]}).encode(),
        json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}]}).encode(),
    ]
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = turns[min(state["n"], 1)]
        state["n"] += 1
        return httpx.Response(200, content=body, request=request)

    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    registry = ToolRegistry()
    registry.register(lambda **kw: "should not run", name="f", description="f")

    messages: list[dict] = [user("go")]
    response = await client.run_tools(messages, registry)

    assert response.text == "ok"
    assert "not valid JSON" in messages[2]["content"]
    await client.aclose()


# ── apiFlavor: request shaping for a plain OpenAI endpoint ────────────────────
#
# The default must stay `deepseek`: every assertion here about the default path is
# also a regression guard that an existing config sends exactly what it sent before.

def _flavored_client(flavor: str | None = None, **provider_extra):
    """A client whose provider declares ``apiFlavor`` (omitted -> not declared)."""
    data = {
        "name": "Local",
        "apiKey": "k",
        "apiType": "chat-completions",
        "models": [{"id": "m", "url": "https://api.test"}],
        **provider_extra,
    }
    if flavor is not None:
        data["apiFlavor"] = flavor
    provider = Provider.from_dict(data)
    return DeepSeekClient.from_provider(
        provider,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )


def _tools():
    registry = ToolRegistry()
    registry.register(lambda **kw: "", name="noop", description="d")
    return registry


def test_api_flavor_deepseek_sends_every_deepseek_field():
    """The default flavour is byte-for-byte the historical request shape."""
    client = _flavored_client("deepseek")
    payload = client.build_payload(
        [user("hi")], thinking=True, reasoning_effort="high", user_id="u 1",
    )
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["user_id"] == "u-1"                 # still _clean_user_id'd


def test_an_omitted_api_flavor_is_deepseek():
    client = _flavored_client()
    assert client.api_flavor == "deepseek"
    payload = client.build_payload([user("hi")], thinking=True, user_id="u1")
    assert "thinking" in payload and "user_id" in payload


def test_api_flavor_deepseek_keeps_provider_defaults_on_the_wire():
    """providers.json injects these through `defaults` — they must survive."""
    client = _flavored_client(
        "deepseek", defaults={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )
    payload = client.build_payload([user("hi")])
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"


def test_api_flavor_openai_omits_every_deepseek_only_field():
    """A strict OpenAI-compatible server rejects all three of these."""
    client = _flavored_client("openai")
    payload = client.build_payload(
        [user("hi")], thinking=True, reasoning_effort="high", user_id="u-1",
    )
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert "user_id" not in payload


def test_api_flavor_openai_drops_deepseek_defaults_too():
    """`defaults` is config, but the flavour is authoritative — nothing leaks."""
    client = _flavored_client(
        "openai", defaults={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )
    payload = client.build_payload([user("hi")], user_id="u-1")
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert "user_id" not in payload


def test_api_flavor_openai_does_not_replay_reasoning_content():
    call = {"id": "c1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}
    payload = _flavored_client("openai").build_payload(
        [user("hi"), assistant(None, tool_calls=[call], reasoning="I should call noop")],
        tools=_tools(),
    )
    turn = payload["messages"][1]
    assert turn["tool_calls"]                       # the tool call itself survives
    assert "reasoning_content" not in turn
    assert "reasoning" not in turn


def test_api_flavor_deepseek_still_replays_reasoning_content():
    call = {"id": "c1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}
    payload = _flavored_client("deepseek").build_payload(
        [user("hi"), assistant(None, tool_calls=[call], reasoning="I should call noop")],
        tools=_tools(),
    )
    assert payload["messages"][1]["reasoning_content"] == "I should call noop"


def test_an_unknown_api_flavor_is_rejected():
    with pytest.raises(ConfigError) as info:
        Provider.from_dict({
            "name": "X", "apiKey": "k", "apiFlavor": "gemini", "models": [{"id": "m"}],
        })
    assert "gemini" in str(info.value)


def test_api_flavor_round_trips_and_does_not_leak_into_extra():
    for spelling, value in (("apiFlavor", "OPENAI "), ("api_flavor", "OpenAI")):
        provider = Provider.from_dict({
            "name": "Local", "apiKey": "k", spelling: value, "models": [{"id": "m"}],
        })
        assert provider.api_flavor == "openai"      # normalised
        assert "apiFlavor" not in provider.extra
        assert "api_flavor" not in provider.extra
        assert provider.sends_reasoning_fields is False
        assert provider.sends_user_id is False
        assert provider.replays_reasoning is False

    default = Provider.from_dict({"name": "D", "apiKey": "k", "models": [{"id": "m"}]})
    assert default.api_flavor == "deepseek"
    assert default.sends_reasoning_fields and default.sends_user_id and default.replays_reasoning


def test_client_without_a_provider_falls_back_to_deepseek():
    client = DeepSeekClient(
        api_key="k", base_url="https://api.test", model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    assert client.api_flavor == "deepseek"
    assert client.replays_reasoning is True
    assert "thinking" in client.build_payload([user("hi")], thinking=True)


# ── tool calls that stream without an `index` ─────────────────────────────────

def test_indexless_tool_calls_stay_separate_when_they_carry_an_id():
    """`index` is optional; defaulting it to 0 used to concatenate two calls."""
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_a", "function": {"name": "resize", "arguments": '{"path":'}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_b", "function": {"name": "crop", "arguments": '{"path":'}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_a", "function": {"arguments": ' "a.png"}'}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_b", "function": {"arguments": ' "b.png"}'}}
    ]}))

    calls = acc.build().tool_calls
    assert [call.id for call in calls] == ["call_a", "call_b"]
    assert [call.name for call in calls] == ["resize", "crop"]
    assert calls[0].parsed_arguments == {"path": "a.png"}
    assert calls[1].parsed_arguments == {"path": "b.png"}
    assert all(call.arguments_valid for call in calls)


def test_indexless_tool_fragment_without_an_id_continues_the_previous_call():
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_a", "function": {"name": "resize", "arguments": '{"pat'}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_b", "function": {"name": "crop", "arguments": '{"pat'}}
    ]}))
    # Neither index nor id: this belongs to the call most recently opened.
    acc.feed(chunk(delta={"tool_calls": [{"function": {"arguments": 'h": "b.png"}'}}]}))

    calls = acc.build().tool_calls
    assert len(calls) == 2
    assert calls[1].id == "call_b"
    assert calls[1].parsed_arguments == {"path": "b.png"}
    assert calls[0].arguments == '{"pat'      # untouched


def test_an_index_less_call_after_an_indexed_one_gets_the_next_slot():
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [
        {"index": 0, "id": "call_a", "function": {"name": "a", "arguments": "{}"}}
    ]}))
    acc.feed(chunk(delta={"tool_calls": [
        {"id": "call_b", "function": {"name": "b", "arguments": "{}"}}
    ]}))

    calls = acc.build().tool_calls
    assert [call.index for call in calls] == [0, 1]
    assert calls[1].id == "call_b"


def test_a_fully_index_less_call_still_gets_an_id():
    """`call_{index}` remains the fallback, so downstream tool dispatch works."""
    acc = DeltaAccumulator()
    acc.feed(chunk(delta={"tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}))
    call = acc.build().tool_calls[0]
    assert call.id == "call_0"
    assert call.name == "x"
    assert call.arguments_valid
