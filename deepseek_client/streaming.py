"""Server-sent-event decoding and streamed-completion assembly.

Two levels of parsing live here:

* :class:`SSEDecoder` turns raw text chunks into SSE events (``data:`` framing,
  multi-line payloads, ``[DONE]`` sentinels, comments).
* :class:`DeltaAccumulator` turns each JSON chunk's ``delta`` into
  :class:`~deepseek_client.types.StreamEvent`s and reassembles the fragments —
  including tool calls, which stream as partial JSON across many chunks — into a
  complete :class:`~deepseek_client.types.ChatResponse`.

:class:`ChatStream` glues both to an ``httpx`` response as an async context
manager so the underlying connection closes even if the consumer breaks early.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from .errors import DeepSeekError, StreamError
from .types import ChatMessage, ChatResponse, StreamEvent, ToolCall, Usage

__all__ = ["SSEEvent", "SSEDecoder", "DeltaAccumulator", "ChatStream", "iter_sse_data"]

logger = logging.getLogger("deepseek_client.streaming")

DONE_SENTINEL = "[DONE]"


# ── SSE framing ───────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SSEEvent:
    """One dispatched ``text/event-stream`` event."""

    data: str
    event: str | None = None
    id: str | None = None


class SSEDecoder:
    """Incremental ``text/event-stream`` parser.

    Feed it whatever the transport produced; it returns whatever complete events
    that unlocked.  Nothing is emitted until a blank line terminates the event,
    so a partial chunk never yields a half-parsed payload.
    """

    __slots__ = ("_buffer", "_data", "_event", "_id")

    def __init__(self) -> None:
        self._buffer = ""
        self._data: list[str] = []
        self._event: str | None = None
        self._id: str | None = None

    def feed(self, chunk: str) -> list[SSEEvent]:
        if not chunk:
            return []
        self._buffer += chunk
        events: list[SSEEvent] = []
        while True:
            newline = self._buffer.find("\n")
            if newline == -1:
                break
            line = self._buffer[:newline]
            self._buffer = self._buffer[newline + 1:]
            event = self._line(line.rstrip("\r"))
            if event is not None:
                events.append(event)
        return events

    def close(self) -> list[SSEEvent]:
        """Flush a trailing line that never got its newline, then the event."""
        events: list[SSEEvent] = []
        if self._buffer:
            line = self._buffer.rstrip("\r")
            self._buffer = ""
            if line:
                event = self._line(line)
                if event is not None:
                    events.append(event)
        final = self._dispatch()
        if final is not None:
            events.append(final)
        return events

    # internals

    def _line(self, line: str) -> SSEEvent | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None  # comment / keep-alive

        field_name, sep, value = line.partition(":")
        if not sep:
            # "data" with no colon is valid and means an empty value.
            field_name, value = line, ""
        elif value.startswith(" "):
            value = value[1:]

        if field_name == "data":
            self._data.append(value)
        elif field_name == "event":
            self._event = value
        elif field_name == "id":
            self._id = value
        # "retry" and unknown fields are ignored.
        return None

    def _dispatch(self) -> SSEEvent | None:
        if not self._data and self._event is None:
            return None
        payload = "\n".join(self._data)
        event = SSEEvent(data=payload, event=self._event, id=self._id)
        self._data = []
        self._event = None
        self._id = None
        return event


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the ``data`` payload of every SSE event in an httpx response."""
    decoder = SSEDecoder()
    async for chunk in response.aiter_text():
        for event in decoder.feed(chunk):
            yield event.data
    for event in decoder.close():
        yield event.data


# ── delta accumulation ────────────────────────────────────────────────────────

@dataclass(slots=True)
class _ToolCallBuffer:
    """Mutable accumulator for one streamed tool call."""

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""

    def to_call(self) -> ToolCall:
        return ToolCall(
            id=self.id or f"call_{self.index}",
            name=self.name,
            arguments=self.arguments,
            index=self.index,
        )


class DeltaAccumulator:
    """Rebuild a complete response from streamed ``chat.completion.chunk`` objects."""

    __slots__ = (
        "content", "reasoning", "usage", "finish_reason",
        "model", "id", "created", "last_chunk", "_calls", "_last_index",
    )

    def __init__(self, model: str = "") -> None:
        self.content = ""
        self.reasoning = ""
        self.usage = Usage()
        self.finish_reason: str | None = None
        self.model = model
        self.id = ""
        self.created = 0
        self.last_chunk: dict[str, Any] = {}
        self._calls: dict[int, _ToolCallBuffer] = {}
        #: Index of the buffer most recently opened (or extended) — the target a
        #: continuation fragment with neither ``index`` nor ``id`` belongs to.
        self._last_index: int | None = None

    # ── feeding ──

    def feed(self, chunk: dict[str, Any]) -> list[StreamEvent]:
        """Absorb one decoded chunk; return the events it produced."""
        events: list[StreamEvent] = []
        self.last_chunk = chunk

        if chunk.get("id"):
            self.id = str(chunk["id"])
        if chunk.get("model"):
            self.model = str(chunk["model"])
        if chunk.get("created"):
            try:
                self.created = int(chunk["created"])
            except (TypeError, ValueError):
                pass

        # Some providers (and llama.cpp) put usage/timings on a trailing chunk
        # that carries no choices at all.
        if chunk.get("usage"):
            self.usage = Usage.from_dict(chunk["usage"])
            events.append(StreamEvent("usage", usage=self.usage, raw=chunk))

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                self.reasoning += reasoning
                events.append(StreamEvent("reasoning", text=reasoning, raw=chunk))

            piece = delta.get("content")
            if piece:
                self.content += piece
                events.append(StreamEvent("content", text=piece, raw=chunk))

            for part in delta.get("tool_calls") or []:
                started = self._merge_tool_call(part)
                if started is not None:
                    events.append(StreamEvent("tool_call", tool_call=started, raw=chunk))

            reason = choice.get("finish_reason")
            if reason:
                self.finish_reason = str(reason)
                events.append(StreamEvent("finish", finish_reason=self.finish_reason, raw=chunk))

        return events

    def _merge_tool_call(self, part: dict[str, Any]) -> ToolCall | None:
        """Merge a tool-call fragment. Returns the call only when it is newly opened.

        ``index`` says which call a fragment belongs to, but it is optional and
        several OpenAI-compatible servers omit it.  Coercing a missing index to
        ``0`` (as this once did) concatenated two separate calls into one, with
        mangled JSON arguments.
        """
        buffer, is_new = self._resolve_call(part)

        if part.get("id"):
            buffer.id = str(part["id"])

        function = part.get("function") or {}
        # Both fields stream in fragments, so append rather than assign.
        if function.get("name"):
            buffer.name += str(function["name"])
        if function.get("arguments"):
            buffer.arguments += str(function["arguments"])

        return buffer.to_call() if is_new else None

    def _resolve_call(self, part: dict[str, Any]) -> tuple[_ToolCallBuffer, bool]:
        """Find (or open) the buffer a fragment extends.

        Resolution order:

        1. an integer ``index`` — the fragment's own slot (the documented shape);
        2. no index but an ``id`` — the slot already carrying that id, else a new
           slot at ``max(existing) + 1``;
        3. neither — a continuation of the call most recently opened.
        """
        index = part.get("index")
        if isinstance(index, int):
            buffer = self._calls.get(index)
            if buffer is None:
                buffer = _ToolCallBuffer(index=index)
                self._calls[index] = buffer
                self._last_index = index
                return buffer, True
            self._last_index = index
            return buffer, False

        call_id = part.get("id")
        if call_id:
            wanted = str(call_id)
            for buffer in self._calls.values():
                if buffer.id == wanted:
                    self._last_index = buffer.index
                    return buffer, False
            index = max(self._calls, default=-1) + 1
            buffer = _ToolCallBuffer(index=index, id=wanted)
            self._calls[index] = buffer
            self._last_index = index
            return buffer, True

        # No index and no id: the arguments of the call just opened.
        if self._last_index is not None and self._last_index in self._calls:
            return self._calls[self._last_index], False
        buffer = _ToolCallBuffer(index=0)
        self._calls[0] = buffer
        self._last_index = 0
        return buffer, True

    # ── finishing ──

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [self._calls[i].to_call() for i in sorted(self._calls)]

    def build(self) -> ChatResponse:
        """Freeze the accumulated state into a :class:`ChatResponse`."""
        message = ChatMessage(
            role="assistant",
            content=self.content,
            reasoning=self.reasoning,
            tool_calls=self.tool_calls,
            finish_reason=self.finish_reason,
        )
        return ChatResponse(
            message=message,
            usage=self.usage,
            model=self.model,
            id=self.id,
            created=self.created,
            raw=self.last_chunk,
        )


# ── the context manager ───────────────────────────────────────────────────────

class ChatStream:
    """Async iterator over :class:`StreamEvent` that owns the HTTP response.

    Typical use::

        async with client.stream(messages, model="deepseek-flash") as stream:
            async for event in stream:
                if event.type == "content":
                    print(event.text, end="", flush=True)
            response = stream.response      # fully assembled

    Iterating to completion, breaking out early, or raising inside the body all
    close the connection.  ``stream.response`` is always populated afterwards.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout: httpx.Timeout | float | None = None,
        on_raw_chunk: Callable[[str], None] | None = None,
    ) -> None:
        self._http = http
        self._url = url
        self._payload = payload
        self._headers = headers
        self._timeout = timeout
        self._on_raw_chunk = on_raw_chunk

        self._cm: Any = None
        self._response: httpx.Response | None = None
        self._iterator: AsyncIterator[str] | None = None
        self._decoder = SSEDecoder()
        self._accumulator = DeltaAccumulator(model=str(payload.get("model") or ""))

        self._pending: list[StreamEvent] = []
        self._saw_done = False
        self._finished = False

        #: The assembled result.  ``None`` until the stream is finished.
        self.response: ChatResponse | None = None

    # lifecycle

    async def __aenter__(self) -> "ChatStream":
        self._cm = self._http.stream(
            "POST", self._url, json=self._payload, headers=self._headers, timeout=self._timeout
        )
        try:
            self._response = await self._cm.__aenter__()
            if self._response.status_code >= 400:
                body = await self._response.aread()
                await self._cm.__aexit__(None, None, None)
                # Imported lazily to avoid a config <-> client import cycle.
                from .client import _retry_after, raise_for_status
                raise_for_status(
                    self._response.status_code,
                    body,
                    url=self._url,
                    retry_after=_retry_after(self._response),
                )
        except BaseException:
            if self._response is None:
                await self._cm.__aexit__(None, None, None)
            raise

        self._iterator = self._response.aiter_text().__aiter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        # Populate the result even when the caller broke out of the loop early,
        # so a cancelled generation still yields the partial text.
        if not self._finished:
            try:
                self._finalise()
            except DeepSeekError:
                if exc is None:
                    raise
        return await self._cm.__aexit__(exc_type, exc, tb)

    # iteration

    def __aiter__(self) -> "ChatStream":
        return self

    async def __anext__(self) -> StreamEvent:
        while True:
            if self._pending:
                return self._pending.pop(0)
            if self._finished:
                raise StopAsyncIteration
            if self._saw_done:
                self._finalise()
                continue

            assert self._iterator is not None, "ChatStream used outside its context manager"
            try:
                chunk = await self._iterator.__anext__()
            except StopAsyncIteration:
                self._finalise()
                continue

            if self._on_raw_chunk is not None:
                try:
                    self._on_raw_chunk(chunk)
                except Exception:  # a broken tap must not kill the stream
                    logger.exception("on_raw_chunk callback failed")

            for event in self._decoder.feed(chunk):
                self._consume(event)

    # internals

    def _consume(self, event: SSEEvent) -> None:
        payload = event.data.strip()
        if not payload:
            return
        if payload == DONE_SENTINEL:
            self._saw_done = True
            return

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("skipping unparseable SSE payload: %.200r", payload)
            return

        if not isinstance(chunk, dict):
            return

        # Mid-stream errors arrive as a top-level "error" object with no choices.
        if chunk.get("error") and not chunk.get("choices"):
            detail = chunk["error"]
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail)
            raise StreamError(str(detail))

        self._pending.extend(self._accumulator.feed(chunk))

    def _finalise(self) -> None:
        if self._finished:
            return
        # Drain anything the decoder is still holding before freezing.
        try:
            for event in self._decoder.close():
                self._consume(event)
        except DeepSeekError:
            raise
        self.response = self._accumulator.build()
        self._pending.append(
            StreamEvent(
                "done",
                usage=self.response.usage,
                finish_reason=self.response.finish_reason,
            )
        )
        self._finished = True
        self._saw_done = True


# ── convenience ───────────────────────────────────────────────────────────────

async def collect(
    stream: ChatStream,
    on_text: Callable[[str], Awaitable[None] | None] | None = None,
    on_reasoning: Callable[[str], Awaitable[None] | None] | None = None,
) -> ChatResponse:
    """Drive a :class:`ChatStream` to completion, optionally tapping deltas.

    Equivalent to an ``async for`` over the stream, but returns the assembled
    response directly for callers that do not need the individual events.
    """
    async with stream:
        async for event in stream:
            if event.type == "content" and on_text is not None:
                result = on_text(event.text)
                if inspect.isawaitable(result):
                    await result
            elif event.type == "reasoning" and on_reasoning is not None:
                result = on_reasoning(event.text)
                if inspect.isawaitable(result):
                    await result
        assert stream.response is not None
        return stream.response
