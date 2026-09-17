"""Result types returned by the client.

These are plain dataclasses rather than ``TypedDict``s so that callers get
attribute access, ``repr``, and type-checker support, while still being trivially
convertible back to wire dicts via ``to_dict()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Usage",
    "ToolCall",
    "ChatMessage",
    "ChatResponse",
    "StreamEvent",
    "StreamEventType",
]

StreamEventType = Literal["reasoning", "content", "tool_call", "usage", "finish", "done"]


@dataclass(slots=True)
class Usage:
    """Token accounting for one request.

    The DeepSeek-specific cache fields are populated on models that support
    context caching; they are harmless zeros elsewhere.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Usage":
        if not data:
            return cls()
        details = data.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
            prompt_cache_hit_tokens=int(data.get("prompt_cache_hit_tokens") or 0),
            prompt_cache_miss_tokens=int(data.get("prompt_cache_miss_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class ToolCall:
    """A single function call the model asked for.

    ``arguments`` stays a raw JSON *string* because that is what streams in
    fragments; use :attr:`parsed_arguments` once the call is complete.  A
    truncated stream can leave the string unparseable, which is why the parse is
    a property that can fail rather than an eager conversion.
    """

    id: str
    name: str
    arguments: str = ""
    index: int = 0

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Decode ``arguments`` into a dict, or ``{}`` if it is empty/invalid."""
        raw = (self.arguments or "").strip()
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {"value": value}

    @property
    def arguments_valid(self) -> bool:
        raw = (self.arguments or "").strip()
        if not raw:
            return True  # no-arg call
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "ToolCall":
        fn = data.get("function") or {}
        return cls(
            id=str(data.get("id") or ""),
            name=str(fn.get("name") or ""),
            arguments=str(fn.get("arguments") or ""),
            index=index,
        )


@dataclass(slots=True)
class ChatMessage:
    """An assistant turn, normalised across streaming and non-streaming calls."""

    role: str = "assistant"
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        if self.reasoning:
            # Emitted under the API's own field name.  This is load-bearing, not
            # cosmetic: when a request carries `tools`, every prior assistant turn
            # must echo its `reasoning_content` back or the API answers 400.  A
            # tool loop that drops it fails on the *second* step, which is exactly
            # the kind of bug that only shows up against the live endpoint.
            out["reasoning_content"] = self.reasoning
        return out


@dataclass(slots=True)
class ChatResponse:
    """A completed (non-streamed, or fully assembled streamed) completion."""

    message: ChatMessage
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    id: str = ""
    created: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def reasoning(self) -> str:
        return self.message.reasoning

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls

    @property
    def finish_reason(self) -> str | None:
        return self.message.finish_reason

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        msg = choice.get("message") or {}

        calls = [
            ToolCall.from_dict(tc, index=i)
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        # Some providers put the text in `reasoning`; DeepSeek uses `reasoning_content`.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

        return cls(
            message=ChatMessage(
                role=msg.get("role") or "assistant",
                content=msg.get("content") or "",
                reasoning=reasoning,
                tool_calls=calls,
                finish_reason=choice.get("finish_reason"),
            ),
            usage=Usage.from_dict(data.get("usage")),
            model=data.get("model") or "",
            id=data.get("id") or "",
            created=int(data.get("created") or 0),
            raw=data,
        )


@dataclass(slots=True)
class StreamEvent:
    """One increment emitted while a streamed completion is being consumed.

    ``type`` is one of:

    ``reasoning``  reasoning-model chain-of-thought delta (``text``)
    ``content``    visible answer delta (``text``)
    ``tool_call``  the model began a new tool call (``tool_call`` holds id/name)
    ``usage``      token accounting, normally on the final chunk
    ``finish``     the model reported why it stopped
    ``done``       terminal event; the full result is on ``ChatStream.response``
    """

    type: StreamEventType
    text: str = ""
    usage: Usage | None = None
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)

    def __str__(self) -> str:  # convenient for quick debugging
        if self.type in ("content", "reasoning"):
            return self.text
        if self.type == "tool_call":
            return f"<tool_call {self.tool_call.name if self.tool_call else '?'}>"
        return f"<{self.type}>"
