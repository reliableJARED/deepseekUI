"""Tool definitions, a registry, and the agentic call loop.

The wrapper exposes two shapes on purpose:

* :class:`ToolRegistry` — collect tool schemas and dispatch by name.  This is what
  an MCP bridge plugs into: list the MCP server's tools, wrap each one's handler,
  and the registry is the whole integration.
* :func:`run_tool_loop` — drive a conversation until the model stops asking for
  tools.  Handles the assistant/tool message bookkeeping, which is easy to get
  wrong by hand and produces hard 400s when it is.

Handlers may be sync or async, and may return a string, a list of content blocks
(so a tool can hand back an image), or a :class:`ToolResult`.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from .errors import ToolError
from .messages import estimate_tokens, tool as tool_message
from .types import ChatResponse, ToolCall

__all__ = ["ToolResult", "Tool", "ToolRegistry", "tool_schema", "run_tool_loop", "ASYNCIO_TOOL"]

logger = logging.getLogger("deepseek_client.tools")

#: Marker for the 'asyncio' helper tool name, documented below.
ASYNCIO_TOOL = "asyncio"


@dataclass(slots=True)
class ToolResult:
    """What a tool handler returns.

    ``content`` may be a string or a list of content blocks.  Blocks are
    JSON-encoded on the wire, which is how a tool returns an image to the model.
    Set ``is_error`` to tell the model the call failed without aborting the loop.
    """

    content: str | list[dict[str, Any]] = ""
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(content=f"Error: {message}", is_error=True)


@dataclass(slots=True)
class Tool:
    """One callable the model may invoke."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    handler: Callable[..., Any] | None = field(default=None, repr=False)

    def schema(self) -> dict[str, Any]:
        """OpenAI ``tools`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        if self.handler is None:
            raise ToolError(f"Tool {self.name!r} has no handler attached")
        outcome = self.handler(**dict(arguments))
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return _coerce_result(outcome)


def _coerce_result(value: Any) -> ToolResult:
    """Normalise whatever a handler returned into a :class:`ToolResult`."""
    if isinstance(value, ToolResult):
        return value
    if value is None:
        return ToolResult(content="")
    if isinstance(value, str):
        return ToolResult(content=value)
    if isinstance(value, (list, tuple)):
        return ToolResult(content=list(value))
    if isinstance(value, dict):
        return ToolResult(content=json.dumps(value, ensure_ascii=False, default=str))
    return ToolResult(content=str(value))


# ── schema helpers ────────────────────────────────────────────────────────────

def tool_schema(
    name: str,
    description: str = "",
    parameters: Mapping[str, Any] | None = None,
    *,
    properties: Mapping[str, Any] | None = None,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-Schema parameter object.

    Two calling styles::

        tool_schema("resize", "Resize an image", parameters=FULL_SCHEMA)
        tool_schema("resize", "Resize an image", properties={...}, required=["path"])
    """
    if parameters is not None:
        schema = dict(parameters)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required or []),
    }


# ── registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """A named collection of tools, usable as the ``@register`` decorator target."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for entry in tools:
            self.add(entry)

    # collection protocol

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # registration

    def add(self, entry: Tool) -> Tool:
        if not entry.name:
            raise ToolError("Tool must have a name")
        if entry.name in self._tools:
            logger.warning("overwriting already-registered tool %r", entry.name)
        self._tools[entry.name] = entry
        return entry

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(self._tools) or "<none>"
            raise ToolError(f"Unknown tool {name!r}. Registered: {available}") from None

    def register(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        properties: Mapping[str, Any] | None = None,
        required: Sequence[str] | None = None,
    ) -> Any:
        """Register a handler, bare or as a decorator.

            registry.register(my_fn, name="resize", description="...", parameters={...})

            @registry.register(name="resize", description="...")
            def resize(path: str, width: int) -> str: ...
        """
        def _wrap(target: Callable[..., Any]) -> Callable[..., Any]:
            schema = tool_schema(
                name or target.__name__,
                description or (inspect.getdoc(target) or "").split("\n\n")[0],
                parameters,
                properties=properties,
                required=required,
            )
            self.add(
                Tool(
                    name=name or target.__name__,
                    description=description or (inspect.getdoc(target) or "").split("\n\n")[0],
                    parameters=schema,
                    handler=target,
                )
            )
            return target

        return _wrap(fn) if fn is not None else _wrap

    # exposure

    def schemas(self) -> list[dict[str, Any]]:
        return [entry.schema() for entry in self._tools.values()]

    def filtered(self, allow: Sequence[str] | None = None) -> "ToolRegistry":
        """A new registry containing only ``allow`` — useful for per-conversation tool scoping."""
        if allow is None:
            return ToolRegistry(self._tools.values())
        return ToolRegistry(self._tools[n] for n in allow if n in self._tools)

    # dispatch

    async def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Invoke a tool by name, converting any exception into an error result.

        Errors become results rather than exceptions because the model can almost
        always recover from a bad argument if it is told what went wrong.
        """
        try:
            entry = self.get(name)
        except ToolError as exc:
            return ToolResult.error(str(exc))

        try:
            return await entry.invoke(arguments or {})
        except TypeError as exc:
            # Usually a mismatch between the model's arguments and the signature.
            return ToolResult.error(f"{name}: bad arguments {dict(arguments or {})!r} — {exc}")
        except Exception as exc:
            logger.exception("tool %r raised", name)
            return ToolResult.error(f"{name} failed: {type(exc).__name__}: {exc}")


# ── the agent loop ────────────────────────────────────────────────────────────

async def run_tool_loop(
    client: Any,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    model: str | None = None,
    max_steps: int = 8,
    on_assistant: Callable[[ChatResponse], Awaitable[None] | None] | None = None,
    on_tool: Callable[[ToolCall, ToolResult], Awaitable[None] | None] | None = None,
    **chat_kwargs: Any,
) -> ChatResponse:
    """Call the model, execute any requested tools, and repeat until it answers.

    ``messages`` is mutated in place so the caller keeps the full transcript,
    including the assistant turns and tool results.  Returns the final
    :class:`ChatResponse` — the one that carried no tool calls — or the last
    response if ``max_steps`` was reached first.
    """
    if not len(registry):
        return await client.chat(messages, model=model, **chat_kwargs)

    schemas = registry.schemas()
    response: ChatResponse | None = None

    for step in range(max_steps):
        response = await client.chat(
            messages, model=model, tools=schemas, **chat_kwargs
        )
        messages.append(response.message.to_dict())

        if on_assistant is not None:
            outcome = on_assistant(response)
            if inspect.isawaitable(outcome):
                await outcome

        if not response.tool_calls:
            return response

        for call in response.tool_calls:
            if not call.arguments_valid:
                result = ToolResult.error(
                    f"arguments were not valid JSON and were discarded: {call.arguments[:200]!r}"
                )
            else:
                result = await registry.call(call.name, call.parsed_arguments)

            messages.append(tool_message(call.id, result.content))

            if on_tool is not None:
                outcome = on_tool(call, result)
                if inspect.isawaitable(outcome):
                    await outcome

        logger.debug(
            "tool loop step %d complete (%d messages, ~%d tokens)",
            step + 1, len(messages), estimate_tokens(messages),
        )

    logger.warning("tool loop hit max_steps=%d without a final answer", max_steps)
    assert response is not None
    return response
