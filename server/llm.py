"""One conversational turn: compose, sanitise, rehydrate, stream, persist.

The shape of this module is forced by three facts about the API:

**Reasoning is mandatory in tool conversations.** With ``tools`` present, every
assistant turn that produced ``reasoning_content`` must echo it back on the next
request or the API answers 400. So the assistant message is persisted *with* its
chain of thought and replayed verbatim. It is never treated as a display-only
artefact.

**Thinking is on by default.** Nothing needs to be sent to enable it, and a
conversation that did not ask for it still streams ``reasoning_content``.

**Images cannot be re-sent cheaply.** Every step of a tool loop re-uploads the
full transcript, so images are re-inlined from disk each step rather than cached in
the message list — the transcript on disk stays small and greppable, and the
newest-first budget stops a long session from growing without bound.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from deepseek_client import DeepSeekClient, DeepSeekError
from deepseek_client.messages import ensure_tool_pairing, estimate_tokens, sanitize_messages
from deepseek_client.types import ToolCall

from .media import DISPLAY_KEY, display_note, split_display_blocks
from .rehydrate import MediaBudget, rehydrate, strip_ui_blocks

__all__ = ["ChatEngine", "TurnResult", "compose_system", "sse"]

logger = logging.getLogger("deepseek_ui.llm")

TODO_MARKER_START = "\n\n===== YOUR CURRENT TASK TRACKER =====\n"
TODO_MARKER_END = "\n===== END TASK TRACKER =====\n"

#: Titles longer than this are trimmed when auto-naming a conversation.
TITLE_LIMIT = 60

#: The conversation the current task is serving. A ContextVar rather than an
#: attribute, because a tool handler reading "the active conversation" must see the
#: right one even when two conversations stream concurrently.
_active_uuid: ContextVar[str | None] = ContextVar("deepseek_ui_active_uuid", default=None)


def sse(event: str, data: Any) -> str:
    """Encode one Server-Sent Event. ``json.dumps`` escapes the newlines for us."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def compose_system(base: str, todo: str = "") -> str:
    """Join the base prompt and the task tracker.

    Kept as two fields on disk so re-syncing the tracker is idempotent — rebuilding
    ``messages[0]`` never accumulates duplicate blocks.
    """
    parts: list[str] = []
    if base.strip():
        parts.append(base.strip())
    if todo.strip():
        parts.append(f"{TODO_MARKER_START}{todo.strip()}{TODO_MARKER_END}")
    return "\n\n".join(parts)


@dataclass(slots=True)
class TurnResult:
    """What a completed turn produced."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    reasoning: str = ""
    steps: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    truncated: bool = False
    error: str = ""


class ChatEngine:
    """Drives turns for one process. Holds the client and the tool registry."""

    def __init__(
        self,
        settings,
        store,
        media,
        client: DeepSeekClient,
        *,
        tool_registry=None,
        mcp_manager=None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.media = media
        self.client = client
        self.tool_registry = tool_registry
        self.mcp = mcp_manager

    def active_uuid(self) -> str | None:
        """The conversation this task is serving. Tool handlers read this."""
        return _active_uuid.get()

    # ── message assembly ──

    def prepare_messages(
        self,
        conv,
        *,
        pending: Sequence[Mapping[str, Any]] = (),
        system_override: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Stored transcript -> API-ready message list.

        ``pending`` holds messages produced during the current turn that have not
        been written to disk yet, so a tool loop sees its own history.
        """
        base = conv.sys_base if system_override is None else system_override
        system_text = compose_system(base or "", conv.sys_todo or "")

        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.extend(dict(m) for m in conv.messages)
        messages.extend(dict(m) for m in pending)

        messages = sanitize_messages(
            messages, context=f"conv={conv.uuid}",
            # DeepSeek requires prior turns' `reasoning_content` back whenever the
            # request carries tools; other endpoints would rather not see it.  The
            # `True` fallback keeps a bare/mock client (no such attribute) unchanged.
            keep_reasoning=getattr(self.client, "replays_reasoning", True),
        )
        messages = rehydrate(messages, self.media, self.settings, media_root=self.store.root)

        # Last, and deliberately so. Nothing here guarantees the tool-call invariant
        # on the way *in* — a raw-index truncation, an import, a concurrent user
        # turn, or a tool loop that was cancelled mid-step can all leave an
        # assistant turn whose calls are unanswered, and the API rejects that on
        # every subsequent request rather than only the broken one. Repairing here
        # means every outbound request is valid no matter how the transcript got
        # that way, including transcripts already broken on disk.
        messages = ensure_tool_pairing(messages, context=f"conv={conv.uuid}")
        return messages

    def context_status(self, conv, model: str | None = None) -> dict[str, Any]:
        """How full the window is, for the UI's context meter.

        Never raises. A context meter is a nice-to-have, and it must not be the thing
        that breaks saving a message when the model is not configured yet.
        """
        empty = {"ratio": 0.0, "tokens": 0, "limit": 0, "safe": True, "messages": 0}
        try:
            messages = self.prepare_messages(conv, model=model)
        except Exception as exc:
            logger.warning("could not prepare messages for %s: %s", conv.uuid, exc)
            return empty

        empty["messages"] = len(messages)

        try:
            spec = self.client.spec_for(model or conv.model or None)
        except Exception as exc:
            logger.debug("context status without a configured model: %s", exc)
            return empty

        tokens = estimate_tokens(messages)
        ratio = tokens / spec.max_input_tokens if spec.max_input_tokens else 0.0
        return {
            "ratio": round(ratio, 4),
            "tokens": tokens,
            "limit": spec.max_input_tokens,
            "safe": ratio <= self.settings.context_safety_ratio,
            "messages": len(messages),
        }

    # ── the turn ──

    async def stream_turn(
        self,
        uuid: str,
        *,
        model: str | None = None,
        system_override: str | None = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
        tools_enabled: bool = True,
        pending_prefix: Sequence[Mapping[str, Any]] = (),
        max_steps: int | None = None,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Run one turn, yielding SSE frames.

        The generator persists everything it produces, so an interrupted client
        still leaves a coherent transcript behind.
        """
        token = _active_uuid.set(uuid)
        try:
            async for frame in self._stream_turn_inner(
                uuid,
                model=model,
                system_override=system_override,
                reasoning_effort=reasoning_effort,
                thinking=thinking,
                tools_enabled=tools_enabled,
                pending_prefix=pending_prefix,
                max_steps=max_steps,
                temperature=temperature,
                extra_body=extra_body,
            ):
                yield frame
        finally:
            _active_uuid.reset(token)

    async def _stream_turn_inner(
        self,
        uuid: str,
        *,
        model: str | None,
        system_override: str | None,
        reasoning_effort: str | None,
        thinking: bool | None,
        tools_enabled: bool,
        pending_prefix: Sequence[Mapping[str, Any]],
        max_steps: int | None,
        temperature: float | None,
        extra_body: Mapping[str, Any] | None,
    ) -> AsyncIterator[str]:
        try:
            conv = self.store.load(uuid)
        except Exception as exc:
            yield sse("error", {"message": str(exc), "status": 404})
            return

        resolved_model = model or conv.model or None

        # Resolving the model happens before the loop's own error handling, and it is
        # exactly where a missing API key surfaces. Report it as a frame rather than
        # letting it escape a response whose 200 has already been sent.
        try:
            spec = self.client.spec_for(resolved_model)
        except Exception as exc:
            logger.error("model unavailable for %s: %s", uuid, exc)
            yield sse("error", {
                "message": f"model unavailable — {exc}",
                "hint": (
                    "Paste a key into the DeepSeek API key field in Settings "
                    "(no restart needed), or set DEEPSEEK_API_KEY in .env."
                ),
            })
            return

        step_limit = max_steps or self.settings.effective("max_tool_steps")

        # Build the tool set: MCP servers plus the built-in media tools.
        registry = None
        if tools_enabled:
            registry = self.tool_registry
            if registry is not None and not len(registry):
                registry = None

        tool_names = (
            [entry["function"]["name"] for entry in registry.schemas()]
            if registry is not None else []
        )
        yield sse("meta", {
            "uuid": uuid,
            "model": spec.id,
            "vision": spec.vision,
            "thinking": thinking if thinking is not None else spec.thinking,
            "reasoning_effort": reasoning_effort or spec.default_reasoning_effort,
            "tools": tool_names,
            "max_steps": step_limit,
        })

        pending: list[dict[str, Any]] = [dict(m) for m in pending_prefix]
        result = TurnResult()
        loop = True
        step = 0
        last_messages: list[dict[str, Any]] = []
        #: Media handed to the user during this turn. Collected across *every* step
        #: because the reply the user is waiting for is usually not the step that
        #: produced the media: ``web_fetch`` returns its images on step 1 and the
        #: answer that mentions them arrives on step 2.
        shown: list[dict[str, Any]] = []

        async def persist() -> None:
            """Write whatever the current step produced.

            Flushing per step rather than per turn means a client that closes the
            tab mid-loop still has a transcript that ends at a coherent boundary
            instead of losing the assistant turn that requested the tools.

            "Coherent" is not free, though: the assistant turn is queued *before*
            its tools run and the results trickle in one at a time, so an
            interruption (a closed tab aborting the fetch, the Stop button,
            ``CancelledError`` from a shutdown — and ``ToolRegistry.call`` catches
            ``Exception``, not ``CancelledError``) can land between the two. What
            used to be written then was a half-finished group, and because the
            same history is replayed on every later request, that conversation
            would 400 forever. Closing the group first keeps the interruption
            from being permanent: the calls that did not finish get a stub result
            that says so.
            """
            if not pending:
                return
            batch = ensure_tool_pairing(pending, context=f"turn={uuid}")
            await self.store.extend(uuid, batch)
            pending.clear()

        async def flush_shown() -> None:
            """Hang the media this turn displayed on its final assistant turn.

            Once, at the end, rather than as each result arrives: the turn may have
            several assistant messages (one per step) and exactly one of them is the
            reply the user reads to, so anywhere else would put the picture under a
            message that says nothing.
            """
            if not shown:
                return
            try:
                await self._attach_shown(uuid, shown)
            except Exception:                            # pragma: no cover - defensive
                logger.exception("could not attach displayed media for %s", uuid)

        try:
            if pending:
                await persist()
                conv = self.store.load(uuid)

            while loop and step < step_limit:
                step += 1
                messages = self.prepare_messages(
                    conv, pending=pending, system_override=system_override, model=resolved_model
                )
                last_messages = messages

                status = self._status_for(messages, spec)
                if status["ratio"] > self.settings.context_safety_ratio:
                    yield sse("warning", {
                        "message": (
                            f"Context is {status['ratio'] * 100:.0f}% full "
                            f"({status['tokens']:,} of {status['limit']:,} tokens). "
                            "Older media is being dropped automatically."
                        ),
                        **status,
                    })

                if step > 1:
                    yield sse("step", {"step": step, "max_steps": step_limit})

                response = None
                async with self.client.stream(
                    messages,
                    model=resolved_model,
                    tools=registry.schemas() if registry is not None else None,
                    reasoning_effort=reasoning_effort,
                    thinking=thinking,
                    temperature=temperature,
                    extra_body=extra_body,
                    user_id=uuid,
                ) as stream:
                    async for event in stream:
                        if event.type == "reasoning":
                            result.reasoning += event.text
                            yield sse("reasoning", {"text": event.text})
                        elif event.type == "content":
                            result.text += event.text
                            yield sse("content", {"text": event.text})
                        elif event.type == "tool_call" and event.tool_call:
                            # Only fires when a call opens; arguments are still
                            # streaming, so this is a "thinking" hint for the UI.
                            yield sse("tool_pending", {
                                "id": event.tool_call.id,
                                "index": getattr(event.tool_call, "index", None),
                            })
                        elif event.type == "usage" and event.raw:
                            usage = (event.raw.get("usage") or {})
                            if usage:
                                result.usage = usage
                                yield sse("usage", usage)

                    response = stream.response

                if response is None:                     # pragma: no cover - defensive
                    raise DeepSeekError("the stream ended without producing a response")

                # Persist the assistant turn *with* reasoning_content.
                assistant = response.message.to_dict()
                pending.append(assistant)

                if not response.tool_calls:
                    loop = False
                    break

                for call in response.tool_calls:
                    yield sse("tool_call", {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.parsed_arguments if call.arguments_valid else {},
                        "raw_arguments": call.arguments if not call.arguments_valid else "",
                        "step": step,
                    })

                    result.tool_calls += 1
                    outcome = await self._run_tool(registry, call)
                    stored, shown_here = self._store_tool_result(uuid, call, outcome)
                    pending.append(stored)
                    shown.extend(shown_here)

                    yield sse("tool_result", {
                        "id": call.id,
                        "name": call.name,
                        "is_error": outcome.is_error,
                        "blocks": _display_blocks(stored),
                        "media": shown_here,
                        "step": step,
                    })

                # Persist the assistant turn and its tool results together, then
                # reload so the next step sees the same transcript the file has.
                await persist()
                conv = self.store.load(uuid)

            else:
                if loop:
                    result.truncated = True
                    yield sse("warning", {
                        "message": (
                            f"Stopped after {step_limit} tool steps without a final answer. "
                            "Raise the tool-step limit in Settings or narrow the request."
                        )
                    })

            await persist()
            await flush_shown()

            if result.usage:
                yield sse("usage", result.usage)

            title = await self._maybe_title(uuid, conv)
            if title:
                yield sse("title", {"uuid": uuid, "title": title})

            yield sse("done", {
                "uuid": uuid,
                "steps": step,
                "tool_calls": result.tool_calls,
                "truncated": result.truncated,
                "text": result.text,
                "reasoning": result.reasoning,
                "usage": result.usage,
                "tokens_estimate": self._estimate(last_messages),
            })

        except DeepSeekError as exc:
            # Anything already produced is worth keeping — a partial answer plus an
            # error is more useful than losing the turn.
            try:
                await persist()
                await flush_shown()
            except Exception:                            # pragma: no cover
                logger.exception("could not persist a partial turn for %s", uuid)
            logger.warning("turn failed for %s: %s", uuid, exc)
            yield sse("error", {
                "message": str(exc),
                "status": getattr(exc, "status", None),
                "retryable": getattr(exc, "retryable", False),
                "partial_text": result.text,
                "partial_reasoning": result.reasoning,
            })
        except BaseException as exc:
            # Includes CancelledError, which is what a closed browser tab looks
            # like here. Persist before letting it propagate.
            try:
                await persist()
                await flush_shown()
            except Exception:                            # pragma: no cover
                logger.exception("could not persist an interrupted turn for %s", uuid)
            if not isinstance(exc, Exception):           # cancellation or shutdown
                raise
            logger.exception("unexpected failure serving %s", uuid)
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}", "status": None})

    # ── helpers ──

    def _status_for(self, messages: Sequence[Mapping[str, Any]], spec) -> dict[str, Any]:
        tokens = estimate_tokens(messages)
        ratio = tokens / spec.max_input_tokens if spec.max_input_tokens else 0.0
        return {
            "ratio": round(ratio, 4),
            "tokens": tokens,
            "limit": spec.max_input_tokens,
            "messages": len(messages),
        }

    def _estimate(self, messages: Sequence[Mapping[str, Any]]) -> int:
        try:
            return estimate_tokens(messages)
        except Exception:                                # pragma: no cover
            return 0

    async def _run_tool(self, registry, call: ToolCall):
        from deepseek_client.tools import ToolResult

        if registry is None:
            return ToolResult.error(f"no tools are available to call {call.name!r}")
        if not call.arguments_valid:
            return ToolResult.error(
                f"arguments for {call.name!r} were not valid JSON and were discarded: "
                f"{call.arguments[:200]!r}"
            )
        return await registry.call(call.name, call.parsed_arguments)

    def _store_tool_result(self, uuid: str, call: ToolCall, outcome):
        """Persist a tool result, keeping media on disk and text in the message.

        Returns ``(stored message, media the user was shown)``. Media meant for the
        user is *removed* from the message rather than left in it: the transcript is
        replayed on every later request, and media left here is media the model would
        be charged for seeing on every one of them. What replaces it is a line naming
        the path, which is the part the model can act on.
        """
        content = outcome.content

        if isinstance(content, list):
            try:
                blocks = self.media.ingest_tool_blocks(uuid, content, call.name)
            except Exception as exc:                     # pragma: no cover - defensive
                logger.warning("could not persist tool media: %s", exc)
                blocks = [{"type": "text", "text": f"[media could not be stored: {exc}]"}]
            shown, blocks = split_display_blocks(blocks)
            if shown:
                blocks = [
                    {"type": "text", "text": display_note(block)} for block in shown
                ] + blocks
            payload = json.dumps(blocks, ensure_ascii=False)
        else:
            payload = str(content or "")
            shown = []

        stored = {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": payload}
        return stored, shown

    async def _attach_shown(self, uuid: str, shown: list[dict[str, Any]]) -> None:
        """Record what the user was shown on the assistant turn they read.

        Stored under a ``_``-prefixed key so it is invisible to the API: ``sanitize_messages``
        and :func:`~server.rehydrate._rehydrate_assistant` both drop anything that
        starts with an underscore, which makes this pure UI state rather than part of
        the conversation the model replays.

        The *last* assistant turn is the right home for it because the frontend renders
        it above that turn's text — which is what "show me the picture, then answer"
        means. When a turn ends on a tool step instead (the step limit was hit), the
        last assistant turn is still the last thing the user sees, so it still works.
        """

        def _attach(conv) -> None:
            for message in reversed(conv.messages):
                if message.get("role") != "assistant":
                    continue
                existing = message.get("_display")
                message["_display"] = list(existing or []) + shown
                return

        await self.store.mutate(uuid, _attach)

    async def _maybe_title(self, uuid: str, conv) -> str:
        """Name a conversation from its first user message."""
        if conv.title and conv.title != "New conversation":
            return ""
        fresh = self.store.load(uuid)
        for message in fresh.messages:
            if message.get("role") != "user":
                continue
            text = _first_text(message.get("content"))
            if not text:
                continue
            title = text.strip().replace("\n", " ")
            if len(title) > TITLE_LIMIT:
                title = title[: TITLE_LIMIT - 1].rstrip() + "\u2026"

            def _set(c, value=title):
                c.title = value

            await self.store.mutate(uuid, _set)
            return title
        return ""


def _first_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text") or "")
    return ""


def _display_blocks(stored: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The blocks a tool result should show in the UI.

    Stored media keeps its ``/memory/...`` URL, so the browser renders the same file
    the model saw without any base64 crossing the wire. A remote block keeps its
    *original* URL plus the few fields that make a player work — the poster, the
    stream proxy, the embed — which is why this is a whitelist rather than a copy:
    the stored line is what the model reads, and it must not grow.

    Returns ``[]`` if the content does not parse as a block list. A plain string —
    a message with no JSON blocks at all — is reported to the caller as no blocks,
    since there is nothing for the media grid to render.
    """
    content = stored.get("content")
    if not isinstance(content, str):
        return []
    text = content.strip()
    if not text.startswith("["):
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []

    out: list[dict[str, Any]] = []
    for block in decoded:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": str(block.get("text") or "")})
            continue
        if btype == "embed":
            out.append({
                "type": "embed",
                "url": block.get("url", ""),
                "embed_url": block.get("embed_url", ""),
                "provider": block.get("provider", ""),
                "title": block.get("title", ""),
                "author": block.get("author", ""),
                "poster": block.get("poster", ""),
                "name": block.get("name", ""),
                "note": block.get("note", ""),
                "source": block.get("source", ""),
                "display": block.get("display", False),
            })
            continue
        if btype not in ("image", "video", "audio"):
            continue
        shown: dict[str, Any] = {
            "type": btype,
            "url": block.get("url", ""),
            "name": block.get("name", ""),
            "mime": block.get("mimeType") or block.get("mime") or "",
        }
        for key in (
            "poster",
            "stream",
            "duration",
            "width",
            "height",
            "source",
            "codec",
            "caption",
            "note",
        ):
            value = block.get(key)
            if value not in (None, "", 0, False):
                shown[key] = value
        if block.get(DISPLAY_KEY):
            shown[DISPLAY_KEY] = True
        out.append(shown)
    return out


def make_client_factory(settings) -> Callable[[], DeepSeekClient]:
    """Build a lazily-created, cached client bound to ``providers.json``.

    One client serves every model the provider declares — ``spec_for(model)`` picks
    the right endpoint and limits per request — so there is deliberately no per-model
    cache here, which would only open redundant connection pools.

    The returned callable carries two extra attributes: ``aclose`` (used on
    shutdown) and ``invalidate`` (used after `.env` changes). They are the same
    operation — drop the cached client and close its pool — because a client whose
    key has just been replaced must not keep serving requests.
    """
    cache: dict[str, DeepSeekClient] = {}

    def factory() -> DeepSeekClient:
        if "client" not in cache:
            cache["client"] = DeepSeekClient.from_providers(
                settings.providers_path,
                model=settings.model or None,
                env_file=settings.env_file,
                max_retries=2,
            )
        return cache["client"]

    async def drop() -> None:
        """Forget the cached client, closing its connection pool first."""
        client = cache.pop("client", None)
        if client is None:
            return
        try:
            await client.aclose()
        except Exception:                                # pragma: no cover - shutdown noise
            logger.debug("error closing the replaced HTTP client", exc_info=True)

    factory.aclose = drop
    factory.invalidate = drop

    return factory
