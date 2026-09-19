"""Content-block and message builders, plus history sanitisation.

Everything here operates on plain dicts in OpenAI ``chat-completions`` shape,
because that is exactly what goes on the wire.  The builders exist so callers
never hand-roll ``{"type": "image_url", "image_url": {"url": ...}}`` and get it
subtly wrong, and :func:`sanitize_messages` exists so a long agentic history
with tool calls cannot produce a 400 on the *next* request.

Nothing in this module performs I/O beyond an explicit ``image_file`` read.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "CHARS_PER_TOKEN",
    "text",
    "image_url",
    "image_data",
    "image_bytes",
    "image_file",
    "image_file_id",
    "image_file_data",
    "content",
    "system",
    "user",
    "assistant",
    "tool",
    "sanitize_messages",
    "ensure_tool_pairing",
    "INTERRUPTED_NOTE",
    "estimate_tokens",
    "image_tokens",
    "merge_text",
    "merge_reasoning",
    "IMAGE_TOKEN_CEILING",
]

logger = logging.getLogger("deepseek_client.messages")

#: Rough characters-per-token estimate.  Deliberately conservative (real English
#: averages closer to 4) so the resulting budget errs on the side of caution.
CHARS_PER_TOKEN = 3.5

#: Hard ceiling the API applies to any single image after its own resize pass.
IMAGE_TOKEN_CEILING = 1_024
#: Pixels per token after the API's resize to roughly 1300x1300.
_PIXELS_PER_TOKEN = 1_650

#: Keys the wrapper adds purely for its own bookkeeping.
INTERNAL_KEYS = frozenset({"_timings", "_usage", "_is_summary", "_meta"})

#: The chain-of-thought field.  This is a *real* API field, not internal:
#: when a request carries ``tools``, ``reasoning_content`` from every previous
#: turn must be echoed back or the API returns a 400.  ``reasoning`` is accepted
#: as an alias on the way in and normalised to ``reasoning_content`` on the way out.
REASONING_KEYS = ("reasoning_content", "reasoning")

#: Image blocks are rejected outright in these roles.
_IMAGE_ROLES = ("user",)
_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image", "input_image"})

#: Content of the result synthesised for a tool call that was never answered.
#: Deliberately speaks about the *record*, not about the tool: the tool may well
#: have run (and had an effect) after the answer was lost, and pretending the call
#: never happened would be a lie the model then builds on.
INTERRUPTED_NOTE = "[no result recorded — the tool call was interrupted]"


# ── content blocks ────────────────────────────────────────────────────────────

def text(value: str) -> dict[str, Any]:
    """A text block."""
    return {"type": "text", "text": "" if value is None else str(value)}


def image_url(url: str, *, detail: str | None = None) -> dict[str, Any]:
    """An image block referencing a URL (``http(s)://``, ``data:``, or a local path).

    ``detail`` controls the server-side preprocessing:

    ``"low"``
        Downscale to 512x512 before inference — faster and cheaper.
    ``"high"`` / ``"original"``
        Keep the original image.  ``high`` exists for compatibility and is
        currently equivalent to ``original``.
    ``"auto"``
        Currently equivalent to ``original``.

    Omit it and the server decides.  ``detail`` is ignored for Files API images.
    """
    inner: dict[str, Any] = {"url": url}
    if detail:
        inner["detail"] = detail
    return {"type": "image_url", "image_url": inner}


def image_file_id(file_id: str, *, filename: str | None = None) -> dict[str, Any]:
    """Reference an image previously uploaded with the Files API.

    Preferred over inlining when the same image is reused across requests, when a
    single image exceeds 32 MiB (the inline cap — Files API allows 64 MiB), or
    when the request body would otherwise exceed 48 MiB.
    """
    block: dict[str, Any] = {"type": "file", "file_id": file_id}
    if filename:
        block["filename"] = filename
    return block


def image_file_data(
    data: bytes | bytearray | str,
    filename: str = "image.jpg",
    mime: str = "image/jpeg",
) -> dict[str, Any]:
    """A Files-API-style ``file`` block carrying the image inline as base64.

    ``file_id`` and ``file_data`` are mutually exclusive on the wire, so this
    helper never sets both.
    """
    if isinstance(data, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(data)).decode("ascii")
    else:
        encoded = str(data).strip()
        if encoded.startswith("data:"):
            return {"type": "file", "file_data": encoded, "filename": filename}
        if encoded.startswith(("http://", "https://")):
            # Not inline data — fall back to a URL reference rather than lying.
            return image_url(encoded)
    return {"type": "file", "file_data": f"data:{mime};base64,{encoded}", "filename": filename}


def image_data(
    data: bytes | bytearray | str,
    mime: str = "image/jpeg",
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    """An image block from raw bytes, a base64 string, or an existing data URI."""
    if isinstance(data, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(data)).decode("ascii")
    else:
        encoded = str(data).strip()
        if encoded.startswith("data:"):
            # Already a complete data URI — do not double-wrap it.
            return image_url(encoded, detail=detail)
        if encoded.startswith(("http://", "https://")):
            return image_url(encoded, detail=detail)
    return image_url(f"data:{mime};base64,{encoded}", detail=detail)


def image_bytes(data: bytes | bytearray, mime: str = "image/jpeg", **kw: Any) -> dict[str, Any]:
    """Alias for :func:`image_data` restricted to bytes, for readability."""
    return image_data(data, mime, **kw)


def image_file(path: str | Path, *, detail: str | None = None) -> dict[str, Any]:
    """Read an image from disk and inline it as a base64 data URI."""
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
    return image_data(file_path.read_bytes(), mime, detail=detail)


def content(*blocks: Any) -> str | list[dict[str, Any]]:
    """Collapse blocks into message content.

    A single text block degrades to a plain string, which is what every server
    accepts and what keeps simple messages readable in the stored history.
    ``None`` becomes ``""`` so the result is always JSON-serialisable.
    """
    flat: list[dict[str, Any]] = []
    for block in blocks:
        if block is None:
            continue
        if isinstance(block, (list, tuple)):
            flat.extend(b for b in block if b is not None)
        else:
            flat.append(block)

    if not flat:
        return ""
    if len(flat) == 1 and flat[0].get("type") == "text":
        return flat[0]["text"]
    return flat


# ── messages ──────────────────────────────────────────────────────────────────

def system(value: str | Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "system", "content": value if isinstance(value, str) else list(value)}


def user(value: str | Sequence[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "user",
        "content": value if isinstance(value, str) else list(value),
    }
    msg.update(extra)
    return msg


def assistant(
    value: str | Sequence[dict[str, Any]] | None = None,
    *,
    tool_calls: Sequence[Any] | None = None,
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Build an assistant turn.

    ``content`` is omitted entirely when falsy: the API rejects a message that
    carries ``content: ""`` alongside ``tool_calls``.

    ``reasoning`` is written as ``reasoning_content`` — the field name the API
    actually uses.  It must be echoed back on every turn of a tool-calling
    conversation, so :func:`sanitize_messages` preserves it by default.
    """
    msg: dict[str, Any] = {"role": "assistant"}

    if value is not None:
        if isinstance(value, str):
            if value:
                msg["content"] = value
        else:
            blocks = list(value)
            if blocks:
                msg["content"] = blocks

    if tool_calls:
        msg["tool_calls"] = [
            call.to_dict() if hasattr(call, "to_dict") else dict(call) for call in tool_calls
        ]
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def tool(tool_call_id: str, value: str | Sequence[dict[str, Any]] | Any) -> dict[str, Any]:
    """Build a tool-result turn.

    List content is JSON-encoded, which is the convention for multi-part tool
    output: it lets image/video blocks ride along in a slot that the schema
    otherwise types as a string.
    """
    if isinstance(value, str):
        payload = value
    elif isinstance(value, (list, tuple)):
        payload = json.dumps(list(value), ensure_ascii=False)
    else:
        payload = json.dumps(value, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": tool_call_id, "content": payload}


def merge_text(value: Any) -> str:
    """Flatten any content value down to its concatenated text.

    Used for previews, conversation titles, and summarisation prompts — contexts
    where media blocks are noise.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text") or "")
        if value.get("type") == "image_url":
            return "[image]"
        return ""
    parts: list[str] = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "image_url":
                parts.append("[image]")
            elif block.get("type") in ("image", "video", "audio", "embed"):
                parts.append(f"[{block.get('type')}]")
    return " ".join(p for p in parts if p)


# ── sanitisation ──────────────────────────────────────────────────────────────

def _normalise_reasoning(msg: dict[str, Any]) -> None:
    """Rename the ``reasoning`` alias to ``reasoning_content`` in place."""
    if "reasoning" in msg:
        value = msg.pop("reasoning")
        msg.setdefault("reasoning_content", value)


def _strip_images(msg: dict[str, Any], role: str, where: str) -> None:
    """Remove image blocks from a role that the API rejects them in.

    Images are only valid in ``user`` messages: anywhere else is a hard 400.  We
    drop the block and keep the message rather than failing the whole request.
    """
    value = msg.get("content")
    if not isinstance(value, list):
        return
    kept = [b for b in value if not (isinstance(b, dict) and b.get("type") in _IMAGE_BLOCK_TYPES)]
    if len(kept) != len(value):
        logger.warning(
            "%sdropped image block(s) from a %r message — images are only valid in "
            "user messages and would otherwise be rejected with HTTP 400",
            where, role,
        )
        msg["content"] = kept


def sanitize_messages(
    messages: Iterable[dict[str, Any]],
    *,
    context: str = "",
    keep_reasoning: bool = True,
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` that is safe to send upstream.

    Fixes, in order:

    * strips wrapper-internal bookkeeping keys (anything ``_``-prefixed);
    * normalises the ``reasoning`` alias to ``reasoning_content``;
    * **keeps** ``reasoning_content`` unless ``keep_reasoning=False``.  The API
      requires the chain of thought to be echoed back on every turn when the
      request carries ``tools``, and ignores it when it does not — so keeping it
      is safe in both cases, and dropping it is what causes a 400 mid tool loop;
    * drops image blocks from ``system``/``assistant`` messages, where the API
      rejects them;
    * drops assistant turns that are empty (no text *and* no tool calls);
    * drops individual ``tool_calls`` whose ``arguments`` are not valid JSON,
      which happens when a stream is cut mid-argument;
    * drops ``tool`` results whose ``tool_call_id`` no longer matches any
      surviving tool call, since an orphaned result is a hard 400;
    * drops ``tool`` results that name no call at all (an empty or missing
      ``tool_call_id``), which answer nothing and are a 400 of their own.

    It deliberately does *not* repair the opposite case — a tool call with no
    result. :func:`ensure_tool_pairing` does that, and it runs separately because
    it is the one step that can *add* to the transcript.
    """
    where = f"{context}: " if context else ""
    valid_ids: set[str] = set()
    out: list[dict[str, Any]] = []

    for original in messages:
        msg = {k: v for k, v in original.items() if not k.startswith("_")}
        _normalise_reasoning(msg)
        role = msg.get("role")

        if keep_reasoning and "reasoning_content" in msg and role != "assistant":
            # Only assistant turns carry chain of thought.
            msg.pop("reasoning_content", None)
        elif not keep_reasoning:
            for key in REASONING_KEYS:
                msg.pop(key, None)

        if role == "assistant":
            _strip_images(msg, role, where)
            raw_calls = msg.get("tool_calls") or []

            if raw_calls:
                good: list[dict[str, Any]] = []
                for call in raw_calls:
                    args = (call.get("function") or {}).get("arguments", "") or ""
                    name = (call.get("function") or {}).get("name", "?")
                    try:
                        if args.strip():
                            json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning(
                            "%sstripped tool_call %r — arguments are not valid JSON "
                            "(truncated stream?): %r",
                            where, name, args[:120],
                        )
                        continue
                    good.append(call)
                    if call.get("id"):
                        valid_ids.add(call["id"])

                if good:
                    msg["tool_calls"] = good
                    out.append(msg)
                elif msg.get("content"):
                    # Every call was malformed but there is prose — keep the prose.
                    msg.pop("tool_calls", None)
                    out.append(msg)
                else:
                    logger.warning("%sdropped empty assistant turn (all tool_calls malformed)", where)
            elif msg.get("content"):
                out.append(msg)
            else:
                logger.warning("%sdropped empty assistant turn", where)

        elif role == "tool":
            call_id = msg.get("tool_call_id") or ""
            if not call_id:
                # A result that names no call answers nothing, so it is a 400 of
                # its own. It used to be kept (`if call_id and ...` fell through
                # to the `else`), which let a mangled result reach the wire.
                logger.warning("%sdropped a tool result with no tool_call_id", where)
            elif call_id not in valid_ids:
                logger.warning(
                    "%sdropped orphaned tool result — tool_call_id %r matches no tool call",
                    where, call_id,
                )
            else:
                out.append(msg)

        else:
            if role in ("system", "developer"):
                _strip_images(msg, role, where)
            out.append(msg)

    return out


def ensure_tool_pairing(
    messages: Iterable[dict[str, Any]],
    *,
    context: str = "",
    interrupted_note: str = INTERRUPTED_NOTE,
) -> list[dict[str, Any]]:
    """Return ``messages`` with the API's tool-call invariant enforced.

    The API validates the **whole transcript** on every request, not just the turn
    being added: an assistant message carrying ``tool_calls`` must be followed by
    exactly one ``tool`` message answering each of its ``tool_call_id``s.  A
    transcript that breaks this is a hard 400 *forever*, because the same broken
    history is replayed on every later request — one interruption that lands in
    the middle of a tool loop can poison a conversation permanently.

    :func:`sanitize_messages` handles the opposite direction (a result whose call
    is gone); this is the direction it deliberately does not touch, because only
    the caller knows whether losing a call or inventing a result is acceptable.

    The policy here, chosen once and applied everywhere:

    * an unanswered call gets a **stub result** rather than being dropped.  The
      model asked for that call and may have had it answered in the real world, so
      a result saying "no result was recorded" is honest, and it keeps the call
      visible in the history where deleting it would silently rewrite what the
      model did;
    * a ``tool_calls`` entry with no ``id`` cannot be answered at all, so it goes;
      if that empties the turn, the turn survives only for its prose;
    * a ``tool`` message is dropped when it answers nothing — no id, an id no
      surviving call asked for, or a duplicate answer to an id already answered.

    Idempotent: running it over its own output changes nothing.
    """
    where = f"{context}: " if context else ""
    items = list(messages)
    out: list[dict[str, Any]] = []
    i = 0

    while i < len(items):
        message = items[i]
        if not isinstance(message, dict):
            out.append(message)
            i += 1
            continue

        role = message.get("role")

        if role == "tool":
            # Nothing before it asked for this: its call was dropped elsewhere, or
            # a raw index sliced the group in half.
            logger.warning(
                "%sdropped orphaned tool result %r — no assistant turn requested it",
                where, message.get("tool_call_id"),
            )
            i += 1
            continue

        calls = message.get("tool_calls") if role == "assistant" else None
        if not calls:
            out.append(message)
            i += 1
            continue

        # The run of results following this turn, keyed by the id each one answers.
        answers: dict[str, dict[str, Any]] = {}
        j = i + 1
        while j < len(items) and isinstance(items[j], dict) and items[j].get("role") == "tool":
            answer = items[j]
            call_id = answer.get("tool_call_id")
            if not call_id:
                logger.warning("%sdropped a tool result with no tool_call_id", where)
            elif call_id in answers:
                logger.warning(
                    "%sdropped a duplicate result for tool_call_id %r", where, call_id,
                )
            else:
                answers[call_id] = answer
            j += 1

        usable = [c for c in calls if isinstance(c, dict) and c.get("id")]
        if len(usable) != len(calls):
            logger.warning(
                "%sdropped %d tool_call(s) with no id — nothing can answer them",
                where, len(calls) - len(usable),
            )

        if not usable:
            stripped = {k: v for k, v in message.items() if k != "tool_calls"}
            if stripped.get("content") or stripped.get("reasoning_content"):
                out.append(stripped)
            else:
                logger.warning(
                    "%sdropped an assistant turn whose tool_calls were all unusable", where,
                )
            i = j
            continue

        kept: list[dict[str, Any]] = []
        for call in usable:
            call_id = call["id"]
            answer = answers.pop(call_id, None)
            if answer is None:
                name = (call.get("function") or {}).get("name") or "?"
                logger.warning(
                    "%ssynthesised a result for the unanswered tool_call_id %r (%s) — "
                    "the call was interrupted or never finished",
                    where, call_id, name,
                )
                answer = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": interrupted_note,
                }
            kept.append(answer)

        out.append(message if len(usable) == len(calls) else {**message, "tool_calls": usable})
        out.extend(kept)

        for leftover in answers:
            logger.warning(
                "%sdropped an orphaned tool result %r — its call is gone", where, leftover,
            )
        i = j

    return out


# ── token estimation ──────────────────────────────────────────────────────────

def image_tokens(width: int | None = None, height: int | None = None) -> int:
    """Estimate the token cost of one image.

    The API resizes every image before inference — upscaling anything below
    roughly 544x544 and downscaling anything larger to about 1300x1300 total
    pixels — which caps the cost at :data:`IMAGE_TOKEN_CEILING`.  With no
    dimensions we assume the worst case, because under-estimating is what causes
    a surprise context overflow.
    """
    if not width or not height:
        return IMAGE_TOKEN_CEILING
    pixels = max(1, int(width) * int(height))
    return max(1, min(IMAGE_TOKEN_CEILING, round(pixels / _PIXELS_PER_TOKEN)))


def estimate_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Rough token count for a message list, base64 excluded.

    Image cost comes from :func:`image_tokens` rather than the length of the
    base64 payload, which would overstate it by orders of magnitude.  Reasoning
    text is charged in full because it does occupy the context window on
    tool-calling turns.
    """
    total = 0.0
    for msg in messages:
        value = msg.get("content")
        if isinstance(value, str):
            total += len(value) / CHARS_PER_TOKEN
        elif isinstance(value, (list, tuple)):
            for block in value:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    total += len(str(block.get("text") or "")) / CHARS_PER_TOKEN
                elif btype in _IMAGE_BLOCK_TYPES:
                    # A data URI can carry its size in the map before the "base64,".
                    total += image_tokens()
                elif btype == "file":
                    total += image_tokens()

        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str):
            total += len(reasoning) / CHARS_PER_TOKEN

        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            total += (len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))) / CHARS_PER_TOKEN
    return int(total)


def merge_reasoning(messages: Iterable[dict[str, Any]]) -> str:
    """Concatenate the chain of thought across assistant turns.

    Handy for a UI "show reasoning" pane or for persisting a transcript without
    re-walking the message list.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        value = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n\n".join(parts)
