"""Turning stored messages back into something DeepSeek will accept.

Three transformations happen here, and each one exists because of a documented
API constraint rather than a preference:

1. **URLs become data URIs.** Stored messages reference ``/memory/...``; the model
   needs bytes.
2. **Images move to ``user`` messages.** DeepSeek returns 400 for an image in a
   ``system`` or ``assistant`` message. A tool that produced a screenshot has its
   text stay in the ``tool`` role and its images emitted as a following ``user``
   turn, which is the only shape that works — and that turn goes *after the whole
   group*, never between two results of a parallel call, because a user turn in
   the middle of a run of results ends the run and leaves the later calls
   unanswered.
3. **A budget is enforced, newest-first.** Replaying every image from a long
   session would blow past the request size limit. The most recent scene always
   keeps its media; older ones degrade to a text note saying media was omitted.
4. **Text files become their own characters.** An attached `.md`, `.txt` or
   `.gitignore` is read off disk and inlined as text. This is the difference
   between "a file is attached" and a file the model can actually read, and it is
   the only reason an extensionless file like `.gitignore` is usable at all. Text
   shares the newest-first discipline with media and says so when it is cut short.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import replace
from typing import Any, Iterable, Sequence

from deepseek_client.messages import image_tokens

__all__ = ["rehydrate", "strip_ui_blocks", "count_images", "MediaBudget"]

logger = logging.getLogger("deepseek_ui.rehydrate")

#: Rough per-item costs used for the greedy budget walk, in characters of base64.
_IMAGE_COST = 90_000
_FRAME_COST = 40_000

#: Estimated token cost of the text note that replaces an omitted image.
_OMITTED_NOTE = "[media already shown earlier — omitted here to save context]"

#: Below this there is no point showing a fragment of a text file, so it degrades to
#: a note the model can act on instead.
_MIN_PARTIAL_TEXT = 2_000

#: Kinds that carry bytes no amount of text handling can make readable.
_NON_TEXT_KINDS = ("image", "video", "audio")


class MediaBudget:
    """Tracks how much inline content may still be emitted, newest-first."""

    def __init__(self, characters: int) -> None:
        self.remaining = max(0, int(characters))

    def claim(self, cost: int) -> bool:
        if cost > self.remaining:
            return False
        self.remaining -= cost
        return True


def count_images(messages: Iterable[dict[str, Any]]) -> int:
    """How many image blocks a message list will produce, for limit checks."""
    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("image_url", "input_image"):
                total += 1
    return total


def strip_ui_blocks(content: Any) -> Any:
    """Drop blocks that exist only for the UI.

    Our own ``_file`` / ``_sources`` / ``_tool`` blocks are display affordances and
    are meaningless to the API. Anything whose type starts with ``_`` is ours.
    """
    if not isinstance(content, list):
        return content
    kept = [
        block
        for block in content
        if not (isinstance(block, dict) and str(block.get("type") or "").startswith("_"))
    ]
    return kept


def _text_of(blocks: Sequence[Any]) -> str:
    return "\n".join(
        str(b.get("text") or "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def rehydrate(
    messages: Sequence[dict[str, Any]],
    media,
    settings,
    *,
    budget: MediaBudget | None = None,
    text_budget: MediaBudget | None = None,
    media_root: Any = None,
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` that is safe and useful to send upstream.

    ``media`` is a :class:`~server.media.MediaStore`.
    """
    budget = budget or MediaBudget(settings.media_char_budget)

    # Pass 1 — decide which tool results still get their media inlined.
    # Newest-first, so the most recent thing the model saw is always present.
    media_indices: list[int] = []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        blocks = _decode_tool_content(message.get("content"))
        if blocks and any(b.get("type") in ("image", "video") for b in blocks):
            media_indices.append(index)

    keep_media: set[int] = set()
    for index in media_indices:
        blocks = _decode_tool_content(messages[index].get("content")) or []
        cost = 0
        for block in blocks:
            if block.get("type") == "image":
                cost += _IMAGE_COST
            elif block.get("type") == "video":
                cost += _FRAME_COST * settings.model_video_max_frames
        if budget.claim(cost):
            keep_media.add(index)
        else:
            break            # budget exhausted: everything older degrades too

    # Pass 2 — the same newest-first question for text attachments. Reading happens
    # here, once per request, so the walk below only has to format what was chosen.
    text_plan = _plan_text_attachments(
        messages,
        media,
        settings,
        text_budget or MediaBudget(getattr(settings, "text_char_budget", 200_000)),
    )

    out: list[dict[str, Any]] = []
    #: Images a tool produced, held back until its whole group has been emitted.
    #: The synthetic ``user`` turn they need is the one placement the API accepts
    #: for an image — but putting it *inside* a run of tool results breaks the
    #: call/result pairing (`assistant(T: A,B) → tool A → user → tool B` leaves B
    #: answering nothing a user turn has not already ended), which is its own 400.
    #: With parallel calls that is the normal case, not an edge case.
    held_images: list[dict[str, Any]] = []

    def _flush_images() -> None:
        if not held_images:
            return
        out.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Images returned by the tool above:"},
                *held_images,
            ],
        })
        held_images.clear()

    for index, message in enumerate(messages):
        role = message.get("role")

        if role != "tool":
            # Anything that is not a tool result ends the group, so the images the
            # group produced have to be emitted before it.
            _flush_images()

        if role == "system":
            out.append({"role": "system", "content": _text_only(message.get("content"))})
            continue

        if role == "user":
            out.append(_rehydrate_user(message, media, settings, plan=text_plan.get(index)))
            continue

        if role == "assistant":
            out.append(_rehydrate_assistant(message))
            continue

        if role != "tool":
            out.append(dict(message))
            continue

        # ── tool results ──
        blocks = _decode_tool_content(message.get("content"))
        if blocks is None:
            cleaned = dict(message)
            cleaned["content"] = _text_only(message.get("content"))
            out.append(cleaned)
            continue

        inline = index in keep_media
        text_parts: list[str] = []
        image_blocks: list[dict[str, Any]] = []

        for block in blocks:
            btype = block.get("type")

            if btype == "text":
                text_parts.append(str(block.get("text") or ""))

            elif btype == "image":
                if not inline:
                    text_parts.append(f"{_OMITTED_NOTE} ({block.get('url', '')})")
                    continue
                prepared = _image_block_from_url(block.get("url", ""), media, settings)
                if prepared is None:
                    text_parts.append(f"[image missing on disk: {block.get('url', '')}]")
                else:
                    image_blocks.append(prepared)

            elif btype == "video":
                if not inline:
                    text_parts.append(f"[video shown earlier — omitted here: {block.get('url', '')}]")
                    continue
                frames = _video_frames(block.get("url", ""), media, settings)
                if not frames:
                    text_parts.append("[video produced but no frames could be extracted]")
                else:
                    text_parts.append(f"[video — {len(frames)} sampled frames follow]")
                    image_blocks.extend(frames)

            elif btype == "audio":
                text_parts.append("[audio output — the model cannot hear this]")

            else:
                text_parts.append(str(block.get("text") or block.get("url") or ""))

        cleaned = dict(message)
        cleaned["content"] = "\n".join(p for p in text_parts if p) or "(no output)"
        out.append(cleaned)

        if image_blocks:
            # The one placement the API accepts: images in a user message. Held
            # until the group ends rather than emitted here — see `held_images`.
            held_images.extend(image_blocks)

    _flush_images()
    return _enforce_image_limit(out, settings)


def _rehydrate_user(
    message: dict[str, Any], media, settings, *, plan: dict[int, tuple[Any, bool]] | None = None
) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        cleaned = dict(message)
        cleaned["content"] = strip_ui_blocks(content)
        return cleaned

    blocks: list[dict[str, Any]] = []
    for position, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "text":
            blocks.append({"type": "text", "text": str(block.get("text") or "")})

        elif btype in ("image_url", "input_image"):
            prepared = _image_block_from_url(
                (block.get("image_url") or {}).get("url", ""), media, settings,
                detail=(block.get("image_url") or {}).get("detail"),
            )
            if prepared is None:
                blocks.append({"type": "text", "text": f"[image missing: {(block.get('image_url') or {}).get('url', '')}]"})
            else:
                blocks.append(prepared)

        elif btype == "_file":
            # Pass 2 already read it and decided whether it fits.
            entry = (plan or {}).get(position)
            blocks.extend(
                _rehydrate_file_block(
                    block, media, settings,
                    doc=entry[0] if entry else None,
                    chosen=bool(entry and entry[1]),
                    considered=entry is not None,
                )
            )

        elif str(btype or "").startswith("_"):
            continue                       # UI-only, drop

        else:
            blocks.append(dict(block))

    # A list that is nothing but text is cheaper and more portable as a string.
    if blocks and all(b.get("type") == "text" for b in blocks):
        joined = "\n".join(str(b.get("text") or "") for b in blocks).strip()
        return {**{k: v for k, v in message.items() if k != "content"}, "content": joined}

    cleaned = dict(message)
    cleaned["content"] = blocks
    return cleaned


def _plan_text_attachments(
    messages: Sequence[dict[str, Any]], media, settings, budget: MediaBudget
) -> dict[int, dict[int, tuple[Any, bool]]]:
    """Read every text attachment and decide, newest-first, which ones fit.

    Returns ``{message index: {block position: (document, chosen)}}``. A document of
    ``None`` means the file is not text (or is gone). Nothing is read twice: the
    plan holds the characters the walk below will print.
    """
    plan: dict[int, dict[int, tuple[Any, bool]]] = {}

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for position, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "_file":
                continue
            if str(block.get("kind") or "") in _NON_TEXT_KINDS:
                continue

            doc = _read_text_document(block, media, settings)
            if doc is None:
                plan.setdefault(index, {})[position] = (None, False)
                continue

            # Claim what would actually be emitted, not what the file might contain.
            if budget.claim(len(doc.text)):
                plan.setdefault(index, {})[position] = (doc, True)
                continue

            # Too big for what is left. Half a README still beats a note saying one
            # is attached, so spend what remains rather than dropping it outright.
            remaining = budget.remaining
            if remaining >= _MIN_PARTIAL_TEXT:
                partial = replace(doc, text=doc.text[:remaining], truncated=True)
                budget.claim(len(partial.text))
                plan.setdefault(index, {})[position] = (partial, True)
            else:
                plan.setdefault(index, {})[position] = (doc, False)

    return plan


def _read_text_document(block: dict[str, Any], media, settings):
    """Read a ``_file`` block's text, or ``None`` when its bytes are not text.

    The stored ``kind`` is a useful hint but not the decision: a conversation saved
    before text was classified on the way in still says ``"file"``, and a client may
    have labelled a ``.log`` as something else entirely. The bytes settle it.
    """
    url = str(block.get("url") or "")
    limit = getattr(settings, "model_text_max_chars", 40_000)

    if url.startswith("data:"):
        from .media import decode_data_uri

        decoded = decode_data_uri(url)
        return _document_from_bytes(decoded[1], limit) if decoded is not None else None

    path = media.path_for_url(url) if url else None
    if path is None:
        return None
    return media.read_text(path, max_chars=limit)


def _document_from_bytes(data: bytes, limit: int):
    """Same shape as ``MediaStore.read_text``, for bytes already in hand."""
    from .media import TextDocument, text_document

    decoded = text_document(data)
    if decoded is None:
        return None

    text, encoding = decoded
    return TextDocument(
        text=text[:limit],
        encoding=encoding,
        bytes=len(data),
        characters=len(text),
        lines=len(text.splitlines()),
        truncated=len(text) > limit,
    )


def _human_bytes(size: int) -> str:
    step = 0
    value = float(max(0, size))
    while value >= 1024 and step < 4:
        value /= 1024
        step += 1
    if step == 0:
        return f"{value:,.0f} bytes"
    return f"{value:,.1f} {'KMGT'[step - 1]}B"


def _text_file_blocks(name: str, doc) -> list[dict[str, Any]]:
    """The file itself, with a header that says what it is and whether it is whole."""
    header = (
        f"[file {name!r} — {_human_bytes(doc.bytes)}, {doc.lines:,} lines, {doc.encoding}"
        + (", partly shown" if doc.truncated else "") + "]"
    )
    text = f"{header}\n{doc.text}"
    if doc.truncated:
        text += (
            f"\n[truncated: showing {len(doc.text):,} of at least {doc.characters:,} characters — "
            f"call read_file({name!r}, offset={len(doc.text)}) for the rest]"
        )
    return [{"type": "text", "text": text}]


def _rehydrate_file_block(
    block: dict[str, Any],
    media,
    settings,
    *,
    doc: Any = None,
    chosen: bool = True,
    considered: bool = False,
) -> list[dict[str, Any]]:
    """Turn our display-only ``_file`` block into something the model can read."""
    kind = str(block.get("kind") or "file")
    name = str(block.get("name") or kind)
    url = str(block.get("url") or "")

    if kind == "video":
        frames = _video_frames(url, media, settings, marker=name)
        if not frames:
            return [{"type": "text", "text": f"[video {name!r} could not be sampled]"}]
        return [
            {"type": "text", "text": f"[video {name!r} — {len(frames)} sampled frames follow]"},
            *frames,
        ]

    if kind == "audio":
        return [{"type": "text", "text": f"[audio {name!r} attached — the model cannot hear audio]"}]

    if kind == "image":
        prepared = _image_block_from_url(url, media, settings)
        if prepared:
            return [prepared]
        return [{"type": "text", "text": f"[image {name!r} missing on disk]"}]

    if doc is not None:
        if not chosen:
            return [{
                "type": "text",
                "text": (
                    f"[file {name!r} attached — not inlined here to save context; "
                    f"read it with read_file({name!r})]"
                ),
            }]
        return _text_file_blocks(name, doc)

    if considered:
        # Read it, and it was not text — say that plainly rather than pretending a
        # file the model cannot use is attached and leaving it to guess.
        size = _size_of(url, media)
        if size is None:
            return [{"type": "text", "text": f"[file {name!r} is no longer on disk]"}]
        return [{
            "type": "text",
            "text": (
                f"[file {name!r} attached — {_human_bytes(size)} of binary data, "
                f"which cannot be read as text]"
            ),
        }]

    return [{"type": "text", "text": f"[file {name!r} attached]"}]


def _size_of(url: str, media) -> int | None:
    path = media.path_for_url(url)
    if path is None:
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None


def _rehydrate_assistant(message: dict[str, Any]) -> dict[str, Any]:
    """Assistant turns keep reasoning and tool calls, and never carry images."""
    cleaned = {k: v for k, v in message.items() if not k.startswith("_")}
    content = cleaned.get("content")

    if isinstance(content, list):
        text = _text_of(content)
        if text:
            cleaned["content"] = text
        else:
            cleaned.pop("content", None)
        if any(isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in content):
            logger.warning("dropped image blocks from an assistant turn — the API rejects those")

    # `reasoning` is our shorthand; the wire field is `reasoning_content`.
    if cleaned.get("reasoning") and not cleaned.get("reasoning_content"):
        cleaned["reasoning_content"] = cleaned.pop("reasoning")
    return cleaned


def _image_block_from_url(url: str, media, settings, *, detail: str | None = None):
    """Load an image from a stored URL and return an API-ready block."""
    if not url:
        return None

    if url.startswith("data:"):
        from .media import decode_data_uri

        decoded = decode_data_uri(url)
        if decoded is None:
            return None
        mime, data = decoded
    elif url.startswith("http://") or url.startswith("https://"):
        # An external URL the API can fetch itself; pass it through untouched.
        block: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
        if detail:
            block["image_url"]["detail"] = detail
        return block
    else:
        path = media.path_for_url(url)
        if path is None:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        import mimetypes

        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"

    resized, new_mime = media.resize_image_bytes(data, settings.model_image_max_dim)
    if new_mime:
        mime = new_mime

    # Inline images are capped at 32 MiB by the API; a drop is better than a 400.
    if len(resized) > media.limits.max_inline_bytes:
        logger.warning(
            "skipping an oversized image (%d bytes > %d)",
            len(resized), media.limits.max_inline_bytes,
        )
        return None

    encoded = base64.b64encode(resized).decode("ascii")
    block = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    if detail:
        block["image_url"]["detail"] = detail
    return block


def _video_frames(url: str, media, settings, *, marker: str = "") -> list[dict[str, Any]]:
    path = media.path_for_url(url)
    if path is None:
        return []
    frames = media.video_frames(
        path,
        target_fps=settings.model_video_fps,
        max_dim=settings.model_video_max_dim,
        max_frames=settings.model_video_max_frames,
    )
    out: list[dict[str, Any]] = []
    for frame in frames:
        encoded = base64.b64encode(frame).decode("ascii")
        out.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })
    if marker:
        logger.debug("sampled %d frames from %r", len(out), marker)
    return out


def _decode_tool_content(content: Any) -> list[dict[str, Any]] | None:
    """Tool results are stored as JSON strings; decode to a block list."""
    if isinstance(content, list):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text.startswith("["):
        return None
    import json

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, list) else None


def _text_only(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content
    if isinstance(content, list):
        text = _text_of(content)
        if text:
            return text
        return "" if not content else "[content]"
    return str(content)


def _enforce_image_limit(messages: list[dict[str, Any]], settings) -> list[dict[str, Any]]:
    """Drop the oldest images once a request would exceed the documented cap.

    The API allows 600 images per request; past that it errors. Dropping the
    oldest (rather than failing the call) keeps a long session usable.
    """
    limit = settings.model_max_images
    total = count_images(messages)
    if total <= limit:
        return messages

    logger.warning("request would carry %d images (limit %d); dropping the oldest", total, limit)
    excess = total - limit
    out: list[dict[str, Any]] = []

    for message in messages:
        content = message.get("content")
        if excess > 0 and isinstance(content, list):
            kept: list[dict[str, Any]] = []
            for block in content:
                is_image = isinstance(block, dict) and block.get("type") in ("image_url", "input_image")
                if is_image and excess > 0:
                    excess -= 1
                    kept.append({"type": "text", "text": _OMITTED_NOTE})
                    continue
                kept.append(block)
            out.append({**message, "content": kept})
        else:
            out.append(message)
    return out


def estimate_image_cost(width: int | None = None, height: int | None = None) -> int:
    """Expose the wrapper's image-token model for context budgeting."""
    return image_tokens(width, height)
