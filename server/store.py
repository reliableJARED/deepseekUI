"""Conversation persistence: one directory per conversation, one JSON file each.

Layout, mirroring the reference implementation::

    memory/
      <uuid>/
        conversation.json
        user_img_1699999999999.jpg
        tool_screenshot_0_1699999999999.png

Two properties matter and are the reason this is a class rather than loose
functions:

* **Atomic writes.** ``conversation.json`` is written to a sibling temp file and
  then ``os.replace``-d, so a crash mid-write can never leave a truncated file.
* **Per-conversation locking.** Every mutation holds an ``asyncio.Lock`` for that
  uuid. Without it, two concurrent tasks each do load -> mutate -> save with an
  ``await`` in the middle and silently overwrite one another.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid as uuidlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = ["ConversationStore", "ConversationError", "NotFoundError", "InvalidUuidError"]

#: Conservative: lowercase hex plus dashes and underscores. Enough for a uuid4 and
#: for the readable ids a human might paste in, while making it impossible to
#: escape the media root with `..`, a drive letter, or a separator.
_UUID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ConversationError(Exception):
    """Base class for conversation-store failures."""


class NotFoundError(ConversationError):
    """The requested conversation does not exist."""


class InvalidUuidError(ConversationError):
    """The supplied id is not a safe single path segment."""


def new_uuid() -> str:
    return uuidlib.uuid4().hex


@dataclass(slots=True)
class Conversation:
    """A conversation as stored on disk."""

    uuid: str
    title: str = "New conversation"
    model: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    #: Base system prompt, without the task-tracker block appended.
    sys_base: str = ""
    #: Task-tracker block. Stored separately so re-syncing it is idempotent and
    #: the prompt cannot grow on every turn.
    sys_todo: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    #: Free-form extras a future version might add; preserved on round-trip.
    extra: dict[str, Any] = field(default_factory=dict)

    # ── serialisation ──

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "uuid": self.uuid,
            "title": self.title,
            "model": self.model,
            "created": self.created,
            "updated": self.updated,
            "sys_base": self.sys_base,
            "sys_todo": self.sys_todo,
            "messages": self.messages,
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        known = {
            "uuid", "title", "model", "created", "updated",
            "sys_base", "sys_todo", "messages",
        }
        messages = data.get("messages")
        return cls(
            uuid=str(data.get("uuid") or new_uuid()),
            title=str(data.get("title") or "New conversation"),
            model=str(data.get("model") or ""),
            created=float(data.get("created") or time.time()),
            updated=float(data.get("updated") or time.time()),
            sys_base=str(data.get("sys_base") or ""),
            sys_todo=str(data.get("sys_todo") or ""),
            messages=list(messages) if isinstance(messages, list) else [],
            extra={k: v for k, v in data.items() if k not in known},
        )

    # ── summaries ──

    @property
    def is_empty(self) -> bool:
        return not self.messages

    def preview(self, limit: int = 140) -> str:
        """First chunk of the first user message, for a conversation list."""
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            text = _message_text(msg)
            if text:
                return text[:limit]
        return ""

    def summary(self) -> dict[str, Any]:
        """The lightweight shape the sidebar needs — no message bodies."""
        return {
            "uuid": self.uuid,
            "title": self.title,
            "model": self.model,
            "created": self.created,
            "updated": self.updated,
            "message_count": len(self.messages),
            "preview": self.preview(),
            "has_system_prompt": bool(self.sys_base.strip()),
        }


def _message_text(msg: dict[str, Any]) -> str:
    value = msg.get("content")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [
            str(b.get("text") or "")
            for b in value
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(p for p in parts if p).strip()
    return ""


def _cut_breaks_a_group(messages: Sequence[Any], limit: int) -> bool:
    """True when keeping only the first ``limit`` messages breaks tool pairing.

    Two shapes break it. An assistant turn carrying ``tool_calls`` that lands last
    has nothing left to answer its calls. And a ``tool`` result that lands last
    while *another* result follows it is the middle of a parallel run: the siblings
    are cut away, and the call that wanted them is left unanswered.

    A ``tool`` result at the very end is fine — the group it belongs to is complete
    — which matters because that is exactly where "regenerate" tends to cut.
    """
    last = messages[limit - 1]
    if not isinstance(last, dict):
        return False
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return True
    if last.get("role") != "tool":
        return False
    following = messages[limit] if limit < len(messages) else None
    return isinstance(following, dict) and following.get("role") == "tool"


def safe_keep(messages: Sequence[Any], keep: int) -> int:
    """The largest index ``<= keep`` that leaves the transcript tool-paired.

    A conversation may only be cut where no tool group is left half-finished,
    because the API re-validates the whole history on every request and an
    assistant turn whose calls are unanswered is a 400 *forever*. A caller's raw
    message index cannot be trusted to land on such a boundary — ``drop_tail`` one
    too far keeps the calls and drops some results — so the index is snapped back
    instead. The cost is dropping a few messages more than asked for; the
    alternative is a conversation that can never be continued.
    """
    limit = max(0, min(int(keep), len(messages)))
    while limit > 0 and _cut_breaks_a_group(messages, limit):
        limit -= 1
    return limit


class ConversationStore:
    """Loads, lists, and mutates conversations under a single media root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    # ── paths ──

    def validate(self, uuid: str) -> str:
        """Return ``uuid`` unchanged, or raise if it is not a safe path segment."""
        if not uuid or not _UUID_RE.match(uuid):
            raise InvalidUuidError(f"unsafe conversation id: {uuid!r}")
        return uuid

    def dir(self, uuid: str) -> Path:
        return self.root / self.validate(uuid)

    def file(self, uuid: str) -> Path:
        return self.dir(uuid) / "conversation.json"

    def exists(self, uuid: str) -> bool:
        try:
            return self.file(uuid).is_file()
        except InvalidUuidError:
            return False

    # ── locking ──

    def lock(self, uuid: str) -> asyncio.Lock:
        """The lock guarding every mutation of this conversation."""
        lock = self._locks.get(uuid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[uuid] = lock
        return lock

    # ── reads ──

    def load(self, uuid: str) -> Conversation:
        path = self.file(uuid)
        if not path.is_file():
            raise NotFoundError(f"no such conversation: {uuid}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConversationError(f"conversation {uuid} is corrupt: {exc}") from exc
        if not isinstance(data, dict):
            raise ConversationError(f"conversation {uuid} is not a JSON object")
        data.setdefault("uuid", uuid)
        return Conversation.from_dict(data)

    def load_or_none(self, uuid: str) -> Conversation | None:
        try:
            return self.load(uuid)
        except (NotFoundError, InvalidUuidError):
            return None

    def list(self) -> list[Conversation]:
        """Every readable conversation, newest first."""
        found: list[Conversation] = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or not (entry / "conversation.json").is_file():
                continue
            try:
                found.append(self.load(entry.name))
            except (ConversationError, InvalidUuidError):
                continue          # skip corrupt or unexpected directories
        found.sort(key=lambda c: c.updated, reverse=True)
        return found

    def summaries(self) -> list[dict[str, Any]]:
        return [c.summary() for c in self.list()]

    # ── writes ──

    def save(self, conv: Conversation) -> None:
        """Write atomically: temp file in the same directory, then ``os.replace``."""
        conv.updated = time.time()
        directory = self.dir(conv.uuid)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "conversation.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(conv.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)      # atomic on the same filesystem

    def create(
        self,
        *,
        title: str = "",
        model: str = "",
        sys_base: str = "",
        uuid: str | None = None,
    ) -> Conversation:
        """Create a conversation.

        ``uuid`` lets a caller pin the id, which is what a client that generates its
        own ids up front needs — otherwise a message posted to an unknown id would
        land in a conversation the client never learns the name of. It is validated
        with the same rules as every other id, so it cannot be used to escape.
        """
        conv = Conversation(
            uuid=self.validate(uuid) if uuid else new_uuid(),
            title=title or "New conversation",
            model=model,
            sys_base=sys_base,
        )
        self.save(conv)
        return conv

    def delete(self, uuid: str) -> bool:
        """Remove the conversation and its media. Returns False if absent."""
        import shutil

        directory = self.dir(uuid)
        if not directory.exists():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        self._locks.pop(uuid, None)
        return True

    # ── atomic mutations ──

    async def mutate(self, uuid: str, fn) -> Conversation:
        """Apply ``fn(conv)`` under the conversation's lock and save the result.

        ``fn`` may be sync or async. Raising from ``fn`` leaves the file untouched.
        """
        import inspect

        async with self.lock(uuid):
            conv = self.load(uuid)
            outcome = fn(conv)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if isinstance(outcome, Conversation):
                conv = outcome
            self.save(conv)
            return conv

    async def append(self, uuid: str, message: dict[str, Any]) -> Conversation:
        """Append one message. Returns the updated conversation."""
        def _add(conv: Conversation) -> None:
            conv.messages.append(message)

        return await self.mutate(uuid, _add)

    async def extend(self, uuid: str, messages: Iterable[dict[str, Any]]) -> Conversation:
        items = list(messages)

        def _add(conv: Conversation) -> None:
            conv.messages.extend(items)

        return await self.mutate(uuid, _add)

    async def truncate(self, uuid: str, keep: int) -> Conversation:
        """Keep the first ``keep`` messages and drop the rest.

        Used by "regenerate": truncate to just before the trailing assistant turn,
        then issue a new completion.

        ``keep`` is snapped back to a tool-group boundary by :func:`safe_keep`,
        so no caller can leave a conversation that 400s every later request.
        """
        def _cut(conv: Conversation) -> None:
            conv.messages = conv.messages[: safe_keep(conv.messages, keep)]

        return await self.mutate(uuid, _cut)

    async def replace_messages(
        self, uuid: str, messages: list[dict[str, Any]]
    ) -> Conversation:
        """Swap the whole message list — used when editing history.

        Only dicts are kept (a client-supplied list is untrusted input) and the
        result is stored exactly as given: repairing *this* list is the caller's
        job, because the repair is the same one every other outbound request gets
        and it belongs next to the request, not in the persistence layer.
        """
        cleaned = [m for m in messages if isinstance(m, dict)]

        def _swap(conv: Conversation) -> None:
            conv.messages = cleaned

        return await self.mutate(uuid, _swap)
