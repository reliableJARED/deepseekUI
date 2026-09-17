"""MCP over Streamable HTTP.

Scope is deliberately narrow: HTTP transport only. Stdio servers are a different
lifecycle problem (process supervision, restart policy, stdout framing) and mixing
the two makes both harder to reason about.

Two details of the SDK shape this module:

* A session lives inside an ``AsyncExitStack`` holding a task group open. It is not
  a request-scoped object, so connections are opened once at startup and held for
  the life of the process.
* ``mcp`` 2.x transports take an ``httpx2`` client, while the wrapper uses
  ``httpx``. These are separate packages, so an explicit client is built only when
  headers are configured and otherwise the SDK's own default is used.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import time
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from deepseek_client.config import expand_env_refs
from deepseek_client.tools import Tool, ToolRegistry, ToolResult

__all__ = [
    "MCPToolError",
    "MCPServerSpec",
    "MCPConnection",
    "MCPManager",
    "MCPDocument",
    "load_mcp_config",
    "read_mcp_document",
    "probe_server",
    "result_to_blocks",
]

logger = logging.getLogger("deepseek_ui.mcp")

#: Longest we will wait for a single tool call before giving up on the server.
DEFAULT_CALL_TIMEOUT = 120.0

#: Longest we will wait for the *handshake* (``initialize`` + ``tools/list``).
#: Deliberately shorter than :data:`DEFAULT_CALL_TIMEOUT`: a server that is slow to
#: run a tool is fine and keeps its full budget, but a handshake that takes minutes
#: is a mistyped URL, and the settings panel must not sit there for it.
CONNECT_TIMEOUT = 20.0

#: The same idea for the settings panel's "Test" button.
PROBE_TIMEOUT = 10.0

#: Lines of per-server history kept for the UI's connection log.
LOG_LIMIT = 40

#: A ``${VAR}`` that survived expansion, meaning the variable was unset.
_UNRESOLVED_REF = re.compile(r"\$\{[^}]+\}")


class MCPToolError(Exception):
    """An MCP server rejected a call or could not be reached."""


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class MCPServerSpec:
    """One Streamable HTTP server, as written in ``mcp.json``."""

    name: str
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    #: Only expose these tools. Empty means "all of them".
    allowed_tools: tuple[str, ...] = ()
    #: Prefix applied to tool names, so two servers cannot collide.
    prefix: str = ""
    #: Per-tool call timeout, in seconds.
    timeout: float = DEFAULT_CALL_TIMEOUT

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int = 0) -> "MCPServerSpec":
        url = str(data.get("url") or data.get("endpoint") or "").strip()
        name = str(data.get("name") or data.get("id") or "").strip()
        if not name:
            name = f"mcp{index + 1}"
        # Tolerate the older `transport: sse` spelling by ignoring it; we speak
        # Streamable HTTP to whatever endpoint is given.
        allowed = data.get("allowedTools") or data.get("allowed_tools") or ()
        if isinstance(allowed, str):
            allowed = [allowed]

        # A header still holding `${...}` means the variable was not set. Sending it
        # verbatim would produce a confusing 401; dropping it produces a clear one.
        headers: dict[str, str] = {}
        for key, value in (data.get("headers") or {}).items():
            text = str(value)
            if _UNRESOLVED_REF.search(text):
                logger.warning(
                    "MCP server %r: header %r references a variable that is not set; "
                    "omitting the header",
                    name, key,
                )
                continue
            headers[str(key)] = text

        return cls(
            name=name,
            url=url,
            headers=headers,
            enabled=bool(data.get("enabled", True)),
            allowed_tools=tuple(str(t) for t in allowed),
            prefix=str(data.get("prefix") or ""),
            timeout=float(data.get("timeout") or DEFAULT_CALL_TIMEOUT),
        )

    @property
    def slug(self) -> str:
        """A tool-name-safe server identifier."""
        cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in self.name)
        return cleaned.strip("_").lower() or "mcp"

    def to_dict(self) -> dict[str, Any]:
        """The entry as it should be written back to ``mcp.json``.

        Defaults are omitted rather than spelled out: this file is meant to stay
        readable and hand-editable, and a wall of ``"enabled": true`` makes the two
        lines that actually matter harder to find.
        """
        entry: dict[str, Any] = {"name": self.name, "url": self.url}
        if self.headers:
            entry["headers"] = dict(self.headers)
        if not self.enabled:
            entry["enabled"] = False
        if self.prefix:
            entry["prefix"] = self.prefix
        if self.allowed_tools:
            entry["allowedTools"] = list(self.allowed_tools)
        if self.timeout != DEFAULT_CALL_TIMEOUT:
            entry["timeout"] = self.timeout
        return entry


#: The three shapes ``mcp.json`` turns up in. Remembering which one a file used is
#: what lets the settings panel write it back without reformatting a hand-edited file.
SHAPE_SERVERS = "servers"
SHAPE_OBJECT = "mcpServers"
SHAPE_LIST = "list"


@dataclass(slots=True)
class MCPDocument:
    """``mcp.json`` kept as a *document* rather than as a list of specs.

    Two programs edit this file: the settings panel and the user, by hand. So the
    read keeps everything it does not understand — ``_comment`` blocks, unknown
    top-level keys, the ``mcpServers`` spelling, entries that are not objects — and
    the write puts the file back in the shape it was found in. Anything else would
    silently delete a comment the first time somebody saved a server through the UI,
    which is exactly the sort of thing that makes people stop trusting a GUI.
    """

    path: Path | None = None
    shape: str = SHAPE_SERVERS
    extra: dict[str, Any] = field(default_factory=dict)
    #: Server entries verbatim as they appear on disk, except that a `name` is filled
    #: in where one was missing, so every entry has a stable handle in the UI.
    entries: list[Any] = field(default_factory=list)
    #: Set when the file exists but could not be parsed. Writes are refused while this
    #: is set — saving one server must not replace a file somebody is mid-edit on.
    error: str = ""

    @property
    def exists(self) -> bool:
        return bool(self.path) and Path(self.path).is_file()

    def _expanded(self, environ: Mapping[str, str] | None) -> list[Any]:
        """A deep copy of the entries with ``${VAR}`` references resolved."""
        try:
            return expand_env_refs(copy.deepcopy(self.entries), environ, strict=False)
        except Exception as exc:                        # pragma: no cover - defensive
            logger.warning("could not expand environment references in %s: %s", self.path, exc)
            return copy.deepcopy(self.entries)

    def specs(self, environ: Mapping[str, str] | None = None) -> list[MCPServerSpec]:
        """The enabled specs, with ``${VAR}`` references resolved.

        Expansion happens here rather than on read so that the document itself keeps
        the placeholders: resolving on the way in would leave the settings panel
        looking at ``Bearer sk-...`` where the file says ``Bearer ${MY_TOKEN}``, and
        the next save would write the secret to disk in the clear.
        """
        built: list[MCPServerSpec] = []
        for index, entry in enumerate(self._expanded(environ)):
            if not isinstance(entry, dict):
                continue
            spec = MCPServerSpec.from_dict(entry, index=index)
            if not spec.url:
                logger.warning("skipping MCP server %r: no url", spec.name)
                continue
            if spec.enabled:
                built.append(spec)
        return built

    def spec(self, name: str, environ: Mapping[str, str] | None = None) -> MCPServerSpec | None:
        """One named entry as a spec — whether or not it is enabled.

        Unlike :meth:`specs` this ignores the ``enabled`` flag, because the settings
        panel offers a "Test" button on disabled servers too: that is how somebody
        checks a server before switching it back on.
        """
        for index, entry in enumerate(self._expanded(environ)):
            if isinstance(entry, dict) and str(entry.get("name") or "") == name:
                return MCPServerSpec.from_dict(entry, index=index)
        return None

    def find(self, name: str) -> dict[str, Any] | None:
        for entry in self.entries:
            if isinstance(entry, dict) and entry.get("name") == name:
                return entry
        return None

    def upsert(self, entry: Mapping[str, Any], *, replacing: str = "") -> None:
        """Add ``entry``, or replace the entry named ``replacing`` with it.

        Keyed by the *old* name so that renaming is one operation instead of a delete
        followed by an add — a rename that failed halfway would otherwise leave the
        server gone from the file.
        """
        target = self.find(replacing or str(entry.get("name") or ""))
        if target is None:
            self.entries.append(dict(entry))
            return
        target.clear()
        target.update(entry)

    def remove(self, name: str) -> bool:
        for index, entry in enumerate(self.entries):
            if isinstance(entry, dict) and entry.get("name") == name:
                del self.entries[index]
                return True
        return False

    def document(self) -> Any:
        """The JSON value :meth:`save` writes."""
        if self.shape == SHAPE_LIST:
            return self.entries
        if self.shape == SHAPE_OBJECT:
            servers: dict[str, Any] = {}
            for entry in self.entries:
                if isinstance(entry, dict):
                    servers[str(entry.get("name") or "")] = {
                        k: v for k, v in entry.items() if k != "name"
                    }
            return {**self.extra, SHAPE_OBJECT: servers}
        return {**self.extra, "servers": self.entries}

    def save(self) -> None:
        """Write the file back, atomically.

        Written to a sibling and moved into place, because this file is the only
        record of the user's servers and a crash mid-write would truncate it to
        nothing. ``os.replace`` is atomic on Windows when the destination exists,
        and on POSIX always.
        """
        if self.path is None:
            raise ValueError("this document has no path")
        if self.error:
            raise ValueError(f"refusing to overwrite an unreadable config ({self.error})")

        path = Path(self.path)
        payload = json.dumps(self.document(), indent=2, ensure_ascii=False) + "\n"
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, path)


def read_mcp_document(path: str | Path | None) -> MCPDocument:
    """Read ``mcp.json``, keeping it in the shape it was written in.

    Values are left exactly as written. A header of ``Bearer ${MY_TOKEN}`` has to
    survive a round trip through the settings panel untouched.
    """
    if not path:
        return MCPDocument()

    file = Path(path)
    document = MCPDocument(path=file)
    if not file.is_file():
        logger.info("no MCP config at %s; running without MCP", file)
        return document

    try:
        # `utf-8-sig` for the same reason the provider config uses it: Notepad and
        # PowerShell's `Set-Content -Encoding utf8` both write a BOM, and a BOM makes
        # `json.loads` fail on a file that is otherwise perfectly valid.
        data = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        document.error = f"{type(exc).__name__}: {exc}"
        logger.error("could not read MCP config %s: %s", file, exc)
        return document

    entries: list[Any]
    if isinstance(data, list):
        document.shape, entries = SHAPE_LIST, list(data)
    elif isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        document.shape = SHAPE_OBJECT
        entries = [
            {"name": str(key), **value} if isinstance(value, dict) else {"name": str(key), "url": value}
            for key, value in data["mcpServers"].items()
        ]
        document.extra = {k: v for k, v in data.items() if k != SHAPE_OBJECT}
    elif isinstance(data, dict):
        document.shape = SHAPE_SERVERS
        raw = data.get("servers")
        entries = list(raw) if isinstance(raw, list) else []
        document.extra = {k: v for k, v in data.items() if k != "servers"}
    else:
        document.error = f"expected an object or a list at the top level, found {type(data).__name__}"
        logger.error("MCP config %s: %s", file, document.error)
        return document

    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and not (entry.get("name") or entry.get("id")):
            # Named here rather than on save, so a nameless entry still gets a card in
            # the settings panel instead of being invisible and un-editable.
            entry["name"] = f"mcp{index + 1}"

    document.entries = entries
    return document


def load_mcp_config(path: str | Path | None, *, environ: Mapping[str, str] | None = None) -> list[MCPServerSpec]:
    """Read ``mcp.json``. Returns an empty list when the file is absent.

    Accepts either ``{"servers": [...]}`` or a bare list, and understands the
    ``mcpServers`` object form used by most MCP clients::

        {"mcpServers": {"files": {"url": "http://..."}}}

    ``${VAR}`` references are expanded through the same helper the provider config
    uses, because MCP servers are commonly authenticated and a bearer token has no
    business sitting in a plain text file that gets shared or committed.
    """
    return read_mcp_document(path).specs(environ)


# ── result translation ────────────────────────────────────────────────────────

def result_to_blocks(result: Any) -> list[dict[str, Any]]:
    """Flatten an MCP ``CallToolResult`` into our internal block list.

    Images arrive base64-encoded and are handed on as ``image`` blocks, which the
    media store persists and the rehydrator turns back into ``image_url`` blocks in
    a ``user`` message — the only placement the API accepts.
    """
    blocks: list[dict[str, Any]] = []

    for item in getattr(result, "content", None) or []:
        kind = getattr(item, "type", None)

        if kind == "text" or (kind is None and hasattr(item, "text")):
            blocks.append({"type": "text", "text": str(getattr(item, "text", ""))})

        elif kind == "image":
            blocks.append({
                "type": "image",
                "data": str(getattr(item, "data", "")),
                "mimeType": str(getattr(item, "mime_type", "") or "image/png"),
            })

        elif kind == "audio":
            blocks.append({
                "type": "audio",
                "data": str(getattr(item, "data", "")),
                "mimeType": str(getattr(item, "mime_type", "") or "audio/wav"),
            })

        elif kind == "resource":
            resource = getattr(item, "resource", None)
            uri = str(getattr(resource, "uri", "") or "")
            mime = str(getattr(resource, "mime_type", "") or "")
            blob = getattr(resource, "blob", None)
            text = getattr(resource, "text", None)

            if blob and mime.startswith("image/"):
                blocks.append({"type": "image", "data": str(blob), "mimeType": mime})
            elif blob and mime.startswith("video/"):
                blocks.append({"type": "video", "data": str(blob), "mimeType": mime})
            elif blob:
                blocks.append({"type": "text", "text": f"[binary resource {uri} ({mime}) — not displayable]"})
            elif text is not None:
                blocks.append({"type": "text", "text": f"[{uri}]\n{text}"})
            else:
                blocks.append({"type": "text", "text": f"[resource {uri}]"})

        elif kind == "resource_link":
            blocks.append({
                "type": "text",
                "text": f"[link {getattr(item, 'name', '')}: {getattr(item, 'uri', '')}]",
            })

        else:
            blocks.append({"type": "text", "text": str(item)})

    # Some servers return only structured content.
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured and not blocks:
        blocks.append({"type": "text", "text": json.dumps(structured, ensure_ascii=False, default=str)})

    return blocks


# ── connection ────────────────────────────────────────────────────────────────

class MCPConnection:
    """A live Streamable HTTP session to one MCP server."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._tools: list[Any] = []
        self._lock = asyncio.Lock()
        self._http_client: Any = None
        self.connected = False
        self.error: str = ""
        #: Recent history, newest last. Kept because "it does not work" is not an
        #: answer the settings panel can show the user — the reason is.
        self.log: list[dict[str, Any]] = []

    def _note(self, text: str) -> None:
        self.log.append({"at": time.time(), "text": text})
        del self.log[:-LOG_LIMIT]

    def _reset_log(self) -> None:
        self.log = []

    def note(self, text: str) -> None:
        """Record a line from outside the connection (used by the manager)."""
        self._note(text)

    # ── lifecycle ──

    async def connect(self) -> None:
        """Open the session and cache the tool listing. Never raises."""
        # The transport cancels the in-flight handshake when it cannot reach the
        # server, and that cancellation is delivered to whoever is waiting for the
        # answer. Left alone, `connect` would raise CancelledError for a server that is
        # merely down — and CancelledError derives from BaseException, so it would sail
        # past every `except Exception` in this module and surface as a 500. Running the
        # handshake in its own task separates the two cases: a cancelled *attempt* is a
        # failure to report, while a cancellation of *us* is still a cancellation.
        attempt = asyncio.ensure_future(self._handshake())
        try:
            await attempt
        except asyncio.CancelledError:
            if not attempt.cancelled():
                attempt.cancel()
                raise
            self._fail("the server did not answer — the handshake was cancelled before it finished")

    async def _handshake(self) -> None:
        """Enter the session, initialize, and cache the tool list.

        Raises only for a cancellation of this attempt. Every other failure is recorded
        on the connection and reported through `error` and the log.
        """
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:                     # pragma: no cover
            self._fail(f"mcp package not installed: {exc}")
            return

        self._note(f"connecting to {self.spec.url}")
        headers = dict(self.spec.headers)
        http_client = None
        if headers:
            self._http_client = http_client = self._build_http_client(headers)
            if http_client is None:
                self._note("this server sets headers, but httpx2 is not installed")

        stack = AsyncExitStack()
        try:
            streams = await stack.enter_async_context(
                streamable_http_client(self.spec.url, http_client=http_client)
            )
            read_stream, write_stream = streams[0], streams[1]
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    # The handshake gets the shorter budget; `call_tool` passes the
                    # configured timeout per call, so tool calls are unaffected.
                    read_timeout_seconds=min(self.spec.timeout, CONNECT_TIMEOUT),
                )
            )
            await session.initialize()
            listing = await session.list_tools()
            self._tools = list(getattr(listing, "tools", []) or [])
        except BaseException as exc:
            # Closed either way: a half-entered transport left open makes anyio
            # complain at loop shutdown, and that noise hides the real error.
            with suppress(Exception):
                await stack.aclose()
            if isinstance(exc, asyncio.CancelledError):
                raise
            self._fail(f"{type(exc).__name__}: {exc}")
            return

        self._stack = stack
        self._session = session
        self.connected = True
        self.error = ""
        names = [str(getattr(t, "name", "")) for t in self._tools]
        self._note(f"connected — {len(self._tools)} tool(s): {', '.join(names) or 'none'}")
        logger.info(
            "MCP server %r connected at %s (%d tools)",
            self.spec.name, self.spec.url, len(self._tools),
        )

    def _fail(self, reason: str) -> None:
        """Record a failed connection once, in the two places that display it."""
        self.error = reason
        self._note(f"failed — {reason}")
        logger.error("MCP server %r failed to connect — %s", self.spec.name, reason)

    @staticmethod
    def _build_http_client(headers: Mapping[str, str]):
        """An ``httpx2`` client carrying static headers, when that package exists.

        ``mcp`` 2.x transports want ``httpx2``, which is a distinct package from the
        ``httpx`` the rest of this project uses. If it is missing we simply skip the
        headers rather than reaching for the wrong client type.
        """
        try:
            import httpx2
        except ImportError:
            logger.warning("httpx2 unavailable; MCP headers will not be sent")
            return None
        return httpx2.AsyncClient(headers=dict(headers), follow_redirects=True)

    async def close(self) -> None:
        stack, self._stack = self._stack, None
        client, self._http_client = self._http_client, None
        self._session = None
        self.connected = False
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:                   # pragma: no cover - shutdown noise
                logger.debug("error closing MCP server %r: %s", self.spec.name, exc)
        if client is not None:
            # The transport does not own a client handed to it, and the settings panel
            # reconnects on every save, so leaking one client per reload would add up.
            with suppress(Exception):
                await client.aclose()

    # ── tools ──

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    def tool_name(self, remote_name: str) -> str:
        """The name we expose to the model for a remote tool."""
        return f"{self.spec.prefix or self.spec.slug}__{remote_name}"

    def remote_name(self, exposed: str) -> str:
        prefix = f"{self.spec.prefix or self.spec.slug}__"
        return exposed[len(prefix):] if exposed.startswith(prefix) else exposed

    def exposed_specs(self) -> list[Any]:
        """Remote tools after the allow-list filter."""
        allow = set(self.spec.allowed_tools)
        return [t for t in self._tools if not allow or getattr(t, "name", "") in allow]

    def status(self) -> dict[str, Any]:
        """This connection, as the settings panel needs to see it."""
        return {
            "name": self.spec.name,
            "url": self.spec.url,
            "enabled": self.spec.enabled,
            "connected": self.connected,
            "error": self.error,
            "prefix": self.spec.prefix or self.spec.slug,
            "timeout": self.spec.timeout,
            "tools": [str(getattr(t, "name", "")) for t in self.exposed_specs()],
            "log": list(self.log),
        }

    async def call(self, remote_name: str, arguments: Mapping[str, Any] | None = None):
        if self._session is None:
            raise MCPToolError(f"MCP server {self.spec.name!r} is not connected")
        async with self._lock:
            try:
                result = await self._session.call_tool(
                    remote_name, dict(arguments or {}), read_timeout_seconds=self.spec.timeout
                )
            except Exception as exc:
                raise MCPToolError(f"{self.spec.name}:{remote_name} failed — {exc}") from exc
        if getattr(result, "is_error", False):
            raise MCPToolError(f"{self.spec.name}:{remote_name} returned an error")
        return result


# ── manager ───────────────────────────────────────────────────────────────────

class MCPManager:
    """Owns every connection and turns them into wrapper-ready tools."""

    def __init__(
        self,
        specs: Sequence[MCPServerSpec] | None = None,
        *,
        config_path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.specs = list(specs or [])
        self.connections: list[MCPConnection] = []
        #: Where the specs came from, so a reload can pick up a file the settings
        #: panel (or the user's editor) just changed.
        self.config_path = Path(config_path) if config_path else None
        self.environ = dict(environ or {})
        self._ingest = None

    # ── lifecycle ──

    def load_specs(self) -> list[MCPServerSpec]:
        """Re-read the config file. Falls back to the current list when there is none."""
        if self.config_path is None:
            return list(self.specs)
        return load_mcp_config(self.config_path, environ=self.environ)

    def _open_connections(self) -> None:
        self.connections = [MCPConnection(spec) for spec in self.specs]

    async def _connect_all(self) -> None:
        if not self.connections:
            return
        await asyncio.gather(*(c.connect() for c in self.connections), return_exceptions=True)
        live = sum(1 for c in self.connections if c.connected)
        logger.info("MCP ready: %d/%d servers connected", live, len(self.connections))

    async def start(self) -> None:
        """Connect to every configured server, in parallel. Failures are logged."""
        if not self.specs and self.config_path is not None:
            # Built with a path but no specs: read them now rather than starting up
            # with nothing to do and looking like a misconfiguration.
            self.specs = self.load_specs()
        if not self.specs:
            logger.info("no MCP servers configured")
            return
        self._open_connections()
        await self._connect_all()

    async def reload(self, specs: Sequence[MCPServerSpec] | None = None) -> None:
        """Drop every connection and reconnect, optionally against a new spec list.

        With no argument the spec list is re-read from the config file, which is what
        both the settings panel and a hand edit followed by "Reload" need. Each
        server's log is carried across, so the panel shows the whole history of a
        server rather than only what happened since the last reconnect.
        """
        history = {c.spec.name: list(c.log) for c in self.connections}
        await self.stop()
        self.specs = list(specs) if specs is not None else self.load_specs()
        self._open_connections()
        for connection in self.connections:
            carried = history.get(connection.spec.name)
            if carried:
                connection.log = carried
                connection._note("reloading")
        await self._connect_all()

    async def stop(self) -> None:
        for connection in self.connections:
            await connection.close()
        self.connections = []

    # ── tool exposure ──

    def set_ingest(self, ingest) -> None:
        """Register the callback that persists media a tool returns.

        Signature: ``ingest(uuid, blocks, tool_name) -> blocks``.
        """
        self._ingest = ingest

    def tools(self, *, uuid_provider=None) -> list[Tool]:
        """Build one :class:`Tool` per remote tool.

        ``uuid_provider`` is a zero-argument callable returning the conversation id
        currently being served, so tool output lands in the right media directory.
        """
        built: list[Tool] = []
        for connection in self.connections:
            if not connection.connected:
                continue
            for remote in connection.exposed_specs():
                built.append(self._make_tool(connection, remote, uuid_provider))
        return built

    def _make_tool(self, connection: MCPConnection, remote: Any, uuid_provider) -> Tool:
        remote_name = str(getattr(remote, "name", "") or "")
        exposed = connection.tool_name(remote_name)
        description = str(getattr(remote, "description", "") or "").strip()
        schema = getattr(remote, "input_schema", None) or getattr(remote, "inputSchema", None) or {}

        async def handler(**arguments: Any) -> ToolResult:
            result = await connection.call(remote_name, arguments)
            blocks = result_to_blocks(result)

            uuid = uuid_provider() if callable(uuid_provider) else None
            if uuid and self._ingest is not None:
                try:
                    blocks = self._ingest(uuid, blocks, connection.spec.slug)
                except Exception as exc:               # pragma: no cover - defensive
                    logger.warning("could not persist media from %s: %s", exposed, exc)

            # Tool content has to be JSON when it carries blocks.
            return ToolResult(
                content=blocks if len(blocks) != 1 or blocks[0].get("type") != "text"
                else str(blocks[0].get("text") or ""),
                is_error=False,
                meta={"server": connection.spec.name, "tool": remote_name},
            )

        return Tool(
            name=exposed,
            description=description or f"MCP tool {remote_name} from {connection.spec.name}",
            parameters=dict(schema) if isinstance(schema, dict) and schema else
            {"type": "object", "properties": {}},
            handler=handler,
        )

    def register_into(self, registry: ToolRegistry, **kwargs: Any) -> ToolRegistry:
        for tool in self.tools(**kwargs):
            registry.add(tool)
        return registry

    # ── diagnostics ──

    def status(self) -> list[dict[str, Any]]:
        return [connection.status() for connection in self.connections]


async def probe_server(spec: MCPServerSpec) -> dict[str, Any]:
    """Connect to one server *without* touching the manager or the config file.

    Backs the settings panel's "Test" button. The answer to "does this URL work?" has
    to be obtainable before committing to it, otherwise trying a server means editing
    the file, reloading, and editing it back.
    """
    connection = MCPConnection(replace(spec, timeout=min(spec.timeout, PROBE_TIMEOUT)))
    await connection.connect()
    try:
        return connection.status()
    finally:
        await connection.close()


def build_registry(
    manager: MCPManager | None,
    extra: Iterable[Tool] = (),
    **kwargs: Any,
) -> ToolRegistry:
    """Combine built-in tools with whatever the MCP servers expose."""
    registry = ToolRegistry(list(extra))
    if manager is not None:
        manager.register_into(registry, **kwargs)
    return registry
