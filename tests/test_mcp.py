"""Tests for MCP config loading, result translation, and the live transport.

Most of what goes wrong is pure translation and needs no socket. The transport's
behaviour — its timeouts, and what happens to a session when one fires — only exists
over a real socket, so a handful of tests here run a minimal Streamable HTTP server
on a real port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from pathlib import Path

import pytest

from deepseek_client.tools import ToolRegistry
from server.mcp import (
    MCPConnection,
    MCPServerSpec,
    MCPManager,
    MCPToolError,
    build_registry,
    load_mcp_config,
    read_mcp_document,
    result_to_blocks,
)


class FakeTool:
    def __init__(self, name):
        self.name = name


def write(tmp_path, payload, name: str = "mcp.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def written(path) -> object:
    """The file at `path` parsed back, for asserting what a save actually produced."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── loading ───────────────────────────────────────────────────────────────────

def test_missing_file_is_not_an_error(tmp_path):
    assert load_mcp_config(tmp_path / "absent.json") == []


def test_none_path_is_not_an_error():
    assert load_mcp_config(None) == []


def test_malformed_json_is_not_fatal(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert load_mcp_config(path) == []


def test_servers_list_form(tmp_path):
    path = write(tmp_path, {"servers": [{"name": "fs", "url": "http://localhost:1/mcp"}]})
    specs = load_mcp_config(path)
    assert len(specs) == 1
    assert specs[0].name == "fs"
    assert specs[0].enabled is True


def test_bare_list_form(tmp_path):
    path = write(tmp_path, [{"name": "one", "url": "http://localhost:1/mcp"}])
    assert [s.name for s in load_mcp_config(path)] == ["one"]


def test_mcp_servers_object_form(tmp_path):
    """The shape most MCP clients use, where the key is the name."""
    path = write(tmp_path, {"mcpServers": {"files": {"url": "http://localhost:1/mcp"}}})
    specs = load_mcp_config(path)
    assert len(specs) == 1
    assert specs[0].name == "files"


def test_entries_without_a_url_are_skipped(tmp_path):
    path = write(tmp_path, {"servers": [{"name": "bad"}, {"name": "good", "url": "http://x/mcp"}]})
    assert [s.name for s in load_mcp_config(path)] == ["good"]


def test_disabled_servers_are_dropped(tmp_path):
    path = write(tmp_path, {"servers": [
        {"name": "off", "url": "http://x/mcp", "enabled": False},
        {"name": "on", "url": "http://y/mcp", "enabled": True},
    ]})
    assert [s.name for s in load_mcp_config(path)] == ["on"]


def test_allowed_tools_accepts_both_spellings(tmp_path):
    snake = write(tmp_path, {"servers": [{"name": "a", "url": "http://x/mcp", "allowed_tools": "one"}]}, "snake.json")
    camel = write(tmp_path, {"servers": [{"name": "b", "url": "http://x/mcp", "allowedTools": ["one", "two"]}]}, "camel.json")
    assert load_mcp_config(snake)[0].allowed_tools == ("one",)
    assert load_mcp_config(camel)[0].allowed_tools == ("one", "two")


def test_names_are_generated_when_missing(tmp_path):
    path = write(tmp_path, {"servers": [{"url": "http://x/mcp"}, {"url": "http://y/mcp"}]})
    assert [s.name for s in load_mcp_config(path)] == ["mcp1", "mcp2"]


def test_legacy_transport_key_is_ignored(tmp_path):
    """`transport: sse` is the old spelling; the endpoint is what matters."""
    path = write(tmp_path, {"servers": [{"name": "a", "url": "http://x/mcp", "transport": "sse"}]})
    specs = load_mcp_config(path)
    assert len(specs) == 1
    assert specs[0].url == "http://x/mcp"


def test_endpoint_is_accepted_as_an_alias_for_url(tmp_path):
    path = write(tmp_path, {"servers": [{"name": "a", "endpoint": "http://x/mcp"}]})
    assert load_mcp_config(path)[0].url == "http://x/mcp"


def test_env_references_in_headers_are_expanded(tmp_path):
    path = write(tmp_path, {"servers": [{
        "name": "a", "url": "http://x/mcp",
        "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
    }]})
    specs = load_mcp_config(path, environ={"TEST_MCP_TOKEN": "s3cret"})
    assert specs[0].headers["Authorization"] == "Bearer s3cret"


def test_headers_with_an_unset_variable_are_dropped_not_sent_literally(tmp_path):
    """Sending `${TOKEN}` verbatim turns a clear 401 into a baffling one."""
    path = write(tmp_path, {"servers": [{
        "name": "a", "url": "http://x/mcp",
        "headers": {"Authorization": "Bearer ${NOT_SET_ANYWHERE}", "X-Trace": "keep-me"},
    }]})
    specs = load_mcp_config(path, environ={})
    assert "Authorization" not in specs[0].headers
    assert specs[0].headers["X-Trace"] == "keep-me"


def test_an_unset_variable_in_one_server_does_not_disable_the_others(tmp_path):
    path = write(tmp_path, {"servers": [
        {"name": "a", "url": "http://x/mcp", "headers": {"A": "${NOPE}"}},
        {"name": "b", "url": "http://y/mcp"},
    ]})
    assert [s.name for s in load_mcp_config(path, environ={})] == ["a", "b"]


# ── spec ──────────────────────────────────────────────────────────────────────

def test_slug_is_tool_name_safe():
    assert MCPServerSpec(name="My Files!").slug == "my_files"
    assert MCPServerSpec(name="").slug == "mcp"
    assert MCPServerSpec(name="already_fine").slug == "already_fine"


def test_tool_names_are_prefixed_so_servers_cannot_collide():
    conn = MCPConnection(MCPServerSpec(name="one", url="http://x/mcp", prefix="a"))
    assert conn.tool_name("search") == "a__search"
    assert conn.remote_name("a__search") == "search"

    # Without an explicit prefix the slug is used.
    bare = MCPConnection(MCPServerSpec(name="two", url="http://y/mcp"))
    assert bare.tool_name("search") == "two__search"
    assert bare.remote_name("two__search") == "search"


def test_remote_name_is_a_noop_for_a_foreign_name():
    conn = MCPConnection(MCPServerSpec(name="one", url="http://x/mcp", prefix="a"))
    assert conn.remote_name("other__search") == "other__search"


def test_allow_list_filters_the_exposed_tools():
    conn = MCPConnection(MCPServerSpec(
        name="one", url="http://x/mcp", allowed_tools=("search",),
    ))
    conn._tools = [FakeTool("search"), FakeTool("delete_everything")]
    assert [t.name for t in conn.exposed_specs()] == ["search"]


def test_an_empty_allow_list_exposes_every_tool():
    conn = MCPConnection(MCPServerSpec(name="one", url="http://x/mcp"))
    conn._tools = [FakeTool("a"), FakeTool("b")]
    assert [t.name for t in conn.exposed_specs()] == ["a", "b"]


# ── result translation ────────────────────────────────────────────────────────

class FakeText:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeImage:
    type = "image"

    def __init__(self, data, mime="image/png"):
        self.data = data
        self.mime_type = mime


class FakeResource:
    type = "resource"

    def __init__(self, uri, text=None, blob=None):
        self.resource = type("R", (), {"uri": uri, "text": text, "blob": blob})()


class FakeLink:
    type = "resource_link"

    def __init__(self, uri, name="doc"):
        self.uri = uri
        self.name = name


class FakeResult:
    def __init__(self, content, structured=None):
        self.content = content
        self.structured_content = structured


def test_text_content_becomes_a_text_block():
    blocks = result_to_blocks(FakeResult([FakeText("hello")]))
    assert blocks == [{"type": "text", "text": "hello"}]


def test_image_content_uses_snake_case_mime_type():
    """The SDK field is `mime_type`; older docs say `mimeType`."""
    blocks = result_to_blocks(FakeResult([FakeImage("AAA=", "image/jpeg")]))
    assert blocks == [{"type": "image", "data": "AAA=", "mimeType": "image/jpeg"}]


def test_resource_with_inline_text_is_inlined():
    blocks = result_to_blocks(FakeResult([FakeResource("file:///a.txt", text="body")]))
    assert blocks[0]["type"] == "text"
    assert "body" in blocks[0]["text"]


def test_resource_link_keeps_the_uri():
    blocks = result_to_blocks(FakeResult([FakeLink("file:///a.txt", "notes")]))
    assert "file:///a.txt" in blocks[0]["text"]


def test_structured_content_is_used_when_there_is_no_content():
    result = FakeResult([], structured={"answer": 42})
    blocks = result_to_blocks(result)
    assert len(blocks) == 1
    assert json.loads(blocks[0]["text"]) == {"answer": 42}


def test_unknown_content_types_degrade_to_text():
    class Weird:
        type = "something_new"

        def __str__(self):
            return "weird"

    blocks = result_to_blocks(FakeResult([Weird()]))
    assert blocks[0]["text"] == "weird"


def test_empty_result_produces_no_blocks():
    assert result_to_blocks(FakeResult([])) == []


# ── manager without any servers ───────────────────────────────────────────────

async def test_manager_with_no_specs_is_a_noop():
    manager = MCPManager([])
    await manager.start()
    await manager.stop()
    assert manager.status() == []
    assert manager.tools() == []


async def test_manager_status_reports_a_failed_connection(tmp_path):
    """An unreachable server must show up as an error, not crash the app."""
    spec = MCPServerSpec(name="dead", url="http://127.0.0.1:1/mcp", timeout=1.0)
    manager = MCPManager([spec])
    await manager.start()
    try:
        status = manager.status()
        assert len(status) == 1
        assert status[0]["name"] == "dead"
        assert status[0]["connected"] is False
        assert status[0]["error"]
    finally:
        await manager.stop()


async def test_a_failed_connection_still_reports_zero_tools():
    manager = MCPManager([MCPServerSpec(name="dead", url="http://127.0.0.1:1/mcp", timeout=1.0)])
    await manager.start()
    try:
        assert manager.tools() == []
    finally:
        await manager.stop()


# ── registry building ────────────────────────────────────────────────────────

async def test_build_registry_without_a_manager_returns_the_extras():
    from deepseek_client.tools import Tool

    extra = Tool(name="resize", description="d", handler=lambda: "ok")
    registry = build_registry(None, extra=[extra])
    assert "resize" in registry
    assert len(registry) == 1


async def test_build_registry_includes_connected_mcp_tools(tmp_path):
    """A dead server contributes nothing, so the built-ins must still survive."""
    from deepseek_client.tools import Tool

    manager = MCPManager([MCPServerSpec(name="dead", url="http://127.0.0.1:1/mcp", timeout=1.0)])
    await manager.start()
    try:
        extra = Tool(name="inspect_media", description="d", handler=lambda: "ok")
        registry = build_registry(manager, extra=[extra], uuid_provider=lambda: "x")
        assert "inspect_media" in registry
        assert len(registry) == 1
    finally:
        await manager.stop()


# ── the document, as the settings panel edits it ──────────────────────────────
#
# Two programs write this file: the UI and the user. The tests below pin the
# properties that make that safe — nothing invented, nothing deleted, nothing
# half-written.

def test_document_round_trip_keeps_comments_and_unknown_keys(tmp_path):
    """A save from the UI must not throw away the parts it does not understand."""
    original = {
        "_comment": ["MCP servers. Tokens go in .env, reference them as ${VAR}."],
        "servers": [{"name": "a", "url": "http://x/mcp"}],
        "someFutureKey": {"nested": True},
    }
    path = write(tmp_path, original)

    document = read_mcp_document(path)
    document.upsert({"name": "b", "url": "http://y/mcp"})
    document.save()

    reloaded = written(path)
    assert reloaded["_comment"] == original["_comment"]
    assert reloaded["someFutureKey"] == {"nested": True}
    assert [entry["name"] for entry in reloaded["servers"]] == ["a", "b"]


def test_document_keeps_env_placeholders_instead_of_baking_in_secrets(tmp_path):
    """Resolving `${VAR}` on read would write the secret back to disk on the next save."""
    path = write(tmp_path, {"servers": [{
        "name": "a", "url": "http://x/mcp",
        "headers": {"Authorization": "Bearer ${MY_MCP_TOKEN}"},
    }]})

    document = read_mcp_document(path)
    document.save()

    entry = written(path)["servers"][0]
    assert entry["headers"]["Authorization"] == "Bearer ${MY_MCP_TOKEN}"

    # Still expands for the client, just not on disk.
    specs = document.specs({"MY_MCP_TOKEN": "s3cret"})
    assert specs[0].headers["Authorization"] == "Bearer s3cret"


def test_document_writes_the_mcpservers_object_shape_back(tmp_path):
    """The file's shape is the user's; a save must not reformat it into ours."""
    path = write(tmp_path, {"mcpServers": {"files": {"url": "http://x/mcp"}}})
    document = read_mcp_document(path)
    document.upsert({"name": "web", "url": "http://y/mcp", "prefix": "ws"})
    document.save()

    reloaded = written(path)
    assert set(reloaded) == {"mcpServers"}
    assert reloaded["mcpServers"]["files"] == {"url": "http://x/mcp"}
    assert reloaded["mcpServers"]["web"]["prefix"] == "ws"
    # The name is the key in this shape, not a field.
    assert "name" not in reloaded["mcpServers"]["web"]


def test_document_writes_the_bare_list_shape_back(tmp_path):
    path = write(tmp_path, [{"name": "a", "url": "http://x/mcp"}])
    document = read_mcp_document(path)
    document.upsert({"name": "b", "url": "http://y/mcp"})
    document.save()
    assert [entry["name"] for entry in written(path)] == ["a", "b"]


def test_document_creates_the_file_from_nothing(tmp_path):
    path = tmp_path / "mcp.json"
    document = read_mcp_document(path)
    assert document.exists is False

    document.upsert({"name": "a", "url": "http://x/mcp"})
    document.save()

    assert written(path) == {"servers": [{"name": "a", "url": "http://x/mcp"}]}


def test_document_save_leaves_no_temp_file_behind(tmp_path):
    path = write(tmp_path, {"servers": []})
    document = read_mcp_document(path)
    document.upsert({"name": "a", "url": "http://x/mcp"})
    document.save()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_bom_does_not_make_the_config_unreadable(tmp_path):
    """Notepad and PowerShell both write a BOM, which plain `json.loads` rejects."""
    path = tmp_path / "mcp.json"
    payload = json.dumps({"servers": [{"name": "a", "url": "http://x/mcp"}]})
    path.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))

    document = read_mcp_document(path)
    assert document.error == ""
    assert [spec.name for spec in document.specs()] == ["a"]

    # And it survives a save, so the UI is not stuck on a file it can only read.
    document.upsert({"name": "b", "url": "http://y/mcp"})
    document.save()
    assert [entry["name"] for entry in written(path)["servers"]] == ["a", "b"]


def test_document_refuses_to_overwrite_a_file_it_could_not_read(tmp_path):
    """Saving one server must not replace a file the user is still repairing."""
    path = tmp_path / "mcp.json"
    path.write_text("{ not json at all", encoding="utf-8")

    document = read_mcp_document(path)
    assert document.error

    with pytest.raises(ValueError):
        document.save()
    assert path.read_text(encoding="utf-8") == "{ not json at all"


def test_upsert_replaces_by_the_old_name_so_a_rename_is_one_operation(tmp_path):
    path = write(tmp_path, {"servers": [
        {"name": "old", "url": "http://x/mcp"},
        {"name": "other", "url": "http://y/mcp"},
    ]})
    document = read_mcp_document(path)
    document.upsert({"name": "new", "url": "http://x/mcp"}, replacing="old")

    names = [entry["name"] for entry in document.entries]
    assert names == ["new", "other"]


def test_remove_reports_whether_it_found_anything(tmp_path):
    path = write(tmp_path, {"servers": [{"name": "a", "url": "http://x/mcp"}]})
    document = read_mcp_document(path)
    assert document.remove("a") is True
    assert document.remove("a") is False


def test_entries_without_a_name_are_given_one(tmp_path):
    """Otherwise the panel has a card it cannot address or edit."""
    path = write(tmp_path, {"servers": [{"url": "http://x/mcp"}, {"url": "http://y/mcp"}]})
    assert [entry["name"] for entry in read_mcp_document(path).entries] == ["mcp1", "mcp2"]


def test_an_unreadable_document_reports_no_specs(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("[1, 2, 3", encoding="utf-8")
    document = read_mcp_document(path)
    assert document.error
    assert document.specs() == []


def test_a_top_level_scalar_is_an_error_not_a_crash(tmp_path):
    path = write(tmp_path, 42)
    document = read_mcp_document(path)
    assert document.error
    assert document.specs() == []


def test_document_spec_ignores_the_enabled_flag(tmp_path):
    """The panel offers Test on a disabled server: that is how it is checked first."""
    path = write(tmp_path, {"servers": [
        {"name": "off", "url": "http://x/mcp", "enabled": False},
    ]})
    document = read_mcp_document(path)
    assert document.specs() == []
    assert document.spec("off").url == "http://x/mcp"
    assert document.spec("absent") is None


def test_spec_to_dict_omits_defaults_so_the_file_stays_readable():
    plain = MCPServerSpec(name="a", url="http://x/mcp").to_dict()
    assert plain == {"name": "a", "url": "http://x/mcp"}

    full = MCPServerSpec(
        name="a", url="http://x/mcp", headers={"X": "1"}, enabled=False,
        prefix="p", allowed_tools=("t",), timeout=30.0,
    ).to_dict()
    assert full == {
        "name": "a", "url": "http://x/mcp", "headers": {"X": "1"},
        "enabled": False, "prefix": "p", "allowedTools": ["t"], "timeout": 30.0,
    }


def test_spec_to_dict_round_trips_through_the_loader():
    spec = MCPServerSpec(
        name="a", url="http://x/mcp", headers={"X": "1"},
        prefix="p", allowed_tools=("t",), timeout=30.0,
    )
    assert MCPServerSpec.from_dict(spec.to_dict()) == spec


# ── connection log ────────────────────────────────────────────────────────────

async def test_a_failed_connection_explains_itself():
    """\"It does not work\" is not an answer the panel can show anyone."""
    connection = MCPConnection(MCPServerSpec(name="dead", url="http://127.0.0.1:1/mcp", timeout=1.0))
    await connection.connect()
    try:
        assert connection.error
        text = " ".join(entry["text"] for entry in connection.log)
        assert "failed" in text
        assert connection.status()["connected"] is False
        assert connection.status()["log"]
    finally:
        await connection.close()


def test_the_connection_log_is_bounded():
    connection = MCPConnection(MCPServerSpec(name="a", url="http://x/mcp"))
    for index in range(200):
        connection.note(f"line {index}")
    assert len(connection.log) == 40
    assert connection.log[-1]["text"] == "line 199"


async def test_reload_carries_the_log_across_reconnects(tmp_path):
    """The panel should show the history of a server, not just the last reconnect."""
    path = write(tmp_path, {"servers": [{"name": "dead", "url": "http://127.0.0.1:1/mcp"}]})
    manager = MCPManager(config_path=path)
    await manager.start()
    try:
        assert manager.connections
        manager.connections[0].note("a marker")
        await manager.reload()
        texts = [entry["text"] for entry in manager.connections[0].log]
        assert "a marker" in texts
        assert "reloading" in texts
    finally:
        await manager.stop()


async def test_reload_re_reads_the_config_file(tmp_path):
    path = write(tmp_path, {"servers": [{"name": "a", "url": "http://127.0.0.1:1/mcp"}]})
    manager = MCPManager(config_path=path)
    await manager.start()
    try:
        assert [c.spec.name for c in manager.connections] == ["a"]

        write(tmp_path, {"servers": [
            {"name": "a", "url": "http://127.0.0.1:1/mcp"},
            {"name": "b", "url": "http://127.0.0.1:1/mcp"},
        ]})
        await manager.reload()
        assert [c.spec.name for c in manager.connections] == ["a", "b"]
    finally:
        await manager.stop()


async def test_a_manager_without_a_path_keeps_the_specs_it_was_given():
    spec = MCPServerSpec(name="a", url="http://127.0.0.1:1/mcp")
    manager = MCPManager([spec])
    assert manager.load_specs() == [spec]


# ── the transport, over a real socket ─────────────────────────────────────────
#
# These are the only tests here that cannot be pure translation: the SDK's transport
# spawns the POST for a tool call into the task group that owns the session, so a
# socket-level timeout inside that POST does not fail the call — it kills the
# session. Nothing about that is visible through an ASGI-level client.

_TOOLS = [
    {"name": "slow", "description": "waits, then answers",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "fast", "description": "answers at once",
     "inputSchema": {"type": "object", "properties": {}}},
]


class StubServer:
    """A minimal Streamable HTTP MCP server, on a real port."""

    def __init__(self, url: str, task, server) -> None:
        self.url = url
        self._task = task
        self._server = server

    async def stop(self) -> None:
        self._server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._task, timeout=10)


@contextlib.asynccontextmanager
async def live_mcp_server(*, slow_seconds: float = 0.0):
    """Serve one `slow` tool and one `fast` tool over Streamable HTTP."""
    uvicorn = pytest.importorskip("uvicorn")
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def handler(request: Request):
        if request.method in ("GET", "OPTIONS"):
            return Response(status_code=204)
        body = await request.json()
        method, request_id = body.get("method"), body.get("id")
        if method == "initialize":
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "stub", "version": "1"},
                "capabilities": {"tools": {}},
            }})
        if method == "notifications/initialized":
            return Response(status_code=204)
        if method == "tools/list":
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"tools": _TOOLS}})
        if method == "tools/call":
            name = body["params"]["name"]
            if name == "slow":
                await asyncio.sleep(slow_seconds)
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {
                "content": [{"type": "text", "text": f"{name} ok"}], "isError": False,
            }})
        return JSONResponse(
            {"jsonrpc": "2.0", "id": request_id,
             "error": {"code": -32601, "message": f"Method not found: {method}"}},
            status_code=404,
        )

    app = Starlette(routes=[Route("/mcp", handler, methods=["POST", "GET", "OPTIONS"])])
    # Bind-and-release to get a port nothing else is using; uvicorn then takes it.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(400):
            if server.started:
                break
            await asyncio.sleep(0.025)
        assert server.started, "the stub server never came up"
        yield StubServer(f"http://127.0.0.1:{port}/mcp", task, server)
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=10)


def headers_spec(url: str, **kwargs) -> MCPServerSpec:
    """A spec that carries a header, which is what forces our own HTTP client.

    Every API-keyed server looks like this, and that is not incidental: a server
    that needs a key is usually one that answers slowly.
    """
    return MCPServerSpec(name="stub", url=url, headers={"x-api-key": "test"}, **kwargs)


async def test_a_slow_tool_call_does_not_take_the_session_down_with_it():
    """The httpx2 default read timeout is 5 s; a slow API-keyed tool exceeds it.

    Before the timeout was set explicitly this failed at 5.0 s with "Connection
    closed", and the *next* call failed instantly too — the session was already gone.
    """
    async with live_mcp_server(slow_seconds=6.0) as stub:
        connection = MCPConnection(headers_spec(stub.url, timeout=120.0))
        await connection.connect()
        try:
            assert connection.connected, connection.error
            slow = await connection.call("slow", {})
            assert slow.content[0].text == "slow ok"
            # The point of the test: the session survived the slow call.
            fast = await connection.call("fast", {})
            assert fast.content[0].text == "fast ok"
            assert connection.connected is True
        finally:
            await connection.close()


async def test_the_headers_client_outlasts_the_calls_own_budget():
    """The socket read must not fire before the dispatcher's per-request timeout.

    Both are fatal in different ways: losing the race means a dead session, so the
    margin is the fix, not a tidiness setting.
    """
    connection = MCPConnection(headers_spec("http://127.0.0.1:1/mcp", timeout=120.0))
    client = connection._build_http_client({"x-api-key": "test"})
    if client is None:
        pytest.skip("httpx2 is not installed")
    try:
        assert client.timeout.read == pytest.approx(120.0 + 30.0)
        # ...and a mistyped URL still reports quickly rather than after 150 s.
        assert client.timeout.connect < 30.0
    finally:
        await client.aclose()


async def test_a_server_that_really_died_is_reported_not_still_shown_as_connected():
    """A session that is really gone must stop reading as "connected".

    "Connected", next to a call that fails instantly, is the least actionable answer
    the panel can give.
    """
    async with live_mcp_server() as stub:
        connection = MCPConnection(headers_spec(stub.url, timeout=5.0))
        await connection.connect()
        try:
            assert connection.connected is True
            await stub.stop()
            with pytest.raises(MCPToolError):
                await connection.call("fast", {})
            assert connection.connected is False
            assert connection.status()["connected"] is False
            assert "Reload" in connection.error
            assert any("connection lost" in entry["text"] for entry in connection.log)
        finally:
            await connection.close()

