"""End-to-end tests for the HTTP surface.

These exercise the routes through a real ASGI app rather than calling handlers
directly, because the things most likely to break here are exactly the things unit
tests miss: route ordering, media that survives a round trip through disk, path
traversal, and the shape of the SSE stream.

No test touches the network. A turn against a configured provider would need a live
key, so the streaming path is only asserted to fail cleanly.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest

from server.app import create_app

PNG_1400x900 = None  # filled by the fixture below


def make_png(width: int = 1400, height: int = 900) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), (30, 90, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


PROVIDERS = {
    "providers": [
        {
            "name": "DeepSeek",
            "vendor": "customendpoint",
            # A literal key: these tests must not depend on the developer's shell.
            "apiKey": "test-key-not-real",
            "apiType": "chat-completions",
            "models": [
                {
                    "id": "deepseek-flash",
                    "name": "deepseek-flash",
                    "url": "https://api.deepseek.com",
                    "toolCalling": True,
                    "vision": True,
                    "maxInputTokens": 1000000,
                    "maxOutputTokens": 393216,
                },
                {
                    "id": "deepseek-v4-pro",
                    "name": "deepseek-v4-pro",
                    "url": "https://api.deepseek.com",
                    "toolCalling": True,
                    "vision": False,
                    "maxInputTokens": 1000000,
                    "maxOutputTokens": 393216,
                },
            ],
        }
    ]
}


def build_app(tmp_path: Path, *, api_key: str = "test-key-not-real"):
    """An app rooted entirely inside ``tmp_path``."""
    providers = tmp_path / "providers.json"
    payload = json.loads(json.dumps(PROVIDERS))
    payload["providers"][0]["apiKey"] = api_key
    providers.write_text(json.dumps(payload), encoding="utf-8")

    return create_app(
        memory_root=str(tmp_path / "memory"),
        providers_path=str(providers),
        env_file=str(tmp_path / "does-not-exist.env"),
        frontend_dir=str(tmp_path / "frontend"),
        mcp_config_path=str(tmp_path / "no-mcp.json"),
    )


@pytest.fixture
def app(tmp_path):
    return build_app(tmp_path)


@pytest.fixture
async def client(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def new_conversation(client, **body) -> str:
    response = await client.post("/api/conversations", json=body)
    assert response.status_code == 200, response.text
    return response.json()["uuid"]


# ── props ─────────────────────────────────────────────────────────────────────

async def test_props_reports_the_configured_models(client):
    response = await client.get("/api/props")
    assert response.status_code == 200

    data = response.json()
    assert data["configured"] is True

    by_id = {model["id"]: model for model in data["models"]}
    assert set(by_id) == {"deepseek-flash", "deepseek-v4-pro"}

    # The vision split is the reason the UI has to warn before attaching an image.
    assert by_id["deepseek-flash"]["vision"] is True
    assert by_id["deepseek-v4-pro"]["vision"] is False
    assert by_id["deepseek-flash"]["tool_calling"] is True
    assert by_id["deepseek-flash"]["max_output_ceiling"] == 393216

    assert data["reasoning_efforts"] == ["none", "low", "high", "max"]
    assert "resize_image" in data["tools"]
    assert data["mcp"] == []


async def test_props_stays_useful_when_the_key_is_missing(tmp_path):
    """The frontend must still boot offline, so this returns 200, not 500."""
    app = build_app(tmp_path, api_key="${DEFINITELY_NOT_SET_ANYWHERE}")
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/api/props")

    assert response.status_code == 200
    data = response.json()
    assert data["configured"] is False
    assert "error" in data
    # Static config still has to be present or the UI cannot render its controls.
    assert data["reasoning_efforts"] == ["none", "low", "high", "max"]
    assert "resize_image" in data["tools"]
    assert data["limits"]["model_max_images"] == 600


async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── conversation lifecycle ────────────────────────────────────────────────────

async def test_create_list_and_fetch(client):
    uuid = await new_conversation(client, title="Smoke", sys_base="You are terse.")

    listed = await client.get("/api/conversations")
    assert [c["uuid"] for c in listed.json()["conversations"]] == [uuid]

    fetched = await client.get(f"/api/conversations/{uuid}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["title"] == "Smoke"
    assert body["system_prompt"] == "You are terse."
    assert body["messages"] == []


async def test_empty_system_prompt_composes_to_empty_string(client):
    uuid = await new_conversation(client)
    body = (await client.get(f"/api/conversations/{uuid}")).json()
    assert body["system_prompt"] == ""


async def test_append_message_counts_and_reports_context(client):
    uuid = await new_conversation(client)

    response = await client.post(f"/api/conversations/{uuid}/messages", json={"content": "hello"})
    assert response.status_code == 200
    assert response.json()["message_count"] == 1

    context = response.json()["context"]
    assert context["limit"] == 1000000
    assert context["safe"] is True
    # Just the user turn: an empty system prompt must not be sent as a message.
    assert context["messages"] == 1


async def test_appending_to_a_missing_conversation_creates_one(client):
    response = await client.post("/api/conversations/deadbeef/messages", json={"content": "hi"})
    assert response.status_code == 200
    assert response.json()["uuid"] == "deadbeef"
    assert (await client.get("/api/conversations/deadbeef")).status_code == 200


async def test_rename_and_reject_empty_title(client):
    uuid = await new_conversation(client)

    assert (await client.post(f"/api/conversations/{uuid}/title", json={})).status_code == 400

    response = await client.post(f"/api/conversations/{uuid}/title", json={"title": "Renamed"})
    assert response.status_code == 200
    assert response.json()["title"] == "Renamed"


async def test_set_system_prompt_replaces_the_stored_base(client):
    uuid = await new_conversation(client, sys_base="old")

    response = await client.post(
        f"/api/conversations/{uuid}/system",
        json={"sys_base": "new base", "sys_todo": "task one"},
    )
    assert response.status_code == 200
    assert response.json()["sys_base"] == "new base"

    body = (await client.get(f"/api/conversations/{uuid}")).json()
    # The tracker lives between markers so the model can find and rewrite it, and so
    # re-syncing it on every turn is idempotent instead of growing the prompt.
    assert body["system_prompt"].startswith("new base\n\n")
    assert "===== YOUR CURRENT TASK TRACKER =====" in body["system_prompt"]
    assert "task one" in body["system_prompt"]
    assert body["system_prompt"].rstrip().endswith("===== END TASK TRACKER =====")


async def test_set_model_validates_against_the_provider(client):
    uuid = await new_conversation(client)

    bad = await client.post(f"/api/conversations/{uuid}/model", json={"model": "gpt-9"})
    assert bad.status_code == 400

    good = await client.post(f"/api/conversations/{uuid}/model", json={"model": "deepseek-v4-pro"})
    assert good.status_code == 200
    assert (await client.get(f"/api/conversations/{uuid}")).json()["model"] == "deepseek-v4-pro"


async def test_truncate_by_keep_and_by_drop_tail(client):
    uuid = await new_conversation(client)
    for text in ("one", "two", "three"):
        await client.post(f"/api/conversations/{uuid}/messages", json={"content": text})

    dropped = await client.post(f"/api/conversations/{uuid}/truncate", json={"drop_tail": 1})
    assert dropped.json()["message_count"] == 2

    kept = await client.post(f"/api/conversations/{uuid}/truncate", json={"keep": 1})
    assert kept.json()["message_count"] == 1

    body = (await client.get(f"/api/conversations/{uuid}")).json()
    assert body["messages"][0]["content"] == "one"


async def test_replace_messages_requires_a_list(client):
    uuid = await new_conversation(client)
    assert (await client.put(f"/api/conversations/{uuid}/messages", json={"messages": "no"})).status_code == 400

    response = await client.put(
        f"/api/conversations/{uuid}/messages",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["message_count"] == 1


def tool_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }


async def put_messages(client, uuid: str, messages: list[dict]) -> dict:
    """Seed a conversation with a verbatim transcript, bypassing the HTTP layer."""
    response = await client.put(f"/api/conversations/{uuid}/messages", json={"messages": messages})
    assert response.status_code == 200, response.text
    return response.json()


async def test_truncating_mid_tool_group_snaps_back(client):
    """`drop_tail` is a raw count, and one too far slices a tool group in half.

    That half-group is not a one-off failure: the same history is replayed on every
    later request, so the conversation would 400 forever. The boundary is snapped
    back instead.
    """
    uuid = await new_conversation(client)
    await put_messages(client, uuid, [
        {"role": "user", "content": "look these up"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("c1"), tool_call("c2")]},
        {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "name": "lookup", "content": "two"},
    ])

    # 3 of 4 messages keeps c1's result and drops c2's — the exact slice that used
    # to be written to disk and then rejected on every subsequent request.
    dropped = await client.post(f"/api/conversations/{uuid}/truncate", json={"drop_tail": 1})
    assert dropped.json()["message_count"] == 1

    body = (await client.get(f"/api/conversations/{uuid}")).json()
    assert [m["role"] for m in body["messages"]] == ["user"]


async def test_editing_history_repairs_an_unanswered_call(client):
    """A client-supplied transcript is the one write door broken input comes through."""
    uuid = await new_conversation(client)
    await put_messages(client, uuid, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("c1"), tool_call("c2")]},
        {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "one"},
    ])

    body = (await client.get(f"/api/conversations/{uuid}")).json()
    results = [m for m in body["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]
    assert results[1]["content"].startswith("[no result recorded")


async def test_importing_a_half_finished_group_repairs_it(client):
    """An export taken mid-loop carries the half-group with it."""
    response = await client.post("/api/conversations/import", json={
        "title": "Interrupted",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("c1")]},
        ],
    })
    assert response.status_code == 200

    imported = response.json()
    assert [m["role"] for m in imported["messages"]] == ["user", "assistant", "tool"]
    assert imported["messages"][-1]["tool_call_id"] == "c1"


async def test_delete_removes_the_directory(client, app):
    uuid = await new_conversation(client)
    storage = Path(app.state.app_state.settings.memory_root)

    assert (await client.delete(f"/api/conversations/{uuid}")).status_code == 200
    assert not (storage / uuid).exists()
    assert (await client.get(f"/api/conversations/{uuid}")).status_code == 404


# ── media ─────────────────────────────────────────────────────────────────────

async def test_inline_base64_becomes_a_disk_file(client, app):
    """Base64 must never reach conversation.json — it would bloat every reload."""
    uuid = await new_conversation(client)
    uri = "data:image/png;base64," + base64.b64encode(make_png(1400, 900)).decode()

    response = await client.post(
        f"/api/conversations/{uuid}/messages",
        json={
            "content": [
                {"type": "text", "text": "what colour?"},
                {"type": "image_url", "image_url": {"url": uri, "detail": "high"}},
            ]
        },
    )
    assert response.status_code == 200, response.text

    stored = response.json()["message"]["content"]
    assert "base64" not in json.dumps(stored)
    assert stored[1]["image_url"]["url"].startswith(f"/memory/{uuid}/")
    assert stored[1]["detail"] if "detail" in stored[1] else True
    # Dimensions are recorded so the tool budget and the UI can use them.
    assert (stored[1]["width"], stored[1]["height"]) == (1400, 900)

    # The file on disk is the original, untouched — no resizing while ingesting.
    served = await client.get(stored[1]["image_url"]["url"])
    assert served.status_code == 200
    assert served.content == make_png(1400, 900)
    assert served.headers["cache-control"] == "private, max-age=3600"


# ── serving media back ────────────────────────────────────────────────────────

# A minimal RIFF/WEBP header. The sniffer only reads the first 32 bytes, and the
# regression these guard is the response header, not the decoded pixels.
WEBP_HEADER = (
    b"RIFF" + (92).to_bytes(4, "little") + b"WEBP"
    + b"VP8 " + (80).to_bytes(4, "little") + b"\x00" * 80
)


def store_file(app, uuid: str, name: str, data: bytes) -> str:
    """Drop a file straight into a conversation's media directory."""
    directory = Path(app.state.app_state.settings.memory_root) / uuid
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(data)
    return f"/memory/{uuid}/{name}"


async def test_webp_is_served_as_an_image(client, app):
    """Windows' stdlib table has no ``.webp`` entry, so this used to fall through to
    the default and serve genuine WebP images as ``text/plain`` — unrenderable."""
    uuid = await new_conversation(client)
    url = store_file(app, uuid, "upload.webp", WEBP_HEADER)

    response = await client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"


async def test_served_type_comes_from_content_not_the_extension(client, app):
    """The extension is only whatever the upload happened to be called."""
    uuid = await new_conversation(client)
    url = store_file(app, uuid, "mislabelled.bin", make_png(8, 8))

    response = await client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


async def test_unrecognised_bytes_fall_back_to_the_extension(client, app):
    uuid = await new_conversation(client)
    url = store_file(app, uuid, "notes.txt", b"just some words")

    response = await client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


async def test_unknown_bytes_with_no_helpful_extension_are_octet_stream(client, app):
    uuid = await new_conversation(client)
    url = store_file(app, uuid, "blob", b"\x01\x02\x03\x04" * 8)

    response = await client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


async def test_download_uses_the_same_content_type(client, app):
    uuid = await new_conversation(client)
    store_file(app, uuid, "upload.webp", WEBP_HEADER)

    response = await client.get(f"/download/{uuid}/upload.webp")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert "attachment" in response.headers["content-disposition"]


def test_served_mime_degrades_to_the_extension_when_unreadable(tmp_path):
    """An unreadable file must not raise — the extension is the last resort."""
    from server.routes import served_mime

    assert served_mime(tmp_path / "absent.png") == "image/png"
    assert served_mime(tmp_path / "absent") == "application/octet-stream"


async def test_unsupported_image_type_is_rejected_with_a_useful_message(client):
    uuid = await new_conversation(client)
    uri = "data:image/tiff;base64," + base64.b64encode(b"II*\x00nonsense").decode()

    response = await client.post(
        f"/api/conversations/{uuid}/messages",
        json={"content": [{"type": "image_url", "image_url": {"url": uri}}]},
    )
    assert response.status_code == 400
    assert "JPEG, PNG, GIF, and WebP" in response.json()["error"]


async def test_media_filenames_cannot_escape_the_conversation(client):
    uuid = await new_conversation(client)

    for attempt in (f"/memory/{uuid}/../../../providers.json", "/memory/..%2f..%2fproviders.json"):
        response = await client.get(attempt)
        assert response.status_code in (400, 404), attempt


async def test_upload_sniffs_the_type_instead_of_trusting_the_name(client):
    """DeepSeek detects image formats from content; a lying extension must not fool us."""
    uuid = await new_conversation(client)
    fake_mp4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("innocent.png", fake_mp4, "image/png")},
    )
    assert response.status_code == 200
    assert "video" in response.json()["mime"]


async def test_upload_records_image_dimensions(client):
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("shot.png", make_png(300, 200), "image/png")},
    )
    assert response.status_code == 200
    block = response.json()["block"]
    assert (block["width"], block["height"]) == (300, 200)
    assert block["type"] == "image_url"


async def test_attach_adds_a_user_turn_in_one_step(client):
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/attach",
        files={"file": ("note.png", make_png(64, 64), "image/png")},
        data={"text": "here you go", "kind": "image"},
    )
    assert response.status_code == 200
    assert response.json()["message_count"] == 1

    body = (await client.get(f"/api/conversations/{uuid}")).json()
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "here you go"}


async def test_upload_rejects_an_empty_body(client):
    uuid = await new_conversation(client)
    response = await client.post(f"/api/conversations/{uuid}/upload", data={})
    assert response.status_code == 400
    assert "no file" in response.json()["error"]


# ── text uploads ──────────────────────────────────────────────────────────────

async def test_upload_stores_an_extensionless_text_file_as_text(client):
    """`.gitignore` has no suffix at all, so only the content can classify it."""
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": (".gitignore", b"__pycache__/\n*.pyc\n", "application/octet-stream")},
    )
    assert response.status_code == 200, response.text

    block = response.json()["block"]
    assert block["kind"] == "text"
    assert block["type"] == "_file"
    assert block["mime"] == "text/plain"
    # With no suffix of its own it still needs a readable one on disk.
    assert block["url"].endswith(".txt")


async def test_upload_classifies_a_plain_text_file_from_its_bytes(client):
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("requirements.txt", b"httpx==0.27.0\n", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["block"]["kind"] == "text"


async def test_a_text_upload_reports_its_size_and_line_count(client):
    """The chip captions itself from the block, so the UI never fetches the file
    back just to count its lines. The fields are additive and optional."""
    uuid = await new_conversation(client)
    body = b"one\ntwo\nthree\n"

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("notes.txt", body, "text/plain")},
    )
    assert response.status_code == 200, response.text
    block = response.json()["block"]
    assert block["bytes"] == len(body)
    assert block["lines"] == 3


async def test_a_windows_text_upload_counts_its_lines_once(client):
    """CRLF is one line ending, not two: splitting on `\n` would say 3 here."""
    uuid = await new_conversation(client)
    body = b"a = 1\r\nb = 2\r\n"

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("thing.py", body, "text/x-python")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["block"]["lines"] == 2


async def test_a_binary_upload_is_not_given_a_line_count(client):
    """Nothing that is not read as text is captioned as text: a binary file has no
    lines, and its size is already on the response."""
    uuid = await new_conversation(client)
    body = b"\x01\x02\x03\x04" * 64

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("blob.bin", body, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["bytes"] == len(body)
    block = response.json()["block"]
    assert "bytes" not in block
    assert "lines" not in block


async def test_an_image_upload_is_not_given_a_line_count(client):
    """The same additions must not appear on a kind that has no lines at all."""
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("shot.png", make_png(20, 20), "image/png")},
    )
    assert response.status_code == 200, response.text
    block = response.json()["block"]
    assert block["type"] == "image_url"
    assert "bytes" not in block and "lines" not in block


async def test_an_extensionless_binary_upload_is_not_called_text(client):
    """The bytes decide — a run of control characters is not a config file."""
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("blob", b"\x01\x02\x03\x04" * 64, "application/octet-stream")},
    )
    assert response.status_code == 200
    assert response.json()["block"]["kind"] != "text"


async def test_attach_keeps_a_markdown_file_as_text(client):
    uuid = await new_conversation(client)

    response = await client.post(
        f"/api/conversations/{uuid}/attach",
        files={"file": ("README.md", b"# Title\n\nsome notes\n", "text/markdown")},
    )
    assert response.status_code == 200, response.text

    body = (await client.get(f"/api/conversations/{uuid}")).json()
    block = body["messages"][0]["content"][0]
    assert block["type"] == "_file"
    assert block["kind"] == "text"
    assert block["url"].endswith(".md")


async def test_attach_stores_the_size_and_line_count_with_the_block(client):
    """Reopening a conversation has to caption the chip the same way, so the
    numbers are written with the conversation rather than computed on the way out."""
    uuid = await new_conversation(client)
    body = b"a = 1\nb = 2\n"

    response = await client.post(
        f"/api/conversations/{uuid}/attach",
        files={"file": ("thing.py", body, "text/x-python")},
    )
    assert response.status_code == 200, response.text

    stored = (await client.get(f"/api/conversations/{uuid}")).json()
    block = stored["messages"][0]["content"][0]
    assert block["bytes"] == len(body)
    assert block["lines"] == 2
    # ...and the bytes themselves are still nowhere near the stored conversation.
    assert "a = 1" not in json.dumps(stored)


async def test_an_attached_text_file_reaches_the_model_as_its_own_characters(client, app):
    """The bug this whole path exists to fix: the model used to be told only that
    a file was attached, with no way to see what it said."""
    uuid = await new_conversation(client)
    body = b"httpx==0.27.0\nstarlette>=0.37\n"

    response = await client.post(
        f"/api/conversations/{uuid}/attach",
        files={"file": ("requirements.txt", body, "text/plain")},
        data={"text": "what pins do we have?"},
    )
    assert response.status_code == 200, response.text

    state = app.state.app_state
    prepared = state.engine.prepare_messages(state.store.load(uuid))
    sent = json.dumps(prepared)

    assert "httpx==0.27.0" in sent
    assert "starlette>=0.37" in sent
    assert "what pins do we have?" in sent
    # ...and the stored conversation still holds a reference, never the bytes.
    stored = (state.store.root / uuid / "conversation.json").read_text(encoding="utf-8-sig")
    assert "httpx==0.27.0" not in stored


async def test_a_served_text_file_comes_back_as_text(client, app):
    uuid = await new_conversation(client)
    url = store_file(app, uuid, "notes.txt", b"just some words")

    response = await client.get(url)
    assert response.status_code == 200
    assert response.text == "just some words"


async def test_resize_preserves_aspect_ratio(client):
    response = await client.post(
        "/api/media/resize",
        files={"file": ("big.png", make_png(2000, 1000), "image/png")},
        data={"max_dim": "512"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["before"] == {"width": 2000, "height": 1000}
    assert body["after"] == {"width": 512, "height": 256}
    # No conversation id was given, so the result comes back inline.
    assert body["data_uri"].startswith("data:image/jpeg;base64,")
    assert body["bytes"] < len(make_png(2000, 1000))


async def test_resize_can_store_into_a_conversation(client):
    uuid = await new_conversation(client)

    response = await client.post(
        "/api/media/resize",
        files={"file": ("big.png", make_png(1600, 1600), "image/png")},
        data={"max_dim": "256", "uuid": uuid},
    )
    assert response.status_code == 200, response.text
    url = response.json()["url"]
    assert url.startswith(f"/memory/{uuid}/")
    assert (await client.get(url)).status_code == 200


async def test_frames_endpoint_samples_a_real_clip(client, make_clip):
    """The tool and the HTTP route share one sampler, so this proves the wiring."""
    uuid = await new_conversation(client)

    uploaded = await client.post(
        f"/api/conversations/{uuid}/upload",
        files={"file": ("colour.mp4", make_clip(seconds=3).read_bytes(), "video/mp4")},
    )
    assert uploaded.status_code == 200, uploaded.text
    source = uploaded.json()["block"]["url"]

    response = await client.post(
        "/api/media/frames",
        json={"uuid": uuid, "url": source, "fps": 1, "max_dim": 64, "max_frames": 16},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["count"] == 3
    for url in body["frames"]:
        assert url.startswith(f"/memory/{uuid}/")
        served = await client.get(url)
        assert served.status_code == 200
        assert served.headers["content-type"] == "image/jpeg"


async def test_frames_endpoint_refuses_a_video_from_another_conversation(client, make_clip):
    uuid = await new_conversation(client)
    other = await new_conversation(client)

    uploaded = await client.post(
        f"/api/conversations/{other}/upload",
        files={"file": ("colour.mp4", make_clip(seconds=1).read_bytes(), "video/mp4")},
    )
    assert uploaded.status_code == 200, uploaded.text

    response = await client.post(
        "/api/media/frames",
        json={"uuid": uuid, "url": uploaded.json()["block"]["url"]},
    )
    assert response.status_code == 404


async def test_frames_endpoint_refuses_to_walk_out_of_the_conversation(client):
    uuid = await new_conversation(client)

    for attempt in ("/memory/../providers.json", "../../providers.json", "/etc/passwd"):
        response = await client.post("/api/media/frames", json={"uuid": uuid, "url": attempt})
        assert response.status_code == 404, attempt


# ── export / import ───────────────────────────────────────────────────────────

async def test_export_then_import_issues_a_new_id(client):
    uuid = await new_conversation(client, title="Original", sys_base="be brief")
    await client.post(f"/api/conversations/{uuid}/messages", json={"content": "hello"})

    exported = (await client.get(f"/api/conversations/{uuid}/export")).json()
    assert exported["format"] == "deepseek-ui/v1"

    response = await client.post("/api/conversations/import", json=exported)
    assert response.status_code == 200
    imported = response.json()

    assert imported["uuid"] != uuid
    assert imported["title"] == "Original"
    assert imported["system_prompt"] == "be brief"
    assert len(imported["messages"]) == len(exported["messages"])

    # Both exist afterwards — importing must never overwrite the source.
    listed = (await client.get("/api/conversations")).json()["conversations"]
    assert {c["uuid"] for c in listed} == {uuid, imported["uuid"]}


async def test_import_rejects_a_payload_without_messages(client):
    response = await client.post("/api/conversations/import", json={"title": "nope"})
    assert response.status_code == 400


# ── errors ────────────────────────────────────────────────────────────────────

async def test_error_responses_are_json(client):
    response = await client.get("/api/conversations/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.json()


async def test_malformed_json_body_is_not_a_crash(client):
    uuid = await new_conversation(client)
    response = await client.post(
        f"/api/conversations/{uuid}/messages",
        content=b"{not json at all",
        headers={"content-type": "application/json"},
    )
    # An unparseable body has no content, so it fails the required-field check.
    assert response.status_code == 400


async def test_missing_body_is_not_a_crash(client):
    uuid = await new_conversation(client)
    response = await client.post(f"/api/conversations/{uuid}/messages")
    assert response.status_code == 400


async def test_unknown_route_is_404(client):
    assert (await client.get("/api/nope")).status_code == 404


# ── streaming ─────────────────────────────────────────────────────────────────

async def test_chat_streams_sse_headers(client):
    uuid = await new_conversation(client)

    response = await client.post(f"/api/chat/{uuid}", json={"content": "hi"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Without this an intermediate proxy would buffer the whole answer.
    assert response.headers["x-accel-buffering"] == "no"
    assert "no-transform" in response.headers["cache-control"]
    assert uuid in (await client.get("/api/conversations")).json()["conversations"][0]["uuid"]


async def test_chat_on_a_missing_conversation_is_404_not_a_stream(client):
    response = await client.post("/api/chat/aaaabbbbcccc", json={"content": "hi"})
    assert response.status_code == 404


async def test_chat_reports_an_unusable_model_as_an_error_frame(tmp_path):
    """The 200 is already sent by the time the model is resolved, so this must be a frame."""
    app = build_app(tmp_path, api_key="${DEFINITELY_NOT_SET_ANYWHERE}")
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            uuid = await new_conversation(c)
            response = await c.post(f"/api/chat/{uuid}", json={"content": "hi"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "model unavailable" in response.text


async def test_chat_appends_the_incoming_message_before_streaming(client):
    """The user turn must be persisted even when the model call then fails."""
    uuid = await new_conversation(client)

    await client.post(f"/api/chat/{uuid}", json={"content": "remember this"})

    on_disk = json.loads(
        (Path(client._transport.app.state.app_state.settings.memory_root) / uuid / "conversation.json").read_text("utf-8")
    )
    assert on_disk["messages"][0]["content"] == "remember this"


# ── mcp ───────────────────────────────────────────────────────────────────────

def build_app_with_mcp(tmp_path: Path, payload=None):
    """An app whose MCP config lives inside ``tmp_path``.

    Servers here are either disabled or pointed at a closed port, so no test in this
    section reaches the network: a disabled server is never dialled, and a refused
    connection fails in microseconds.
    """
    path = tmp_path / "mcp.json"
    if payload is not None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    app = create_app(
        memory_root=str(tmp_path / "memory"),
        providers_path=str(write_providers(tmp_path)),
        env_file=str(tmp_path / "does-not-exist.env"),
        frontend_dir=str(tmp_path / "frontend"),
        mcp_config_path=str(path),
    )
    return app, path


def write_providers(tmp_path: Path) -> Path:
    providers = tmp_path / "providers.json"
    payload = json.loads(json.dumps(PROVIDERS))
    payload["providers"][0]["apiKey"] = "test-key-not-real"
    providers.write_text(json.dumps(payload), encoding="utf-8")
    return providers


@pytest.fixture
async def mcp_client(tmp_path):
    """A client whose MCP config file already holds one disabled server."""
    app, path = build_app_with_mcp(tmp_path, {"servers": [
        {"name": "offline", "url": "http://127.0.0.1:1/mcp", "enabled": False},
    ]})
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.mcp_path = path
            yield c


def servers_on_disk(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["servers"]


async def test_mcp_status_is_empty_without_a_config(client):
    response = await client.get("/api/mcp")
    assert response.status_code == 200
    assert response.json()["servers"] == []


async def test_mcp_reload_keeps_the_builtin_tools(client):
    response = await client.post("/api/mcp/reload")
    assert response.status_code == 200
    assert "resize_image" in response.json()["tools"]


async def test_props_exposes_the_config_path_for_the_panel(tmp_path):
    """The panel has to say which file it is editing."""
    app, path = build_app_with_mcp(tmp_path, {"servers": []})
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            data = (await c.get("/api/props")).json()

    assert data["mcp_config"] == {
        "path": str(path), "exists": True, "error": "", "editable": True,
    }


async def test_a_disabled_server_is_listed_but_not_connected(mcp_client):
    """Disabled servers have to render, or they could never be re-enabled."""
    data = (await mcp_client.get("/api/mcp")).json()
    assert [s["name"] for s in data["servers"]] == ["offline"]
    server = data["servers"][0]
    assert server["enabled"] is False
    assert server["connected"] is False
    assert server["log"] == []


async def test_mcp_add_server_writes_the_file(mcp_client):
    response = await mcp_client.post("/api/mcp/servers", json={
        "name": "web", "url": "http://127.0.0.1:1/mcp", "prefix": "ws",
    })
    assert response.status_code == 200, response.text

    assert [s["name"] for s in servers_on_disk(mcp_client.mcp_path)] == ["offline", "web"]
    assert "web" in [s["name"] for s in response.json()["servers"]]


async def test_mcp_add_server_rejects_a_duplicate_name(mcp_client):
    response = await mcp_client.post("/api/mcp/servers", json={
        "name": "offline", "url": "http://127.0.0.1:1/mcp",
    })
    assert response.status_code == 409
    assert len(servers_on_disk(mcp_client.mcp_path)) == 1


@pytest.mark.parametrize("body,expected", [
    ({"url": "http://x/mcp"}, "name"),
    ({"name": "a"}, "url"),
    ({"name": "a", "url": "ftp://x/mcp"}, "http"),
    ({"name": "a", "url": "localhost:8569"}, "http"),
])
async def test_mcp_add_server_validates_its_input(mcp_client, body, expected):
    response = await mcp_client.post("/api/mcp/servers", json=body)
    assert response.status_code == 400
    assert expected in response.json()["error"].lower()
    assert len(servers_on_disk(mcp_client.mcp_path)) == 1


async def test_mcp_add_server_keeps_a_placeholder_header_verbatim(mcp_client):
    """Resolving `${VAR}` here would write the secret into the file."""
    await mcp_client.post("/api/mcp/servers", json={
        "name": "authed", "url": "http://127.0.0.1:1/mcp",
        "headers": {"Authorization": "Bearer ${MY_MCP_TOKEN}"},
    })
    entry = servers_on_disk(mcp_client.mcp_path)[-1]
    assert entry["headers"] == {"Authorization": "Bearer ${MY_MCP_TOKEN}"}


async def test_mcp_add_server_splits_a_comma_separated_allow_list(mcp_client):
    await mcp_client.post("/api/mcp/servers", json={
        "name": "tidy", "url": "http://127.0.0.1:1/mcp",
        "headers": {"X-A": "1", "X-B": "2"}, "allowedTools": "one, two\n three ",
    })
    entry = servers_on_disk(mcp_client.mcp_path)[-1]
    assert entry["headers"] == {"X-A": "1", "X-B": "2"}
    assert entry["allowedTools"] == ["one", "two", "three"]


async def test_headers_must_be_an_object(mcp_client):
    response = await mcp_client.post("/api/mcp/servers", json={
        "name": "a", "url": "http://127.0.0.1:1/mcp", "headers": "X-A: 1",
    })
    assert response.status_code == 400
    assert len(servers_on_disk(mcp_client.mcp_path)) == 1


async def test_mcp_update_server_can_rename(mcp_client):
    response = await mcp_client.put("/api/mcp/servers/offline", json={
        "name": "online", "url": "http://127.0.0.1:1/mcp", "enabled": False,
    })
    assert response.status_code == 200, response.text
    assert [s["name"] for s in servers_on_disk(mcp_client.mcp_path)] == ["online"]


async def test_mcp_update_server_404s_for_an_unknown_name(mcp_client):
    response = await mcp_client.put("/api/mcp/servers/ghost", json={
        "name": "ghost", "url": "http://127.0.0.1:1/mcp",
    })
    assert response.status_code == 404


async def test_mcp_update_server_rejects_a_name_that_is_taken(mcp_client):
    await mcp_client.post("/api/mcp/servers", json={"name": "other", "url": "http://127.0.0.1:1/mcp"})
    response = await mcp_client.put("/api/mcp/servers/offline", json={
        "name": "other", "url": "http://127.0.0.1:1/mcp",
    })
    assert response.status_code == 409
    assert sorted(s["name"] for s in servers_on_disk(mcp_client.mcp_path)) == ["offline", "other"]


async def test_enabling_a_server_records_only_the_exception(mcp_client):
    """The file stays readable: defaults are omitted, not written out in full."""
    response = await mcp_client.post("/api/mcp/servers/offline/enabled", json={"enabled": True})
    assert response.status_code == 200, response.text

    entry = servers_on_disk(mcp_client.mcp_path)[0]
    assert "enabled" not in entry
    assert entry == {"name": "offline", "url": "http://127.0.0.1:1/mcp"}

    await mcp_client.post("/api/mcp/servers/offline/enabled", json={"enabled": False})
    assert servers_on_disk(mcp_client.mcp_path)[0]["enabled"] is False


async def test_mcp_enabled_requires_a_boolean(mcp_client):
    response = await mcp_client.post("/api/mcp/servers/offline/enabled", json={})
    assert response.status_code == 400


async def test_enabling_an_unknown_server_404s(mcp_client):
    response = await mcp_client.post("/api/mcp/servers/ghost/enabled", json={"enabled": True})
    assert response.status_code == 404


async def test_mcp_delete_server_removes_it(mcp_client):
    response = await mcp_client.delete("/api/mcp/servers/offline")
    assert response.status_code == 200, response.text
    assert servers_on_disk(mcp_client.mcp_path) == []

    assert (await mcp_client.delete("/api/mcp/servers/offline")).status_code == 404


async def test_deleting_the_last_server_leaves_a_readable_file(mcp_client):
    await mcp_client.delete("/api/mcp/servers/offline")
    assert json.loads(mcp_client.mcp_path.read_text(encoding="utf-8")) == {"servers": []}


async def test_a_hand_edited_file_is_picked_up_by_reload(mcp_client):
    """The original workflow — edit the JSON, press Reload — has to keep working."""
    mcp_client.mcp_path.write_text(json.dumps({"servers": [
        {"name": "hand", "url": "http://127.0.0.1:1/mcp", "enabled": False},
    ]}), encoding="utf-8")

    response = await mcp_client.post("/api/mcp/reload")
    assert response.status_code == 200
    assert [s["name"] for s in response.json()["servers"]] == ["hand"]


async def test_saving_preserves_comments_the_panel_does_not_understand(mcp_client):
    mcp_client.mcp_path.write_text(json.dumps({
        "_comment": "keep me",
        "servers": [{"name": "offline", "url": "http://127.0.0.1:1/mcp", "enabled": False}],
    }), encoding="utf-8")

    await mcp_client.post("/api/mcp/servers/offline/enabled", json={"enabled": False})
    assert json.loads(mcp_client.mcp_path.read_text(encoding="utf-8"))["_comment"] == "keep me"


async def test_creating_the_first_server_works_without_one_before(tmp_path):
    """`mcp.json` need not exist: adding the first server creates it."""
    app, path = build_app_with_mcp(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            assert (await c.get("/api/mcp")).json()["config"]["exists"] is False
            response = await c.post("/api/mcp/servers", json={
                "name": "first", "url": "http://127.0.0.1:1/mcp",
            })

    assert response.status_code == 200, response.text
    assert servers_on_disk(path) == [{"name": "first", "url": "http://127.0.0.1:1/mcp"}]


async def test_an_unreadable_config_is_reported_and_left_alone(tmp_path):
    """A half-finished hand edit must not be clobbered by the panel."""
    app, path = build_app_with_mcp(tmp_path)
    path.write_text("{ still typing", encoding="utf-8")

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            listing = await c.get("/api/mcp")
            assert listing.status_code == 200
            assert listing.json()["config"]["editable"] is False
            assert listing.json()["config"]["error"]

            # Every write path refuses rather than guessing.
            add = await c.post("/api/mcp/servers", json={"name": "a", "url": "http://127.0.0.1:1/mcp"})
            toggle = await c.post("/api/mcp/servers/a/enabled", json={"enabled": True})
            delete = await c.delete("/api/mcp/servers/a")
            update = await c.put("/api/mcp/servers/a", json={"name": "a", "url": "http://127.0.0.1:1/mcp"})

    assert [add.status_code, toggle.status_code, delete.status_code, update.status_code] == [409] * 4
    assert path.read_text(encoding="utf-8") == "{ still typing"


async def test_mcp_test_rejects_a_non_http_url(mcp_client):
    """Validation happens before any socket is opened."""
    response = await mcp_client.post("/api/mcp/test", json={"url": "not-a-url"})
    assert response.status_code == 400


async def test_mcp_test_404s_for_an_unknown_server(mcp_client):
    assert (await mcp_client.post("/api/mcp/test", json={"name": "ghost"})).status_code == 404


async def test_mcp_test_can_probe_a_definition_that_was_never_saved(mcp_client):
    """Testing before saving is the whole point of the Test button."""
    response = await mcp_client.post("/api/mcp/test", json={
        "url": "http://127.0.0.1:1/mcp", "timeout": 1,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is False
    assert body["error"]
    assert len(servers_on_disk(mcp_client.mcp_path)) == 1  # nothing was added


async def test_mcp_test_probes_a_disabled_server_on_request(mcp_client):
    """A disabled server is exactly the one you want to test before enabling."""
    response = await mcp_client.post("/api/mcp/test", json={"name": "offline"})
    assert response.status_code == 200
    assert response.json()["connected"] is False


async def test_reload_reports_what_the_panel_should_now_show(mcp_client):
    response = await mcp_client.post("/api/mcp/reload")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"servers", "tools", "config"}
    assert body["config"]["path"] == str(mcp_client.mcp_path)
    assert "resize_image" in body["tools"]


# ── frontend ──────────────────────────────────────────────────────────────────

async def test_index_serves_a_placeholder_when_unbuilt(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "deepseekUI" in response.text


async def test_frontend_files_are_served_when_present(tmp_path):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>built</h1>", encoding="utf-8")
    (frontend / "app.js").write_text("console.log(1)", encoding="utf-8")

    app = build_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            assert "built" in (await c.get("/")).text
            asset = await c.get("/assets/app.js")
            assert asset.status_code == 200
            assert asset.text == "console.log(1)"
            # httpx normalises `..` away before sending, so this arrives as
            # /providers.json. Either way it must not be served.
            assert (await c.get("/assets/../providers.json")).status_code in (400, 404)
