"""Tests for reading files as text.

The bug these exist to prevent: an attached `.md`, `.txt` or `.gitignore` used to
arrive as an `unknown` file and reach the model as the sentence "[file 'x'
attached]" — no characters, no content. Two things were wrong, and both are pinned
here. Classification looked at the extension, which `.gitignore` does not have; and
the rehydrator only knew how to inline images, video and audio.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from server.media import (
    MAX_TEXT_READ,
    MediaStore,
    TextDocument,
    classify_kind,
    decode_text,
    drop_partial_character,
    is_text_mime,
    looks_like_text,
    looks_like_text_name,
    text_document,
    text_facts,
)
from server.rehydrate import MediaBudget, rehydrate
from server.settings import load_settings
from server.store import ConversationStore
from server.tools_builtin import build_media_tools

GITIGNORE = b"__pycache__/\n*.pyc\n.venv/\n"
MAKEFILE = b"all:\n\tpython -m pytest tests -q\n"
README = b"# deepseekUI\n\nA local chat UI.\n"


def make_png(width=8, height=8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    return ConversationStore(tmp_path / "memory")


@pytest.fixture
def media(store):
    return MediaStore(store)


@pytest.fixture
def settings(store):
    return load_settings({}, load_dotenv=False, memory_root=store.root)


@pytest.fixture
def registry(store, media, settings):
    from deepseek_client.tools import ToolRegistry

    store.create(uuid="conv1")
    tools = build_media_tools(store, media, lambda: "conv1", settings=settings)
    reg = ToolRegistry()
    for tool in tools:
        reg.add(tool)
    return reg


def text_of(result) -> str:
    content = result.content
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


# ── classification ───────────────────────────────────────────────────────────

def test_an_extensionless_config_file_is_text():
    """`Path('.gitignore').suffix` is empty, so nothing name-based could ever work."""
    assert classify_kind(GITIGNORE, "", ".gitignore") == "text"


def test_an_extensionless_makefile_is_text():
    assert classify_kind(MAKEFILE, "", "Makefile") == "text"


def test_a_markdown_file_is_text():
    assert classify_kind(README, "text/markdown", "README.md") == "text"


def test_markdown_is_recognised_without_a_helpful_declared_type():
    """Browsers send `application/octet-stream` for anything they do not know."""
    assert classify_kind(README, "application/octet-stream", "notes.md") == "text"


def test_a_png_is_an_image_whatever_it_is_called_and_whatever_it_claims():
    assert classify_kind(make_png(), "text/plain", "notes.txt") == "image"


def test_an_mp4_is_a_video():
    data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64
    assert classify_kind(data, "", "clip.mp4") == "video"


def test_control_bytes_are_binary_even_though_they_decode():
    """A run of 0x01 is valid UTF-8, so a successful decode cannot be the only test."""
    assert classify_kind(b"\x01\x02\x03\x04" * 64, "", "blob") == "binary"


def test_a_pdf_is_binary_not_text():
    """Its header is ASCII and the rest is compressed streams — printing it is mojibake."""
    assert classify_kind(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n", "", "doc.pdf") == "binary"


def test_a_binary_file_keeps_the_clients_label_when_the_bytes_say_nothing():
    """`classify_kind` reports `binary`; the routes fall back to the declared kind."""
    assert classify_kind(b"\xff\xd8\xff\xe0" + b"\x00" * 32, "image/jpeg", "x.jpg") == "image"


def test_the_text_mime_predicate_covers_the_types_that_matter():
    assert is_text_mime("text/plain")
    assert is_text_mime("text/markdown")
    assert is_text_mime("application/json")
    assert is_text_mime("text/plain; charset=utf-8")
    assert not is_text_mime("image/png")
    assert not is_text_mime("application/octet-stream")


# ── code and config files ────────────────────────────────────────────────────

#: The files people actually attach when asking about their project. Every one has
#: to be readable — declaring the extension in a table is not the mechanism, this
#: is a regression net around the content sniff.
CODE_FILES = {
    "module.py": b"import os\n\n\ndef main() -> int:\n    return 0\n",
    "config.json": b'{"a": 1, "nested": {"b": [true, null]}}\n',
    "data.csv": b"id,name,value\n1,alpha,3.5\n2,beta,4.0\n",
    "query.sql": b"SELECT id, name\nFROM users\nWHERE active = true;\n",
    "style.css": b".a { color: #fff; }\n",
    "script.sh": b"#!/usr/bin/env bash\nset -euo pipefail\necho hi\n",
    "dockerfile": b"FROM python:3.11-slim\nWORKDIR /app\n",
    "gemfile": b"source 'https://rubygems.org'\ngem 'rails'\n",
    "bundle.min.js": b"function a(b){return b?1:0}" * 200,
    "tabbed.py": b"def f():\n\treturn 1\n",
    "crlf.py": b"import os\r\nimport sys\r\n",
    "schema.proto": b'message M {\n  string name = 1;\n}\n',
    "settings.yaml": b"server:\n  port: 5000\n",
    "template.html": b"<!doctype html>\n<html><body>hi</body></html>\n",
}


@pytest.mark.parametrize("name", sorted(CODE_FILES))
def test_a_code_or_config_file_is_text(name):
    """Tabs, CRLF, curly braces and minified JS are all ordinary text."""
    assert classify_kind(CODE_FILES[name], "", name) == "text"


@pytest.mark.parametrize("name", sorted(CODE_FILES))
def test_a_code_file_decodes_without_needing_its_extension(name):
    decoded = text_document(CODE_FILES[name])
    assert decoded is not None
    assert decoded[0] == CODE_FILES[name].decode(decoded[1])


def test_a_notebook_is_text_because_it_is_json():
    notebook = json.dumps({"cells": [{"cell_type": "code", "source": ["x = 1"]}]}).encode()
    assert classify_kind(notebook, "", "analysis.ipynb") == "text"


def test_a_bytecode_file_is_not_text():
    """A `.pyc` starts with `\r\r\n`, which is friendly enough to fool a naive sniff;
    the compiled body behind it is not. Printing it would be pure mojibake."""
    pyc = b"\x0d\x0d\x0a" + bytes(range(256)) * 4
    assert classify_kind(pyc, "", "module.pyc") == "binary"


def test_a_zip_is_not_text():
    assert classify_kind(b"PK\x03\x04" + bytes(range(256)) * 4, "", "archive.zip") == "binary"


def test_a_notepad_json_file_loses_its_bom():
    """`json.loads` rejects a BOM, and Notepad writes one — so this is the difference
    between a config file the model can read and one it sees as corrupted."""
    text, encoding = decode_text(b'\xef\xbb\xbf{"a": 1}')
    assert text == '{"a": 1}'
    assert encoding == "utf-8-sig"
    assert json.loads(text) == {"a": 1}


def test_a_latin1_source_file_falls_back_rather_than_failing():
    """An unencoded `é` in a comment is a real thing to find in the wild."""
    text, encoding = decode_text(b"# caf\xe9\nx = 1\n")
    assert encoding == "cp1252"
    assert "caf\u00e9" in text


# ── decoding ─────────────────────────────────────────────────────────────────

def test_plain_utf8_decodes():
    assert decode_text(b"hello") == ("hello", "utf-8")


def test_a_utf8_bom_is_consumed_rather_than_kept():
    """Notepad writes one; keeping it glues U+FEFF to the first line the model sees."""
    text, encoding = decode_text(b"\xef\xbb\xbfhello")
    assert text == "hello"
    assert encoding == "utf-8-sig"
    assert not text.startswith("\ufeff")


def test_a_utf16_bom_is_honoured():
    text, encoding = decode_text("héllo".encode("utf-16"))
    assert text == "héllo"
    assert encoding == "utf-16"


def test_windows_1252_is_the_fallback_for_a_non_utf8_file():
    assert decode_text(b"caf\xe9\n") == ("café\n", "cp1252")


def test_a_nul_in_the_head_rules_out_the_cp1252_fallback():
    assert decode_text(b"caf\xe9\x00\x00\n") is None


def test_a_very_short_pickup_still_decodes():
    assert decode_text(b"\xff") == ("ÿ", "cp1252")


def test_text_document_rejects_what_only_looks_decodable():
    assert text_document(b"ok" * 100) == ("ok" * 100, "utf-8")
    assert text_document(b"\x01" * 200) is None


def test_an_empty_file_is_text_only_when_its_name_says_so():
    assert looks_like_text(b"", "Makefile")
    assert not looks_like_text(b"", "blob")
    assert looks_like_text_name("README.md")
    assert looks_like_text_name(".gitignore")
    assert not looks_like_text_name("archive.zip")


# ── reading a file back ──────────────────────────────────────────────────────

def test_read_text_reports_the_file_it_read(media, tmp_path):
    path = tmp_path / "notes.md"
    path.write_bytes(README)

    document = media.read_text(path)
    assert isinstance(document, TextDocument)
    assert document.text == README.decode()
    assert document.encoding == "utf-8"
    assert document.lines == 3
    assert document.bytes == len(README)
    assert not document.truncated


def test_read_text_refuses_a_binary_file(media, tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x01\x02\x03\x04" * 64)

    assert media.read_text(path) is None


def test_read_text_honours_a_character_cap(media, tmp_path):
    path = tmp_path / "log.txt"
    path.write_bytes(b"x" * 5000)

    document = media.read_text(path, max_chars=1000)
    assert len(document.text) == 1000
    assert document.truncated
    assert document.characters == 5000       # the total, so a caller can say what is left


def test_read_text_keeps_a_utf8_file_utf8_when_the_byte_cap_slices_a_character(media, tmp_path):
    """A stray lead byte must not demote the file to the cp1252 fallback.

    cp1252 accepts nearly anything, so if the half character is left in place the
    decode succeeds there and every remaining byte becomes its own wrong glyph.
    """
    path = tmp_path / "accented.txt"
    path.write_bytes(("\u00e9" * 200).encode("utf-8"))

    document = media.read_text(path, max_bytes=101)
    assert document is not None
    assert document.encoding == "utf-8"
    assert document.text == "\u00e9" * len(document.text)
    assert document.truncated


def test_read_text_on_a_missing_file_is_none_not_an_exception(media, tmp_path):
    assert media.read_text(tmp_path / "absent.txt") is None


def test_drop_partial_character_leaves_a_complete_file_alone():
    assert drop_partial_character(b"complete\n") == b"complete\n"
    assert drop_partial_character(b"") == b""


def test_drop_partial_character_trims_only_the_cut_character():
    data = ("caf\u00e9".encode("utf-8"))[: 3 + 1]        # 'caf' plus the first byte of é
    assert drop_partial_character(data) == b"caf"


def test_drop_partial_character_keeps_the_file_when_nothing_decodes():
    data = b"\xff\xfe\xff\xfe"
    assert drop_partial_character(data) == data


# ── the size and line count a chip captions itself from ──────────────────────

# These travel with the `_file` block so the UI can say "3.7 KB · 122 lines"
# without fetching the file back just to count it. They are additive: a client that
# ignores them still has the url, the name and the mime.

def test_text_facts_count_lines_the_way_an_editor_counts_them():
    assert text_facts(b"one\ntwo\n") == {"bytes": 8, "lines": 2}
    assert text_facts(b"one\r\ntwo\r\n") == {"bytes": 10, "lines": 2}     # a Windows file
    assert text_facts(b"no trailing newline") == {"bytes": 19, "lines": 1}
    assert text_facts(b"") == {"bytes": 0, "lines": 0}


def test_text_facts_refuse_to_count_lines_that_are_not_there():
    """A line count is a claim about text, so it is only made about text."""
    data = b"\x01\x02\x03\x04" * 64
    assert text_facts(data) == {"bytes": len(data)}


def test_text_facts_stop_counting_at_the_read_window():
    """A huge log is not decoded to caption a chip. A count measured over a cut
    sample would be a lower bound rather than a fact, so none is reported."""
    data = b"x" * (MAX_TEXT_READ + 1)
    assert text_facts(data) == {"bytes": len(data)}


def test_text_facts_are_optional_arguments_not_a_ceiling():
    assert text_facts(b"a\nb\nc\n", max_bytes=4) == {"bytes": 6}
    assert text_facts(b"a\nb\nc\n", max_bytes=6) == {"bytes": 6, "lines": 3}


# ── ingestion: deciding on the way in ────────────────────────────────────────

def test_an_inline_text_file_is_stored_as_text(media, store):
    store.create(uuid="conv1")
    block = {
        "type": "_file",
        "kind": "file",                             # the client did not know better
        "name": "requirements.txt",
        "mime": "application/octet-stream",
        "b64": base64.b64encode(b"httpx==0.27.0\n").decode(),
    }

    out = media.ingest_user_content("conv1", [block])[0]
    assert out["kind"] == "text"
    assert out["type"] == "_file"
    assert out["mime"] == "text/plain"
    assert "b64" not in out and "data" not in out
    assert media.path_for_url(out["url"]).read_bytes() == b"httpx==0.27.0\n"


def test_an_inline_binary_file_stays_a_file(media, store):
    store.create(uuid="conv1")
    block = {
        "type": "_file",
        "kind": "file",
        "name": "blob",
        "b64": base64.b64encode(b"\x01\x02\x03\x04" * 64).decode(),
    }

    out = media.ingest_user_content("conv1", [block])[0]
    assert out["kind"] == "file"


def test_an_ingested_text_file_reports_its_size_and_line_count(media, store):
    store.create(uuid="conv1")
    body = b"httpx==0.27.0\nstarlette>=0.37\n"
    block = {
        "type": "_file",
        "kind": "file",
        "name": "requirements.txt",
        "b64": base64.b64encode(body).decode(),
    }

    out = media.ingest_user_content("conv1", [block])[0]
    assert out["bytes"] == len(body)
    assert out["lines"] == 2


def test_an_ingested_binary_file_is_given_no_caption_numbers(media, store):
    store.create(uuid="conv1")
    body = b"\x01\x02\x03\x04" * 64
    block = {
        "type": "_file",
        "kind": "file",
        "name": "blob.bin",
        "b64": base64.b64encode(body).decode(),
    }

    out = media.ingest_user_content("conv1", [block])[0]
    assert out["kind"] == "file"
    assert "bytes" not in out and "lines" not in out


def test_a_reference_to_an_extensionless_text_file_is_marked_text(media, store):
    """No bytes to sniff, but the name is enough for `.gitignore` and `Makefile`."""
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", GITIGNORE, "text/plain", name=".gitignore")

    out = media.ingest_user_content(
        "conv1", [{"type": "_file", "kind": "file", "name": ".gitignore", "url": url}]
    )[0]
    assert out["kind"] == "text"


# ── rehydration: getting the characters to the model ─────────────────────────

def build_file_message(media, name, data, *, kind="text", mime="text/plain", note=None):
    url = media.save_bytes("conv1", data, mime, name=name)
    content = [{"type": "_file", "kind": kind, "name": name, "mime": mime, "url": url}]
    if note:
        content.append({"type": "text", "text": note})
    return {"role": "user", "content": content}


def test_an_attached_text_file_reaches_the_model_as_characters(media, store, settings):
    store.create(uuid="conv1")
    messages = [build_file_message(media, "notes.md", README, mime="text/markdown", note="summarise")]

    out = rehydrate(messages, media, settings)
    sent = out[0]["content"]
    assert "# deepseekUI" in sent
    assert "A local chat UI." in sent
    assert "summarise" in sent


def test_the_inlined_file_says_what_it_is(media, store, settings):
    store.create(uuid="conv1")
    out = rehydrate([build_file_message(media, "notes.md", README)], media, settings)

    sent = out[0]["content"]
    assert "notes.md" in sent
    assert "3 lines" in sent
    assert "utf-8" in sent


def test_a_legacy_file_block_is_still_read_as_text(media, store, settings):
    """Conversations saved before this existed say `kind: "file"`. The bytes decide."""
    store.create(uuid="conv1")
    messages = [build_file_message(media, "legacy.txt", b"pins and needles\n", kind="file")]

    out = rehydrate(messages, media, settings)
    assert "pins and needles" in out[0]["content"]


def test_a_binary_attachment_says_it_cannot_be_read(media, store, settings):
    """Silently saying "a file is attached" is worse than admitting the limit."""
    store.create(uuid="conv1")
    messages = [build_file_message(media, "blob.bin", b"\x01\x02\x03\x04" * 64, kind="file")]

    out = rehydrate(messages, media, settings)
    sent = out[0]["content"]
    assert "binary" in sent
    assert "blob.bin" in sent


def test_an_old_text_file_is_omitted_with_a_way_to_get_it(media, store, settings):
    """Newest-first, and the note tells the model how to read what was dropped."""
    store.create(uuid="conv1")
    messages = [
        build_file_message(media, "old.md", b"ancient history\n"),
        {"role": "assistant", "content": "ok"},
        build_file_message(media, "new.md", b"the latest thing\n"),
    ]

    out = rehydrate(messages, media, settings, text_budget=MediaBudget(20))
    assert "the latest thing" in out[2]["content"]
    assert "ancient history" not in out[0]["content"]
    assert "read_file" in out[0]["content"]


def test_a_file_too_big_for_the_budget_is_shown_partly_rather_than_dropped(media, store, settings):
    store.create(uuid="conv1")
    body = ("line %04d\n" % 0) * 1 + "".join(f"line {i:04d}\n" for i in range(500))
    messages = [build_file_message(media, "big.log", body.encode())]

    out = rehydrate(messages, media, settings, text_budget=MediaBudget(3_000))
    sent = out[0]["content"]
    assert "line 0000" in sent
    assert "partly shown" in sent
    assert "truncated" in sent


def test_a_stored_text_file_never_keeps_its_base64(media, store, settings):
    """The whole architecture rests on bytes living on disk, not in the transcript."""
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", GITIGNORE, "text/plain", name=".gitignore")
    assert url.startswith("/memory/conv1/")

    out = rehydrate(
        [{"role": "user", "content": [{"type": "_file", "kind": "text", "name": ".gitignore", "url": url}]}],
        media,
        settings,
    )
    assert "base64" not in out[0]["content"]


def test_a_missing_file_is_reported_as_missing(media, store, settings):
    store.create(uuid="conv1")
    out = rehydrate(
        [{"role": "user", "content": [{"type": "_file", "kind": "text", "name": "gone.md", "url": "/memory/conv1/gone.md"}]}],
        media,
        settings,
    )
    assert "gone.md" in out[0]["content"]
    assert "no longer on disk" in out[0]["content"]


# ── tools ────────────────────────────────────────────────────────────────────

async def test_inspect_media_reports_a_text_file_as_text(registry, media, store):
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", README, "text/markdown", name="README.md")

    result = await registry.call("inspect_media", {"path": url})
    assert not result.is_error
    assert result.meta["kind"] == "text"
    assert result.meta["charset"] == "utf-8"
    assert result.meta["lines"] == 3


async def test_inspect_media_lists_text_files_as_readable(registry, media, store):
    store.create(uuid="conv1")
    media.save_bytes("conv1", GITIGNORE, "text/plain", name=".gitignore")
    media.save_bytes("conv1", b"\x01\x02\x03\x04" * 64, "", name="blob.bin")

    result = await registry.call("inspect_media", {})
    body = text_of(result)
    assert "text" in body
    assert "unreadable" in body          # the binary one, and only that one


async def test_read_file_returns_the_contents_with_line_numbers(registry, media, store):
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", README, "text/markdown", name="README.md")

    result = await registry.call("read_file", {"path": url})
    assert not result.is_error
    body = text_of(result)
    assert "# deepseekUI" in body
    assert "1  # deepseekUI" in body


async def test_read_file_pages_through_a_long_file(registry, media, store):
    store.create(uuid="conv1")
    body = "".join(f"line {i:05d}\n" for i in range(2_000))
    url = media.save_bytes("conv1", body.encode(), "text/plain", name="long.log")

    first = text_of(await registry.call("read_file", {"path": url, "limit": 1_000}))
    assert "line 00000" in first
    assert "offset=" in first

    offset = first.split("offset=")[1].split(")")[0]
    second = text_of(await registry.call("read_file", {"path": url, "offset": int(offset), "limit": 1_000}))
    assert "line 00000" not in second
    assert "line " in second


async def test_read_file_refuses_a_binary_file(registry, media, store):
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", b"\x01\x02\x03\x04" * 64, "", name="blob.bin")

    result = await registry.call("read_file", {"path": url})
    assert result.is_error
    assert "not readable as text" in text_of(result)


async def test_read_file_cannot_walk_out_of_the_conversation(registry):
    result = await registry.call("read_file", {"path": "../../providers.json"})
    assert result.is_error


async def test_read_file_reports_an_offset_past_the_end(registry, media, store):
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", b"short\n", "text/plain", name="short.txt")

    result = await registry.call("read_file", {"path": url, "offset": 9_999})
    assert result.is_error


# ── the seam: a truncated inline and the read_file that continues it ─────────

def read_file_lines(body: str) -> list[str]:
    """Undo the `  NNNN  ` numbering, keeping the file's own characters."""
    lines = body.split("\n")[1:]                    # drop the header line
    if lines and lines[-1].startswith("["):
        lines.pop()                                 # drop a notice, if one was added
    return [re.sub(r"^ *\d+  ", "", line) for line in lines]


async def test_read_file_reproduces_every_line_of_a_crlf_source(registry, media, store):
    """Line *content* survives; the CR of a CRLF is presentation, like the numbers."""
    store.create(uuid="conv1")
    source = b"import os\r\n\r\ndef main():\r\n    return 0\r\n"
    url = media.save_bytes("conv1", source, "text/plain", name="module.py")

    body = text_of(await registry.call("read_file", {"path": url}))
    assert read_file_lines(body) == source.decode().splitlines()


async def test_inlining_a_crlf_source_keeps_its_characters_exactly(registry, media, store, settings):
    """Unlike the numbered view, the inline is a character-for-character prefix — a
    `.py` file on Windows is CRLF, and a stampede of rewritten endings is the kind of
    thing that makes a model distrust what it is reading. (A text-only message is
    folded into one string and trimmed, so the file's *final* newline is not the
    thing under test here; its interior is.)
    """
    source = b"import os\r\n\r\nVALUE = 1\r\nEND = 2\r\n"
    url = media.save_bytes("conv1", source, "text/plain", name="module.py")

    out = rehydrate(
        [{"role": "user", "content": [{"type": "_file", "kind": "text", "name": "module.py", "url": url}]}],
        media,
        settings,
    )
    sent = out[0]["content"]
    sent = sent if isinstance(sent, str) else "\n".join(b.get("text", "") for b in sent)

    assert "import os\r\n\r\nVALUE = 1\r\nEND = 2" in sent
    assert "import os\n\n" not in sent.replace("\r\n", "\\r\\n")   # no CRLF was flattened


async def test_read_file_finds_a_file_by_the_name_it_was_attached_as(registry, media, store):
    """The continuation hints name the attachment, not the file on disk. A 40 KB
    `routes.py` is stored as `user_file_178...py`, so a hint saying
    `read_file('routes.py', offset=40000)` only works if the attached name resolves.
    """
    conversation = store.load("conv1")
    conversation.messages.append({
        "role": "user",
        "content": media.ingest_user_content("conv1", [{
            "type": "_file",
            "kind": "file",
            "name": "main.py",
            "mime": "text/x-python",
            "b64": base64.b64encode(b"import os\nVALUE = 1\n").decode(),
        }]),
    })
    store.save(conversation)

    stored = conversation.messages[0]["content"][0]["url"]
    assert Path(stored).name not in ("main.py", "main")      # a generated name on disk

    result = await registry.call("read_file", {"path": "main.py"})
    assert not result.is_error
    assert "VALUE = 1" in text_of(result)
    assert "main.py" in text_of(result)


async def test_the_attached_name_still_has_to_be_inside_the_conversation(registry, media, store, settings):
    """Scanning the conversation for a matching name must not widen the boundary."""
    from deepseek_client.tools import ToolRegistry

    store.create(uuid="conv1")
    store.create(uuid="conv2")
    media.save_bytes("conv2", b"secret\n", "text/plain", name="secret.txt")

    other = ToolRegistry()
    for tool in build_media_tools(store, media, lambda: "conv1", settings=settings):
        other.add(tool)

    result = await other.call("read_file", {"path": "secret.txt"})
    assert result.is_error


async def test_an_attached_name_that_is_gone_from_disk_is_an_error(registry, media, store):
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", b"hello\n", "text/plain", name="gone.txt")
    media.path_for_url(url).unlink()

    result = await registry.call("read_file", {"path": "gone.txt"})
    assert result.is_error


async def test_the_numbering_resumes_where_a_truncated_inline_stopped(registry, media, store, settings):
    """The interaction that makes a big .py usable: inline cuts at the per-file cap and
    names a character offset, and `read_file` at that offset carries on from exactly
    there — no gap, nothing repeated.
    """
    store.create(uuid="conv1")
    source = "".join(f"def f{i:05d}():\n    return {i}\n" for i in range(4_000))
    url = media.save_bytes("conv1", source.encode(), "text/plain", name="module.py")

    out = rehydrate(
        [{"role": "user", "content": [{"type": "_file", "kind": "text", "name": "module.py", "url": url}]}],
        media,
        settings,
    )
    sent = out[0]["content"]
    assert f"[truncated: showing {settings.model_text_max_chars:,}" in sent

    offset = int(re.search(r"offset=(\d+)", sent).group(1))
    assert offset == settings.model_text_max_chars
    assert source[:offset] in sent                # the inline really is that prefix

    body = text_of(await registry.call("read_file", {"path": url, "offset": offset}))
    window = "\n".join(read_file_lines(body)[:-1])
    assert source[offset:offset + len(window)] == window


async def test_the_continuation_labels_the_line_the_inline_stopped_on(registry, media, store):
    """The cut lands mid-line, so the first labelled line is that line's remainder —
    mislabelling it would make a caller mis-count the file."""
    store.create(uuid="conv1")
    source = "".join(f"line {i:03d} of a deliberately long file\n" for i in range(400))
    url = media.save_bytes("conv1", source.encode(), "text/plain", name="long.py")

    offset = 1_000
    body = text_of(await registry.call("read_file", {"path": url, "offset": offset}))
    first_label = int(re.match(r" *(\d+)  ", body.split("\n")[1]).group(1))
    assert first_label == source.count("\n", 0, offset) + 1


async def test_read_file_shows_a_python_file_with_its_line_numbers(registry, media, store):
    store.create(uuid="conv1")
    url = media.save_bytes("conv1", b"import os\n\n\ndef main():\n    return 0\n", "text/plain", name="main.py")

    body = text_of(await registry.call("read_file", {"path": url}))
    assert "1  import os" in body
    assert "5      return 0" in body
    assert "5 lines" in body and "utf-8" in body
