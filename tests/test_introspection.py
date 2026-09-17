"""Tests for the self-reflection MCP server (``mcp_server/self_reflection.py``).

Two things are being defended here.

The first is the boundary. This server hands an LLM the contents of a repository,
so every path it accepts must be proved to stay inside the project root and to
stay away from credentials and from other people's conversations. The refusal
cases matter more than the happy ones.

The second is that degradation is honest. The summariser is optional, and when it
cannot be reached the tools must still return the deterministic map, symbols and
imports, labelled as such — "could not ask the model" must never be presented to
the model as "there is nothing here".

Nothing in this file touches the network: ``disabled=True`` on the summariser
short-circuits before any client is built, and the protocol tests never call a
tool that summarises.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mcp_server import self_reflection as sr

# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MODULE = '''"""Doc line one.

More prose that the outline does not need.
"""
import json
import os
from pathlib import Path

import starlette
from starlette.routing import Route

from .helper import thing

LIMIT = 10
label = "x"


class Widget:
    """A widget."""

    @staticmethod
    def make():
        return Widget()

    @classmethod
    def from_name(cls, name):
        return cls()

    def render(self, rows=1):
        """Render it."""
        def inner():
            return rows
        return inner


async def fetch(url, *, timeout=1.0):
    """Fetch something."""


def outer(a):
    def middle(b):
        return lambda c: b + c
    return middle
'''

BROKEN_MODULE = "def broken(:\n    pass\n"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A tiny project root, so a test never walks the real repository by accident."""
    (tmp_path / "server").mkdir()
    (tmp_path / "mcp_server").mkdir()
    (tmp_path / "memory" / "abc").mkdir(parents=True)

    (tmp_path / "server" / "app.py").write_text(SAMPLE_MODULE, encoding="utf-8")
    (tmp_path / "server" / "broken.py").write_text(BROKEN_MODULE, encoding="utf-8")
    (tmp_path / "mcp_server" / "server.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Title\n\nprose\n\n## Sub\n", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("starlette>=0.37\n", encoding="utf-8")

    # Must never be readable, and must not even be listed.
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-realvalue0123456789\n", encoding="utf-8")
    (tmp_path / "server" / "key.pem").write_text("-----BEGIN KEY-----\n", encoding="utf-8")
    (tmp_path / "memory" / "abc" / "conversation.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(sr, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(sr, "_FACTS_CACHE", {})
    monkeypatch.delenv("INTROSPECTION_LLM", raising=False)
    return tmp_path.resolve()


@pytest.fixture
def offline():
    """A summariser that is switched off: no client, no network, no waiting."""
    return sr.Summarizer(providers="", disabled=True)


def text_of(blocks) -> str:
    assert blocks and blocks[0]["type"] == "text"
    return blocks[0]["text"]


def strip_gutter(line: str) -> str:
    """Drop the "    12  " prefix from a numbered source line."""
    return re.sub(r"^\s*\d+  ", "", line)


# ── tool definitions ──────────────────────────────────────────────────────────

def test_exactly_three_tools():
    assert [tool["name"] for tool in sr.TOOLS] == ["introspect", "explain_file", "read_source"]


def test_tool_schemas_are_well_formed():
    for tool in sr.TOOLS:
        assert tool["description"].strip()
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"]["properties"]


def test_only_the_file_tools_require_a_path():
    by_name = {tool["name"]: tool for tool in sr.TOOLS}
    assert by_name["explain_file"]["inputSchema"]["required"] == ["path"]
    assert by_name["read_source"]["inputSchema"]["required"] == ["path"]
    assert not by_name["introspect"]["inputSchema"].get("required")
    assert "include" in by_name["introspect"]["inputSchema"]["properties"]


def test_tools_are_json_serialisable():
    assert json.loads(json.dumps(sr.TOOLS))[0]["name"] == "introspect"


def test_module_docstring_names_every_tool():
    """The docstring is the first thing a model reads; drift here is a real bug."""
    doc = sr.__doc__ or ""
    for tool in sr.TOOLS:
        assert f"``{tool['name']}``" in doc


def test_every_advertised_tool_is_dispatchable(project, offline):
    for tool in sr.TOOLS:
        arguments = {"path": "server/app.py"} if tool["name"] != "introspect" else {}
        text = text_of(sr.call_tool(tool["name"], arguments, offline))
        assert not text.startswith("Unknown tool"), tool["name"]


# ── path containment ──────────────────────────────────────────────────────────

def test_relative_path_inside_root_is_resolved(project):
    assert sr._resolve_path("server/app.py") == project / "server" / "app.py"


def test_absolute_path_inside_root_is_allowed(project):
    assert sr._resolve_path(str(project / "server" / "app.py")) == project / "server" / "app.py"


def test_root_itself_is_allowed(project):
    assert sr._resolve_path(".") == project


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_missing_path_is_refused(project, raw):
    with pytest.raises(sr.Refused, match="no path given"):
        sr._resolve_path(raw)


def test_parent_escape_is_refused(project):
    with pytest.raises(sr.Refused, match="outside the project root"):
        sr._resolve_path("../secrets.txt")


def test_deep_parent_escape_is_refused(project):
    with pytest.raises(sr.Refused, match="outside the project root"):
        sr._resolve_path("server/../../../../etc/passwd")


def test_absolute_path_outside_root_is_refused(project):
    with pytest.raises(sr.Refused, match="outside the project root"):
        sr._resolve_path("C:/Windows/win.ini")


def test_nul_byte_is_refused(project):
    with pytest.raises(sr.Refused, match="NUL byte"):
        sr._resolve_path("server/app.py\x00.txt")


def test_credentials_file_is_refused(project):
    with pytest.raises(sr.Refused, match="credentials"):
        sr._resolve_path(".env")


def test_private_key_suffix_is_refused(project):
    with pytest.raises(sr.Refused, match="credentials"):
        sr._resolve_path("server/key.pem")


def test_memory_is_refused(project):
    with pytest.raises(sr.Refused, match="other conversations"):
        sr._resolve_path("memory/abc/conversation.json")


def test_refusal_names_the_reason_not_the_file_contents(project):
    with pytest.raises(sr.Refused) as caught:
        sr._resolve_path(".env")
    assert "sk-realvalue" not in str(caught.value)


def make_link(link: Path, target: Path) -> bool:
    """Point `link` at `target`, with a junction as the unprivileged fallback."""
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
        return True
    except (OSError, NotImplementedError):
        pass
    if not target.is_dir():
        return False
    # A directory junction needs no privilege on Windows, and Path.resolve()
    # follows it, so it exercises the same containment rule as a symlink.
    completed = subprocess.run(f'cmd /c mklink /J "{link}" "{target}"', shell=True,
                               capture_output=True)
    return completed.returncode == 0


def test_link_escape_is_refused(project, tmp_path):
    """Resolving before comparing is what makes a link out of the tree useless."""
    outside = tmp_path.parent / "introspection_outside"
    (outside / "nested").mkdir(parents=True, exist_ok=True)
    (outside / "nested" / "secret.txt").write_text("secret\n", encoding="utf-8")

    link = project / "link"
    if not make_link(link, outside):
        pytest.skip("cannot create a symlink or junction on this machine")
    assert (link / "nested" / "secret.txt").exists(), "the link must really work"

    with pytest.raises(sr.Refused, match="outside the project root"):
        sr._resolve_path("link/nested/secret.txt")


def test_link_escape_is_not_listed_either(project, tmp_path):
    outside = tmp_path.parent / "introspection_outside_files"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "leak.py").write_text("SECRET = 1\n", encoding="utf-8")

    link = project / "linked"
    if not make_link(link, outside):
        pytest.skip("cannot create a symlink or junction on this machine")
    assert "linked/leak.py" not in sr._iter_files()


# ── what the map is allowed to see ────────────────────────────────────────────

def test_listing_hides_credentials_and_conversations(project):
    rels = sr._iter_files()
    assert "server/app.py" in rels
    assert ".env" not in rels
    assert not any(rel.endswith(".pem") for rel in rels)
    assert not any(rel.startswith("memory/") for rel in rels)


def test_listing_is_sorted_and_relative(project):
    rels = sr._iter_files()
    assert rels == sorted(rels)
    assert all(not Path(rel).is_absolute() for rel in rels)


def test_listing_honours_a_glob(project):
    rels = sr._iter_files("server/*")
    assert "server/app.py" in rels
    assert "README.md" not in rels


def test_ignored_directories_are_skipped(project):
    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "app.cpython-311.pyc").write_text("x", encoding="utf-8")
    assert not any("__pycache__" in rel for rel in sr._iter_files())


def test_real_project_listing_includes_the_frontend():
    """`frontend/dist/assets/*.js` is hand-written source in this project.

    Excluding "dist" as build output would hide the entire UI from introspection,
    which is why it is absent from IGNORED_DIRS. This fails if someone adds it back.
    """
    rels = sr._iter_files()
    assert "frontend/dist/assets/app.js" in rels
    assert "server/app.py" in rels
    assert not any(rel.startswith("memory/") for rel in rels)


# ── decoding text ─────────────────────────────────────────────────────────────

def test_empty_bytes_decode_to_empty_text():
    assert sr._decode(b"") == ("", "utf-8")


def test_pdf_is_not_text():
    assert sr._decode(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n") is None


def test_nul_bytes_mean_binary():
    assert sr._decode(b"PK\x03\x04\x00\x00data\x00\x00") is None


def test_high_entropy_garbage_is_not_text():
    assert sr._decode(bytes(range(1, 256))) is None


def test_utf8_bom_is_consumed_not_shown():
    text, encoding = sr._decode("hello\n".encode("utf-8-sig"))
    assert (text, encoding) == ("hello\n", "utf-8-sig")


def test_cp1252_fallback():
    text, encoding = sr._decode(b"caf\xe9\n")
    assert encoding == "cp1252"
    assert text == "caf\xe9\n"


def test_missing_file_is_refused(project):
    with pytest.raises(sr.Refused, match="does not exist"):
        sr._read_text_file(project / "server" / "absent.py")


def test_directory_is_refused(project):
    with pytest.raises(sr.Refused, match="is a directory"):
        sr._read_text_file(project / "server")


def test_binary_file_is_refused(project):
    (project / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(sr.Refused, match="not text"):
        sr._read_text_file(project / "blob.bin")


def test_language_detection():
    assert sr.language_of(Path("a.py")) == "python"
    assert sr.language_of(Path("a.JS")) == "javascript"
    assert sr.language_of(Path("Dockerfile")) == "dockerfile"
    assert sr.language_of(Path("unknown.zzz")) == "text"


# ── secret masking ────────────────────────────────────────────────────────────

def test_api_key_is_masked():
    scrubbed, count = sr._redact('api_key = "supersecretvalue"')
    assert count == 1
    assert "supersecretvalue" not in scrubbed
    assert "<redacted>" in scrubbed
    assert "api_key" in scrubbed          # the name stays, only the value goes


def test_sk_token_is_masked():
    scrubbed, count = sr._redact("token: sk-abcdefghijklmnopqrst")
    assert count >= 1
    assert "sk-abcdefghijklmnopqrst" not in scrubbed


def test_bearer_token_is_masked():
    scrubbed, count = sr._redact("Authorization: Bearer abcdefghijklmnop.qrst")
    assert count == 1
    assert "abcdefghijklmnop" not in scrubbed


def test_jwt_is_masked():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abcdefgh"
    scrubbed, count = sr._redact(f"cookie={jwt}")
    assert count >= 1
    assert jwt not in scrubbed


def test_ordinary_text_is_untouched():
    text = "def add(a, b):\n    return a + b\n"
    assert sr._redact(text) == (text, 0)


def test_redaction_note_pluralises():
    assert sr._redaction_note(0) == ""
    assert "1 secret-looking value " in sr._redaction_note(1)
    assert "2 secret-looking values" in sr._redaction_note(2)


def test_read_source_masks_before_returning(project):
    (project / "config.py").write_text('apiKey = "hunter2hunter2"\n', encoding="utf-8")
    text = text_of(sr.tool_read_source({"path": "config.py"}))
    assert "hunter2hunter2" not in text
    assert "<redacted>" in text
    assert "masked" in text


# ── the Python outline ────────────────────────────────────────────────────────

def test_outline_finds_the_public_surface():
    symbols, _ = sr._python_outline(SAMPLE_MODULE)
    kinds = {(symbol.kind, symbol.name) for symbol in symbols}
    assert ("class", "Widget") in kinds
    assert ("function", "outer") in kinds
    assert ("async function", "fetch") in kinds


def test_outline_labels_methods_by_kind():
    symbols, _ = sr._python_outline(SAMPLE_MODULE)
    methods = {symbol.name: symbol.kind for symbol in symbols if "method" in symbol.kind}
    assert methods == {"make": "staticmethod", "from_name": "classmethod", "render": "method"}


def test_outline_finds_constants_and_assignments():
    symbols, _ = sr._python_outline(SAMPLE_MODULE)
    by_name = {symbol.name: symbol.kind for symbol in symbols}
    assert by_name["LIMIT"] == "constant"
    assert by_name["label"] == "assignment"


def test_outline_carries_signatures_and_docstrings():
    symbols, _ = sr._python_outline(SAMPLE_MODULE)
    fetch = next(symbol for symbol in symbols if symbol.name == "fetch")
    assert fetch.signature == "async def fetch(url, *, timeout=1.0):"
    assert fetch.doc == "Fetch something."
    widget = next(symbol for symbol in symbols if symbol.name == "Widget")
    assert widget.signature == "class Widget:"
    render = next(symbol for symbol in symbols if symbol.name == "render")
    assert render.signature == "def render(self, rows=1):"
    assert render.doc == "Render it."
    assert next(s for s in symbols if s.name == "make").decorators == ["staticmethod"]


def test_outline_splits_imports(monkeypatch):
    monkeypatch.setattr(sr, "_third_party_roots", lambda: {"starlette"})
    _, facts = sr._python_outline(SAMPLE_MODULE)
    assert facts["imports"]["stdlib"] == ["json", "os", "pathlib"]
    assert facts["imports"]["third_party"] == ["starlette", "starlette.routing"]
    assert facts["imports"]["local"] == [".helper"]


def test_outline_counts_nested_definitions():
    """`outer` holds `middle`, which holds a lambda: two nested definitions.

    Methods are *not* nested definitions — a method already gets its own line.
    """
    _, facts = sr._python_outline(SAMPLE_MODULE)
    assert facts["nested_definitions"] == 2


def test_outline_reports_a_module_docstring():
    _, facts = sr._python_outline(SAMPLE_MODULE)
    assert facts["doc"].startswith("Doc line one.")
    assert "\n" not in facts["doc"]


def test_outline_of_broken_code_reports_the_error():
    symbols, facts = sr._python_outline(BROKEN_MODULE)
    assert symbols == []
    assert "invalid syntax" in facts["error"]


def test_outline_dispatches_markdown():
    symbols, facts = sr._markdown_outline("# Title\n\nprose\n\n## Sub\n"), {}
    assert [(s.kind, s.name) for s in symbols] == [("heading 1", "Title"), ("heading 2", "Sub")]


def test_markdown_outline_ignores_fenced_headings():
    symbols = sr._markdown_outline("# Real\n\n```\n# Not a heading\n```\n\n## Also real\n")
    assert [symbol.name for symbol in symbols] == ["Real", "Also real"]


def test_outline_dispatches_ini():
    symbols, _ = sr._outline("[pytest]\ntestpaths = tests\n; comment\n[other]\nkey: 1\n",
                             Path("pytest.ini"))
    assert [symbol.kind for symbol in symbols] == ["section", "key", "section", "key"]
    assert symbols[1].name == "testpaths"


def test_outline_never_emits_a_backtick_in_a_description():
    """A stray backtick would end the inline code span and break the transcript."""
    symbols, _ = sr._python_outline('def f():\n    """Uses `x` and `y`."""\n')
    assert "`" not in symbols[0].doc


def test_render_outline_indents_methods_under_their_class():
    symbols = [
        sr.Symbol(kind="class", name="Widget", line=1, signature="class Widget"),
        sr.Symbol(kind="method", name="render", line=3, signature="def render(self)"),
        sr.Symbol(kind="function", name="free", line=9, signature="def free()"),
    ]
    rendered = sr._render_outline(symbols).split("\n")
    assert rendered[0] == "- line 1: `class Widget`"
    assert rendered[1] == "  - line 3: method `def render(self)`"
    assert rendered[2] == "- line 9: `def free()`"


def test_render_outline_says_so_when_empty():
    assert sr._render_outline([]) == "_(no classes, functions, or sections found)_"


# ── read_source ───────────────────────────────────────────────────────────────

def test_read_source_numbers_lines_and_round_trips(project):
    body = "line one\nline two\nline three\n"
    (project / "notes.txt").write_text(body, encoding="utf-8")
    text = text_of(sr.tool_read_source({"path": "notes.txt"}))
    lines = text.split("\n")
    assert lines[0].startswith("notes.txt — ") and "3 lines" in lines[0]
    assert lines[1] == "    1  line one"
    assert lines[2] == "    2  line two"
    assert lines[3] == "    3  line three"
    assert len(lines) == 4, "a trailing newline must not invent a fourth line"
    assert "\n".join(strip_gutter(line) for line in lines[1:]) + "\n" == body


def test_read_source_keeps_a_final_line_that_has_no_newline(project):
    (project / "notes.txt").write_text("alpha\nbeta", encoding="utf-8")
    text = text_of(sr.tool_read_source({"path": "notes.txt"}))
    assert "2 lines" in text and "(lines 1-2)" in text
    assert text.split("\n")[-1] == "    2  beta"


def test_read_source_strips_carriage_returns(project):
    (project / "crlf.txt").write_bytes(b"alpha\r\nbeta\r\n")
    text = text_of(sr.tool_read_source({"path": "crlf.txt"}))
    assert "\r" not in text
    assert "2 lines" in text
    assert text.endswith("    2  beta")


def test_read_source_consumes_a_bom_without_showing_it(project):
    (project / "bom.txt").write_bytes("\ufeffalpha\nbeta\n".encode("utf-8"))
    text = text_of(sr.tool_read_source({"path": "bom.txt"}))
    assert text.split("\n")[1] == "    1  alpha"
    assert "utf-8-sig" in text


def rows(count: int = 50) -> bytes:
    """LF-only bytes: write_text would turn every \\n into \\r\\n on Windows,
    which would make the character offsets in these tests machine-dependent."""
    return "".join(f"row {n}\n" for n in range(1, count + 1)).encode("utf-8")


def test_read_source_header_counts_the_whole_file_not_the_page(project):
    (project / "many.txt").write_bytes(rows())
    text = text_of(sr.tool_read_source({"path": "many.txt", "limit": 20}))
    assert "50 lines" in text
    # The page cuts line 4 in half; it is still reported as line 4.
    assert "(lines 1-4)" in text
    assert text.split("\n")[-2] == "    4  ro"


def test_read_source_pages_from_the_offset_in_its_footer(project):
    (project / "many.txt").write_bytes(rows())
    first = text_of(sr.tool_read_source({"path": "many.txt", "limit": 20}))
    assert "row 1" in first and "row 3" in first
    offset = int(re.search(r"offset=(\d+)", first).group(1))
    second = text_of(sr.tool_read_source({"path": "many.txt", "offset": offset, "limit": 20}))
    assert "(lines 4-7)" in second
    assert strip_gutter(second.split("\n")[1]) == "w 4"    # the rest of line 4
    assert "row 5" in second and "row 6" in second
    assert "[showing a page; the file starts at offset 0]" in second
    assert "showing a page" not in first


def test_read_source_does_not_open_a_page_inside_a_line_break(project):
    """A limit that lands between \\r and \\n must not show a phantom blank line.

    Rows are 8 bytes as CRLF ("row 1\\r\\n" + one digit), so a 7-character page
    stops exactly between the two bytes of the first line break.
    """
    (project / "crlf.txt").write_bytes(b"row 1\r\nrow 2\r\nrow 3\r\n")
    first = text_of(sr.tool_read_source({"path": "crlf.txt", "limit": 7}))
    offset = int(re.search(r"offset=(\d+)", first).group(1))
    second = text_of(sr.tool_read_source({"path": "crlf.txt", "offset": offset}))
    assert "(lines 2-3)" in second
    assert second.split("\n")[1] == "    2  row 2"


def test_read_source_numbering_survives_a_mid_line_offset(project):
    (project / "many.txt").write_bytes(rows())
    # Offset 20 is one character into "row 4", so the remainder is still line 4.
    text = text_of(sr.tool_read_source({"path": "many.txt", "offset": 20, "limit": 20}))
    assert "(lines 4-7)" in text
    assert strip_gutter(text.split("\n")[1]) == "w 4"


def test_read_source_offset_past_the_end_is_explained(project):
    (project / "notes.txt").write_text("short\n", encoding="utf-8")
    text = text_of(sr.tool_read_source({"path": "notes.txt", "offset": 9999}))
    assert "past the end" in text and "offset 0" in text


def test_read_source_reports_an_empty_file(project):
    text = text_of(sr.tool_read_source({"path": "empty.py"}))
    assert "empty file" in text


def test_read_source_clamps_an_absurd_limit(project):
    (project / "notes.txt").write_text("only a little\n", encoding="utf-8")
    text = text_of(sr.tool_read_source({"path": "notes.txt", "limit": 10**9}))
    assert "only a little" in text


def test_read_source_rejects_a_non_numeric_offset(project):
    with pytest.raises(sr.Refused, match="offset must be a whole number"):
        sr.tool_read_source({"path": "server/app.py", "offset": "abc"})


def test_read_source_rejects_a_non_numeric_limit(project):
    with pytest.raises(sr.Refused, match="limit must be a whole number"):
        sr.tool_read_source({"path": "server/app.py", "limit": "abc"})


def test_read_source_refusal_is_returned_to_the_caller_not_raised(project):
    text = text_of(sr.call_tool("read_source", {"path": "../../etc/passwd"}, sr.Summarizer(
        providers="", disabled=True)))
    assert text.startswith("Refused:")
    assert "outside the project root" in text


# ── explain_file ──────────────────────────────────────────────────────────────

def test_explain_file_describes_one_module(project, offline, monkeypatch):
    monkeypatch.setattr(sr, "_third_party_roots", lambda: {"starlette"})
    text = text_of(sr.tool_explain_file({"path": "server/app.py"}, offline))
    assert text.startswith("# server/app.py")
    assert "## What it contains" in text
    assert "`class Widget:`" in text
    assert "**local imports:** .helper" in text
    assert "**third_party imports:** starlette" in text


def test_explain_file_offers_the_verbatim_read(project, offline):
    text = text_of(sr.tool_explain_file({"path": "server/app.py"}, offline))
    assert 'read_source(path="server/app.py")' in text


def test_explain_file_flags_a_module_that_does_not_parse(project, offline):
    text = text_of(sr.tool_explain_file({"path": "server/broken.py"}, offline))
    assert "does not parse" in text


def test_explain_file_outlines_a_config_file(project, offline):
    text = text_of(sr.tool_explain_file({"path": "pytest.ini"}, offline))
    assert "- line 1: `[pytest]`" in text
    assert "- line 2: `testpaths`" in text


def test_explain_file_requires_a_path(project, offline):
    text = text_of(sr.call_tool("explain_file", {}, offline))
    assert text.startswith("Refused: no path given")


# ── introspect ────────────────────────────────────────────────────────────────

def test_overview_maps_the_tree(project, offline):
    text = text_of(sr.tool_project_overview({}, offline))
    assert text.startswith("# ")
    assert "project map" in text
    assert "## Directory tree" in text
    assert "├──" in text or "└──" in text
    assert "server/" in text and "README.md" in text
    assert "## Python modules" in text


def test_overview_lists_modules_with_structure(project, offline):
    text = text_of(sr.tool_project_overview({}, offline))
    assert "`server/app.py`" in text
    assert "1 class, 2 functions, 3 methods" in text
    assert "`server/broken.py`" in text and "syntax error" in text


def test_overview_per_directory_totals_are_sums(project, offline):
    """`server/` in the fixture holds app.py and broken.py; key.pem is not counted."""
    text = text_of(sr.tool_project_overview({}, offline))
    assert re.search(r"server/  \(2 files, \d+ lines\)", text)


def test_overview_hides_credentials_and_says_what_it_skips(project, offline):
    text = text_of(sr.tool_project_overview({}, offline))
    assert ".env" not in text
    assert "key.pem" not in text
    assert "memory" not in text.split("Not walked")[0]
    assert "deliberately unreadable" in text


def test_overview_honours_the_include_filter(project, offline):
    text = text_of(sr.tool_project_overview({"include": "server/*"}, offline))
    assert "Filtered to `server/*`" in text
    assert "README.md" not in text
    assert "server/app.py" in text


# ── honest degradation when there is no model ─────────────────────────────────

def test_disabled_summariser_reports_why(offline):
    assert offline.available is False
    assert "switched off" in offline.reason
    assert offline.model_name == ""


def test_disabled_summariser_refuses_to_summarise(offline):
    with pytest.raises(sr.SummaryUnavailable):
        offline.summarize("system", "user")


def test_overview_without_a_model_still_returns_the_map(project, offline):
    text = text_of(sr.tool_project_overview({}, offline))
    assert "## How it fits together" in text
    assert "Summary unavailable:" in text
    assert "were produced locally" in text
    assert "Summarised by" not in text
    assert "server/app.py" in text


def test_explain_file_without_a_model_still_returns_the_outline(project, offline):
    text = text_of(sr.tool_explain_file({"path": "server/app.py"}, offline))
    assert "## How it works" in text
    assert "Summary unavailable:" in text
    assert "`class Widget:`" in text


def test_config_errors_are_masked_too(monkeypatch):
    """A configuration error can quote the environment, so it is scrubbed."""

    def explode():
        raise RuntimeError("bad key sk-abcdefghijklmnop while loading providers")

    monkeypatch.setattr(sr, "_client_class", explode)
    summarizer = sr.Summarizer(providers="providers.json")
    assert summarizer.available is False
    assert "sk-abcdefghijklmnop" not in summarizer.reason
    assert "<redacted>" in summarizer.reason
    assert "RuntimeError" in summarizer.reason


def test_unknown_tool_never_raises(project, offline):
    assert text_of(sr.call_tool("nope", {}, offline)) == "Unknown tool: nope"


# ── JSON-RPC over the wire ────────────────────────────────────────────────────

def rpc(client, body):
    return client.post("/mcp", json=body)


def parse(raw: str) -> dict:
    """The body may be `....{"jsonrpc": ...}`; the client strips to the first '{'."""
    return json.loads(raw[raw.index("{"):])


@pytest.fixture
def client(project, offline, monkeypatch):
    monkeypatch.setattr(sr, "_DEFAULT_SUMMARIZER", offline)
    with TestClient(sr.app) as test_client:
        yield test_client


def test_initialize_handshake(client):
    reply = parse(rpc(client, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {"protocolVersion": "2025-03-26"}}).text)
    result = reply["result"]
    assert reply["id"] == 1
    assert result["protocolVersion"] == "2025-03-26"
    assert result["serverInfo"]["name"] == "self-reflection-mcp"
    assert result["capabilities"] == {"tools": {}}
    assert "read_source" in result["serverInfo"]["instructions"]


def test_initialized_notification_is_accepted(client):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert response.status_code == 204


def test_tools_list_matches_the_schemas(client):
    reply = parse(rpc(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).text)
    assert reply["result"]["tools"] == sr.TOOLS


def test_tools_call_runs_read_source(client):
    response = rpc(client, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "read_source",
                                       "arguments": {"path": "server/app.py", "limit": 80}}})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    reply = parse(response.text)
    assert reply["result"]["isError"] is False
    assert "server/app.py" in reply["result"]["content"][0]["text"]


def test_tools_call_survives_a_refused_path(client):
    reply = parse(rpc(client, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                               "params": {"name": "read_source",
                                          "arguments": {"path": "../../etc/passwd"}}}).text)
    assert reply["result"]["content"][0]["text"].startswith("Refused:")


def test_tools_call_without_arguments_does_not_crash(client):
    reply = parse(rpc(client, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                               "params": {"name": "read_source"}}).text)
    assert reply["result"]["content"][0]["text"].startswith("Refused:")


def test_unknown_method_is_a_404(client):
    response = rpc(client, {"jsonrpc": "2.0", "id": 6, "method": "no/such/method"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == -32601


def test_malformed_json_is_a_parse_error(client):
    response = client.post("/mcp", content=b"{ not json",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


def test_root_route_answers_too(client):
    reply = parse(client.post("/", json={"jsonrpc": "2.0", "id": 7,
                                         "method": "tools/list"}).text)
    assert [tool["name"] for tool in reply["result"]["tools"]] == [
        "introspect", "explain_file", "read_source"]


def test_get_and_options_are_accepted(client):
    assert client.get("/mcp").status_code == 204
    assert client.options("/mcp").status_code == 204


def test_a_fast_tool_call_has_no_heartbeat_dots(client):
    """Dots are only emitted while a call runs longer than the heartbeat interval."""
    raw = rpc(client, {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                       "params": {"name": "read_source",
                                  "arguments": {"path": "README.md"}}}).text
    assert raw.startswith("{")
