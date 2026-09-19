"""Tests for the web-search MCP server (``mcp_server/mcpserver.py``).

The search path was rewritten around one keyed API (Gemini + Google Search
grounding) because the previous unkeyed scrapers could be refused without raising,
and a refused search was being reported to the user as *no results* — the tool
told them a topic did not exist when the truth was that it never got to ask.

Three things are being defended here.

The first is the key. It does not live in this process: it arrives on every
request as a header, from the MCP server entry in the client's config. So the
header spellings, their precedence over the environment, and the fact that a key
edited in the settings panel reaches the tool on the very next call are all part
of the contract, and all tested end-to-end through the JSON-RPC handler.

The second is honesty about failure. ``SearchUnavailable`` must never surface as
"No results": the model repeats that text to the user as fact. Every failure mode
— no key, a rejected key, an exhausted quota, a missing package, a dead network —
is asserted to produce a message that says it is a setup or rate-limit problem.

The third is that a reply is pure JSON. The old code streamed "." heartbeat
characters in front of the payload while a slow scrape ran, which made the body
``....{"jsonrpc": ...}``. The official MCP client validates the whole body as
JSON, so those calls were unparseable — the tool looked like it never answered.

Nothing here touches the network: ``genai.Client`` is replaced by a stub before
every search, and the protocol tests that do call a tool either patch ``call_tool``
outright or fail before any request is built.
"""

from __future__ import annotations

import builtins
import json

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from mcp_server import mcpserver

# ── fixtures ──────────────────────────────────────────────────────────────────

# The key is normally in the client's config, but the environment is a documented
# fallback. Clearing it globally keeps a developer's real key from leaking into
# every assertion about "no key configured".
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in mcpserver.API_KEY_ENV:
        monkeypatch.delenv(name, raising=False)


class Response:
    """Stands in for a ``GenerateContentResponse``.

    Its candidates are built through the real ``types`` classes so that the
    attribute names ``_grounding_sources`` reads are the SDK's, not this file's.
    """

    def __init__(self, text="an answer", chunks=(), text_raises=False):
        self._text = text
        self._text_raises = text_raises
        self.candidates = []
        if chunks is not None:
            from google.genai import types

            self.candidates = [types.GenerateContentResponse.model_validate({
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": text or ""}]},
                    "grounding_metadata": {"grounding_chunks": list(chunks)},
                }],
            }).candidates[0]]

    @property
    def text(self):
        # `.text` genuinely raises on a reply that was cut off mid-part, rather
        # than returning None, so the stub has to be able to do the same.
        if self._text_raises:
            raise ValueError("the reply was cut off")
        return self._text


class FakeGemini:
    """A ``genai.Client`` that records what it was asked and returns a canned reply.

    The search goes through ``chats.create(...).send_message(...)`` rather than
    ``models.generate_content(...)``, so it is *those* two calls that get recorded.
    """

    def __init__(self, reply=None, error=None):
        self.reply = reply if reply is not None else Response()
        self.error = error
        self.client_kwargs = None
        self.chat_kwargs = None
        self.message = None

    def client_factory(self, **kwargs):
        self.client_kwargs = kwargs
        return _FakeClient(self)

    def create_chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return _FakeChat(self)

    def send_message(self, message):
        self.message = message
        if self.error is not None:
            raise self.error
        return self.reply


class _FakeChat:
    def __init__(self, gemini):
        self._gemini = gemini

    def send_message(self, message):
        return self._gemini.send_message(message)


class _FakeClient:
    def __init__(self, gemini):
        self.chats = self
        self._gemini = gemini

    def create(self, **kwargs):
        return self._gemini.create_chat(**kwargs)


@pytest.fixture
def gemini(monkeypatch):
    """Replace ``genai.Client``, and skip if the package is not installed."""
    genai = pytest.importorskip("google.genai")
    fake = FakeGemini()
    monkeypatch.setattr(genai, "Client", fake.client_factory)
    return fake


def _request(headers=None) -> Request:
    """A bare request carrying ``headers`` — enough for ``_api_key`` to read.

    Key names are lowercased because that is how they arrive over ASGI; matching
    them case-insensitively is the HTTP server's job, not this module's.
    """
    raw = [(str(k).lower().encode(), str(v).encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/mcp", "headers": raw})


def _web(uri="https://a.test/1", title="A"):
    return {"web": {"uri": uri, "title": title}}


# ── the key ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", mcpserver.API_KEY_HEADERS)
def test_every_accepted_header_spelling_works(name):
    assert mcpserver._api_key(_request({name: "K1"})) == "K1"


def test_bearer_authorization_is_accepted():
    assert mcpserver._api_key(_request({"Authorization": "Bearer  K2"})) == "K2"


def test_header_beats_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert mcpserver._api_key(_request({"x-api-key": "from-header"})) == "from-header"


def test_environment_is_the_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    assert mcpserver._api_key(_request()) == "from-env"


def test_a_blank_header_falls_through_to_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert mcpserver._api_key(_request({"x-api-key": "   "})) == "from-env"


def test_no_key_anywhere_is_empty():
    assert mcpserver._api_key(_request()) == ""
    assert mcpserver._api_key(None) == ""


def test_key_is_read_per_call_not_at_import():
    """There is no cached key: editing it in the panel takes effect immediately."""
    assert mcpserver._api_key(_request({"x-api-key": "first"})) == "first"
    assert mcpserver._api_key(_request({"x-api-key": "second"})) == "second"


# ── failure hints ─────────────────────────────────────────────────────────────

def test_a_rejected_key_is_named():
    errors = pytest.importorskip("google.genai.errors")

    exc = errors.APIError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
    assert mcpserver._key_hint(exc) == "the API key was rejected"


def test_an_exhausted_quota_is_named():
    errors = pytest.importorskip("google.genai.errors")

    exc = errors.APIError(429, {"error": {"message": "Quota exceeded for quota metric"}})
    assert mcpserver._key_hint(exc) == "the free-tier daily quota is used up"


def test_an_unrecognised_error_gets_no_hint():
    assert mcpserver._key_hint(ValueError("something odd")) == ""


# ── sources ───────────────────────────────────────────────────────────────────

def test_sources_are_deduped_and_non_web_chunks_are_skipped():
    response = Response(chunks=[
        _web("https://a.test/1", "A"),
        {"maps": {"uri": "https://maps.test/x", "title": "Somewhere"}},
        _web("https://a.test/1", "A again"),
        _web("https://b.test/2", ""),
    ])
    assert mcpserver._grounding_sources(response) == [
        ("A", "https://a.test/1"),
        ("", "https://b.test/2"),
    ]


def test_a_response_without_grounding_metadata_has_no_sources():
    assert mcpserver._grounding_sources(Response(chunks=None)) == []


def test_a_response_without_candidates_has_no_sources():
    response = Response()
    response.candidates = []
    assert mcpserver._grounding_sources(response) == []


# ── the search itself ─────────────────────────────────────────────────────────

def test_no_key_is_unavailable_rather_than_no_results(gemini):
    # `gemini` is here for its import guard: the key is checked after the SDK import.
    with pytest.raises(mcpserver.SearchUnavailable) as raised:
        mcpserver._gemini_search("anything", "")
    message = str(raised.value)
    assert "no Gemini API key" in message
    # It has to say where to get one, or the failure is not actionable.
    assert "aistudio.google.com/apikey" in message


def test_a_missing_package_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def refuse_google(name, *args, **kwargs):
        if name == "google" or name.startswith("google."):
            raise ImportError("No module named 'google'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_google)
    with pytest.raises(mcpserver.SearchUnavailable) as raised:
        mcpserver._gemini_search("anything", "K")
    assert "pip install google-genai" in str(raised.value)


def test_the_answer_and_its_urls_are_both_returned(gemini):
    gemini.reply = Response(text="Python 3.14 is out.", chunks=[
        _web("https://python.test/3.14", "What's new"),
        _web("https://news.test/x", "Coverage"),
    ])
    text = mcpserver._gemini_search("python 3.14", "K")
    lines = text.split("\n")
    assert lines[0] == "Python 3.14 is out."
    assert "" in lines
    assert "Sources:" in lines
    assert "1. What's new — https://python.test/3.14" in lines
    assert "2. Coverage — https://news.test/x" in lines


def test_the_request_is_grounded_and_timed_out(gemini):
    mcpserver._gemini_search("query text", "K")
    # A per-call http_options is what stops a hung request from holding the tool
    # call open forever; genai measures it in milliseconds.
    assert gemini.client_kwargs["api_key"] == "K"
    assert gemini.client_kwargs["http_options"]["timeout"] == mcpserver.GEMINI_TIMEOUT * 1000
    assert gemini.chat_kwargs["model"] == mcpserver.GEMINI_MODEL
    assert gemini.message == "query text"
    # The grounding tool is the whole reason this API was chosen: without it the
    # answer is unverifiable and there are no URLs to cite.
    assert gemini.chat_kwargs["config"] == {"tools": [{"google_search": {}}]}


def test_the_search_uses_a_chat_not_the_deprecated_afc_path(gemini):
    """``models.generate_content`` with ``tools`` is the path the SDK warns about.

    Handing a tool to it routes the call through automatic function calling, so
    every search logged "Direct use of automatic function calling (AFC) in
    Models.generate_content is not recommended. Instead, we recommend to use AFC
    in Chat.send_message." AFC is meaningless here — google_search runs inside
    Gemini and no Python callable is ever passed — so the chat is the right call,
    and this asserts the client was not even offered the other one.
    """
    mcpserver._gemini_search("q", "K")
    assert gemini.chat_kwargs is not None          # chats.create(...) was used
    assert not hasattr(_FakeClient, "models")      # ...and models.* was not


def test_the_source_list_is_capped(gemini):
    gemini.reply = Response(chunks=[_web(f"https://s.test/{i}", f"S{i}") for i in range(20)])
    lines = mcpserver._gemini_search("q", "K").split("\n")
    numbered = [line for line in lines if line[:1].isdigit() and ". " in line]
    assert len(numbered) == mcpserver.SEARCH_MAX_SOURCES
    assert numbered[-1].startswith(f"{mcpserver.SEARCH_MAX_SOURCES}. ")


def test_an_answer_without_sources_has_no_empty_heading(gemini):
    gemini.reply = Response(text="Nothing to cite.", chunks=None)
    text = mcpserver._gemini_search("q", "K")
    assert text == "Nothing to cite."
    assert "Sources:" not in text


def test_sources_without_an_answer_are_still_reported(gemini):
    """A cut-off reply loses the prose but the URLs are still worth handing over."""
    gemini.reply = Response(text="", chunks=[_web("https://a.test/1", "A")])
    text = mcpserver._gemini_search("q", "K")
    assert "no written answer" in text
    assert "https://a.test/1" in text


def test_text_that_raises_does_not_lose_the_sources(gemini):
    gemini.reply = Response(text_raises=True, chunks=[_web("https://a.test/1", "A")])
    text = mcpserver._gemini_search("q", "K")
    assert "https://a.test/1" in text


def test_a_reply_with_nothing_at_all_is_unavailable(gemini):
    gemini.reply = Response(text="", chunks=None)
    with pytest.raises(mcpserver.SearchUnavailable):
        mcpserver._gemini_search("q", "K")


@pytest.mark.parametrize("code,payload", [
    (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}),
    (429, {"error": {"message": "Quota exceeded for quota metric"}}),
])
def test_an_api_error_is_unavailable_and_says_which(gemini, code, payload):
    from google.genai import errors

    gemini.error = errors.APIError(code, payload)
    with pytest.raises(mcpserver.SearchUnavailable) as raised:
        mcpserver._gemini_search("q", "K")
    message = str(raised.value)
    assert f"HTTP {code}" in message
    assert "quota" in message or "rejected" in message


def test_an_unreachable_api_is_unavailable(gemini):
    gemini.error = ConnectionError("name resolution failed")
    with pytest.raises(mcpserver.SearchUnavailable) as raised:
        mcpserver._gemini_search("q", "K")
    assert "could not reach Gemini" in str(raised.value)


# ── web_fetch: a download and a tag strip, and NO model ───────────────────────

def test_there_is_no_llm_in_the_fetch_path():
    """The whole point of the rewrite: nothing here can call a model.

    web_fetch used to hand the page to local llama.cpp (a 4-way parallel map, then
    a reduce) to strip boilerplate. On a big page that needed ~135 s, past the MCP
    client's 120 s per-call budget — and a call that outlives that budget dooms the
    session it was sent on, so every fetch after the first big page failed with
    "Connection closed", including a bare example.com.
    """
    for gone in ("_llm_chat", "_call_llama", "_chunk", "_extract_page",
                 "_SYS_EXTRACT", "_SYS_FINAL", "AICall", "_HAS_AI_CALL",
                 "LLAMA_URL"):
        assert not hasattr(mcpserver, gone), f"{gone} should be gone"


def test_readable_text_drops_scripts_and_keeps_the_prose():
    html = (
        "<html><head><title>T</title>"
        "<script>var x = 'SECRET IN A SCRIPT';</script>"
        "<style>.a { color: red }</style></head>"
        "<body><nav>Home | About</nav>"
        "<p>The actual sentence.</p>"
        "<div>Second paragraph.</div>"
        "<!-- a comment --></body></html>"
    )
    text = mcpserver._readable_text(html)

    assert "The actual sentence." in text
    assert "Second paragraph." in text
    assert "SECRET IN A SCRIPT" not in text      # script contents are not text
    assert "color: red" not in text              # nor are style contents
    assert "<" not in text and ">" not in text   # every tag is gone
    # Block-level elements became breaks, so the two paragraphs are not fused.
    assert "The actual sentence.\nSecond paragraph." in text


def test_a_page_is_returned_as_plain_text_with_no_model_call(monkeypatch):
    monkeypatch.setattr(mcpserver, "_fetch_page",
                        lambda url: ("the page text", []))
    content = mcpserver.call_tool("web_fetch", {"url": "https://a.test/1"})
    assert content == [{"type": "text", "text": "the page text"}]


def test_content_images_come_back_alongside_the_text(monkeypatch):
    monkeypatch.setattr(mcpserver, "_fetch_page",
                        lambda url: ("the page text", ["https://a.test/i.jpg"]))
    monkeypatch.setattr(mcpserver, "_download_images",
                        lambda urls: [{"type": "image", "data": "b64",
                                       "mimeType": "image/jpeg"}])
    content = mcpserver.call_tool("web_fetch", {"url": "https://a.test/1"})
    assert content[0] == {"type": "text", "text": "the page text"}
    assert content[1]["type"] == "image"


def test_a_bare_domain_is_assumed_to_be_https(monkeypatch):
    """The model naturally sends `example.com`; rejecting that was pure friction."""
    seen = {}

    def fetch(url):
        seen["url"] = url
        return ("text", [])

    monkeypatch.setattr(mcpserver, "_fetch_page", fetch)
    mcpserver.call_tool("web_fetch", {"url": "example.com"})
    assert seen["url"] == "https://example.com"


def test_a_non_http_scheme_is_still_refused(monkeypatch):
    def fetch(url):
        raise AssertionError("must not be fetched")

    monkeypatch.setattr(mcpserver, "_fetch_page", fetch)
    assert "Invalid URL" in mcpserver.call_tool("web_fetch", {"url": "ftp://a.test/x"})[0]["text"]


def test_a_missing_url_is_reported():
    assert mcpserver.call_tool("web_fetch", {}, "K")[0]["text"] == "No url provided."


def test_an_empty_page_says_so_rather_than_returning_nothing(monkeypatch):
    monkeypatch.setattr(mcpserver, "_fetch_page", lambda url: ("", []))
    assert "No readable text" in mcpserver.call_tool("web_fetch", {"url": "a.test"})[0]["text"]


def test_a_failed_download_is_a_fetch_failure_not_a_crash(monkeypatch):
    def fetch(url):
        raise OSError("name resolution failed")

    monkeypatch.setattr(mcpserver, "_fetch_page", fetch)
    assert "Fetch failed" in mcpserver.call_tool("web_fetch", {"url": "a.test"})[0]["text"]


# ── call_tool ─────────────────────────────────────────────────────────────────

def test_an_unavailable_search_never_reads_as_no_results():
    content = mcpserver.call_tool("web_search", {"query": "quantum widgets"}, "")
    text = content[0]["text"]
    assert "Search is unavailable" in text
    assert "NOT an absence of results" in text
    # The exact lie the old scrapers told, whenever they were refused.
    assert "No results" not in text


def test_an_available_search_returns_the_summary(monkeypatch):
    monkeypatch.setattr(mcpserver, "_gemini_search", lambda query, api_key: f"summary for {query}")
    content = mcpserver.call_tool("web_search", {"query": "quantum widgets"}, "K")
    assert content == [{"type": "text", "text": "summary for quantum widgets"}]


def test_a_missing_query_is_reported():
    assert mcpserver.call_tool("web_search", {}, "K")[0]["text"] == "No query provided."


def test_an_unknown_tool_is_reported():
    assert "Unknown tool" in mcpserver.call_tool("nope", {}, "K")[0]["text"]


# ── the protocol ──────────────────────────────────────────────────────────────

client = TestClient(mcpserver.app)


def _rpc(body):
    return client.post("/mcp", json=body)


def test_tools_call_answers_with_pure_json():
    """No heartbeat dots: the client validates the whole body as JSON."""
    response = _rpc({
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "web_search", "arguments": {"query": "x"}},
    })
    assert response.status_code == 200
    assert response.text.lstrip().startswith("{")
    assert not response.text.startswith(".")
    payload = json.loads(response.text)
    assert payload["id"] == 7
    assert payload["result"]["isError"] is False


@pytest.mark.parametrize("header", list(mcpserver.API_KEY_HEADERS) + ["Authorization"])
def test_the_key_in_the_request_reaches_the_tool(monkeypatch, header):
    """The path a key edited in the settings panel takes, end to end.

    This is the integration the whole feature rests on: the client stores the key
    in the MCP server entry's headers, sends it on every request, and the tool
    reads it per call — so a new key is in force without restarting anything.
    """
    seen = {}

    def recorder(name, arguments, api_key=""):
        seen["name"] = name
        seen["arguments"] = arguments
        seen["api_key"] = api_key
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(mcpserver, "call_tool", recorder)
    value = "Bearer K9" if header == "Authorization" else "K9"
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "web_search", "arguments": {"query": "q"}}},
        headers={header: value},
    )
    assert response.status_code == 200
    assert seen["name"] == "web_search"
    assert seen["arguments"] == {"query": "q"}
    assert seen["api_key"] == "K9"


def test_a_tool_that_blows_up_becomes_a_tool_error(monkeypatch):
    def boom(*_):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mcpserver, "call_tool", boom)
    payload = json.loads(_rpc({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "web_search", "arguments": {"query": "q"}},
    }).text)
    assert "kaboom" in payload["result"]["content"][0]["text"]


def test_initialize_advertises_both_tools_and_the_unavailable_caveat():
    payload = json.loads(_rpc({"jsonrpc": "2.0", "id": 0, "method": "initialize"}).text)
    info = payload["result"]["serverInfo"]
    assert info["name"] == "websearch-mcp"
    # The model has to be told what "unavailable" means, or it will repeat it as
    # "this topic has no coverage".
    assert "unavailable" in info["instructions"]
    assert "web_fetch" in info["instructions"]


def test_tools_list_names_web_search_and_web_fetch():
    payload = json.loads(_rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).text)
    names = [tool["name"] for tool in payload["result"]["tools"]]
    assert names == ["web_search", "web_fetch"]


def test_an_unknown_method_is_a_json_rpc_error():
    response = _rpc({"jsonrpc": "2.0", "id": 4, "method": "does/not/exist"})
    assert response.status_code == 404
    assert json.loads(response.text)["error"]["code"] == -32601


def test_a_get_is_an_empty_response_not_an_error():
    assert client.get("/mcp").status_code == 204


def test_a_broken_body_is_a_parse_error():
    response = client.post("/mcp", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert json.loads(response.text)["error"]["code"] == -32700

# ── first run: seeding `mcp.json` ─────────────────────────────────────────────
# This server never reads mcp.json — the app does, and sends the key in as a header.
# It creates one anyway on the way up, because on a fresh clone it is the process a
# user starts first, and copying the template there is what tells the app this server
# exists at all.

TEMPLATE = '{"servers": [{"name": "websearch"}]}\n'


def test_ensure_mcp_config_copies_the_template(tmp_path):
    (tmp_path / "mcp.example.json").write_text(TEMPLATE, encoding="utf-8")

    created = mcpserver.ensure_mcp_config(str(tmp_path))

    assert created == str(tmp_path / "mcp.json")
    assert (tmp_path / "mcp.json").read_text(encoding="utf-8") == TEMPLATE


def test_ensure_mcp_config_never_overwrites_an_existing_file(tmp_path):
    (tmp_path / "mcp.example.json").write_text(TEMPLATE, encoding="utf-8")
    (tmp_path / "mcp.json").write_text('{"servers": []}', encoding="utf-8")

    assert mcpserver.ensure_mcp_config(str(tmp_path)) == ""
    assert (tmp_path / "mcp.json").read_text(encoding="utf-8") == '{"servers": []}'


def test_ensure_mcp_config_without_a_template_creates_nothing(tmp_path):
    assert mcpserver.ensure_mcp_config(str(tmp_path)) == ""
    assert not (tmp_path / "mcp.json").exists()


def test_ensure_mcp_config_leaves_no_temp_file_behind(tmp_path):
    """The copy goes through `<name>.tmp` + os.replace, so a crash mid-write cannot
    leave a half-written config for the app to parse on its next start."""
    (tmp_path / "mcp.example.json").write_text(TEMPLATE, encoding="utf-8")

    mcpserver.ensure_mcp_config(str(tmp_path))

    assert not (tmp_path / "mcp.json.tmp").exists()
