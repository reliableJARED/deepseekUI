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
A filter refusal is a third thing again and gets its own message (see the
"no content filters" section), for the same reason: calling it "unavailable"
would send the user hunting for a quota problem that does not exist.

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
from http.server import BaseHTTPRequestHandler

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


# DuckDuckGo is on in production and off for every test that did not ask for it,
# for the same reason genai.Client is stubbed: without this the suite would reach
# the real DuckDuckGo on every search assertion — slow, flaky, and dependent on
# what the index happens to hold today.
@pytest.fixture(autouse=True)
def no_ddg(monkeypatch):
    monkeypatch.setattr(mcpserver, "DDG_ENABLED", False)


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
    ``attempts`` keeps the chat kwargs of every call, which is how a retry can be
    told from a single call; ``error_when`` decides per attempt whether to fail,
    since the retry deliberately sends a different config.
    """

    def __init__(self, reply=None, error=None, error_when=None):
        self.reply = reply if reply is not None else Response()
        self.error = error
        self.error_when = error_when
        self.client_kwargs = None
        self.chat_kwargs = None
        self.message = None
        self.attempts: list = []

    def client_factory(self, **kwargs):
        self.client_kwargs = kwargs
        return _FakeClient(self)

    def create_chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return _FakeChat(self)

    def send_message(self, message):
        self.message = message
        self.attempts.append(self.chat_kwargs)
        if self.error_when is not None:
            failure = self.error_when(self.chat_kwargs)
            if failure is not None:
                raise failure
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


def _filtered_response(reason="SAFETY"):
    """A real ``GenerateContentResponse`` for a reply a filter stopped.

    Built through the SDK's own classes for the same reason the canned replies are:
    ``prompt_feedback.block_reason`` and ``Candidate.finish_reason`` are names this
    module reads, so they must be the SDK's, not this file's. A blocked reply is one
    with no parts at all — ``.text`` comes back None rather than raising.
    """
    from google.genai import types

    return types.GenerateContentResponse.model_validate({
        "prompt_feedback": {"block_reason": reason},
        "candidates": [{"finish_reason": reason}],
    })


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
    # answer is unverifiable and there are no URLs to cite. The safety side of the
    # same config is asserted on its own below.
    assert gemini.chat_kwargs["config"]["tools"] == [{"google_search": {}}]


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


# ── no content filters ────────────────────────────────────────────────────────
# The requirement is that search is NOT filtered, so these assert the request that
# goes out rather than trusting a default. A default is not a promise: it belongs to
# the model, and GEMINI_SEARCH_MODEL can be pointed at another one.

def test_every_adjustable_category_is_turned_off(gemini):
    mcpserver._gemini_search("q", "K")
    settings = gemini.chat_kwargs["config"]["safety_settings"]
    assert {(s.category.value, s.threshold.value) for s in settings} == {
        (name, "OFF") for name in mcpserver.GEMINI_SAFETY_CATEGORIES
    }
    # One entry per category, not one entry repeated four times.
    assert len(settings) == len(mcpserver.GEMINI_SAFETY_CATEGORIES)


def test_the_categories_are_the_sdks_own_names():
    """Built from the enum, so a renamed category fails here, not silently.

    A category string the API does not know is not ignored — ``getattr`` on the enum
    raises, and a test is the only place that can be allowed to happen. This also
    pins the set to the four the API documents as adjustable, so a future category
    is a deliberate addition rather than an accident.
    """
    from google.genai import types

    assert all(name in types.HarmCategory.__members__ for name in mcpserver.GEMINI_SAFETY_CATEGORIES)
    assert mcpserver.GEMINI_SAFETY_CATEGORIES == (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )


def test_the_settings_survive_the_sdks_own_validation(gemini):
    """The stub accepts any dict; the SDK does not.

    ``Chats.create`` validates ``config`` into a real ``GenerateContentConfig``, so
    this proves the payload the server sends is one the API will accept — and that
    the threshold it names is a threshold this version of the package knows.
    """
    from google.genai import types

    mcpserver._gemini_search("q", "K")
    config = types.GenerateContentConfig(**gemini.chat_kwargs["config"])
    assert {s.threshold.value for s in config.safety_settings} == {"OFF"}
    assert {s.category.value for s in config.safety_settings} == set(mcpserver.GEMINI_SAFETY_CATEGORIES)


def test_a_threshold_this_sdk_does_not_know_still_means_no_filtering(monkeypatch, gemini):
    """BLOCK_NONE is the same promise under the name every version accepts."""
    monkeypatch.setattr(mcpserver, "GEMINI_SAFETY_THRESHOLD", "NOT_A_THRESHOLD")
    assert {s.threshold.value for s in mcpserver._safety_settings()} == {"BLOCK_NONE"}


def _thresholds(chat_kwargs) -> set:
    """The threshold values a recorded call carried."""
    settings = (chat_kwargs or {}).get("config", {}).get("safety_settings") or []
    return {setting.threshold.value for setting in settings}


def test_a_model_that_rejects_off_is_retried_with_the_older_spelling(gemini):
    """`OFF` is the current name; BLOCK_NONE is the same promise under the old one.

    A model that does not know `OFF` 400s the whole request rather than ignoring the
    field, so without a retry a vocabulary difference would turn into a failed
    search — and the one thing this must never become is "search is unavailable".
    """
    from google.genai import errors

    def reject_off(chat_kwargs):
        if "OFF" in _thresholds(chat_kwargs):
            return errors.ClientError(400, {"error": {"message": "invalid threshold"}})
        return None

    gemini.error_when = reject_off
    assert mcpserver._gemini_search("q", "K") == "an answer"
    assert len(gemini.attempts) == 2
    assert _thresholds(gemini.attempts[0]) == {"OFF"}
    assert _thresholds(gemini.attempts[1]) == {"BLOCK_NONE"}


def test_the_retry_stops_rather_than_looping(gemini):
    """One retry, not a cycle: the fallback is the last spelling there is."""
    from google.genai import errors

    gemini.error_when = lambda chat_kwargs: errors.ClientError(
        400, {"error": {"message": "still no"}})
    with pytest.raises(mcpserver.SearchUnavailable):
        mcpserver._gemini_search("q", "K")
    assert len(gemini.attempts) == 2


def test_only_a_400_is_retried(gemini):
    """A 500 or a transport error is not a threshold complaint.

    Retrying those would double the wait for an outage and report the second failure
    instead of the first, which is the more informative one.
    """
    from google.genai import errors

    gemini.error_when = lambda chat_kwargs: errors.ServerError(
        500, {"error": {"message": "overloaded"}})
    with pytest.raises(mcpserver.SearchUnavailable):
        mcpserver._gemini_search("q", "K")
    assert len(gemini.attempts) == 1


def test_the_configured_threshold_is_never_silently_narrowed(gemini):
    """The fallback is only ever used when the configured one was rejected.

    Stepping down to a *stricter* setting would answer with a filtered search while
    looking like success, which is the one outcome worse than a visible failure.
    """
    mcpserver._gemini_search("q", "K")
    assert len(gemini.attempts) == 1
    assert _thresholds(gemini.attempts[0]) == {mcpserver.GEMINI_SAFETY_THRESHOLD}


def test_a_filtered_reply_is_a_refusal_not_an_absence(gemini):
    """Every adjustable filter is off, so a block means the API's own floor.

    Reported on its own rather than folded into SearchUnavailable: "unavailable"
    says *setup or quota*, which would send the user looking for a key problem that
    does not exist, and the one lie this whole path exists to prevent is a refusal
    being relayed as "there is nothing on that topic".
    """
    gemini.reply = _filtered_response("SAFETY")
    with pytest.raises(mcpserver.SearchBlocked) as raised:
        mcpserver._gemini_search("q", "K")
    message = str(raised.value)
    assert "SAFETY" in message
    assert "OFF" in message            # says the adjustable ones were already off
    assert "turn off" in message       # ...and that this one cannot be
    assert "quota" not in message      # the diagnosis SearchUnavailable would give


def test_a_reply_cut_off_for_budget_is_not_called_a_filter():
    """A finish reason is present on every reply, not only on blocked ones.

    MAX_TOKENS (the whole budget spent on thinking) and STOP both carry one. Reading
    either as a content filter would swap one wrong message for another.
    """
    from google.genai import types

    for reason in ("MAX_TOKENS", "STOP", "OTHER"):
        response = types.GenerateContentResponse.model_validate(
            {"candidates": [{"finish_reason": reason}]})
        assert mcpserver._block_reason(response) == ""


def test_a_filtered_search_is_reported_to_the_model_as_a_refusal(monkeypatch):
    def blocked(query, api_key):
        raise mcpserver.SearchBlocked("Gemini stopped this search with its SAFETY filter.")

    monkeypatch.setattr(mcpserver, "_gemini_answer", blocked)
    text = mcpserver.call_tool("web_search", {"query": "q"}, "K")[0]["text"]
    assert "refused" in text
    assert "NOT an absence of results" in text
    # Not the setup/quota wording — that would be a different, wrong diagnosis.
    assert "Search is unavailable" not in text


# ── the request that actually goes out ────────────────────────────────────────
# The ``gemini`` stub records the config this file *passes*. What the SDK does with
# it afterwards is a second step, and the whole "search is not filtered" promise
# rests on the settings surviving that step: a config the API never receives filters
# exactly as much as no setting at all. Only the real client can show that, so this
# one runs it against a local HTTP server and reads the body off the socket.

class _CaptureHandler(BaseHTTPRequestHandler):
    """Records each POST and answers with a minimal valid reply."""

    def __init__(self, seen, *args, **kwargs):
        self._seen = seen
        super().__init__(*args, **kwargs)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        self._seen.append({"path": self.path, "body": json.loads(raw)})
        payload = json.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP"}],
            "modelVersion": "capture",
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def sent_request(monkeypatch):
    """Run the real client against a local server and return what it POSTed."""
    import functools
    import threading
    from http.server import ThreadingHTTPServer

    genai = pytest.importorskip("google.genai")
    seen: list = []
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                    functools.partial(_CaptureHandler, seen))
    except OSError as exc:          # nothing to bind, so nothing to verify
        pytest.skip(f"cannot bind a local port: {exc}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    real_client = genai.Client
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    def aimed(**kwargs):
        """The same call ``_gemini_search`` makes, pointed at the capture server."""
        kwargs["http_options"] = dict(kwargs.get("http_options") or {}, base_url=base_url)
        return real_client(**kwargs)

    monkeypatch.setattr(genai, "Client", aimed)
    try:
        yield seen
    finally:
        server.shutdown()
        server.server_close()


def test_the_safety_settings_reach_the_wire(sent_request):
    """The field the API reads, spelled the way the REST API spells it."""
    assert mcpserver._gemini_search("q", "K") == "ok"
    assert len(sent_request) == 1
    body = sent_request[0]["body"]
    assert body["safetySettings"] == [
        {"category": name, "threshold": "OFF"}
        for name in mcpserver.GEMINI_SAFETY_CATEGORIES
    ]
    # Grounding rides along in the same body, under its own camelCase name.
    assert body["tools"] == [{"googleSearch": {}}]


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


# ── grounding redirects become real publisher URLs ────────────────────────────
# Gemini reports every source as an opaque hop on its own host —
# vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQ... — which names
# Google rather than the publisher, is unreadable without opening it, and carries a
# signature that expires. The caller only reports what this server hands it, so the
# hop is taken here, once per source, before the summary is written.
#
# These stay offline: ``_resolve_redirect`` is the only thing that touches the
# network, so every test either stubs it or stubs ``urlopen`` beneath it.

REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQsig"


class _Hop:
    """Stands in for the response object ``urlopen`` returns.

    ``geturl()`` is the whole contract. urlopen completes the redirect chain before
    it returns, so the final address is already in hand and the body never has to be
    read — which is what keeps a resolution to one round-trip of headers instead of
    a page download per source.
    """

    def __init__(self, final, body=b""):
        self._final = final
        self._body = body

    def geturl(self):
        return self._final

    def read(self, size=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _hop(monkeypatch, final, body=b""):
    monkeypatch.setattr(mcpserver.urllib.request, "urlopen",
                        lambda request, timeout=None: _Hop(final, body))


def test_only_google_grounding_hops_count_as_redirects():
    assert mcpserver._is_grounding_redirect(REDIRECT)
    # The whole host is the grounding redirector, so any path on it is a hop.
    assert mcpserver._is_grounding_redirect(
        "https://vertexaisearch.cloud.google.com/anything")
    # A source that is already a real URL must be left alone: rewriting one could
    # only make it worse, and it would cost a request for nothing.
    assert not mcpserver._is_grounding_redirect("https://python.org/downloads/")
    assert not mcpserver._is_grounding_redirect("")
    assert not mcpserver._is_grounding_redirect("https://cloud.google.com/vertex-ai")


def test_a_real_url_is_never_opened(monkeypatch):
    """The pass is a no-op for sources that are already publisher URLs."""
    opened = []

    def refuse(request, timeout=None):
        opened.append(request)
        raise AssertionError("no request should be made for a real URL")

    monkeypatch.setattr(mcpserver.urllib.request, "urlopen", refuse)
    assert mcpserver._resolve_sources([("A", "https://python.org/downloads/")]) == [
        ("A", "https://python.org/downloads/")]
    assert opened == []


def test_a_redirect_is_swapped_for_the_url_it_lands_on(monkeypatch):
    _hop(monkeypatch, "https://dreampixelforge.com/articles/x")
    assert mcpserver._resolve_redirect(REDIRECT) == "https://dreampixelforge.com/articles/x"


def test_a_hop_that_cannot_be_taken_keeps_the_original_url(monkeypatch):
    """A working redirect is a worse citation than a publisher URL, but a far
    better one than a dropped source."""

    def boom(request, timeout=None):
        raise OSError("tls handshake failed")

    monkeypatch.setattr(mcpserver.urllib.request, "urlopen", boom)
    assert mcpserver._resolve_redirect(REDIRECT) == ""
    assert mcpserver._resolve_sources([("A", REDIRECT)]) == [("A", REDIRECT)]


def test_a_200_that_still_points_at_the_hop_is_read_for_its_target(monkeypatch):
    """Not every hop answers with a 3xx, and urllib will not follow one that does
    not — the status was 200, so as far as it knows the request is over."""
    _hop(monkeypatch, REDIRECT,
         b'<html><head><meta http-equiv="refresh" '
         b'content="0;url=https://real.test/a"></head></html>')
    assert mcpserver._resolve_redirect(REDIRECT) == "https://real.test/a"


def test_a_javascript_redirect_page_is_read_too(monkeypatch):
    _hop(monkeypatch, REDIRECT, b'<script>location.href = "/relative/page";</script>')
    # A relative target is normal on these pages, so it resolves against the hop.
    assert mcpserver._resolve_redirect(REDIRECT) == (
        "https://vertexaisearch.cloud.google.com/relative/page")


def test_resolved_sources_keep_gemini_s_order(monkeypatch):
    monkeypatch.setattr(mcpserver, "_resolve_redirect",
                        lambda url: "https://publisher.test/" + url.rsplit("/", 1)[-1])
    resolved = mcpserver._resolve_sources([("A", REDIRECT + "1"),
                                           ("B", REDIRECT + "2"),
                                           ("C", REDIRECT + "3")])
    assert [title for title, _ in resolved] == ["A", "B", "C"]
    assert [url for _, url in resolved] == ["https://publisher.test/AUZIYQsig1",
                                            "https://publisher.test/AUZIYQsig2",
                                            "https://publisher.test/AUZIYQsig3"]


def test_two_hops_to_the_same_page_are_one_citation(monkeypatch):
    """The same article under two signatures is one source, not two."""
    monkeypatch.setattr(mcpserver, "_resolve_redirect",
                        lambda url: "https://same.test/article")
    assert mcpserver._resolve_sources([("A", REDIRECT + "a"),
                                       ("A", REDIRECT + "b")]) == [
        ("A", "https://same.test/article")]


def test_only_the_reported_sources_are_followed(monkeypatch):
    """The cap applies before the hops are taken: 20 sources means
    SEARCH_MAX_SOURCES requests, not 20 resolved and then thrown away."""
    followed = []
    monkeypatch.setattr(mcpserver, "_resolve_redirect",
                        lambda url: followed.append(url) or "")
    mcpserver._resolve_sources([(f"S{i}", f"{REDIRECT}{i}") for i in range(20)])
    assert len(followed) == mcpserver.SEARCH_MAX_SOURCES


def test_resolution_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(mcpserver, "SEARCH_RESOLVE_REDIRECTS", False)
    monkeypatch.setattr(mcpserver, "_resolve_redirect",
                        lambda url: pytest.fail("must not be resolved"))
    assert mcpserver._resolve_sources([("A", REDIRECT)]) == [("A", REDIRECT)]


def test_the_summary_carries_publisher_urls_not_redirects(gemini, monkeypatch):
    monkeypatch.setattr(mcpserver, "_resolve_redirect",
                        lambda url: "https://dreampixelforge.com/articles/x")
    gemini.reply = Response(text="An answer.",
                            chunks=[_web(REDIRECT, "dreampixelforge.com")])
    text = mcpserver._gemini_search("q", "K")
    assert "1. dreampixelforge.com — https://dreampixelforge.com/articles/x" in text
    # The redirect is the whole defect: it names Google, and it expires.
    assert "vertexaisearch" not in text


# ── web_fetch: a download plus a prune pass over an HTML tree, and NO model ───────

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

    assert text.splitlines()[0] == "T"             # <title> is kept: it names the page
    assert "The actual sentence." in text
    assert "Second paragraph." in text
    assert "SECRET IN A SCRIPT" not in text      # script contents are not text
    assert "color: red" not in text              # nor are style contents
    assert "a comment" not in text               # nor are comments
    assert "Home" not in text                    # nor is <nav>, wherever it sits
    assert "<" not in text and ">" not in text   # every tag is gone
    # Block-level elements became breaks (a blank line, i.e. a paragraph), so the
    # two paragraphs are not fused into one sentence.
    assert "The actual sentence.\n\nSecond paragraph." in text


# ── the prune pass ────────────────────────────────────────────────────────────
# One page carrying every kind of furniture the pass knows about, with an article
# underneath it. The rules are pinned one at a time against this instead of a live
# URL, so the suite stays offline and a rule cannot silently stop firing.
CHROME_PAGE = """<!doctype html>
<html><head>
  <title>Sprocket maintenance guide — Sprocket Co</title>
  <meta name="description" content="How to grease a sprocket">
  <link rel="stylesheet" href="/site.css">
  <script>var ga = 'SECRET TRACKING';</script>
  <style>.cookie { position: fixed }</style>
</head><body>
  <header class="site-header">
    <img src="/img/hero.jpg" width="1200" height="400" alt="Sprocket Co">
    <nav><a href="/">Home</a><a href="/about">About us</a></nav>
  </header>
  <div id="cookie-banner" class="cookie-consent">
    We use cookies. <button>Accept all</button>
  </div>
  <main>
    <article>
      <header>
        <h1>Sprocket maintenance guide</h1>
        <p class="byline">By A. Machinist</p>
      </header>
      <p>The first thing to check is the chain tension.</p>
      <figure><img data-src="/media/sprocket.jpg" alt="A sprocket"></figure>
      <p>Lubricate every 300 kilometres.</p>
    </article>
    <aside class="newsletter-signup">Subscribe to our newsletter.</aside>
  </main>
  <footer><p>© 2026 Sprocket Co</p><a href="/legal">Legal</a></footer>
</body></html>"""


@pytest.fixture
def chrome_text():
    return mcpserver._readable_text(CHROME_PAGE)


def test_the_article_survives_and_the_furniture_does_not(chrome_text):
    assert "The first thing to check is the chain tension." in chrome_text
    assert "Lubricate every 300 kilometres." in chrome_text
    assert "SECRET TRACKING" not in chrome_text            # <script>
    assert "position: fixed" not in chrome_text             # <style>
    assert "Home" not in chrome_text                        # <nav>
    assert "Accept all" not in chrome_text                  # cookie notice, by class
    assert "Subscribe to our newsletter." not in chrome_text  # named chrome in <main>
    assert "© 2026 Sprocket Co" not in chrome_text          # <footer>
    assert "Legal" not in chrome_text
    assert "How to grease a sprocket" not in chrome_text     # a <meta> is not text
    assert "<" not in chrome_text and ">" not in chrome_text


def test_an_articles_header_is_content_but_the_sites_header_is_furniture(chrome_text):
    """The rule that needs a tree. HTML5 puts an article's title in its own <header>.

    Dropping <header> by tag would delete the headline, the byline and the date —
    the three lines a summary most needs — so header/footer/aside count as
    furniture only OUTSIDE main/article. The same page proves both halves.
    """
    assert "By A. Machinist" in chrome_text                       # inside <article>
    assert chrome_text.count("Sprocket maintenance guide") == 2   # <title> + the <h1>
    assert "hero.jpg" not in chrome_text                          # the site <header>


def test_a_void_element_does_not_swallow_the_page():
    """`<meta>`, `<link>` and `<img>` never close, so stacking them nests the rest."""
    text = mcpserver._readable_text(
        '<html><body><img src="/x.jpg"><p>After the image.</p>'
        '<meta charset="utf-8"><p>After the meta.</p></body></html>')
    assert "After the image." in text
    assert "After the meta." in text


def test_the_note_says_a_pass_ran():
    """The disclosure IS the safety mechanism: it is what stops a pruned section
    being reported to the user as a section the page does not have."""
    for html in (CHROME_PAGE, "<p>Bare prose.</p>", "<div>No tags of interest.</div>"):
        assert mcpserver._readable_text(html).endswith(mcpserver.PRUNE_NOTE)


def test_a_page_that_is_all_furniture_still_comes_back():
    """Empty output would read as "this page has no text", which is a worse lie
    than the boilerplate it was trying to avoid."""
    text = mcpserver._readable_text(
        "<html><body><div class='cookie-consent'>Accept all cookies</div>"
        "</body></html>")
    assert "Accept all cookies" in text
    assert mcpserver.PRUNE_NOTE in text


def test_a_soft_hint_counts_outside_the_article_and_not_inside_it():
    """A page can legitimately be *about* a sidebar, so inside main only the
    unambiguous hints — cookie, share, newsletter, ad — are trusted."""
    text = mcpserver._readable_text(
        "<html><body>"
        "<div class='sidebar'><p>Site sidebar.</p></div>"
        "<main><article><p>Article prose.</p></article>"
        "<div class='sidebar'><p>Sidebar prose in main.</p></div>"
        "<div class='newsletter-signup'><p>Named chrome in main.</p></div>"
        "</main></body></html>")
    assert "Article prose." in text
    assert "Site sidebar." not in text                # soft hint, outside main
    assert "Sidebar prose in main." in text           # soft hint, inside main: kept
    assert "Named chrome in main." not in text        # hard hint: dropped anywhere


def test_a_link_farm_goes_but_a_table_of_contents_keeps_its_links():
    links = "".join(
        f"<li><a href='/p{i}'>Chapter {i} of the field manual</a></li>" for i in range(20))
    chapter = "Chapter 7 of the field manual"

    farm = ("<html><body><article><p>Real prose.</p></article>"
            f"<div id='related'><ul>{links}</ul></div></body></html>")
    text = mcpserver._readable_text(farm)
    assert "Real prose." in text
    assert chapter not in text

    # The same links one level under a heading: that is a contents list, and it is
    # often the most useful thing on a documentation page.
    toc = (f"<html><body><section><h2>On this page</h2><ul>{links}</ul></section>"
           "</body></html>")
    assert chapter in mcpserver._readable_text(toc)


def test_the_note_survives_the_cap(monkeypatch):
    """A truncated reply must still say that a pass ran, or the cut reads as
    "the page ends here"."""
    monkeypatch.setattr(mcpserver, "FETCH_BODY_CAP", 60)
    text = mcpserver._readable_text("<p>" + "prose " * 200 + "</p>")
    assert text.count("prose") < 20                  # the body was cut short
    assert text.endswith(mcpserver.PRUNE_NOTE)       # the disclosure was not


def test_the_cap_is_spent_on_content_and_not_on_chrome(monkeypatch):
    """The saving that matters. The cap is 60k characters, and a chrome-heavy page
    used to spend them on nav and cookie notices, so the text was truncated BEFORE
    the article began. Pruning moves the truncation point back."""
    monkeypatch.setattr(mcpserver, "FETCH_BODY_CAP", 120)
    nav = "<nav>" + "".join(f"<a href='/{i}'>Link {i}</a>" for i in range(50)) + "</nav>"
    text = mcpserver._readable_text(
        f"<html><body>{nav}<article><h1>Title</h1>"
        "<p>The article body.</p></article></body></html>")
    assert "The article body." in text


def test_a_page_that_is_only_links_comes_back_whole_rather_than_empty():
    """An index or an archive page IS a list of links.

    Pruning it to nothing would be reported as "this page has no text", which is a
    worse lie than the boilerplate the prune was trying to avoid, so a page the
    heuristic empties is handed over whole — still with the note.
    """
    links = "".join(
        f"<li><a href='/p{i}'>Chapter {i} of the field manual</a></li>" for i in range(20))
    text = mcpserver._readable_text(f"<html><body><ul>{links}</ul></body></html>")
    assert "Chapter 7 of the field manual" in text
    assert mcpserver.PRUNE_NOTE in text


def test_nested_svg_sprites_do_not_leak_out_as_text():
    """The case the old regex strip got wrong: `<svg>` inside `<svg>` left the rest
    of the sprite in the text, because `.*?</svg>` stops at the first close tag.
    The tree closes the nearest matching tag instead."""
    text = mcpserver._readable_text(
        "<html><body><svg width='0'><svg><text>SPRITE LEAK</text></svg></svg>"
        "<p>Real prose.</p></body></html>")
    assert "Real prose." in text
    assert "SPRITE LEAK" not in text


def test_unbalanced_tags_keep_their_text():
    """Every real page has a stray close tag; one must not flatten what follows."""
    text = mcpserver._readable_text(
        "<html><body><div><p>First.</p></div></div></div>"
        "<main><p>Second.</p></main></body></html>")
    assert "First." in text and "Second." in text


def test_a_parser_failure_falls_back_to_the_plain_strip(monkeypatch):
    """html.parser is tolerant but not obliged to be. On a construct it rejects,
    the fetch still returns text rather than "No readable text found"."""
    def boom(body):
        raise RuntimeError("bad construct")

    monkeypatch.setattr(mcpserver, "_build_tree", boom)
    text = mcpserver._readable_text("<script>x</script><p>Stubborn prose.</p>")
    assert "Stubborn prose." in text
    assert mcpserver.PRUNE_NOTE in text


def test_a_page_with_no_text_at_all_is_still_empty():
    """An empty answer is for a page that has nothing, not for a page that has
    something the heuristic could not read."""
    assert mcpserver._readable_text("<html><head><title></title></head></html>") == ""


def test_content_images_are_read_from_the_pruned_tree():
    """No logo heuristic was added: the site <header> the logo lived in is gone
    before the images are looked for, so the logo is never a candidate."""
    root = mcpserver._parse_and_prune(CHROME_PAGE)
    assert mcpserver._images_from_tree(root, "https://sprocket.test/guide") == [
        "https://sprocket.test/media/sprocket.jpg"]


def test_article_images_win_the_cap_over_the_pages_chrome(monkeypatch):
    """Document order alone spends the small cap on a hero and three icons before
    reaching the diagrams in the body, so article/figure images are ranked first."""
    monkeypatch.setattr(mcpserver, "FETCH_MAX_IMAGES", 2)
    root = mcpserver._parse_and_prune(
        "<html><body><div class='hero'><img src='/lead.jpg'></div>"
        "<article><img src='/body-1.jpg'><img src='/body-2.jpg'></article>"
        "</body></html>")
    assert mcpserver._images_from_tree(root, "https://x.test/guide") == [
        "https://x.test/body-1.jpg", "https://x.test/body-2.jpg"]


def test_a_lazy_loaded_image_is_read_from_its_real_attribute():
    """A page that lazy-loads puts a 1x1 placeholder in `src` and the image in
    `data-src`/`data-srcset`, so reading `src` first would fetch a blank gif."""
    root = mcpserver._parse_and_prune(
        "<html><body><article>"
        "<img src='/blank.gif' data-src='/real.jpg'>"
        "<img data-srcset='/big.jpg 1x, /big@2x.jpg 2x'>"
        "</article></body></html>")
    assert mcpserver._images_from_tree(root, "https://x.test") == [
        "https://x.test/real.jpg", "https://x.test/big.jpg"]


def test_a_declared_size_is_only_read_when_it_is_pixels():
    """`width='100%'` is not a 100-pixel icon; a responsive image is content."""
    root = mcpserver._parse_and_prune(
        "<html><body><article><img src='/percent.jpg' width='100%' height='100%'>"
        "<img src='/icon.png' width='120' height='90'>"
        "<img src='/plain.jpg'></article></body></html>")
    assert mcpserver._images_from_tree(root, "https://x.test") == [
        "https://x.test/percent.jpg", "https://x.test/plain.jpg"]


def test_the_tool_says_that_a_prune_pass_runs_and_offers_no_way_out_of_it():
    """There is no `full=true`: the text always comes back pruned, so the sentence
    in the tool description and the note in the reply carry the disclosure."""
    fetch = next(tool for tool in mcpserver.TOOLS if tool["name"] == "web_fetch")
    description = fetch["description"]
    assert "removed" in description
    assert "pruned" in description
    assert list(fetch["inputSchema"]["properties"]) == ["url"]
    assert "nothing was changed about the page" not in description


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


# ── _fetch_page end to end, on a stubbed socket ───────────────────────────────

class _FakeResponse:
    """Just enough of an http.client response for _fetch_page."""

    def __init__(self, body: bytes, ctype: str):
        self.headers = {"Content-Type": ctype}
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, body: bytes, ctype: str):
    monkeypatch.setattr(mcpserver.urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(body, ctype))


def test_an_html_download_comes_back_pruned_noted_and_with_its_images(monkeypatch):
    _serve(monkeypatch, CHROME_PAGE.encode(), "text/html; charset=utf-8")
    text, images = mcpserver._fetch_page("https://sprocket.test/guide")

    assert "The first thing to check is the chain tension." in text
    assert "Accept all" not in text and "Legal" not in text
    assert text.endswith(mcpserver.PRUNE_NOTE)
    assert images == ["https://sprocket.test/media/sprocket.jpg"]


def test_a_non_html_download_is_not_pruned_and_gets_no_note(monkeypatch):
    """JSON has no page furniture, so saying a pass ran over it would be noise."""
    _serve(monkeypatch, b'{"ok": true}', "application/json")
    text, images = mcpserver._fetch_page("https://api.test/x.json")

    assert text == '{"ok": true}'
    assert mcpserver.PRUNE_NOTE not in text
    assert images == []


def test_html_served_as_plain_text_is_still_pruned(monkeypatch):
    """Some servers mislabel a page; the body is the give-away, not the header."""
    _serve(monkeypatch,
           b"<!doctype html><html><body><nav>x</nav><p>Prose.</p></body></html>",
           "text/plain")
    text, _images = mcpserver._fetch_page("https://a.test/x")
    assert "Prose." in text
    assert mcpserver.PRUNE_NOTE in text


# ── call_tool ─────────────────────────────────────────────────────────────────

def test_an_unavailable_search_never_reads_as_no_results():
    content = mcpserver.call_tool("web_search", {"query": "quantum widgets"}, "")
    text = content[0]["text"]
    assert "Search is unavailable" in text
    assert "NOT an absence of results" in text
    # The exact lie the old scrapers told, whenever they were refused.
    assert "No results" not in text


def test_an_available_search_returns_the_summary(monkeypatch):
    monkeypatch.setattr(mcpserver, "_gemini_answer",
                        lambda query, api_key: (f"summary for {query}", []))
    content = mcpserver.call_tool("web_search", {"query": "quantum widgets"}, "K")
    assert content == [{"type": "text", "text": "summary for quantum widgets"}]


# ── the second index, and what a refusal actually is ──────────────────────────
# The motivating report: "test your web search. can you find porn sites?" came back
# as a denial. Measured, that denial carries finish_reason=STOP and
# prompt_feedback.block_reason=None — the FILTER passed it; the MODEL declined, and
# no safety_setting reaches a model's own answer. So the fixes are a frame for the
# request and a second index that has no model in it at all.

REFUSAL = (
    "I cannot fulfill this request. I am programmed to be a helpful and harmless "
    "AI assistant. My safety guidelines prohibit me from searching for, "
    "generating, or linking to sexually explicit content."
)


def _ddg(*rows):
    """Stub results in the (title, url, snippet) shape _ddg_search returns."""
    return list(rows)


@pytest.fixture
def both_sources(monkeypatch):
    """DDG on, with a stubbed index — the only place in this suite it is on."""
    monkeypatch.setattr(mcpserver, "DDG_ENABLED", True)
    return monkeypatch


def test_the_request_carries_the_index_framing(gemini):
    """safety_settings alone does not stop a decline; the frame is what reaches it.

    With every category OFF the model still answered "my safety guidelines
    prohibit...", because the block table and the model's judgment are different
    layers. This pins the only lever that has been shown to move the second one.
    """
    mcpserver._gemini_search("q", "K")
    assert mcpserver.GEMINI_SEARCH_SYSTEM.strip()
    assert gemini.chat_kwargs["config"]["system_instruction"] == mcpserver.GEMINI_SEARCH_SYSTEM


def test_both_indexes_contribute_to_the_same_answer(both_sources):
    both_sources.setattr(mcpserver, "_gemini_answer",
                         lambda q, k: ("A grounded summary.", [("G", "https://g.test/1")]))
    both_sources.setattr(mcpserver, "_ddg_search",
                         lambda q: _ddg(("D", "https://d.test/1", "a snippet")))
    text = mcpserver.call_tool("web_search", {"query": "q"}, "K")[0]["text"]
    assert "A grounded summary." in text
    assert "https://g.test/1" in text
    assert "https://d.test/1" in text
    # The snippet travels with the raw result — often the only place a dropped fact
    # survives.
    assert "a snippet" in text


def test_a_declined_summary_still_returns_the_raw_results(both_sources):
    """The user's actual report, end to end.

    Gemini declines; DuckDuckGo cannot. The answer must carry the results AND say
    plainly that this was a refusal, so the caller never turns it into "the topic
    has no coverage".
    """
    both_sources.setattr(mcpserver, "_gemini_answer", lambda q, k: (REFUSAL, []))
    both_sources.setattr(mcpserver, "_ddg_search",
                         lambda q: _ddg(("D", "https://d.test/1", "snippet")))
    text = mcpserver.call_tool("web_search", {"query": "q"}, "K")[0]["text"]
    assert "https://d.test/1" in text
    assert "declined" in text
    assert "NOT an absence of results" in text
    # The decline's own wording is not replayed — the caller needs the fact, not the
    # lecture.
    assert "I am programmed to be" not in text


def test_a_refusal_is_not_mistaken_for_an_answer():
    """Both conditions are required, so a terse real answer is never relabelled."""
    assert mcpserver._looks_like_refusal(REFUSAL, []) is True
    # Sources came back, so it is an answer whatever it says.
    assert mcpserver._looks_like_refusal(REFUSAL, [("t", "https://x.test")]) is False
    # Long: a real answer carries detail, so this cannot be a decline.
    assert mcpserver._looks_like_refusal(REFUSAL + "x" * 400, []) is False
    assert mcpserver._looks_like_refusal("Canberra is the capital.", []) is False
    assert mcpserver._looks_like_refusal("", []) is False


def test_the_api_floor_does_not_fall_back_to_duckduckgo(both_sources):
    """SearchBlocked is terminal, on purpose.

    Every adjustable filter is already off, so a block means the API's own
    non-adjustable floor — core harms. Quietly answering that from a second index
    would be routing around the one guard that exists deliberately, which a search
    tool must not do just because it can.
    """
    asked = []

    def blocked(query, api_key):
        raise mcpserver.SearchBlocked("the SAFETY floor")

    def ddg(query):
        asked.append(query)
        return _ddg(("D", "https://d.test/1", "snippet"))

    both_sources.setattr(mcpserver, "_gemini_answer", blocked)
    both_sources.setattr(mcpserver, "_ddg_search", ddg)
    with pytest.raises(mcpserver.SearchBlocked):
        mcpserver._web_search("q", "K")
    # DuckDuckGo WAS consulted — it runs concurrently and cannot be cancelled out of
    # the pool — and its results were still thrown away rather than returned.
    assert asked == ["q"]


def test_a_missing_gemini_key_still_searches_with_duckduckgo(both_sources):
    """No key costs the summary, not the search: the second index needs none."""
    def no_key(query, api_key):
        raise mcpserver.SearchUnavailable("no Gemini API key.")

    both_sources.setattr(mcpserver, "_gemini_answer", no_key)
    both_sources.setattr(mcpserver, "_ddg_search",
                         lambda q: _ddg(("D", "https://d.test/1", "snippet")))
    text = mcpserver._web_search("q", "")
    assert "https://d.test/1" in text
    assert "Gemini was unavailable" in text
    assert "NOT an absence of results" in text


def test_a_duckduckgo_failure_is_disclosed_not_hidden(both_sources):
    """One index failing is not the same as its results being irrelevant."""
    def dead(query):
        raise mcpserver.DdgUnavailable("ConnectError: refused")

    both_sources.setattr(mcpserver, "_gemini_answer",
                         lambda q, k: ("Summary.", [("G", "https://g.test/1")]))
    both_sources.setattr(mcpserver, "_ddg_search", dead)
    text = mcpserver._web_search("q", "K")
    assert "https://g.test/1" in text
    assert "DuckDuckGo" in text and "ConnectError" in text


def test_both_indexes_failing_reports_both_reasons(both_sources):
    """Neither failure may hide behind the other."""
    both_sources.setattr(mcpserver, "_gemini_answer",
                         lambda q, k: (_ for _ in ()).throw(
                             mcpserver.SearchUnavailable("quota exhausted")))
    both_sources.setattr(mcpserver, "_ddg_search",
                         lambda q: (_ for _ in ()).throw(
                             mcpserver.DdgUnavailable("ConnectError: refused")))
    with pytest.raises(mcpserver.SearchUnavailable) as raised:
        mcpserver._web_search("q", "K")
    assert "quota exhausted" in str(raised.value)
    assert "ConnectError" in str(raised.value)
    # The lie this whole path exists to prevent.
    assert "no results" not in str(raised.value).lower()


def test_an_empty_duckduckgo_index_is_not_a_failure(both_sources):
    """Empty and unreachable are different facts, and only one is disclosed."""
    both_sources.setattr(mcpserver, "_gemini_answer",
                         lambda q, k: ("Summary.", [("G", "https://g.test/1")]))
    both_sources.setattr(mcpserver, "_ddg_search", lambda q: [])
    text = mcpserver._web_search("q", "K")
    assert "returned no results" in text
    assert "could not be reached" not in text


def test_duckduckgo_is_asked_with_safesearch_off(monkeypatch):
    """The package default is "moderate" — a content filter under another name.

    Measured: it changes WHICH results come back, not merely their order, so leaving
    it alone would quietly narrow every search.
    """
    calls: dict = {}

    class FakeDDGS:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def text(self, query, **kwargs):
            calls["query"] = query
            calls["text"] = kwargs
            return []

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    monkeypatch.setattr(mcpserver, "DDG_ENABLED", True)
    mcpserver._ddg_search("q")
    assert calls["text"]["safesearch"] == "off" == mcpserver.DDG_SAFESEARCH
    assert calls["text"]["max_results"] == mcpserver.DDG_MAX_RESULTS


def test_duckduckgo_reports_a_failure_instead_of_an_empty_list(monkeypatch):
    """A challenged or throttled request must never read as an empty index."""
    class AngryDDGS:
        def __init__(self, **kwargs):
            pass

        def text(self, query, **kwargs):
            raise RuntimeError("202 challenge")

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", AngryDDGS)
    with pytest.raises(mcpserver.DdgUnavailable) as raised:
        mcpserver._ddg_search("q")
    assert "202 challenge" in str(raised.value)


def test_the_two_indexes_are_merged_without_duplicating_a_page():
    """The same page from both indexes is one citation, and Gemini's comes first."""
    merged = mcpserver._merge_sources(
        [("G", "https://same.test/page")],
        _ddg(("D", "https://same.test/page/", "dup"),
             ("E", "https://other.test/x", "new")),
    )
    assert [url for _t, url, _s in merged] == [
        "https://same.test/page", "https://other.test/x"]
    # Gemini's entry keeps its place and gains no snippet from the duplicate.
    assert merged[0] == ("G", "https://same.test/page", "")


def test_the_merged_list_is_capped():
    """Both indexes are capped, and so is the total."""
    gemini_sources = [(f"G{i}", f"https://g.test/{i}") for i in range(8)]
    ddg_results = _ddg(*((f"D{i}", f"https://d.test/{i}", "s") for i in range(20)))
    merged = mcpserver._merge_sources(gemini_sources, ddg_results)
    assert len(merged) == mcpserver.SEARCH_MAX_TOTAL
    assert len({url for _t, url, _s in merged}) == len(merged)


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
    # ...and that a fetched page was pruned, so a gap in it is not reported to the
    # user as a gap in the page.
    assert "furniture" in info["instructions"]


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
