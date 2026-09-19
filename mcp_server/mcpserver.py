import os
import re
import base64
import asyncio
import subprocess
import html as html_lib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SERVER_PORT      = 8572
HTTP_UA          = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# ── web_search: one Gemini call, grounded in Google Search ────────────────────
# The model is named here rather than left to the caller because gemini-3.6-flash
# is the one whose Google-Search grounding is free (500 requests/day); a model
# outside that tier would quietly bill the key's project instead.
#
# The API key arrives per request from the MCP client, which stores it on this
# server's entry in mcp.json — see _api_key(). GEMINI_API_KEY / GOOGLE_API_KEY in
# the environment is the fallback, for running this file by hand.
GEMINI_MODEL       = os.environ.get("GEMINI_SEARCH_MODEL", "").strip() or "gemini-3.6-flash"
GEMINI_TIMEOUT     = 90      # seconds for the whole grounded call
SEARCH_MAX_SOURCES = 8       # source URLs echoed back with the answer

# Request headers accepted as the API key, most specific first. Recognising
# several spellings means the generic "Headers" box in the settings panel — not
# only the dedicated API-key field — can carry it.
API_KEY_HEADERS = ("x-api-key", "x-gemini-api-key", "x-goog-api-key")
API_KEY_ENV     = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# web_fetch tuning.
#
# There is deliberately NO model in this path. Fetching a page is a download and a
# tag strip, so it finishes in the time the network takes and can never outlive the
# MCP client's per-call budget.
#
# It used to hand the page to local llama.cpp — a 4-way parallel map over chunks,
# then a reduce — to strip boilerplate, which took ~135 s on a large page. The MCP
# client's per-call timeout is 120 s, and a call that outlives that budget dooms the
# session it was sent on (the abandoned POST's late response lands on the shared
# connection and the transport reads it as end-of-stream), so every fetch after the
# first big page failed with "Connection closed" — including a bare example.com.
# The caller is a large-context model: give it the page and let it do the reading.
FETCH_TIMEOUT      = 30      # socket timeout for the initial page download
FETCH_BODY_CAP     = 60_000  # max chars of cleaned page text returned
OUTPUT_CHAR_CAP    = 80_000  # cap on a web_search summary (a page is capped at FETCH_BODY_CAP)

# media extraction tuning.
# Images are staged on disk with FIXED names (image_1.jpg, image_2.jpg, ...) and
# overwritten on every call, so web_media/ never accumulates past pages' media.
WEB_MEDIA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_media")
FETCH_MAX_IMAGES   = 5       # cap on content images returned per page
IMG_WORKERS        = 4       # parallel image downloads
MIN_IMG_DIM        = 150     # skip <img> with width/height attrs below this (icons/pixels)
MIN_IMG_BYTES      = 2048    # skip downloads smaller than this (trackers/spacers)
IMG_LONGEST_EDGE   = 800     # ffmpeg downscale cap — keeps MCP image blocks small
IMG_BAD_HINTS      = ("logo", "icon", "sprite", "avatar", "banner", "advert",
                      "doubleclick", "tracking", "pixel", "analytics", "badge",
                      "button", "spinner", "placeholder", "emoji", "favicon")


# ─────────────────────────────────────────────
# Tool definition (description well under 250 tokens)
# ─────────────────────────────────────────────
TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the internet for any topic. Returns a written summary followed by "
            "the source URLs it was grounded on — cite them, and open one with "
            "web_fetch when the summary is not enough. "
            "Use it for current events, news, API updates, new developments, or anything "
            "outside your training data. "
            "Pass a specific, descriptive natural-language query with names, dates, and key words. "
            "If this tool reports that search is unavailable, that is a missing key or "
            "an exhausted quota, NOT a lack of results — do not tell the user the topic "
            "has no coverage; ask them for a URL and open it with web_fetch instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Open a URL directly and return the page as readable plain text. "
            "Use this whenever the user gives you a URL, or on any address you already "
            "know — no search required. Accepts a bare domain like example.com. "
            "Content images on the page are returned alongside the text, so you can "
            "see them. "
            "You get the whole page, with its HTML tags removed — so it can still "
            "carry navigation, cookie notices and other page furniture. Read past "
            "what you don't need and summarise it yourself; there is no "
            "pre-summarising step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch (http/https)."},
            },
            "required": ["url"],
        },
    },
]


# ─────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────
def _strip_html(s: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _truncate(text: str, char_cap: int) -> str:
    if len(text) <= char_cap:
        return text
    cut = text[:char_cap]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + " …"


# ─────────────────────────────────────────────
# Web search — one grounded Gemini call
# ─────────────────────────────────────────────
# Gemini runs the search, reads the pages, and writes the answer; its grounding
# metadata carries the source URLs, which come back alongside the summary so the
# caller can cite them — or open one with web_fetch.
#
# Nothing here scrapes a search engine, and that is the whole point: every keyless
# engine now sits behind a bot filter that answers with a 202 challenge page or a
# 429 rather than an error. None of those raise, so a throttled search was
# indistinguishable from an empty one and got reported to the user as "no results"
# — telling them a topic does not exist when the truth was "we were refused".
# One keyed API answers every time, and says why when it cannot.
class SearchUnavailable(RuntimeError):
    """Search could not run — no key, a rejected key, or no quota left.

    Deliberately not the same thing as "no results": the caller must not tell the
    user a topic has no coverage when the truth is that we never got to ask.
    """


def _api_key(request: Request | None = None) -> str:
    """The Gemini key for this call: the client's header first, then the environment.

    The key belongs to the MCP server *entry* in the client's config, so it arrives
    with every request rather than being baked into this process. Reading it per
    call is also what lets a key edited in the settings panel take effect on the
    very next search, with nothing to restart.
    """
    if request is not None:
        for name in API_KEY_HEADERS:
            value = (request.headers.get(name) or "").strip()
            if value:
                return value
        # `Authorization: Bearer <key>` is the other spelling a client can send
        # without knowing about our header names.
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()

    for name in API_KEY_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _grounding_sources(response) -> list[tuple[str, str]]:
    """[(title, url), ...] from a grounded response, de-duplicated, order kept.

    A grounding chunk can describe something that is not a web page (a map place,
    for instance), so only chunks carrying a web URI become sources.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    metadata = getattr(candidates[0], "grounding_metadata", None)

    sources: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in (getattr(metadata, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        url = str(getattr(web, "uri", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append((str(getattr(web, "title", "") or "").strip(), url))
    return sources


def _key_hint(exc: Exception) -> str:
    """A plain-English reason for a Gemini error, or "" when it is not clear-cut."""
    low = str(exc).lower()
    if "api key not valid" in low or "api_key_invalid" in low or "permission denied" in low:
        return "the API key was rejected"
    if "quota" in low or "rate limit" in low or "resource_exhausted" in low:
        return "the free-tier daily quota is used up"
    return ""


def _gemini_search(query: str, api_key: str) -> str:
    """Search once, answer once. Returns the summary AND its source URLs, or raises.

    The URLs are part of the contract rather than a nicety: they are what lets the
    model attribute a claim, and they are the input to web_fetch when the summary
    is not enough. Grounding metadata is the only reason to prefer this over any
    other search API — it is what makes the answer checkable.
    """
    try:
        from google import genai
        from google.genai import errors as genai_errors
    except ImportError as exc:
        raise SearchUnavailable(
            f"the google-genai package is not installed for this server ({exc}). "
            "Run: pip install google-genai"
        ) from exc

    if not api_key:
        raise SearchUnavailable(
            "no Gemini API key. Open the settings panel, press Edit on this MCP "
            "server and paste one into the API key field (free from "
            "https://aistudio.google.com/apikey) — or set GEMINI_API_KEY in the "
            "environment this server runs in."
            "You will also need pre-pay billing enabled, add $5."
             " at https://ai.google.dev/gemini-api/docs/billing#prepay. ',"
        )

    try:
        client = genai.Client(
            api_key=api_key,
            # genai measures this in milliseconds.
            http_options={"timeout": int(GEMINI_TIMEOUT * 1000)},
        )
        # A one-shot Chat, NOT models.generate_content. Handing `tools` to
        # generate_content routes the call through the SDK's automatic function
        # calling, so every single search logged "Direct use of automatic function
        # calling (AFC) in Models.generate_content is not recommended. Instead, we
        # recommend to use AFC in Chat.send_message." AFC is meaningless here
        # anyway: google_search runs inside Gemini, and no Python callable is ever
        # passed for the SDK to call back.
        #
        # send_message is the supported path and it is genuinely simpler than it
        # looks — it notices google_search is AFC-incompatible, sets
        # automatic_function_calling.disable=True itself, and makes the same
        # underlying call. Verified offline against genai 2.24.0 by stubbing
        # Models._generate_content: the Chat path logs nothing, the direct call
        # logs the warning, and the response object is identical either way
        # (`.text` and `.candidates[0].grounding_metadata` both still work, so
        # _grounding_sources below is unchanged).
        #
        # The chat is built and dropped per search, so its history is always empty
        # and no turn can ever leak into the next query.
        chat = client.chats.create(
            model=GEMINI_MODEL,
            config={"tools": [{"google_search": {}}]},
        )
        response = chat.send_message(query)
    except genai_errors.APIError as exc:
        hint = _key_hint(exc)
        raise SearchUnavailable(
            f"Gemini refused the search (HTTP {getattr(exc, 'code', '?')}"
            + (f" — {hint}" if hint else "")
            + f"): {exc}"
        ) from exc
    except Exception as exc:
        raise SearchUnavailable(f"could not reach Gemini: {type(exc).__name__}: {exc}") from exc

    try:
        answer = (response.text or "").strip()
    except Exception:
        # `.text` raises rather than returning None when the reply was cut off
        # mid-part; the sources are still worth reporting without it.
        answer = ""
    sources = _grounding_sources(response)

    if not answer and not sources:
        raise SearchUnavailable(
            "Gemini answered with neither text nor sources — the request was "
            "probably refused before it ran."
        )

    lines = [answer or "(Gemini searched but returned no written answer.)"]
    if sources:
        lines += ["", "Sources:"]
        for index, (title, url) in enumerate(sources[:SEARCH_MAX_SOURCES], 1):
            lines.append(f"{index}. {title} — {url}" if title else f"{index}. {url}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# URL fetch + tag strip
# ─────────────────────────────────────────────
def _readable_text(body: str) -> str:
    """Visible text of an HTML document, with the markup removed.

    Non-content elements (script, style, iframe, form, svg) are dropped whole so
    their contents never appear as text, block-level tags become line breaks so
    paragraphs do not fuse into one another, and every remaining tag is stripped.

    This is the entirety of web_fetch's cleanup. It is mechanical on purpose: an
    LLM pass here cost ~135 s on a large page and blew the MCP client's 120 s
    per-call budget, which is what made every fetch fail after the first big page.
    """
    body = re.sub(r"<(script|style|noscript|iframe|svg|form)[^>]*>.*?</\1>",
                  " ", body, flags=re.S | re.I)
    body = re.sub(r"<(p|div|br|li|tr|h[1-6]|section|article)[^>]*>",
                  "\n", body, flags=re.I)
    text = _strip_html(body)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _fetch_page(url: str) -> tuple[str, list[str]]:
    """Download a URL → (cleaned visible text, content-image URLs).

    Image URLs are harvested from the raw HTML BEFORE tag stripping, with
    logos/icons/ads/tracking pixels filtered out."""
    print("="*20)
    print(f"[_fetch_page]: debug url: {url[:120]}")
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        ctype = resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8", "replace")

    if "html" in ctype or body.lstrip().lower().startswith(("<", "<!doctype")):
        img_urls = _extract_img_urls(body, url)
        text = _readable_text(body)
    else:
        img_urls = []
        text = body  # plain text / json / etc.

    return _truncate(text, FETCH_BODY_CAP), img_urls


# ─────────────────────────────────────────────
# Media extraction — content images as MCP image blocks
# ─────────────────────────────────────────────
def _extract_img_urls(body: str, base_url: str) -> list[str]:
    """Pull main-content image URLs out of raw HTML.

    Skips anything that smells like chrome/boilerplate: logos, icons, sprites,
    avatars, ad/tracker hosts, SVGs, data URIs, and tiny declared dimensions
    (1x1 pixels, spacer gifs, thumbnails)."""
    urls: list[str] = []
    for tag in re.findall(r"<img\b[^>]*>", body, re.I):
        # data-src first (lazy-loaded content images), then plain src
        m = (re.search(r'data-src\s*=\s*["\']([^"\']+)["\']', tag, re.I)
             or re.search(r'\bsrc\s*=\s*["\']([^"\']+)["\']', tag, re.I))
        if not m:
            continue
        src = html_lib.unescape(m.group(1)).strip()
        low = src.lower()
        if not src or src.startswith("data:") or low.split("?")[0].endswith(".svg"):
            continue
        if any(h in low for h in IMG_BAD_HINTS):
            continue
        # declared width/height below MIN_IMG_DIM → icon / tracking pixel
        dims = re.findall(r'\b(?:width|height)\s*=\s*["\']?(\d+)', tag, re.I)
        dims = [int(d) for d in dims]
        if dims and min(dims) < MIN_IMG_DIM:
            continue
        abs_url = urllib.parse.urljoin(base_url, src)
        # percent-encode non-ASCII (e.g. CJK filenames) — urllib's request
        # machinery requires an ASCII-safe URL string
        abs_url = urllib.parse.quote(abs_url, safe=":/?#[]@!$&'()*+,;=%")
        if abs_url not in urls:
            urls.append(abs_url)
        if len(urls) >= FETCH_MAX_IMAGES:
            break
    return urls


def _download_images(urls: list[str]) -> list[dict]:
    """Download images into WEB_MEDIA_DIR as image_N.jpg (fixed names, overwritten
    every call), downscaled via ffmpeg, and return MCP image content blocks.

    Blocks look like: {"type": "image", "data": <base64 jpeg>, "mimeType": ...}
    so the calling LLM receives the media inline alongside the text."""
    os.makedirs(WEB_MEDIA_DIR, exist_ok=True)
    # clear any leftover image_* files from the previous page first
    for old in os.listdir(WEB_MEDIA_DIR):
        if old.startswith("image_"):
            try:
                os.remove(os.path.join(WEB_MEDIA_DIR, old))
            except OSError:
                pass

    # longest edge -> IMG_LONGEST_EDGE, aspect kept, never upscales
    scale = (
        f"scale='if(gt(iw,ih),min({IMG_LONGEST_EDGE},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({IMG_LONGEST_EDGE},ih))'"
    )

    def _grab(item: tuple[int, str]):
        i, url = item
        dest = os.path.join(WEB_MEDIA_DIR, f"image_{i}.jpg")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                raw = resp.read()
            if len(raw) < MIN_IMG_BYTES:
                return None  # near-certainly a tracker/spacer, not content
            # feed bytes via stdin so webp/gif/avif all normalize to jpeg
            res = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", "pipe:0", "-vf", scale, "-frames:v", "1", "-q:v", "4", dest],
                input=raw, capture_output=True)
            if res.returncode != 0 or not os.path.exists(dest):
                return None
            with open(dest, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return {"type": "image", "data": b64, "mimeType": "image/jpeg"}
        except Exception as e:
            print(f"[media] skip {url[:80]}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=IMG_WORKERS) as pool:
        blocks = [b for b in pool.map(_grab, enumerate(urls, 1)) if b]
    print(f"[media] extracted {len(blocks)}/{len(urls)} images -> {WEB_MEDIA_DIR}")
    return blocks


# ─────────────────────────────────────────────
# Tool logic
# ─────────────────────────────────────────────
def call_tool(name: str, arguments: dict, api_key: str = "") -> list:
    print(f"[tool call] name={name} arguments={arguments}")

    if name == "web_search":
        query = (arguments.get("query") or "").strip()
        if not query:
            return [{"type": "text", "text": "No query provided."}]

        try:
            summary = _gemini_search(query, api_key)
        except SearchUnavailable as e:
            # Say what actually happened. "No results" here would be a lie, and the
            # model would pass that lie on as "this topic does not exist".
            print(f"[search] unavailable: {e}")
            return [{"type": "text", "text": (
                f"Search is unavailable — {e} "
                "This is a setup or rate-limit problem, NOT an absence of results: "
                "do not tell the user the topic has no coverage. Either answer from "
                "what you already know, ask them for a specific URL and open it with "
                "web_fetch, or try again later."
            )}]

        return [{"type": "text", "text": _truncate(summary, OUTPUT_CHAR_CAP)}]

    if name == "web_fetch":
        url = (arguments.get("url") or "").strip()
        if not url:
            return [{"type": "text", "text": "No url provided."}]
        # The model usually sends a bare domain ("open example.com"). Assume
        # https rather than rejecting it — that rejection was pure friction for
        # exactly the "open this page" request this tool exists to serve.
        # Only add a scheme when there is none, so ftp:// still fails the check.
        if not re.match(r"^[a-z][a-z0-9+.\-]*://", url, re.I):
            url = "https://" + url.lstrip("/")
        if not url.lower().startswith(("http://", "https://")):
            return [{"type": "text", "text": f"Invalid URL (must be http/https): {url}"}]

        try:
            text, img_urls = _fetch_page(url)
        except Exception as e:
            return [{"type": "text", "text": f"Fetch failed: {e}"}]

        if not text:
            return [{"type": "text", "text": f"No readable text found at: {url}"}]

        # The page goes back as-is (already tag-stripped and capped by _fetch_page).
        # The caller is an LLM with a large context window, so reading and
        # summarising is its job — this tool only fetches.
        blocks = [{"type": "text", "text": text}]
        # attach content images as MCP image blocks so the calling LLM gets the
        # page's media alongside its text.
        if img_urls:
            blocks += _download_images(img_urls)
        return blocks

    return [{"type": "text", "text": f"Unknown tool: {name}"}]


# ─────────────────────────────────────────────
# Tool call
# ─────────────────────────────────────────────
async def _tool_call(req_id, name: str, arguments: dict, api_key: str) -> JSONResponse:
    """Run the (blocking) tool in a worker thread, and answer with plain JSON.

    Both tools do blocking network I/O, so they must stay off the event loop —— but
    the response body is then exactly the JSON-RPC payload, with nothing in front
    of it.

    It used to stream a "." heartbeat ahead of the payload while a slow scrape ran,
    which made the body `....{"jsonrpc": ...}` — not JSON. The official MCP client
    parses that body whole (``validate_json`` on the full response), so any call that
    outlived the heartbeat failed to parse: the tool looked like it never answered.
    A single grounded Gemini call answers in seconds, so the keep-alive bought
    nothing and cost correctness.
    """
    loop = asyncio.get_running_loop()
    try:
        content = await loop.run_in_executor(None, call_tool, name, arguments, api_key)
    except Exception as exc:
        content = [{"type": "text", "text": f"Tool error: {type(exc).__name__}: {exc}"}]

    return JSONResponse({
        "jsonrpc": "2.0", "id": req_id,
        "result": {"content": content, "isError": False},
    })


# ─────────────────────────────────────────────
# JSON-RPC handler
# ─────────────────────────────────────────────
async def handle_jsonrpc(request: Request) -> Response:
    if request.method in ("GET", "OPTIONS"):
        return Response(status_code=204)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
            status_code=400,
        )

    params = body.get("params", {})
    req_id = body.get("id")
    method = body.get("method")

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "serverInfo": {
                    "name": "websearch-mcp",
                    "version": "2.0.0",
                    "instructions": (
                        "Two tools. web_search(query) searches the web and returns a "
                        "written summary followed by the source URLs it was grounded "
                        "on — quote them, and open one with web_fetch when the "
                        "summary is not enough. web_fetch(url) opens a single page "
                        "directly and returns the whole page as readable plain text "
                        "plus its content images; use it whenever the user gives you "
                        "a URL or you already know the address, and it needs no "
                        "search. Its text is the page with the HTML tags removed, so "
                        "it can still include navigation and other page furniture — "
                        "read past that, and do the summarising yourself. If "
                        "web_search reports "
                        "that it is unavailable, that is a missing key or an exhausted "
                        "quota rather than a lack of results: do not tell the user the "
                        "topic has no coverage."
                    ),
                },
                "capabilities": {"tools": {}},
            },
        })

    if method == "notifications/initialized":
        return Response(status_code=204)

    if method == "tools/list":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        # The key travels with the request, so a key changed in the client's settings
        # panel is in force on the very next call.
        return await _tool_call(req_id, name, arguments, _api_key(request))

    return JSONResponse(
        {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}},
        status_code=404,
    )


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = Starlette(routes=[
    Route("/",    endpoint=handle_jsonrpc, methods=["POST", "GET", "OPTIONS"]),
    Route("/mcp", endpoint=handle_jsonrpc, methods=["POST", "GET", "OPTIONS"]),
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# First run: seed mcp.json
# ─────────────────────────────────────────────
# This server never reads mcp.json — the deepseekUI app does, and passes the API key
# in as a request header (see _api_key above). A copy is created here anyway, on the
# way up, because on a fresh clone this file is often the first thing a user starts,
# and the file it writes is what the app needs in order to know this server exists.
#
# Deliberately not imported from server/settings.py::ensure_mcp_file, for the same
# reason the module stands alone: it has to run when the app is broken or absent.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP_CONFIG_NAME = "mcp.json"
MCP_EXAMPLE_NAME = "mcp.example.json"


def ensure_mcp_config(root: str = "") -> str:
    """Copy ``mcp.example.json`` to ``mcp.json`` if the latter is missing.

    Returns the path it created, or ``""`` when there was nothing to do. Never
    raises: a read-only checkout must still be able to serve requests, it just has
    nowhere to keep a server list.
    """
    root = root or _PROJECT_ROOT
    config = os.path.join(root, MCP_CONFIG_NAME)
    # `exists`, not `isfile`: a directory in the way has to stop this too, since
    # writing over it would raise and there is nothing sensible to create around it.
    if os.path.exists(config):
        return ""
    example = os.path.join(root, MCP_EXAMPLE_NAME)
    if not os.path.isfile(example):
        return ""
    try:
        # Written to a temp name and renamed, so a crash mid-copy cannot leave a
        # truncated config behind for the app to parse on its next start.
        with open(example, "rb") as src, open(config + ".tmp", "wb") as dst:
            dst.write(src.read())
        os.replace(config + ".tmp", config)
    except OSError as exc:
        print(f"could not create {config} from {example} ({exc})")
        return ""
    print(f"Created {config} from {example}")
    return config


if __name__ == "__main__":
    ensure_mcp_config()
    print(f"Web search MCP server running on http://localhost:{SERVER_PORT}")
    print(f"Search              : {GEMINI_MODEL} + Google Search grounding "
          f"(timeout {GEMINI_TIMEOUT}s)")
    if not _api_key():
        print("                      NOTE: no API key here. web_search needs one, sent "
              "by the\n                      client as an x-api-key header on each "
              "request (set it via\n                      the settings panel); "
              "GEMINI_API_KEY in this process's\n                      environment also "
              "works. web_fetch is unaffected.")
    print("Fetch               : no key, no model — page text + content images")
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)