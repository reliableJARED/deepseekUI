import os
import re
import json
import math
import time
import base64
import asyncio
import subprocess
import html as html_lib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ai_call.py provides an OpenAI-compatible client for the local llama.cpp
# server. If it isn't importable we fall back to the raw urllib helper below.
try:
    from ai_call import AICall
    _HAS_AI_CALL = True
except Exception as _e:
    print(f"[init] ai_call.py not importable, using raw urllib fallback: {_e}")
    AICall = None
    _HAS_AI_CALL = False

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SERVER_PORT      = 8572
DDG_HTML_URL     = "https://html.duckduckgo.com/html/"
DDG_LITE_URL     = "https://lite.duckduckgo.com/lite/"
LLAMA_URL        = "http://localhost:8080/v1/chat/completions"
HTTP_UA          = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# Optional keyed search APIs. Scraping a consumer search page is now a losing
# game — every engine gates bots (see _search) — so these are tried FIRST when
# present, and the scraped engines become the fallback. Either one gives search
# that answers every time:
#   BRAVE_API_KEY  https://api-dashboard.search.brave.com  (free tier)
#   SEARXNG_URL    a SearXNG instance with the JSON API enabled, e.g.
#                  https://searx.example.org
BRAVE_API_KEY    = os.environ.get("BRAVE_API_KEY", "").strip()
SEARXNG_URL      = os.environ.get("SEARXNG_URL", "").strip()

MAX_RESULTS      = 6      # raw hits pulled
SNIPPET_CAP      = 500    # max chars kept per snippet (keeps a single call small)
TITLE_CAP        = 150    # max chars kept per title

# llama.cpp has a tiny context window — budget everything against it.
LLAMA_CTX_TOKENS   = 3072
GEN_RESERVE        = 256          # tokens reserved for the model's own output
PROMPT_OVERHEAD    = 256          # rough cost of system prompt + framing
CHARS_PER_TOKEN    = 2.5          # conservative (dense snippets tokenize small)
INPUT_TOKEN_BUDGET = LLAMA_CTX_TOKENS - GEN_RESERVE - PROMPT_OVERHEAD   # ~2560
MAX_CHUNK_CHARS    = int(INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN)          # ~6400

PARTIAL_TOKENS   = 6000    # cap on each per-chunk (map) summary — reasoning
                          # models burn tokens on reasoning_content before
                          # answering; tiny caps make every call return empty
FINAL_TOKENS     = 20000    # cap on the final answer (same reasoning overhead)
OUTPUT_CHAR_CAP  = int(FINAL_TOKENS * 4)   # defensive final truncation

# Heartbeat / timeout tuning.
# We do NOT stream the model output. Instead, while the (blocking) search+summary
# runs in a worker thread, the HTTP response to the MCP client dribbles a single
# "." every HEARTBEAT_INTERVAL seconds to keep the connection warm. When the work
# finishes, the final JSON-RPC payload is appended after the dots.
#
# Because the llama call is non-streaming, it sends no bytes until generation is
# fully done, so its OWN socket timeout must be generous — otherwise it dies long
# before the heartbeats matter. Set LLAMA_TIMEOUT to None for "wait forever".
HEARTBEAT_INTERVAL = 30      # seconds between "." heartbeats sent to the client
LLAMA_TIMEOUT      = 900     # per-call socket timeout for the llama.cpp request

# web_fetch tuning.
FETCH_TIMEOUT      = 30      # socket timeout for the initial page download
FETCH_BODY_CAP     = 60_000  # max chars of cleaned page text kept before chunking
EXTRACT_WORKERS    = 4       # parallel LLM calls during the map phase
EXTRACT_TOKENS     = 7000     # cap on each per-chunk (map) extraction — reasoning
                             # models (heretic/ablit/qwen3) burn tokens on
                             # reasoning_content before answering, so this must
                             # be well above PARTIAL_TOKENS or every chunk returns empty

# media extraction tuning.
# Images are staged on disk with FIXED names (image_1.jpg, image_2.jpg, ...) and
# overwritten on every call, so web_media/ never accumulates past pages' media.
WEB_MEDIA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_media")
FETCH_MAX_IMAGES   = 5       # cap on content images returned per page
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
            "Search the internet for any topic and return a concise, summarized answer. "
            "Use it for current events, news, API updates, new developments, or anything "
            "outside your training data. "
            "Pass a specific, descriptive natural-language query with names, dates, and key words. "
            "If this tool reports that search is unavailable, do NOT conclude the topic does "
            "not exist — ask the user for a URL and open it with web_fetch instead."
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
            "Open a URL directly and return its readable main content. "
            "Use this whenever the user gives you a URL, or on any address you already "
            "know — no search required. Accepts a bare domain like example.com. "
            "Ads, navigation, boilerplate, and irrelevant links are stripped by an LLM pass. "
            "Content images on the page are returned alongside the text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch (http/https)."},
                "query": {
                    "type": "string",
                    "description": "Optional: what you're looking for on the page. Guides content extraction.",
                },
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


def _est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _truncate(text: str, char_cap: int) -> str:
    if len(text) <= char_cap:
        return text
    cut = text[:char_cap]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + " …"


# ─────────────────────────────────────────────
# Web search
# ─────────────────────────────────────────────
# Every keyless engine now sits behind a bot filter, and that filter is a NORMAL
# HTTP answer rather than an error: DDG returns 202 with "select all squares
# containing a duck", Mojeek demands JavaScript, Brave answers 429. None of those
# raise, so a throttled search used to be indistinguishable from an empty one and
# got reported as "No results found" — telling the user a topic does not exist
# when the truth was "we were refused". Hence the (results, refused) split below.
SEARCH_TIMEOUT    = 15.0   # per-request socket timeout
SEARCH_RETRY_WAIT = 3.0    # pause before one retry of the whole chain

BLOCK_STATUS  = (202, 429)  # 202 = DDG's challenge page, 429 = too many requests
BLOCK_MARKERS = (           # only consulted when a page parsed to nothing
    "select all squares",
    "confirm this search was made by a human",
    "unusual traffic",
    "are you a robot",
    "javascript is required",
    "captcha",
)


class SearchBlocked(RuntimeError):
    """Every engine refused to answer. Deliberately NOT the same as 'no results'."""

    def __init__(self, refused: list[str]):
        self.refused = refused
        super().__init__(
            "every search engine refused the request ("
            + ", ".join(refused)
            + " returned a bot challenge)"
        )


def _looks_blocked(status: int, body: str) -> bool:
    """True when an engine answered with a bot filter rather than a result page."""
    if status in BLOCK_STATUS:
        return True
    low = body.lower()
    return any(marker in low for marker in BLOCK_MARKERS)


def _search_request(url: str, params: dict, *, headers: dict | None = None,
                    use_post: bool = False) -> tuple[int, str]:
    """One HTTP hit. Returns (status, body), including for HTTP error statuses.

    urllib raises on 4xx/5xx, but for search the status IS the diagnosis — a 429
    is the clearest possible signal that we were throttled — so it is returned
    rather than propagated."""
    encoded = urllib.parse.urlencode(params)
    if use_post:
        target, data = url, encoded.encode()
    else:
        target = url + ("&" if "?" in url else "?") + encoded
        data = None

    req = urllib.request.Request(
        target, data=data, headers={"User-Agent": HTTP_UA, **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            return getattr(resp, "status", 200), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _real_ddg_url(tag: str) -> str:
    """DDG wraps outbound links as /l/?uddg=<url-encoded target> — unwrap those."""
    m = re.search(r'href="([^"]+)"', tag)
    if not m:
        return ""
    href = html_lib.unescape(m.group(1))
    uddg = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
    return uddg[0] if uddg else href


def _parse_ddg(body: str, max_results: int) -> list[tuple[str, str, str]]:
    """Parse DDG result pages, both the html/ and lite/ flavours.

    The two endpoints use different class names (result__a / result-link) and
    different quoting, so each set is tried only if the previous found nothing —
    matching them together would let one page's markup nominate the other's
    selector and quietly drop results.

    Titles and snippets are paired by INDEX, never by zip(): zip() silently stops
    at the shorter list, so a markup change in only the snippet selector would
    discard every result instead of merely losing the snippets."""
    variants = (
        (r'(<a[^>]*class=["\']result__a["\'][^>]*>.*?</a>)',
         r'class=["\']result__snippet["\'][^>]*>(.*?)</a>'),
        (r'(<a[^>]*class=["\']result-link["\'][^>]*>.*?</a>)',
         r'class=["\']result-snippet["\'][^>]*>(.*?)</td>'),
    )

    for anchor_pat, snippet_pat in variants:
        anchors = re.findall(anchor_pat, body, re.S)
        if not anchors:
            continue
        snippets = re.findall(snippet_pat, body, re.S)

        results: list[tuple[str, str, str]] = []
        for i, tag in enumerate(anchors):
            title = _strip_html(tag)[:TITLE_CAP]
            url = _real_ddg_url(tag)
            snippet = _strip_html(snippets[i])[:SNIPPET_CAP] if i < len(snippets) else ""
            if title and url:
                results.append((title, url, snippet))
            if len(results) >= max_results:
                break
        return results

    return []


def _parse_searxng(body: str, max_results: int) -> list[tuple[str, str, str]]:
    data = json.loads(body)
    items = data.get("results") if isinstance(data, dict) else None
    return [
        (str(it.get("title", "")), str(it.get("url", "")), str(it.get("content", "")))
        for it in (items or [])
        if isinstance(it, dict) and it.get("url")
    ][:max_results]


def _parse_brave(body: str, max_results: int) -> list[tuple[str, str, str]]:
    data = json.loads(body)
    items = (data.get("web") or {}).get("results") if isinstance(data, dict) else None
    return [
        (str(it.get("title", "")), str(it.get("url", "")), str(it.get("description", "")))
        for it in (items or [])
        if isinstance(it, dict) and it.get("url")
    ][:max_results]


def _backends(query: str) -> list[dict]:
    """Backend chain, most reliable first.

    The keyed APIs lead because they exist to be called by software and answer
    every time; the scraped engines are the fallback, not the default."""
    chain: list[dict] = []

    if BRAVE_API_KEY:
        chain.append({
            "label": "brave-api",
            "url": "https://api.search.brave.com/res/v1/web/search",
            "params": {"q": query, "count": MAX_RESULTS},
            "headers": {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
            "use_post": False,
            "parse": _parse_brave,
        })

    if SEARXNG_URL:
        chain.append({
            "label": "searxng",
            "url": SEARXNG_URL.rstrip("/") + "/search",
            "params": {"q": query, "format": "json"},
            "headers": {"Accept": "application/json"},
            "use_post": False,
            "parse": _parse_searxng,
        })

    chain += [
        {
            "label": "duckduckgo-html",
            "url": DDG_HTML_URL,
            "params": {"q": query},
            "headers": None,
            "use_post": True,
            "parse": _parse_ddg,
        },
        {
            "label": "duckduckgo-lite",
            "url": DDG_LITE_URL,
            "params": {"q": query},
            "headers": None,
            "use_post": True,
            "parse": _parse_ddg,
        },
    ]
    return chain


def _search(query: str, max_results: int = MAX_RESULTS) -> list[tuple[str, str, str]]:
    """Search the web, trying each backend in turn.

    Returns [(title, url, snippet), ...] — possibly empty, which means every
    reachable backend genuinely had no matches.

    Raises SearchBlocked when the engines refused us, and RuntimeError when none
    could be reached. Keeping those three outcomes distinct is the whole point:
    the caller must not tell the user "no results" when the truth is "blocked".
    """
    print("=" * 20)
    print(f"[_search]: debug query: {query[:75]}")

    chain = _backends(query)

    # One retry of the whole chain. A 429 is usually per-second and clears at
    # once, whereas a captcha does not — so this costs one short sleep in the
    # worst case and rescues the common transient case.
    for attempt in (1, 2):
        refused: list[str] = []
        problems: list[str] = []
        answered = False

        for backend in chain:
            label = backend["label"]
            try:
                status, body = _search_request(
                    backend["url"], backend["params"],
                    headers=backend["headers"], use_post=backend["use_post"],
                )
            except Exception as exc:
                print(f"[search] {label}: {type(exc).__name__}: {exc}")
                problems.append(f"{label}: {type(exc).__name__}")
                continue

            results = backend["parse"](body, max_results) if status == 200 else []
            if results:
                print(f"[search] {label}: {len(results)} result(s) (HTTP {status})")
                return results

            # Nothing usable came back. Only now does the block check matter:
            # a page that yielded results is never a challenge page, so testing
            # markers first would risk false positives from bundled JS.
            if _looks_blocked(status, body):
                print(f"[search] {label}: bot challenge (HTTP {status})")
                refused.append(label)
            else:
                print(f"[search] {label}: reached, no matches (HTTP {status})")
                answered = True

        if answered or not refused or attempt == 2:
            break
        print(f"[search] all backends refused; retrying in {SEARCH_RETRY_WAIT}s")
        time.sleep(SEARCH_RETRY_WAIT)

    if refused and not answered:
        raise SearchBlocked(refused)
    if not answered:
        raise RuntimeError("no search backend could be reached — " + "; ".join(problems))
    return []


# ─────────────────────────────────────────────
# Summarization — map/reduce against a small context window
# ─────────────────────────────────────────────
def _call_llama(system: str, user: str, max_tokens: int) -> str:
    """One NON-streaming chat call to llama.cpp. Raises on failure."""
    print("="*20)
    print(f"[_call_llama]: debug system: {system[:75]}")
    print(f"[_call_llama]: debug user: {user[:75]}")
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }
    req = urllib.request.Request(
        LLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Non-streaming → no bytes arrive until generation is fully done, so this
    # socket timeout effectively caps total generation time. Keep it generous
    # (or None) since the client connection is kept alive by heartbeats anyway.
    with urllib.request.urlopen(req, timeout=LLAMA_TIMEOUT) as resp:
        out = json.loads(resp.read().decode("utf-8", "replace"))

    choice = out["choices"][0]
    message = choice.get("message") or {}
    res = (message.get("content") or "").strip()
    if not res:
        # Reasoning models put thinking in reasoning_content; if that burns the
        # whole token budget the server returns finish_reason="length" with
        # EMPTY content. Raise so callers fall back instead of returning "".
        reasoning = (message.get("reasoning_content") or "").strip()
        detail = (f"LLM returned empty content (finish_reason="
                  f"{choice.get('finish_reason')!r}, max_tokens={max_tokens}).")
        if reasoning:
            detail += (" Reasoning only — raise the token cap.\n"
                       f"Reasoning tail: ...{reasoning[-300:]}")
        raise RuntimeError(detail)
    print("="*20)
    print(f"[_call_llama]: debug response: {res[:75]}")
    return res


def _chunk(items: list[str], max_chars: int) -> list[str]:
    """Group text items into chunks that stay under max_chars."""
    chunks, cur, cur_len = [], [], 0
    for it in items:
        # a single oversized item gets hard-truncated so it always fits
        if len(it) > max_chars:
            it = _truncate(it, max_chars)
        if cur and cur_len + len(it) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(it)
        cur_len += len(it) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


_SYS_PARTIAL = (
    "Extract only the facts from these web results that help answer the query. "
    "Be terse — bullet-style notes, no preamble."
)
_SYS_FINAL = (
    "Answer the query in under 400 tokens using only the provided notes/results. "
    "Be factual and concise. ALWAYS keep the source URLs in your answer — the "
    "caller needs them to fetch pages. If the results don't answer it, say so "
    "briefly and still list the URLs."
)


def _summarize(query: str, results: list[tuple[str, str, str]]) -> str:
    """Condense results, splitting into multiple small calls as needed.
    Falls back to the raw results (with URLs) if the LLM call fails — an empty
    summary is worse than unprocessed snippets."""
    items = [f"{i + 1}. {t}\n{u}\n{s}" for i, (t, u, s) in enumerate(results)]
    raw_join = "\n\n".join(items)

    print("="*20)
    print(f"[_summarize]: debug query: {query[:75]}")

    try:
        chunks = _chunk(items, MAX_CHUNK_CHARS)

        # Single chunk fits → one call.
        if len(chunks) == 1:
            return _call_llama(_SYS_FINAL, f"Query: {query}\n\nResults:\n{chunks[0]}\n\nAnswer:", FINAL_TOKENS)

        # MAP: summarize each chunk separately (multiple smaller requests).
        partials = [
            _call_llama(_SYS_PARTIAL, f"Query: {query}\n\nResults:\n{c}\n\nNotes:", PARTIAL_TOKENS)
            for c in chunks
        ]

        # REDUCE: fold partials together, re-chunking if they still overflow.
        while _est_tokens("\n\n".join(partials)) > INPUT_TOKEN_BUDGET and len(partials) > 1:
            sub = _chunk([f"- {p}" for p in partials], MAX_CHUNK_CHARS)
            if len(sub) >= len(partials):          # no progress → stop, truncate below
                break
            partials = [
                _call_llama(_SYS_PARTIAL, f"Query: {query}\n\nNotes:\n{c}\n\nNotes:", PARTIAL_TOKENS)
                for c in sub
            ]

        combined = _truncate("\n\n".join(partials), MAX_CHUNK_CHARS)
        return _call_llama(_SYS_FINAL, f"Query: {query}\n\nNotes:\n{combined}\n\nAnswer:", FINAL_TOKENS)

    except Exception as e:
        # llama.cpp unreachable/erroring → fall back to raw snippets.
        print(f"[summarize] falling back to raw results: {e}")
        return raw_join


# ─────────────────────────────────────────────
# URL fetch + LLM content extraction (parallel map)
# ─────────────────────────────────────────────
def _llm_chat(system: str, user: str, max_tokens: int) -> str:
    """Chat call via ai_call.AICall (OpenAI-compatible), with raw urllib fallback."""
    if _HAS_AI_CALL:
        client = AICall(base_url="http://localhost:8080",
                        max_tokens=max_tokens,
                        timeout=LLAMA_TIMEOUT)
        return client.chat(prompt=user, system_prompt=system).strip()
    return _call_llama(system, user, max_tokens)


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
        # drop non-content blocks entirely before stripping tags
        body = re.sub(r"<(script|style|noscript|iframe|svg|form)[^>]*>.*?</\1>",
                      " ", body, flags=re.S | re.I)
        # line breaks for block-level elements so text doesn't fuse together
        body = re.sub(r"<(p|div|br|li|tr|h[1-6]|section|article)[^>]*>",
                      "\n", body, flags=re.I)
        text = _strip_html(body)
    else:
        img_urls = []
        text = body  # plain text / json / etc.

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
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

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        blocks = [b for b in pool.map(_grab, enumerate(urls, 1)) if b]
    print(f"[media] extracted {len(blocks)}/{len(urls)} images -> {WEB_MEDIA_DIR}")
    return blocks


_SYS_EXTRACT = (
    "Extract the main readable content from this raw webpage text. "
    "any content that is ads or promotional in nature should be removed. also remove navigation menus, cookie notices, footers, and links that make "
    "Keep only relevant in full form"
    "do not summarize or condense the main text body return it verbatim."
)


def _extract_page(url: str, query: str, text: str) -> str:
    """Parallel map: each chunk is cleaned by its own LLM call, then reduced."""
    chunks = _chunk(re.split(r"\n\n+", text), MAX_CHUNK_CHARS)
    print(f"[_extract_page]: {len(text)} chars -> {len(chunks)} chunk(s)")

    topic = query or f"the main content of {url}"

    # Single chunk fits → one call, straight to the final-style answer.
    if len(chunks) == 1:
        return _llm_chat(
            _SYS_EXTRACT,
            f"Topic: {topic}\n\nWebpage text:\n{chunks[0]}\n\nRelevant content:",
            FINAL_TOKENS,
        )

    # MAP: parallel per-chunk extraction against localhost:8080.
    # A failed chunk (e.g. token-starved reasoning model) degrades to its raw
    # text instead of aborting the whole page extraction.
    def _extract_chunk(c: str) -> str:
        try:
            return _llm_chat(
                _SYS_EXTRACT,
                f"Topic: {topic}\n\nWebpage text:\n{c}\n\nRelevant content:",
                EXTRACT_TOKENS,
            )
        except Exception as e:
            print(f"[extract] chunk fallback to raw: {e}")
            return c

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        partials = list(pool.map(_extract_chunk, chunks))

    # REDUCE: fold the partials into one concise answer.
    combined = _truncate("\n\n".join(partials), MAX_CHUNK_CHARS)
    return _llm_chat(
        _SYS_FINAL,
        f"Query: {topic}\n\nNotes:\n{combined}\n\nAnswer:",
        FINAL_TOKENS,
    )


# ─────────────────────────────────────────────
# Tool logic
# ─────────────────────────────────────────────
def call_tool(name: str, arguments: dict) -> list:
    print(f"[tool call] name={name} arguments={arguments}")

    if name == "web_search":
        query = (arguments.get("query") or "").strip()
        if not query:
            return [{"type": "text", "text": "No query provided."}]

        try:
            results = _search(query)
        except SearchBlocked as e:
            # Say what actually happened. Reporting "no results" here would be a
            # lie, and the model would pass that lie on as "this does not exist".
            print(f"[search] blocked: {e}")
            return [{"type": "text", "text": (
                f"Search is unavailable right now — {e}. "
                "This is a rate limit, NOT an absence of results: do not tell the "
                "user the topic has no coverage. Either answer from what you already "
                "know, ask them for a specific URL and open it with web_fetch, or try "
                "this search again shortly."
            )}]
        except Exception as e:
            return [{"type": "text", "text": f"Search failed: {e}"}]

        if not results:
            return [{"type": "text", "text": f"No results found for: {query}"}]

        summary = _truncate(_summarize(query, results), OUTPUT_CHAR_CAP)
        return [{"type": "text", "text": summary}]

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

        query = (arguments.get("query") or "").strip()

        try:
            text, img_urls = _fetch_page(url)
        except Exception as e:
            return [{"type": "text", "text": f"Fetch failed: {e}"}]

        if not text:
            return [{"type": "text", "text": f"No readable text found at: {url}"}]

        try:
            content = _extract_page(url, query, text)
        except Exception as e:
            # llama.cpp unreachable/erroring → fall back to raw cleaned text.
            print(f"[extract] falling back to raw text: {e}")
            content = text

        blocks = [{"type": "text", "text": _truncate(content, OUTPUT_CHAR_CAP)}]
        # attach content images as MCP image blocks so the calling LLM gets the
        # media relevant to the text alongside it.
        if img_urls:
            blocks += _download_images(img_urls)
        return blocks

    return [{"type": "text", "text": f"Unknown tool: {name}"}]


# ─────────────────────────────────────────────
# Heartbeat-wrapped tool call
# ─────────────────────────────────────────────
async def _tool_call_with_heartbeat(req_id, name: str, arguments: dict) -> StreamingResponse:
    """
    Run the (blocking) tool in a worker thread while streaming a "." heartbeat to
    the client every HEARTBEAT_INTERVAL seconds. When the tool finishes, the full
    JSON-RPC response is appended after the dots.

    NOTE: the resulting body is `....{"jsonrpc": ...}` — i.e. NOT pure JSON. The
    leading dots are intentional keep-alive bytes. The client must strip anything
    before the first '{' before json-parsing, e.g.:
        text = raw_body[raw_body.index("{"):]
        msg  = json.loads(text)
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, call_tool, name, arguments)

    async def gen():
        task = asyncio.ensure_future(future)
        # Dribble a dot every HEARTBEAT_INTERVAL seconds until the work completes.
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_INTERVAL)
            if not done:
                print("[heartbeat] .")
                yield b"."

        try:
            content = task.result()
        except Exception as e:
            content = [{"type": "text", "text": f"Tool error: {e}"}]

        payload = {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": content, "isError": False},
        }
        yield json.dumps(payload).encode()

    # application/json because strict MCP clients reject text/plain, even though
    # the body is dots + JSON, not pure JSON. The client must strip anything
    # before the first '{' before json-parsing (see docstring above).
    return StreamingResponse(gen(), media_type="application/json")


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
                    "version": "1.1.0",
                    "instructions": (
                        "Two tools. web_search(query) searches the web and returns a "
                        "summarized answer (<=250 tokens) with sources. web_fetch(url) "
                        "opens one page directly and returns its readable text and "
                        "content images — use it whenever the user gives you a URL or "
                        "you already know the address, and it needs no search. If "
                        "web_search reports that it is unavailable, that is a rate "
                        "limit rather than a lack of results: do not tell the user the "
                        "topic has no coverage. Ask for a URL and use web_fetch instead."
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
        # Stream "." heartbeats while the search+summary runs, then dump the result.
        return await _tool_call_with_heartbeat(req_id, name, arguments)

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

if __name__ == "__main__":
    print(f"Web search MCP server running on http://localhost:{SERVER_PORT}")
    print(f"Search backends     : {', '.join(b['label'] for b in _backends('probe'))}")
    if not (BRAVE_API_KEY or SEARXNG_URL):
        print("                      NOTE: no BRAVE_API_KEY / SEARXNG_URL set, so search "
              "relies on\n                      scraped engine pages, which are captcha-"
              "gated and will fail\n                      intermittently. web_fetch is "
              "unaffected and always works.")
    print(f"Summarizer (llama)  : {LLAMA_URL}  (ctx={LLAMA_CTX_TOKENS} tok)")
    print(f"Heartbeat           : '.' every {HEARTBEAT_INTERVAL}s; llama timeout={LLAMA_TIMEOUT}s")
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)