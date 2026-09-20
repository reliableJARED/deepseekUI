import os
import re
import base64
import asyncio
import subprocess
import html as html_lib
import urllib.parse
import urllib.request
from html.parser import HTMLParser
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
# The model is named here rather than left to the caller because Gemini 3.1 Flash-Lite
# is the one whose Google-Search grounding is free (500 requests/day); a model
# outside that tier would quietly bill the key's project instead.
#
# The API key arrives per request from the MCP client, which stores it on this
# server's entry in mcp.json — see _api_key(). GEMINI_API_KEY / GOOGLE_API_KEY in
# the environment is the fallback, for running this file by hand.
GEMINI_MODEL       = os.environ.get("GEMINI_SEARCH_MODEL", "").strip() or "gemini-3.1-flash-lite"
GEMINI_TIMEOUT     = 90      # seconds for the whole grounded call
SEARCH_MAX_SOURCES = 8       # source URLs echoed back with the answer

# Grounding sources do not arrive as publisher URLs. Each one is wrapped in an
# opaque Google hop — vertexaisearch.cloud.google.com/grounding-api-redirect/
# AUZIYQ... — which names no publisher and carries a signature that expires. The
# caller only ever reports what this server hands it, so the hop is taken here,
# once per source, and the address a browser would land on is what goes back.
# See _resolve_redirect().
SEARCH_RESOLVE_REDIRECTS = True  # False = pass the grounding URLs through untouched
REDIRECT_TIMEOUT   = 15      # seconds per source hop
REDIRECT_WORKERS   = 8       # sources followed in parallel
REDIRECT_SCAN_BYTES = 8192   # body prefix read when a 200 hides the redirect

# Request headers accepted as the API key, most specific first. Recognising
# several spellings means the generic "Headers" box in the settings panel — not
# only the dedicated API-key field — can carry it.
API_KEY_HEADERS = ("x-api-key", "x-gemini-api-key", "x-goog-api-key")
API_KEY_ENV     = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# web_fetch tuning.
#
# There is deliberately NO model in this path. Fetching a page is a download and a
# mechanical prune of the page furniture, so it finishes in the time the network
# takes and can never outlive the MCP client's per-call budget.
#
# It used to hand the page to local llama.cpp — a 4-way parallel map over chunks,
# then a reduce — to strip boilerplate, which took ~135 s on a large page. The MCP
# client's per-call timeout is 120 s, and a call that outlives that budget dooms the
# session it was sent on (the abandoned POST's late response lands on the shared
# connection and the transport reads it as end-of-stream), so every fetch after the
# first big page failed with "Connection closed" — including a bare example.com.
# The caller is a large-context model: give it the page and let it do the reading.
#
# The page is PRUNED of page furniture before it goes out (see the HTML section
# below), which is a token saving rather than a summary: the pass is mechanical,
# it is fast enough to stay inside the budget by construction, and what it removed
# is disclosed in the reply. It also moves the cap: a chrome-heavy page spends its
# 60k characters on nav and cookie notices and gets truncated before the article
# starts, which is the real cost of leaving the furniture in.
FETCH_TIMEOUT      = 30      # socket timeout for the initial page download
FETCH_BODY_CAP     = 60_000  # max chars of pruned page text returned
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
            "web_fetch when the summary is not enough. Each source is the publisher's "
            "own URL, already resolved, so it can be quoted or opened as it stands. "
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
            "Navigation, cookie notices, footers and similar page furniture are "
            "removed before the text comes back, so it is the page's main content "
            "rather than a copy of everything on the screen — the result says so, "
            "and something the user can see may have been pruned away. "
            "Images from that content come back alongside the text, so you can see "
            "them. "
            "Read and summarise the returned text yourself; there is no "
            "pre-summarising step."
            "This tool returns text only, and it does not run JavaScript: on a page "
            "whose content is rendered by a script — a video player, most social feeds — "
            "little survives the prune and the title may be all you get. It cannot see "
            "inside a video either. For a YouTube, Vimeo or other video URL, use "
            "display_media when the user asked to be shown the video — it gives them a "
            "player — or reduce_video_frames when you need to watch the video's frames "
            "yourself; to show the user a URL's image, use display_media. "
            "Images come back only when a parsed HTML page references them, so fetching "
            "an image URL directly returns bytes rather than a picture."
            "Reddit Specific Search: when fetching reddit pages reddit.com/r/StableDiffusion/comments/1wli1vi/an_hour_in_to_qwen21/ ->MUST CONVERT to arctic-shift using the post ID->arctic-shift.photon-reddit.com/api/posts/ids?ids=1wli1vi"
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
# HTML → text: build a tree, prune the furniture, render what is left
# ─────────────────────────────────────────────
# A regex cannot do this job. `<script>.*?</script>` is not nesting-safe, so an
# `<svg>` holding a nested `<svg>` — which is what an icon sprite is — ends the
# match at the INNER close tag and leaks the rest of the sprite out as text. And
# the question that decides the most here, "is this `<header>` the site's masthead
# or the article's own title?", needs ancestors, which a regex does not have. Both
# come free from a tree, and html.parser is in the standard library and tolerant of
# the malformed markup real pages ship.
#
# The rules are ordered below by how much each one can lose:
#   1. tags whose contents a browser does not show either     — cannot lose anything
#   2. `nav`, plus header/footer/aside outside the article     — the spec's own names
#   3. id/class/role words (cookie, newsletter, share, …)      — learned hints
#   4. link density                                            — a link list, not prose
# Only rule 1 is safe by construction, so the reply says what was removed
# (PRUNE_NOTE) instead of hiding it: the model can then tell the user the page may
# hold more, rather than reporting the gap as a fact about the page.

# Never page text: a browser does not render these either, so dropping them cannot
# lose anything a reader can see.
PRUNE_DROP_TAGS = frozenset((
    "script", "style", "noscript", "template", "iframe", "svg", "canvas",
    "form", "button", "select", "option", "optgroup", "textarea", "label",
    "fieldset", "legend", "object", "embed", "applet", "audio", "video",
    "dialog", "marquee",
))

# `nav` is furniture wherever it appears. header/footer/aside are furniture only
# OUTSIDE the article: HTML5 puts an article's title, byline and date in its own
# <header> and its pull quotes in <aside>, so dropping those by tag would delete
# the very lines a summary most needs.
PRUNE_ALWAYS_TAGS = frozenset(("nav",))
PRUNE_MAIN_TAGS   = frozenset(("main", "article"))
PRUNE_UNLESS_MAIN = frozenset(("header", "footer", "aside"))

# Hints matched against WHOLE words split out of id/class/role/aria-label, so
# `cookie-banner` matches `cookie` while `shareholder-report` does not match
# `share`. The first group is furniture anywhere; the second counts only outside
# the article, because a page can legitimately be *about* something named `nav`.
PRUNE_ALWAYS_HINTS = frozenset((
    "cookie", "cookies", "consent", "gdpr", "newsletter", "subscribe",
    "subscription", "signup", "modal", "popup", "overlay", "breadcrumb",
    "breadcrumbs", "pagination", "pager", "share", "sharing", "social",
    "skip", "advert", "advertisement", "ads", "adsbygoogle", "sponsor",
    "sponsored", "promo", "promotional", "outbrain", "taboola",
))
PRUNE_UNLESS_MAIN_HINTS = frozenset((
    "nav", "navbar", "navigation", "menu", "menubar", "sidebar", "masthead",
    "toolbar", "topbar", "header", "footer", "banner", "drawer", "hamburger",
))

# Link density. A block that is mostly links is a link list — a menu, a footer, a
# related-posts strip — not prose. Two exemptions keep the useful case: a block
# carrying a HEADING is not a link list, and neither is the level below it. A table
# of contents is
#   <section><h2>On this page</h2><ul><li><a>…</a></li></ul></section>
# where the links live in the <ul>, not in the block that carries the heading — and
# on a documentation page it is often the most useful thing there is.
PRUNE_LINK_DENSITY   = 0.5
PRUNE_LINK_MIN_CHARS = 200
PRUNE_HEADING_TAGS   = frozenset(("h1", "h2", "h3", "h4", "h5", "h6"))

# Rendered with a line break before and after, so blocks do not fuse into one
# paragraph. Everything else is inline and contributes only its text.
PRUNE_BLOCK_TAGS = frozenset((
    "address", "article", "aside", "blockquote", "br", "dd", "details", "div",
    "dl", "dt", "fieldset", "figcaption", "figure", "footer", "h1", "h2", "h3",
    "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "summary", "table", "tbody", "td", "tfoot", "th", "thead",
    "title", "tr", "ul",
))

# Told to the model on every fetch. The tool cannot know what the heuristic
# removed, but it can say that something was removed — which is what stops a pruned
# section from being reported to the user as a section the page does not have.
PRUNE_NOTE = (
    "The contents of this web page have been passed through HTMLParser to attempt "
    "to remove inconsequential content, if the user can see something on this page "
    "you can't that is why"
)

_HTML_VOID_TAGS = frozenset((
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
))


class _DomNode:
    """One element — or one run of text, which carries tag "" — in a parsed page."""

    __slots__ = ("tag", "attrs", "children", "text")

    def __init__(self, tag: str = "", attrs: dict | None = None, text: str = ""):
        self.tag = tag
        self.attrs = attrs or {}
        self.children: list[_DomNode] = []
        self.text = text


class _DomBuilder(HTMLParser):
    """HTML → tree. Character references are decoded by the parser itself."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _DomNode("[document]")
        self._stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _DomNode(tag, {name.lower(): (value or "") for name, value in attrs})
        self._stack[-1].children.append(node)
        if tag not in _HTML_VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self._stack[-1].children.append(
            _DomNode(tag, {name.lower(): (value or "") for name, value in attrs}))

    def handle_endtag(self, tag):
        # Unbalanced markup is the norm: close the nearest matching open tag and
        # ignore a close tag with no opener, rather than unwinding the stack — one
        # stray `</div>`, which every page has, would otherwise flatten the rest of
        # the document into it.
        for depth in range(len(self._stack) - 1, 0, -1):
            if self._stack[depth].tag == tag:
                del self._stack[depth:]
                return

    def handle_data(self, data):
        self._stack[-1].children.append(_DomNode(text=data))


def _node_text(node: _DomNode) -> str:
    if not node.tag:
        return node.text
    return "".join(_node_text(child) for child in node.children)


def _link_text(node: _DomNode) -> str:
    if node.tag == "a":
        return _node_text(node)
    if not node.tag:
        return ""
    return "".join(_link_text(child) for child in node.children)


def _has_heading(node: _DomNode) -> bool:
    if node.tag in PRUNE_HEADING_TAGS:
        return True
    return any(_has_heading(child) for child in node.children)


def _attr_words(node: _DomNode) -> set[str]:
    """The whole words of a node's id/class/role/aria-label, lower-cased."""
    raw = " ".join(node.attrs.get(name, "")
                   for name in ("id", "class", "role", "aria-label")).lower()
    return {word for word in re.split(r"[^a-z0-9]+", raw) if word}


def _is_link_farm(node: _DomNode) -> bool:
    """True for a block that is mostly links — a menu or a related-posts strip.

    Measured on whitespace-normalised text: the indentation between `<li>` tags is
    real characters in the tree and would otherwise dilute the ratio until nothing
    ever looked like a link list.
    """
    text = re.sub(r"\s+", " ", _node_text(node)).strip()
    if len(text) < PRUNE_LINK_MIN_CHARS or _has_heading(node):
        return False
    linked = len(re.sub(r"\s+", " ", _link_text(node)).strip())
    return linked / len(text) > PRUNE_LINK_DENSITY


def _prune(node: _DomNode, in_main: bool = False) -> None:
    """Drop furniture from the tree in place. `in_main` tracks the article's ancestry."""
    # Children of a block that has a heading of its own are exempt from the link
    # density rule — see PRUNE_LINK_DENSITY. This has to be decided BEFORE the
    # children are walked, because the rule is applied to them on the way back up.
    titled = any(child.tag in PRUNE_HEADING_TAGS for child in node.children)
    kept: list[_DomNode] = []
    for child in node.children:
        if not child.tag:                      # a text run, never furniture
            kept.append(child)
            continue
        if child.tag in PRUNE_DROP_TAGS or child.tag in PRUNE_ALWAYS_TAGS:
            continue
        main = in_main or child.tag in PRUNE_MAIN_TAGS
        if not main and child.tag in PRUNE_UNLESS_MAIN:
            continue
        words = _attr_words(child)
        if words & PRUNE_ALWAYS_HINTS or (not main and words & PRUNE_UNLESS_MAIN_HINTS):
            continue
        _prune(child, main)
        # Only a BLOCK can be a link list. Without that guard the rule reaches the
        # page's own structure: on a page that is mostly links — an index, an
        # archive, a link directory — `<html>` itself looks like a link farm and
        # the entire document is dropped, which is not a prune.
        if not titled and child.tag in PRUNE_BLOCK_TAGS and _is_link_farm(child):
            continue
        kept.append(child)
    node.children = kept


def _build_tree(body: str) -> _DomNode:
    builder = _DomBuilder()
    builder.feed(body)
    builder.close()
    return builder.root


def _parse_and_prune(body: str) -> _DomNode:
    root = _build_tree(body)
    _prune(root)
    return root


def _render_text(root: _DomNode) -> str:
    """The tree's visible text, with block tags rendered as line breaks."""
    chunks: list[str] = []

    def walk(node: _DomNode) -> None:
        if not node.tag:
            chunks.append(node.text)
            return
        block = node.tag in PRUNE_BLOCK_TAGS
        if block:
            chunks.append("\n")
        for child in node.children:
            walk(child)
        if block:
            chunks.append("\n")

    walk(root)
    text = "".join(chunks).replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


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


# ── grounding redirects ───────────────────────────────────────────────────────
# A grounding chunk's URI is not the page. It is a redirect on Google's own host:
#
#   dreampixelforge.com — https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQ...
#
# That URL is useless to the caller in three separate ways. It names Google, not
# the publisher, so a citation built from it attributes the claim to the wrong
# site. It is opaque, so nothing downstream — the user, a log, the model's own
# follow-up fetch — can tell where it goes without opening it. And it is signed and
# time-limited, so it rots: a link that worked when the answer was written is dead
# by the time anyone clicks it.
#
# The redirects themselves do work, so the fix is to take the hop rather than to ask
# Gemini for something it does not have. One request per source, in parallel, and
# the publisher URL replaces the redirect before the summary is written.
GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


def _is_grounding_redirect(url: str) -> bool:
    """True for the opaque hops that need following.

    Deliberately narrow: a source that is already a real URL is left exactly as
    Gemini reported it, so a citation the model could have used as-is is never
    rewritten — and never costs a request.
    """
    if "grounding-api-redirect" in url.lower():
        return True
    return (urllib.parse.urlsplit(url).hostname or "").lower() == GROUNDING_REDIRECT_HOST


def _redirect_page_target(body: str, base_url: str) -> str:
    """The destination of a page that redirects from inside its own HTML.

    Not every hop answers with a 3xx. Some serve a small document carrying
    ``<meta http-equiv="refresh" content="0;url=...">`` or a ``location.href =
    "..."`` instead, and urllib has no reason to follow either — the status was 200,
    so as far as it is concerned the request is over. Checking both spellings is
    what makes those sources resolve like the rest.
    """
    match = (re.search(r"""<meta[^>]+http-equiv\s*=\s*["']?refresh["']?[^>]*?"""
                       r"""content\s*=\s*["'][^"']*?url\s*=\s*([^"'>\s]+)""", body, re.I)
             # `location.href = "..."`, `location = "..."`, `location.replace("...")`
             # and `location.assign("...")` — one alternation, so group(1) is
             # always the URL whatever the page used.
             or re.search(r"""(?:location(?:\.href)?\s*=\s*|"""
                          r"""location\.(?:replace|assign)\(\s*)["']([^"']+)["']""",
                          body, re.I))
    if not match:
        return ""
    target = html_lib.unescape(match.group(1)).strip()
    # A relative target is normal here, so resolve it against the hop itself.
    return urllib.parse.urljoin(base_url, target) if target else ""


def _resolve_redirect(url: str) -> str:
    """Follow one grounding redirect and return the publisher URL it lands on.

    The body is not read: ``urlopen`` completes the whole redirect chain before it
    returns, so ``geturl()`` already holds the final address and the connection can
    be closed straight away. That keeps a source to a round-trip of headers rather
    than a full page download — which matters, because this runs on every search
    and the search has to stay inside the MCP client's per-call budget.

    A 200 that still points at the hop itself is the one case that needs the body —
    see _redirect_page_target(); only a capped prefix of it is read.

    Returns "" both when the URL is not a hop at all and when a hop cannot be taken,
    so the caller keeps the original URL either way. A source that needs no request
    must never cost one: that is what _is_grounding_redirect() is for.
    """
    if not _is_grounding_redirect(url):
        return ""

    try:
        request = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
        with urllib.request.urlopen(request, timeout=REDIRECT_TIMEOUT) as response:
            final = str(response.geturl() or "").strip()
            if not final or final == url:
                head = response.read(REDIRECT_SCAN_BYTES).decode("utf-8", "replace")
                final = _redirect_page_target(head, url) or final
        return final
    except Exception as exc:
        print(f"[search] source redirect not followed ({url[:80]}): "
              f"{type(exc).__name__}: {exc}")
        return ""


def _resolve_sources(sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Replace every grounding redirect with the publisher's own URL.

    Followed in parallel — one hop each, so the added wall clock is a single
    round-trip for the batch rather than one per source — with Gemini's ordering
    preserved. Two hops that land on the same page collapse into one entry: the same
    article cited twice under two different signatures is one citation.

    A source that is already a real URL, or whose hop cannot be taken, comes back
    unchanged. A working redirect is a worse citation than a publisher URL, but a
    far better one than nothing.
    """
    sources = sources[:SEARCH_MAX_SOURCES]
    if not SEARCH_RESOLVE_REDIRECTS or not sources:
        return sources

    with ThreadPoolExecutor(max_workers=REDIRECT_WORKERS) as pool:
        landed = list(pool.map(lambda item: _resolve_redirect(item[1]), sources))

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for (title, original), final in zip(sources, landed):
        url = final or original
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        resolved.append((title, url))

    taken = sum(1 for final in landed if final)
    if taken:
        print(f"[search] followed {taken}/{len(sources)} grounding redirects "
              f"to publisher URLs")
    return resolved


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

    What goes back is the publishers' own URLs. Gemini reports each source as an
    opaque Google redirect, and those are followed here so the caller never has to
    know a hop was involved — see _resolve_sources().
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

    # Last step before the summary is written, so nothing downstream — the model,
    # the transcript, the user's click — ever sees a redirect. This is also where
    # the source cap is applied, so the hops taken are only the ones reported.
    sources = _resolve_sources(sources)

    lines = [answer or "(Gemini searched but returned no written answer.)"]
    if sources:
        lines += ["", "Sources:"]
        for index, (title, url) in enumerate(sources, 1):
            lines.append(f"{index}. {title} — {url}" if title else f"{index}. {url}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# URL fetch — a page in, pruned text and content images out
# ─────────────────────────────────────────────
def _fallback_text(body: str) -> str:
    """The old mechanical strip, kept as a last resort.

    Used only when the tree pass renders NOTHING, which means the heuristic met a
    page it does not understand rather than an empty page. Keeping the furniture is
    better than a false "no readable text found" for a page a browser renders in
    full, and the note still says that a pass ran.
    """
    body = re.sub(r"<(script|style|noscript|iframe|svg|form)[^>]*>.*?</\1>",
                  " ", body, flags=re.S | re.I)
    body = re.sub(r"<(p|div|br|li|tr|h[1-6]|section|article)[^>]*>",
                  "\n", body, flags=re.I)
    return _strip_html(body)


def _with_note(text: str) -> str:
    """Cap the text and say that a prune pass ran over it — see PRUNE_NOTE."""
    return _truncate(text, FETCH_BODY_CAP) + "\n\n" + PRUNE_NOTE


def _render_page(body: str, base_url: str) -> tuple[str, list[str]]:
    """A page's HTML → (its pruned text, its content-image URLs).

    Pure — no network — so the whole pass can be pinned on a saved page in tests.
    The pruning is disclosed in the returned text (PRUNE_NOTE) because it is a
    heuristic: text the model never sees must not come back missing without a word,
    or the absence gets reported to the user as a fact about the page.
    """
    try:
        root = _parse_and_prune(body)
    except Exception as exc:
        # html.parser is tolerant, but it is not obliged to be. On a construct it
        # rejects, strip tags mechanically rather than failing the fetch — no tree,
        # so no images, but the text still arrives and the note still stands.
        print(f"[fetch] HTML parse failed ({type(exc).__name__}: {exc}) — "
              f"falling back to a plain tag strip")
        text = _fallback_text(body)
        return (_with_note(text), []) if text else ("", [])

    text = _render_text(root)
    if not text:
        # Nothing survived the prune. That means the page IS its furniture — a link
        # index, an archive, a 404 page — not that the page was empty, so hand it
        # over whole (the note still says a pass ran) with its images taken from the
        # UNPRUNED tree, because that is the page the model is being given. Parsing
        # twice is the price of not keeping two trees alive on the normal path.
        text = _fallback_text(body)
        if not text:
            return "", []
        return _with_note(text), _images_from_tree(_build_tree(body), base_url)

    return _with_note(text), _images_from_tree(root, base_url)


def _readable_text(body: str) -> str:
    """Visible text of an HTML document, with the page furniture pruned out.

    Exactly what a fetch of this page would hand the model — same pass, same note.
    """
    return _render_page(body, "")[0]


def _fetch_page(url: str) -> tuple[str, list[str]]:
    """Download a URL → (pruned visible text, content-image URLs).

    Images come out of the PRUNED tree, so a logo that lived in the site header is
    gone without needing a logo heuristic."""
    print("="*20)
    print(f"[_fetch_page]: debug url: {url[:120]}")
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        ctype = resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8", "replace")

    if "html" in ctype or body.lstrip().lower().startswith(("<", "<!doctype")):
        return _render_page(body, url)

    # plain text / json / etc. — no tags to prune, so no note either
    return _truncate(body, FETCH_BODY_CAP), []


# ─────────────────────────────────────────────
# Media extraction — content images as MCP image blocks
# ─────────────────────────────────────────────
# Lazy-loading spellings first: a page that lazy-loads puts a placeholder in `src`
# and the real image in `data-src`, so reading `src` first fetches a 1x1 gif.
_IMG_SRC_ATTRS = ("data-src", "data-lazy-src", "data-original",
                  "data-srcset", "srcset", "src")


def _img_src(node: _DomNode) -> str:
    for name in _IMG_SRC_ATTRS:
        raw = (node.attrs.get(name) or "").strip()
        if not raw:
            continue
        if name.endswith("srcset"):
            # "a.jpg 1x, b.jpg 2x" — the first candidate is the one a browser would
            # use at the smallest size, i.e. the real image.
            raw = raw.split(",")[0].strip().split(" ")[0]
        if raw:
            return html_lib.unescape(raw)
    return ""


def _declared_too_small(node: _DomNode) -> bool:
    """True only for a purely numeric width/height below MIN_IMG_DIM.

    `width="100%"` is handled by the bytes filter instead of being read as 100 and
    thrown away — a responsive image is not an icon.
    """
    dims = [int(value) for name in ("width", "height")
            if (value := node.attrs.get(name, "")).isdigit()]
    return bool(dims) and min(dims) < MIN_IMG_DIM


def _images_from_tree(root: _DomNode, base_url: str) -> list[str]:
    """Content-image URLs out of the PRUNED tree, article images ranked first.

    Walking what survived the prune is what removes a logo or a social icon without
    a logo heuristic: the header it lived in is already gone. Ranking is the other
    half of it — the cap is small (FETCH_MAX_IMAGES), and document order alone
    tends to spend it on a hero image and three icons before reaching the diagrams
    in the body, so images inside article/main/figure/picture come first.
    """
    ranked: list[tuple[int, int, str]] = []
    order = 0

    def walk(node: _DomNode, priority: int) -> None:
        nonlocal order
        if node.tag:
            if node.tag in PRUNE_MAIN_TAGS or node.tag in ("figure", "picture"):
                priority = 1
            if node.tag == "img" and not _declared_too_small(node):
                src = _img_src(node)
                if src:
                    ranked.append((priority, order, src))
                    order += 1
            for child in node.children:
                walk(child, priority)

    walk(root, 0)
    ranked.sort(key=lambda item: (-item[0], item[1]))

    urls: list[str] = []
    for _priority, _order, src in ranked:
        low = src.lower()
        if src.startswith("data:") or low.split("?")[0].endswith(".svg"):
            continue
        if any(hint in low for hint in IMG_BAD_HINTS):
            continue
        # percent-encode non-ASCII (e.g. CJK filenames) — urllib's request
        # machinery requires an ASCII-safe URL string
        abs_url = urllib.parse.quote(urllib.parse.urljoin(base_url, src),
                                     safe=":/?#[]@!$&'()*+,;=%")
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

        # The page goes back as-is (already pruned, noted and capped by
        # _fetch_page). The caller is an LLM with a large context window, so reading
        # and summarising is its job — this tool only fetches.
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
                        "on — the publishers' own addresses, already resolved, so quote "
                        "them as they stand; open one with web_fetch when the "
                        "summary is not enough. web_fetch(url) opens a single page "
                        "directly and returns it as readable plain text plus its "
                        "content images; use it whenever the user gives you "
                        "a URL or you already know the address, and it needs no "
                        "search. Its text is the page with the HTML tags removed "
                        "AND the page furniture pruned out — navigation, cookie "
                        "notices, footers — and it ends by saying so. Treat it as "
                        "the page's main content, not as a copy of everything on "
                        "the screen, and do the summarising yourself; if something "
                        "the user can see is missing, that is why. If "
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
    print("Fetch               : no key, no model — page text pruned by HTMLParser "
          "+ content images")
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)