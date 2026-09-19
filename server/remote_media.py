"""Remote media — showing a video the server never downloads.

``display_media`` and ``reduce_video_frames`` used to be local-only: a path, or a
``/memory/`` URL. A link to a video on the web therefore had no way in at all —
which is why an ``.mp4`` handed to ``web_fetch`` came back as a page of mojibake,
and why "find me a video of X and show it to me" was unsatisfiable even though the
search half worked.

Three things make a URL workable without the file:

* **Metadata is an HTTP question.** One ``HEAD`` answers the content type, the
  length and whether byte ranges are honoured. Nothing has to be downloaded to
  decide whether a link is a video, a web page, or a paywall.

* **A poster and a few frames need a decoder, not the file.** ``ffmpeg -ss <t> -i
  <url> -frames:v 1`` seeks to the bytes around ``<t>`` and decodes one picture, so
  eight samples of a 20-minute 1080p source cost a few megabytes instead of a few
  gigabytes.

* **Playback is the browser's job.** The player is handed the original URL, with a
  thin local proxy behind it for hosts that refuse a browser request. That proxy
  streams bytes through without ever holding the file.

What the server keeps is a manifest: the canonical URL, the content type, the
duration, the dimensions, a poster and the sampled frames. The tool writes those
into the conversation with :meth:`server.media.MediaStore.save_bytes` — the same
call every other attachment goes through. This is ``display_media`` with one more
entry point, not a second media subsystem.

The byte ceiling is *enforced*, not measured afterwards. Every read the server
performs is charged to a :class:`ByteBudget`, and the decoder reads the source
through :func:`proxy_base` so ffmpeg's own traffic is charged too. A probe that
would pass the ceiling is cut off mid-stream and reported, so "a 20-minute 1080p
source" cannot quietly turn into a 2 GB download.

Everything here is synchronous and network-bound. Callers on the event loop must
hand it to a worker thread (``asyncio.to_thread``), which is what the tools do.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence

logger = logging.getLogger("deepseek_ui.remote_media")

__all__ = [
    "RemoteMediaError",
    "RemoteLimits",
    "RemoteMedia",
    "ByteBudget",
    "classify_url",
    "resolve_remote",
    "proxy_base",
    "shutdown_proxy",
    "wait_for_proxy",
    "sources",
    "forget",
    "is_remote_url",
    "TRANSPORT",
]

#: One `resolve_remote` call has to finish inside its caller's own timeout, because
#: the MCP dispatcher gives a tool 120 s and a call that overruns does not merely
#: fail — it tears the session down. Every deadlineable step below is computed against
#: a single instant, so "HTTP probe + index + poster + eight seeks" cannot add up to
#: more than this, whatever the individual caps say.
DEFAULT_TOTAL_BUDGET = 90.0

#: Fetching as a browser is not deception, it is the difference between a working
#: probe and a 403. Plenty of CDNs answer a bare urllib request with "go away" and
#: the same request from a normal UA with the file.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

#: The proxy is a pipe, so 64 KB is only about how often a socket is written to.
_CHUNK = 64 * 1024

#: A manifest is small — a playlist, or an XML document. Reading more of one than
#: this is reading someone else's live TV guide.
_MANIFEST_PREFIX = 8 * 1024

#: How much of an HTML page is worth reading to look for `og:video`. Anything that
#: declares its video later than this in the document does not declare it usefully.
_PAGE_PREFIX = 96 * 1024

_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",
    "video/matroska",
    "video/mpeg",
    "video/ogg",
    "video/3gpp",
    "video/3gpp2",
    "video/x-msvideo",
    "video/avi",
    "video/x-ms-wmv",
    "video/x-flv",
    "video/mp2t",
    "video/x-m4v",
    "application/mp4",
    "application/ogg",
}

_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "audio/aac",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/flac",
    "audio/x-flac",
}

#: Segmented streams. Not a file: a playlist listing files.
_MANIFEST_TYPES = {
    "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
    "audio/mpegurl",
    "audio/x-mpegurl",
    "application/dash+xml",
    "application/vnd.ms-sstr+xml",
}

_VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".mov", ".mkv", ".ogv", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}
_AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga", ".opus", ".flac"}
_MANIFEST_EXTS = {".m3u8", ".mpd", ".ism", ".m3u"}

_PAGE_TYPES = {"text/html", "application/xhtml+xml"}

#: Names that mean "the stream is encrypted and only a licensed player may decode
#: it". Reported differently from a 403, because the fix is different: there is no
#: fix, and pretending otherwise wastes the user's time.
_DRM_HINTS = (
    "widevine",
    "playready",
    "com.widevine",
    "com.microsoft.playready",
    "com.apple.fps",
    "fairplay",
    "sample-aes",
    "urn:uuid:edef8ba9",
    "urn:uuid:9a04f079",
    "cenc",
    "clearkey",
)


class RemoteMediaError(RuntimeError):
    """A remote source that could not become a block, with a reason to act on.

    ``kind`` is the part the caller branches on:

    ``blocked``     the source refused us (401/403/451) or will not be embedded
    ``drm``         the stream is encrypted; only a licensed player can decode it
    ``unsupported`` a real media type this player cannot open (HLS, DASH)
    ``not_media``   the URL is a web page, or nothing at all
    ``too_big``     answering would cost more than the byte ceiling allows
    ``network``     the host could not be reached, or answered with an error
    ``decode``      fetched, but no decoder could read a frame out of it
    """

    def __init__(self, message: str, *, kind: str = "error", status: int = 0, url: str = ""):
        super().__init__(message)
        self.kind = kind
        self.status = int(status or 0)
        self.url = url

    def __str__(self) -> str:  # pragma: no cover - inherited behaviour, kept explicit
        return str(self.args[0])


# ── limits ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RemoteLimits:
    """What a probe is allowed to cost, in bytes and in seconds.

    The default byte ceiling is deliberately far below a single minute of 1080p
    H.264 (~5 MB at 8 Mbit/s): the whole point is that sampling is a *seek-and-
    decode* operation, so a source needing more than this is a source we are
    reading the wrong way.
    """

    #: Total bytes the server may pull to produce metadata, a poster and frames.
    max_bytes: int = 24 * 1024 * 1024
    #: Wall-clock ceiling for one HTTP request (HEAD/range/thumbnail).
    timeout: float = 20.0
    #: Wall-clock ceiling for the metadata probe, which walks to the container index.
    probe_timeout: float = 45.0
    #: Wall-clock ceiling for one decoded frame.
    frame_timeout: float = 25.0
    #: Longest edge of a stored poster or frame.
    frame_max_dim: int = 512
    #: Ceiling on frames, whatever the caller asks for.
    frames_max: int = 8
    #: One thumbnail may not be bigger than this (they are ~20 KB in practice).
    thumb_max_bytes: int = 4 * 1024 * 1024
    #: Serve playback through the local proxy. Off means "hand the browser the
    #: original URL and nothing else", which is what a test wants.
    proxy: bool = True
    #: Follow one `og:video` hop out of an HTML page.
    follow_pages: bool = True
    #: Permit `localhost` and LAN addresses. Off by default: the two URLs this tool
    #: would otherwise be pointed at on a developer's machine all day are this
    #: server itself and a local model daemon, and neither is a video.
    allow_private: bool = False
    #: Wall-clock ceiling for one `resolve_remote` call, all steps included.
    total_timeout: float = DEFAULT_TOTAL_BUDGET

    @classmethod
    def from_settings(cls, settings: Any) -> "RemoteLimits":
        """Read the knobs off a :class:`server.settings.Settings` instance."""

        def pick(name: str, default: Any) -> Any:
            value = getattr(settings, name, None)
            return default if value is None else value

        return cls(
            max_bytes=int(pick("remote_media_max_bytes", cls.max_bytes)),
            timeout=float(pick("remote_media_timeout", cls.timeout)),
            probe_timeout=float(pick("remote_media_probe_timeout", cls.probe_timeout)),
            frame_max_dim=int(pick("remote_frame_max_dim", cls.frame_max_dim)),
            frames_max=int(pick("remote_frames_max", cls.frames_max)),
            proxy=bool(pick("remote_media_proxy", cls.proxy)),
            allow_private=bool(pick("remote_media_allow_private", cls.allow_private)),
            total_timeout=float(pick("remote_media_total_timeout", cls.total_timeout)),
        )


class ByteBudget:
    """A running total with a hard stop.

    Raising on the offending read rather than returning a flag is what makes the
    ceiling real: the read loop cannot forget to check, and the failure carries the
    numbers that explain the refusal.
    """

    __slots__ = ("limit", "used")

    def __init__(self, limit: int | None):
        self.limit = int(limit) if limit else None
        self.used = 0

    def charge(self, count: int) -> None:
        self.used += int(count)
        if self.limit is not None and self.used > self.limit:
            raise RemoteMediaError(
                f"this source needed more than the {_mb(self.limit)} limit for a "
                f"remote video ({_mb(self.used)} and counting). Sampling is supposed "
                "to read a few hundred kilobytes per frame; a source that costs "
                "more than this is usually a segmented stream or a host ignoring "
                "byte ranges — download the file and attach it instead.",
                kind="too_big",
            )

    @property
    def left(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)


def _mb(count: int) -> str:
    if count >= 1024 * 1024:
        return f"{count / 1048576:.1f} MB"
    if count >= 1024:
        return f"{count / 1024:.0f} KB"
    return f"{count} B"


# ── HTTP ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Head:
    """What a ``HEAD`` (or a one-byte ranged GET) told us."""

    url: str
    status: int
    content_type: str
    length: int | None
    ranges: bool
    prefix: bytes = b""


class Transport:
    """The one HTTP primitive everything else is built on.

    ``open`` deliberately returns what urllib returns — a response object with
    ``status``/``headers``/``read``. Tests replace the module-level :data:`TRANSPORT`
    with an object that returns a fake with the same three attributes, which is how
    the whole of this file is tested without a network.
    """

    def open(self, url: str, *, headers: dict[str, str] | None = None, method: str = "GET", timeout: float = 20.0):
        request = urllib.request.Request(url, headers=dict(headers or {}), method=method)
        return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - scheme checked by the caller


#: Swapped out in tests. One module-level seam, because a per-call seam would let a
#: code path forget to pass it and quietly become untestable.
TRANSPORT: Transport = Transport()


def _fail_open(exc: BaseException, url: str) -> RemoteMediaError:
    """Translate a urllib failure into something worth showing a person."""
    if isinstance(exc, urllib.error.HTTPError):
        return _fail_status(exc.code, url)
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return RemoteMediaError(
                f"{_host(url)} took too long to answer. It may be rate limiting this "
                "machine, or the link may be an expired signed URL.",
                kind="network",
                url=url,
            )
        if isinstance(reason, socket.gaierror):
            return RemoteMediaError(f"could not resolve {_host(url)}.", kind="network", url=url)
        if isinstance(reason, (ConnectionRefusedError, ConnectionResetError)):
            return RemoteMediaError(
                f"{_host(url)} refused the connection. It may be offline, or the host "
                "may block requests that did not come from a browser session.",
                kind="network",
                url=url,
            )
        return RemoteMediaError(f"could not reach {_host(url)}: {reason}", kind="network", url=url)
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return RemoteMediaError(f"{_host(url)} took too long to answer.", kind="network", url=url)
    return RemoteMediaError(f"could not reach {_host(url)}: {exc}", kind="network", url=url)


def _fail_status(code: int, url: str) -> RemoteMediaError:
    """A status code, said out loud. Refusal is never reported as emptiness."""
    host = _host(url)
    if code in (401, 407):
        return RemoteMediaError(
            f"{host} wants a login (HTTP {code}). A link that only works inside a "
            "signed-in session — or behind a paywall — cannot be opened from here. "
            "Download the file, or save the page's own video URL, and attach that "
            "instead.",
            kind="blocked",
            status=code,
            url=url,
        )
    if code == 403:
        return RemoteMediaError(
            f"{host} refused the request (HTTP 403). Sources do this when the link "
            "has expired, when the Referer is missing, or when the request comes "
            "from a region they block. A freshly copied URL from the page itself "
            "often works; so does attaching a downloaded copy.",
            kind="blocked",
            status=code,
            url=url,
        )
    if code == 451:
        return RemoteMediaError(
            f"{host} says this is unavailable for legal reasons (HTTP 451).",
            kind="blocked",
            status=code,
            url=url,
        )
    if code == 429:
        return RemoteMediaError(
            f"{host} is rate limiting this machine (HTTP 429). Wait a moment and try "
            "the same URL again — nothing about the request needs to change.",
            kind="blocked",
            status=code,
            url=url,
        )
    if code in (404, 410):
        return RemoteMediaError(
            f"{host} has nothing at that URL (HTTP {code}). Check the link — search "
            "results and shares tend to carry the page, not the file.",
            kind="not_media",
            status=code,
            url=url,
        )
    if code == 416:
        return RemoteMediaError(
            f"{host} rejected a byte range (HTTP 416). The host advertises range "
            "support but does not honour it, so frames cannot be sampled "
            "efficiently. Download the file and attach it.",
            kind="unsupported",
            status=code,
            url=url,
        )
    return RemoteMediaError(f"{host} answered HTTP {code}.", kind="network", status=code, url=url)


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or url


def _try_open(url: str, *, headers: dict[str, str] | None = None, method: str = "GET", timeout: float = 20.0):
    """``(handle, None)`` or ``(None, exception)``.

    A 404 is data about the world, not a defect, so status codes come back as
    ordinary values and are turned into messages by the caller that knows what it
    was asking for.
    """
    try:
        return TRANSPORT.open(url, headers=headers, method=method, timeout=timeout), None
    except Exception as exc:  # noqa: BLE001 - every failure is translated, none is swallowed
        return None, exc


def _drain(handle: Any, budget: ByteBudget, limit: int) -> bytes:
    """Read at most ``limit`` bytes, charging every one of them to ``budget``."""
    chunks: list[bytes] = []
    got = 0
    while got < limit:
        chunk = handle.read(min(_CHUNK, limit - got))
        if not chunk:
            break
        budget.charge(len(chunk))
        got += len(chunk)
        chunks.append(chunk)
    return b"".join(chunks)


def _header_map(handle: Any) -> dict[str, str]:
    try:
        return {str(k).lower(): str(v) for k, v in handle.headers.items()}
    except Exception:  # noqa: BLE001 - a fake in a test may not have headers at all
        return {}


def _status_of(handle: Any, default: int = 200) -> int:
    return int(getattr(handle, "status", None) or getattr(handle, "code", None) or default)


def _total_length(headers: dict[str, str]) -> int | None:
    """The *whole* size, from either ``Content-Length`` or ``Content-Range``."""
    content_range = headers.get("content-range") or ""
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    raw = headers.get("content-length")
    if raw and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _base_headers(url: str = "") -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "video/*,audio/*,image/*,application/vnd.apple.mpegurl,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        # Video is already compressed; a Content-Encoding would just make the byte
        # accounting a lie.
        "Accept-Encoding": "identity",
    }
    parts = urllib.parse.urlsplit(url)
    if parts.scheme and parts.netloc:
        # Same-origin Referer: the single most common reason a CDN answers 403 to a
        # perfectly good URL is that nothing referred it.
        headers["Referer"] = f"{parts.scheme}://{parts.netloc}/"
    return headers


def _left(deadline: float | None, cap: float) -> float:
    """Seconds left before ``deadline``, clamped into ``[1, cap]``.

    Never returns a non-positive number: a zero timeout is a different thing from a
    short one — it means "give up now" to some libraries and "wait forever" to
    others — and the smallest useful wait is a second either way. ``None`` means
    nobody is holding a deadline, which is a plain uncapped step.
    """
    if deadline is None:
        return max(1.0, cap)
    return max(1.0, min(cap, deadline - time.monotonic()))


def _expired(deadline: float | None, *, reserve: float = 1.5) -> bool:
    """True once there is no longer enough time left to start another decode."""
    return False if deadline is None else time.monotonic() >= deadline - reserve


def _probe(url: str, *, headers: dict[str, str], limits: RemoteLimits, budget: ByteBudget,
           deadline: float | None = None) -> Head:
    """Learn the content type, the length and whether ranges work.

    A ``HEAD`` and no body, unless the host rejects ``HEAD`` (405/501/400) — which
    plenty do — in which case one byte of a range is the polite substitute, and
    that byte is charged like everything else.
    """
    handle, exc = _try_open(
        url, headers=headers, method="HEAD", timeout=_left(deadline, limits.timeout)
    )
    if handle is None and isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 405, 501):
        handle, exc = _try_open(
            url,
            headers={**headers, "Range": "bytes=0-0"},
            method="GET",
            timeout=_left(deadline, limits.timeout),
        )
    if handle is None:
        raise _fail_open(exc, url)

    with closing(handle):
        status = _status_of(handle)
        if status >= 400:
            raise _fail_status(status, url)
        hdrs = _header_map(handle)
        content_type = (hdrs.get("content-type") or "").split(";")[0].strip().lower()
        length = _total_length(hdrs)
        ranges = (hdrs.get("accept-ranges") or "").lower().startswith("bytes") or status == 206
        prefix = b""
        if status == 206 and not hdrs.get("content-range"):
            # Some hosts answer a ranged request with 200 and the whole file. Reading
            # one byte then closes the socket, which is the best available outcome.
            prefix = _drain(handle, budget, 1)
        return Head(
            url=getattr(handle, "url", url) or url,
            status=status,
            content_type=content_type,
            length=length,
            ranges=ranges,
            prefix=prefix,
        )


# ── classification ────────────────────────────────────────────────────────────


def is_remote_url(value: Any) -> bool:
    """True for the URLs this module can act on. Deliberately narrow: no ``file:``."""
    return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))


_YOUTUBE_ID = re.compile(r"(?:v=|/embed/|/shorts/|/live/|/v/)([A-Za-z0-9_-]{6,})")


def classify_url(url: str) -> tuple[str, str, str]:
    """``(provider, id, embed_url)`` for a video host we can embed, else empties.

    Only hosts that publish both an oEmbed record and a stable embed URL are here.
    A host that needs a page scrape or a script to play is not "embeddable", it is
    "hope", and a block that renders a blank iframe is worse than a link.
    """
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    query = parts.query or ""

    if host == "youtu.be" or host.endswith(".youtu.be"):
        video_id = (parts.path.strip("/").split("/") or [""])[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id or ""):
            return "youtube", video_id, f"https://www.youtube-nocookie.com/embed/{video_id}"
        return "", "", ""

    if host == "youtube.com" or host.endswith(".youtube.com"):
        match = _YOUTUBE_ID.search(parts.path or "")
        if match is None:
            match = _YOUTUBE_ID.search("?" + query)  # `?v=` without a leading path segment
        if match:
            return "youtube", match.group(1), f"https://www.youtube-nocookie.com/embed/{match.group(1)}"
        return "", "", ""

    if host == "vimeo.com" or host.endswith(".vimeo.com"):
        match = re.search(r"/(?:video/)?(\d{6,})", parts.path or "")
        if match:
            return "vimeo", match.group(1), f"https://player.vimeo.com/video/{match.group(1)}"
    return "", "", ""


def _extension(url: str) -> str:
    path = urllib.parse.urlsplit(url).path or ""
    suffix = Path(path).suffix.lower()
    if suffix and len(suffix) <= 6:
        return suffix
    return ""


def _kind_for(content_type: str, url: str) -> str:
    """``video`` | ``audio`` | ``manifest`` | ``page`` | ``image`` | ``""``.

    The extension breaks ties: hosts serve MP4 as ``application/octet-stream`` often
    enough that trusting the header alone would refuse real videos.
    """
    ctype = (content_type or "").split(";")[0].strip().lower()
    ext = _extension(url)
    if ctype in _MANIFEST_TYPES or ext in _MANIFEST_EXTS:
        return "manifest"
    if ctype in _VIDEO_TYPES:
        return "video"
    if ctype in _AUDIO_TYPES:
        return "audio"
    if ctype.startswith("video/"):
        return "video"
    if ctype.startswith("audio/"):
        return "audio"
    if ctype.startswith("image/"):
        return "image"
    if ctype in _PAGE_TYPES or ctype.startswith("text/html"):
        return "page"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _MANIFEST_EXTS:
        return "manifest"
    return ""


_OG_META = re.compile(
    r"""<meta\b[^>]*?\b(?:property|name)\s*=\s*["'](og:video(?::secure_url|:url)?|twitter:player:stream|og:image|twitter:image)["'][^>]*?>""",
    re.IGNORECASE | re.DOTALL,
)
_OG_CONTENT = re.compile(r"""\bcontent\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def _page_media(html: str) -> tuple[str, str]:
    """``(video_url, image_url)`` from OpenGraph/Twitter cards.

    Only the first of each, and only when the tag carries a ``content`` attribute —
    a card that half-matches is a card that produces a broken player.
    """
    video = ""
    image = ""
    for tag in _OG_META.finditer(html or ""):
        content_match = _OG_CONTENT.search(tag.group(0))
        if content_match is None:
            continue
        value = content_match.group(1).strip()
        if not value.lower().startswith(("http://", "https://")):
            continue
        name = tag.group(1).lower()
        if name.startswith("og:video") or name == "twitter:player:stream":
            video = video or value
        else:
            image = image or value
    return video, image


# ── the local playback proxy ──────────────────────────────────────────────────


@dataclass
class RemoteSource:
    """A registered upstream URL, plus the accounting for reading it.

    The token is what reaches the browser and what ffmpeg is handed. Two tokens are
    registered for one probe: the analysis token carries the :class:`ByteBudget`,
    the stream token does not. Otherwise eight sampled frames would eat the budget
    and then the player would be cut off mid-playback by accounting for work that
    already finished.
    """

    token: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str = ""
    name: str = ""
    timeout: float = 20.0
    budget: ByteBudget | None = None
    status: int = 0
    error: str = ""
    created: float = 0.0

    @property
    def analysis(self) -> bool:
        return self.budget is not None


_SOURCES: dict[str, RemoteSource] = {}
_SOURCES_LOCK = threading.Lock()

#: Analysis tokens are dropped as soon as a probe finishes, so only playback tokens
#: accumulate — at most two per video the user has actually been shown.
_SOURCE_KEEP = 64


def sources() -> dict[str, RemoteSource]:
    """The live registry. Exposed for tests; not part of the tool surface."""
    return _SOURCES


def _remember(source: RemoteSource) -> str:
    with _SOURCES_LOCK:
        _SOURCES[source.token] = source
        if len(_SOURCES) > _SOURCE_KEEP:
            oldest = sorted(_SOURCES.values(), key=lambda s: s.created)
            for stale in oldest[: len(_SOURCES) - _SOURCE_KEEP]:
                _SOURCES.pop(stale.token, None)
    return source.token


def forget(token: str) -> None:
    with _SOURCES_LOCK:
        _SOURCES.pop(token, None)


class _ProxyHandler(BaseHTTPRequestHandler):
    """Serves one registered source, forwarding ranges, counting bytes.

    This is a pipe and nothing else: no rewriting, no caching, no directory
    listing. The only thing it adds over a redirect is that the request leaves with
    our headers and that its cost is visible.
    """

    protocol_version = "HTTP/1.1"
    server_version = "deepseek-ui-remote"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("remote proxy: " + fmt, *args)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self._serve(with_body=False)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self._serve(with_body=True)

    def _serve(self, *, with_body: bool) -> None:
        token = urllib.parse.urlsplit(self.path).path.strip("/").split("/")[0]
        with _SOURCES_LOCK:
            source = _SOURCES.get(token)
        if source is None:
            self._fail(404, "unknown or expired media token")
            return

        headers = dict(source.headers)
        byte_range = self.headers.get("Range")
        if byte_range:
            headers["Range"] = byte_range

        # Always GET upstream, even for a client HEAD: a host that refuses HEAD is
        # common, and the body is simply not read in that case.
        handle, exc = _try_open(source.url, headers=headers, method="GET", timeout=source.timeout)
        if handle is None:
            source.status = getattr(exc, "code", 0) or 0
            source.error = str(exc)
            self._fail(source.status or 502, f"upstream refused: {source.error}")
            return

        with closing(handle):
            status = _status_of(handle)
            source.status = status
            hdrs = _header_map(handle)
            self.send_response(status)
            for name in (
                "Content-Type",
                "Content-Length",
                "Content-Range",
                "Accept-Ranges",
                "Last-Modified",
                "ETag",
            ):
                value = hdrs.get(name.lower())
                if value:
                    self.send_header(name, value)
            if not hdrs.get("content-type"):
                self.send_header("Content-Type", source.content_type or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            if not with_body or not hdrs.get("content-length"):
                # HTTP/1.1 framing needs a length we can promise. Without one from
                # upstream — a chunked source, or a HEAD — the only honest answer is
                # to close the connection when this response ends.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if with_body:
                self._pump(handle, source)

    def _pump(self, handle: Any, source: RemoteSource) -> None:
        try:
            while True:
                try:
                    chunk = handle.read(_CHUNK)
                except Exception as exc:  # noqa: BLE001 - upstream died mid-stream
                    source.error = str(exc)
                    break
                if not chunk:
                    break
                if source.budget is not None:
                    try:
                        source.budget.charge(len(chunk))
                    except RemoteMediaError as exc:
                        source.error = str(exc)
                        break
                self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The player was closed or seeked away. Not an error, and not worth a
            # log line: scrubbing a video does this several times a second.
            pass
        # A stopped stream with a Content-Length already promised leaves the
        # connection unusable, so it must not be reused for the next request.
        self.close_connection = True

    def _fail(self, status: int, message: str) -> None:
        body = message.encode("utf-8", "replace")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client gone
            pass


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_PROXY: "_ProxyServer | None" = None
_PROXY_FAILED = False
_PROXY_LOCK = threading.Lock()


def proxy_base() -> str:
    """The proxy's base URL, starting it on first use. ``""`` when it cannot run.

    Bound to ``127.0.0.1`` on an ephemeral port: nothing outside this machine can
    reach it, and the tokens in it are random. Failure is not fatal — everything
    degrades to "the browser talks to the origin directly".
    """
    global _PROXY, _PROXY_FAILED
    if _PROXY_FAILED:
        return ""
    with _PROXY_LOCK:
        if _PROXY is None:
            try:
                _PROXY = _ProxyServer(("127.0.0.1", 0), _ProxyHandler)
            except OSError as exc:
                logger.warning("remote media proxy could not start (%s); remote videos "
                               "will be played from their original URL", exc)
                _PROXY_FAILED = True
                return ""
            thread = threading.Thread(target=_PROXY.serve_forever, name="remote-media-proxy", daemon=True)
            thread.start()
        host, port = _PROXY.server_address[:2]
        return f"http://{host}:{port}"


def shutdown_proxy() -> None:
    """Stop the proxy. For tests and shutdown; safe to call twice."""
    global _PROXY
    with _PROXY_LOCK:
        server, _PROXY = _PROXY, None
    if server is not None:
        server.shutdown()
        server.server_close()


def _register(url: str, headers: dict[str, str], *, name: str, content_type: str, timeout: float,
              budget: ByteBudget | None) -> str:
    # Unguessable, because the proxy fetches with our headers and is reachable by
    # anything on this machine that can reach a loopback port.
    token = secrets.token_hex(12)
    return _remember(
        RemoteSource(
            token=token,
            url=url,
            headers=dict(headers),
            content_type=content_type,
            name=name,
            timeout=timeout,
            budget=budget,
            created=time.time(),
        )
    )


def _proxy_url(token: str, limits: RemoteLimits) -> str:
    if not limits.proxy:
        return ""
    base = proxy_base()
    return f"{base}/{token}" if base else ""


# ── decoding ──────────────────────────────────────────────────────────────────


def _ffmpeg(*names: str) -> str:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return ""


def _run(cmd: Sequence[str], timeout: float, *, what: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(list(cmd), capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise RemoteMediaError(
            f"the decoder did not finish {what} within {timeout:.0f}s. A distant or "
            "rate-limited host looks exactly like this; try the URL once more, or "
            "download the file and attach it.",
            kind="network",
        )
    except OSError as exc:
        raise RemoteMediaError(f"could not run {cmd[0]}: {exc}", kind="decode")


def _ffprobe(src: str, limits: RemoteLimits, *, timeout: float | None = None) -> dict[str, Any] | None:
    """Container metadata, or ``None`` when ffprobe is not installed.

    Reads the index and the first packets only — this is the cheap half of the
    probe, and it is what makes ``inspect_media`` cost roughly nothing.
    """
    exe = _ffmpeg("ffprobe", "ffprobe.exe")
    if not exe:
        return None
    proc = _run(
        [
            exe,
            "-v", "error",
            "-rw_timeout", str(int(limits.timeout * 1_000_000)),
            "-print_format", "json",
            "-show_entries",
            "format=duration,format_name:stream=codec_type,codec_name,width,height,nb_frames,duration,avg_frame_rate",
            "-i", src,
        ],
        limits.probe_timeout if timeout is None else timeout,
        what="reading the video's index",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError:
        return None


def _probe_facts(data: dict[str, Any] | None) -> dict[str, Any]:
    """Pull ``duration``/``width``/``height``/``codec`` out of ffprobe's answer."""
    facts: dict[str, Any] = {"duration": None, "width": None, "height": None, "codec": ""}
    if not data:
        return facts
    streams = [s for s in (data.get("streams") or []) if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video:
        facts["width"] = _as_int(video.get("width"))
        facts["height"] = _as_int(video.get("height"))
        facts["codec"] = str(video.get("codec_name") or "")
        facts["duration"] = _as_float(video.get("duration"))
        frames = _as_int(video.get("nb_frames"))
        rate = _parse_rate(video.get("avg_frame_rate"))
        if facts["duration"] is None and frames and rate:
            facts["duration"] = frames / rate
    if facts["duration"] is None:
        facts["duration"] = _as_float((data.get("format") or {}).get("duration"))
    return facts


def _parse_rate(value: Any) -> float | None:
    text = str(value or "")
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            top, bottom = float(num), float(den)
        except ValueError:
            return None
        return top / bottom if bottom else None
    try:
        return float(text) or None
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _decode_one(src: str, at: float, max_dim: int, limits: RemoteLimits, *, timeout: float | None = None) -> bytes | None:
    """One JPEG, decoded from a seek to ``at`` seconds.

    ``-ss`` before ``-i`` is the whole trick: ffmpeg asks for the bytes around that
    timestamp instead of downloading up to it.
    """
    exe = _ffmpeg("ffmpeg", "ffmpeg.exe")
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="dsui-remote-") as work:
        target = Path(work) / "frame.jpg"
        proc = _run(
            [
                exe,
                "-nostdin",
                "-loglevel", "error",
                "-ss", f"{max(0.0, at):.3f}",
                "-i", src,
                "-frames:v", "1",
                "-an", "-sn", "-dn",
                "-vf", f"scale='min({int(max_dim)},iw)':-2",
                "-f", "image2",
                "-c:v", "mjpeg",
                "-q:v", "4",
                "-y", str(target),
            ],
            limits.frame_timeout if timeout is None else timeout,
            what=f"decoding the frame at {at:.1f}s",
        )
        if proc.returncode != 0 or not target.exists():
            logger.debug("frame at %.1fs failed: %s", at, proc.stderr.decode("utf-8", "replace")[:400])
            return None
        data = target.read_bytes()
    return data or None


def _decode_with_opencv(src: str, times: Sequence[float], max_dim: int, limits: RemoteLimits) -> list[bytes]:
    """The fallback path when there is no ffmpeg binary on ``PATH``.

    ``cv2`` is optional and its own build of FFmpeg can open an HTTP URL, so it is
    worth trying before giving up — but it reads far more than ffmpeg's seek does,
    which is why it is second and why the byte budget is what stops it.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional dependency, absence is normal
        return []

    capture = cv2.VideoCapture()
    try:
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(limits.timeout * 1000))
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(limits.frame_timeout * 1000))
        if not capture.open(src):
            return []
        out: list[bytes] = []
        for at in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, at) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, float(max_dim) / max(1, max(width, height)))
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok and encoded is not None:
                out.append(encoded.tobytes())
        return out
    except Exception as exc:  # noqa: BLE001 - OpenCV raises everything as cv2.error or worse
        logger.debug("opencv could not sample %s: %s", src, exc)
        return []
    finally:
        capture.release()


def _frame_times(duration: float | None, count: int) -> list[float]:
    """``count`` sample points that avoid the first and last moments.

    The very first frame of a video is a fade, a title card or black, and the last
    is a fade-out; sampling the interior is the difference between "what happens in
    it" and "what colour the studio ident is".
    """
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    if not duration or duration <= 0:
        return [float(i) for i in range(count)]
    span = max(0.0, duration - 0.5)
    return [span * (i + 0.5) / count for i in range(count)]


def _dedupe(frames: Iterable[bytes]) -> list[bytes]:
    """Drop frames that are byte-identical, preserving order.

    A static shot sampled eight times is one picture; sending eight copies of it to
    the model is eight times the cost for none of the information.
    """
    seen: set[bytes] = set()
    out: list[bytes] = []
    for frame in frames:
        key = frame[:512] + bytes([len(frame) % 251])
        if key in seen:
            continue
        seen.add(key)
        out.append(frame)
    return out


# ── embed hosts ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbedHost:
    """Everything needed to describe a video on a host we cannot download.

    ``frames`` is honest about where the pictures come from: YouTube publishes four
    stills from the video at fixed points, and they are real frames at real
    timestamps. That is a genuinely useful answer for "what happens in it" without
    touching the stream, and it is reported as exactly that rather than as "8
    sampled frames".
    """

    oembed: str
    play: str
    poster: str
    frames: tuple[str, ...] = ()
    frames_note: str = ""


_EMBED_HOSTS: dict[str, EmbedHost] = {
    "youtube": EmbedHost(
        oembed="https://www.youtube.com/oembed?url={url}&format=json",
        play="https://www.youtube-nocookie.com/embed/{id}",
        poster="https://i.ytimg.com/vi/{id}/hqdefault.jpg",
        frames=(
            "https://i.ytimg.com/vi/{id}/hq1.jpg",
            "https://i.ytimg.com/vi/{id}/hq2.jpg",
            "https://i.ytimg.com/vi/{id}/hq3.jpg",
        ),
        frames_note=(
            "YouTube publishes four stills from this video (around 1/8, 3/8, 5/8 and 7/8 "
            "of its length); those are the frames above. The video itself was not fetched "
            "— it can only be played, not downloaded, from here."
        ),
    ),
    "vimeo": EmbedHost(
        oembed="https://vimeo.com/api/oembed.json?url={url}",
        play="https://player.vimeo.com/video/{id}",
        poster="",
    ),
}


def _fetch_bytes(
    url: str,
    *,
    budget: ByteBudget,
    limit: int,
    timeout: float,
) -> tuple[bytes, str]:
    """``(bytes, content_type)`` or an empty pair. Used for thumbnails only."""
    handle, exc = _try_open(url, headers=_base_headers(url), timeout=timeout)
    if handle is None:
        logger.debug("thumbnail %s failed: %s", url, exc)
        return b"", ""
    with closing(handle):
        if _status_of(handle) >= 400:
            return b"", ""
        content_type = (_header_map(handle).get("content-type") or "").split(";")[0].strip().lower()
        try:
            return _drain(handle, budget, limit), content_type
        except RemoteMediaError:
            raise
        except Exception as exc:  # noqa: BLE001 - a thumbnail is never worth failing a probe over
            logger.debug("thumbnail %s read failed: %s", url, exc)
            return b"", ""


def _resolve_embed(
    url: str,
    provider: str,
    video_id: str,
    embed_url: str,
    *,
    limits: RemoteLimits,
    want_frames: int,
    deadline: float | None = None,
) -> RemoteMedia:
    host = _EMBED_HOSTS[provider]
    deadline = deadline if deadline is not None else time.monotonic() + limits.total_timeout
    budget = ByteBudget(limits.max_bytes)
    title = author = ""
    oembed: dict[str, Any] = {}

    query = urllib.parse.quote(url, safe="")
    handle, exc = _try_open(
        host.oembed.format(url=query, id=video_id),
        headers=_base_headers(url),
        timeout=_left(deadline, limits.timeout),
    )
    if handle is None:
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404, 451):
            raise RemoteMediaError(
                f"{provider} will not serve metadata for that video (HTTP {exc.code}) — "
                "which is what private, deleted, age-restricted and region-blocked "
                "videos look like from here. Open the page in a browser to check which "
                "one it is.",
                kind="blocked",
                status=exc.code,
                url=url,
            )
        logger.debug("oembed for %s failed (%s); continuing without a title", url, exc)
    else:
        with closing(handle):
            if _status_of(handle) < 400:
                try:
                    oembed = json.loads(_drain(handle, budget, 64 * 1024).decode("utf-8", "replace"))
                except (ValueError, RemoteMediaError):
                    oembed = {}
        if isinstance(oembed, dict):
            title = str(oembed.get("title") or "")
            author = str(oembed.get("author_name") or "")

    poster_url = str(oembed.get("thumbnail_url") or "") or host.poster.format(id=video_id)
    poster, poster_type = _fetch_bytes(
        poster_url, budget=budget, limit=limits.thumb_max_bytes, timeout=_left(deadline, limits.timeout)
    )

    frames: list[bytes] = []
    note_bits: list[str] = []
    if want_frames > 0 and host.frames:
        fetched: list[bytes] = []
        for template in host.frames[: max(0, want_frames)]:
            data, _ = _fetch_bytes(
                template.format(id=video_id),
                budget=budget,
                limit=limits.thumb_max_bytes,
                timeout=_left(deadline, limits.timeout),
            )
            if data and _looks_like_image(data, poster_type):
                fetched.append(data)
        # Deduplicated for the same reason the direct path is: a video that is one
        # colour must not be described as four different frames.
        frames = _dedupe(fetched)
        if frames:
            note_bits.append(host.frames_note)
            if len(frames) < len(fetched):
                extra = len(fetched) - len(frames)
                note_bits.append(
                    f"{extra} of the {len(fetched)} stills published for this video were "
                    f"identical, so {extra} fewer {'is' if extra == 1 else 'are'} shown."
                )
    elif want_frames > 0:
        note_bits.append(
            f"{provider} publishes one still image for this video and no frame samples, so "
            "only the poster is available. The video itself cannot be read without "
            "downloading it, which is not something this tool does."
        )

    if not poster:
        note_bits.append(
            "No poster frame could be fetched for this embed; the player will show its "
            "own placeholder."
        )

    return RemoteMedia(
        url=url,
        kind="embed",
        content_type="text/html",
        name=title or f"{provider} video {video_id}",
        duration=_as_float(oembed.get("duration")) if isinstance(oembed, dict) else None,
        width=_as_int(oembed.get("width") if isinstance(oembed, dict) else None),
        height=_as_int(oembed.get("height") if isinstance(oembed, dict) else None),
        poster=poster or None,
        frames=frames,
        provider=provider,
        video_id=video_id,
        embed_url=embed_url or host.play.format(id=video_id),
        title=title,
        author=author,
        playable=True,
        note=" ".join(note_bits),
        bytes_read=budget.used,
    )


def _looks_like_image(data: bytes, content_type: str = "") -> bool:
    if not data:
        return False
    if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    return content_type.startswith("image/")


# ── direct media ──────────────────────────────────────────────────────────────


def _manifest_failure(
    url: str, *, headers: dict[str, str], limits: RemoteLimits, budget: ByteBudget, deadline: float
) -> NoReturn:
    """Raise the right error for an HLS/DASH playlist, DRM or not.

    Checked by reading the playlist, because "is this DRM" is not a header. The
    distinction matters: an encrypted stream is not coming, and a plain one is
    merely not a file.
    """
    kind = "unsupported"
    prefix = ""
    handle, _ = _try_open(url, headers=headers, timeout=_left(deadline, limits.timeout))
    if handle is not None:
        with closing(handle):
            if _status_of(handle) < 400:
                try:
                    prefix = _drain(handle, budget, _MANIFEST_PREFIX).decode("utf-8", "replace")
                except RemoteMediaError:
                    prefix = ""
                except Exception as exc:  # noqa: BLE001 - the refusal below is the answer either way
                    logger.debug("manifest prefix read failed: %s", exc)

    lowered = prefix.lower()
    hit = next((hint for hint in _DRM_HINTS if hint in lowered), "")
    if hit:
        raise RemoteMediaError(
            f"this stream is encrypted for a licensed player (found '{hit}' in its "
            "playlist). Widevine, PlayReady and FairPlay content cannot be decoded "
            "here at all — not by the player and not for frame samples. Use a source "
            "that is not DRM protected.",
            kind="drm",
            url=url,
        )
    if "widevine" in lowered or "playready" in lowered:  # pragma: no cover - covered above
        raise RemoteMediaError("this stream is DRM protected.", kind="drm", url=url)

    raise RemoteMediaError(
        "this is a segmented stream (an .m3u8 or .mpd playlist), not a video file. A "
        "browser <video> element cannot open one, and the individual segments live at "
        "URLs the playlist hides from anything but a player that follows it. Open the "
        "URL in a browser, or download the file and attach it.",
        kind="unsupported",
        url=url,
    )


def _resolve_page(
    url: str,
    *,
    headers: dict[str, str],
    limits: RemoteLimits,
    budget: ByteBudget,
    poster_wanted: bool,
    depth: int,
    deadline: float,
) -> RemoteMedia | None:
    """Follow an HTML page's ``og:video``, or take its ``og:image`` if that is all it has.

    Returns ``None`` when the page declares nothing, so the caller can say so — an
    empty result and a refusal must never look the same.
    """
    if not limits.follow_pages or depth > 1:
        return None
    handle, exc = _try_open(url, headers=headers, timeout=_left(deadline, limits.timeout))
    if handle is None:
        raise _fail_open(exc, url)
    with closing(handle):
        if _status_of(handle) >= 400:
            raise _fail_status(_status_of(handle), url)
        html = _drain(handle, budget, _PAGE_PREFIX).decode("utf-8", "replace")

    video_url, image_url = _page_media(html)
    if video_url:
        embedded = _resolve_direct(
            video_url,
            limits=limits,
            budget=budget,
            poster_wanted=poster_wanted,
            frames_wanted=0,
            target_fps=1.0,
            depth=depth + 1,
            deadline=deadline,
            note=f"The page at {url} declares this video, so that is what is shown.",
        )
        return embedded
    if image_url:
        data, content_type = _fetch_bytes(
            image_url, budget=budget, limit=limits.thumb_max_bytes, timeout=_left(deadline, limits.timeout)
        )
        if data:
            return RemoteMedia(
                url=image_url,
                kind="image",
                content_type=content_type or "image/jpeg",
                name=Path(urllib.parse.urlsplit(image_url).path).name or "image",
                poster=data,
                playable=False,
                note=(
                    f"{url} is a web page with no video in it — it declares the image above. "
                    "Shown because it is the closest thing on the page to what was asked for."
                ),
                bytes_read=budget.used,
            )
    return None


def _resolve_direct(
    url: str,
    *,
    limits: RemoteLimits,
    budget: ByteBudget,
    poster_wanted: bool,
    frames_wanted: int,
    target_fps: float,
    depth: int = 0,
    note: str = "",
    deadline: float | None = None,
) -> RemoteMedia:
    headers = _base_headers(url)
    deadline = deadline if deadline is not None else time.monotonic() + limits.total_timeout
    head = _probe(url, headers=headers, limits=limits, budget=budget, deadline=deadline)
    kind = _kind_for(head.content_type, head.url)

    if kind == "page":
        page = _resolve_page(
            head.url,
            headers=headers,
            limits=limits,
            budget=budget,
            poster_wanted=poster_wanted,
            depth=depth,
            deadline=deadline,
        )
        if page is not None:
            return page
        raise RemoteMediaError(
            f"that URL is a web page, not a video ({_host(url)} served "
            f"{head.content_type or 'text/html'}). Nothing on the page declares a "
            "video or an image in its metadata, so there is nothing here to show. Use "
            "web_fetch to read it, or find the link to the video itself.",
            kind="not_media",
            url=url,
        )

    if kind == "manifest":
        _manifest_failure(head.url, headers=headers, limits=limits, budget=budget, deadline=deadline)
        raise AssertionError("unreachable: _manifest_failure always raises")

    if kind == "image":
        # `display_media` on an image URL is a normal request; hand back the bytes.
        data, content_type = _fetch_bytes(
            head.url, budget=budget, limit=limits.max_bytes, timeout=_left(deadline, limits.timeout)
        )
        if not data:
            raise RemoteMediaError(f"{_host(url)} served an image that could not be read.", kind="decode", url=url)
        return RemoteMedia(
            url=head.url,
            kind="image",
            content_type=content_type or head.content_type or "image/jpeg",
            name=Path(urllib.parse.urlsplit(head.url).path).name or "image",
            length=head.length,
            poster=data,
            playable=False,
            note=note,
            bytes_read=budget.used,
        )

    if kind == "":
        described = head.content_type or "no content type"
        raise RemoteMediaError(
            f"{_host(url)} served {described}, which is not a video or an image. If it "
            "is a web page, web_fetch can read it; if it is a file, attach it or give a "
            "path to it.",
            kind="not_media",
            url=url,
        )

    # ── it is a video or an audio file ────────────────────────────────────────
    name = _name_for(head.url, head.content_type, kind)
    stream_token = _register(
        head.url,
        headers,
        name=name,
        content_type=head.content_type,
        timeout=limits.timeout,
        budget=None,
    )
    probe_token = _register(
        head.url,
        headers,
        name=name,
        content_type=head.content_type,
        timeout=limits.timeout,
        budget=budget,
    )
    try:
        return _sample(
            head,
            kind=kind,
            name=name,
            headers=headers,
            limits=limits,
            budget=budget,
            poster_wanted=poster_wanted,
            frames_wanted=frames_wanted,
            target_fps=target_fps,
            stream_token=stream_token,
            probe_token=probe_token,
            note=note,
            deadline=deadline,
        )
    except BaseException:
        # Nothing may keep a token for a probe that failed: the registry is what the
        # proxy serves from, and a half-registered source is a URL reachable from the
        # browser with no manifest behind it.
        forget(stream_token)
        raise
    finally:
        # The analysis token exists only for this call. Keeping it would leave a
        # budgeted URL reachable from the browser, i.e. a player that stops after
        # one ceiling's worth of bytes.
        forget(probe_token)


def _sample(
    head: Head,
    *,
    kind: str,
    name: str,
    headers: dict[str, str],
    limits: RemoteLimits,
    budget: ByteBudget,
    poster_wanted: bool,
    frames_wanted: int,
    target_fps: float,
    stream_token: str,
    probe_token: str,
    note: str,
    deadline: float,
) -> RemoteMedia:
    """Pull the poster and the frames for a source already classified as media.

    Everything the decoder reads goes through the analysis token, i.e. through the
    local proxy, so the byte ceiling in ``budget`` covers ffmpeg's traffic and not
    merely our own HTTP calls.
    """
    proxy = _proxy_url(probe_token, limits)
    source = proxy or head.url
    bits = [note] if note else []

    facts: dict[str, Any] = {"duration": None, "width": None, "height": None, "codec": ""}
    probed = _ffprobe(source, limits, timeout=_left(deadline, limits.probe_timeout))
    if probed is not None:
        facts = _probe_facts(probed)

    want = _sample_count(frames_wanted, facts["duration"], target_fps, limits)
    poster: bytes | None = None
    frames: list[bytes] = []
    skipped = 0

    if kind == "video" and (poster_wanted or want):
        # A tenth of the way in: past the studio ident and the fade from black, and
        # still representative. Without a duration, the first frame is the only
        # honest choice.
        at = facts["duration"] * 0.1 if facts["duration"] else 0.0
        poster = _decode_one(source, at, limits.frame_max_dim, limits, timeout=_left(deadline, limits.frame_timeout))

    if kind == "video" and want:
        times = _frame_times(facts["duration"], want)
        decoded: list[bytes] = []
        for at in times:
            if _expired(deadline):
                # Stopping early is the right failure here: the alternative is a call
                # that outlives the dispatcher that is waiting for it, which in this
                # server does not fail the call, it drops the session.
                skipped = want - len(decoded)
                break
            frame = _decode_one(source, at, limits.frame_max_dim, limits, timeout=_left(deadline, limits.frame_timeout))
            if frame:
                decoded.append(frame)
        if not decoded and not skipped:
            decoded = _decode_with_opencv(source, times, limits.frame_max_dim, limits)
        frames = _dedupe(decoded)

    if skipped:
        bits.append(
            f"{skipped} of the {want} frames were skipped to stay inside the "
            f"{limits.total_timeout:.0f}-second limit for a remote probe. Lower "
            "target_fps or max_frames, or download the file and attach it."
        )

    if kind == "video" and want and not frames and poster is None:
        raise RemoteMediaError(
            f"{_host(head.url)} served {head.content_type or 'a video'} but no frame could "
            "be decoded from it. The container may be truncated, or its codec may need "
            "a decoder this machine does not have (ffmpeg was tried first, then "
            "OpenCV). Download the file and attach it, and the local tools can read it.",
            kind="decode",
            url=head.url,
        )
    if want and len(frames) < want:
        bits.append(
            f"{len(frames)} of {want} requested frames came back distinct — the rest were "
            "identical, so they would have cost tokens without adding anything."
        )
    if kind == "audio" and want:
        bits.append("Audio has no frames to sample; inspect_media reports its details instead.")

    stream_url = _proxy_url(stream_token, limits)
    if not stream_url:
        bits.append(
            "No local stream proxy is available, so the player will request the original "
            "URL directly; some hosts refuse that."
        )
    return RemoteMedia(
        url=head.url,
        kind=kind,
        content_type=head.content_type,
        name=name,
        duration=_as_float(facts.get("duration")),
        width=_as_int(facts.get("width")),
        height=_as_int(facts.get("height")),
        length=head.length,
        poster=poster,
        frames=frames,
        stream_url=stream_url,
        codec=str(facts.get("codec") or ""),
        ranges=head.ranges,
        note=" ".join(bit for bit in bits if bit),
        bytes_read=budget.used,
    )


def _sample_count(requested: int, duration: float | None, target_fps: float, limits: RemoteLimits) -> int:
    """How many frames to pull.

    "Scaled by duration": the rate is frames-per-second *of source*, so a ten-second
    clip gets ten samples and a twenty-minute film gets the ceiling rather than twelve
    hundred. The ceiling defaults to 8, which is where "a handful is enough to say
    what happens in it" lands. A request of zero — the "just show me" case — is zero:
    playback costs no tokens at all.
    """
    if requested <= 0:
        return 0
    ceiling = max(1, min(int(limits.frames_max), int(requested)))
    if not duration or duration <= 0 or target_fps <= 0:
        return ceiling
    natural = int(round(duration * target_fps))
    return max(1, min(ceiling, natural))


def _name_for(url: str, content_type: str, kind: str) -> str:
    leaf = Path(urllib.parse.urlsplit(url).path).name
    if leaf and "." in leaf and len(leaf) <= 120:
        return urllib.parse.unquote(leaf)
    ext = {
        "video": {"video/webm": ".webm", "video/quicktime": ".mov", "video/x-matroska": ".mkv"}.get(content_type, ".mp4"),
        "audio": {"audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/ogg": ".ogg"}.get(
            content_type, ".m4a"
        ),
    }.get(kind, "")
    return f"remote{ext or '.bin'}"


# ── the entry point ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class RemoteMedia:
    """The manifest: what the URL is, and the few pictures that describe it.

    ``url`` is the *original* source. The player is meant to use it, and
    ``stream_url`` only as a fallback for hosts that refuse a browser. Nothing here
    is a copy of the file.
    """

    url: str
    kind: str
    content_type: str = ""
    name: str = ""
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    length: int | None = None
    #: One JPEG, describing the video at about a tenth of the way in.
    poster: bytes | None = None
    #: Frames the caller asked for. Always JPEG, and only ever a handful.
    frames: list[bytes] = field(default_factory=list)
    #: The local proxy's URL for this source. ``""`` when the proxy is unavailable,
    #: in which case the player has only the original.
    stream_url: str = ""
    codec: str = ""
    ranges: bool = False
    playable: bool = True
    provider: str = ""
    video_id: str = ""
    embed_url: str = ""
    title: str = ""
    author: str = ""
    note: str = ""
    bytes_read: int = 0

    @property
    def is_video(self) -> bool:
        return self.kind == "video"

    def facts(self) -> dict[str, Any]:
        """The JSON `inspect_media` prints. Deliberately the same keys as a local probe."""
        info: dict[str, Any] = {
            "name": self.name,
            "url": self.url,
            "kind": self.kind,
            "source": "remote",
            "content_type": self.content_type,
            "bytes": self.length or 0,
            "bytes_fetched": self.bytes_read,
        }
        if self.kind == "embed":
            info.update(
                {
                    "provider": self.provider,
                    "video_id": self.video_id,
                    "embed_url": self.embed_url,
                    "title": self.title,
                    "author": self.author,
                }
            )
        if self.duration:
            info["duration"] = round(self.duration, 3)
            info["duration_text"] = _format_duration(self.duration)
        if self.width and self.height:
            info["width"] = self.width
            info["height"] = self.height
        if self.codec:
            info["codec"] = self.codec
        if self.kind in ("video", "audio"):
            info["range_requests"] = bool(self.ranges)
            info["stream"] = bool(self.stream_url)
        if self.note:
            info["note"] = self.note
        return info

    def block(self, *, poster_url: str = "", frame_urls: Sequence[str] = ()) -> dict[str, Any]:
        """The display block for this source. ``poster_url`` is a ``/memory/`` URL."""
        common: dict[str, Any] = {"name": self.name, "source": "remote"}
        if self.note:
            common["note"] = self.note
        if self.kind == "embed":
            return {
                "type": "embed",
                "url": self.url,
                "embed_url": self.embed_url,
                "provider": self.provider,
                "video_id": self.video_id,
                "title": self.title,
                "author": self.author,
                "poster": poster_url,
                "mime": "text/html",
                **common,
            }
        block: dict[str, Any] = {
            "type": self.kind,
            "url": self.url,
            "mime": self.content_type,
            "poster": poster_url,
            **common,
        }
        if self.stream_url:
            block["stream"] = self.stream_url
        if self.duration:
            block["duration"] = round(self.duration, 3)
        if self.width and self.height:
            block["width"] = self.width
            block["height"] = self.height
        if self.length:
            block["bytes"] = self.length
        if frame_urls:
            block["frames"] = list(frame_urls)
        return block


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def resolve_remote(
    url: str,
    *,
    limits: RemoteLimits | None = None,
    want_frames: int = 0,
    target_fps: float = 1.0,
    poster: bool = True,
) -> RemoteMedia:
    """Produce the manifest for ``url``. Raises :class:`RemoteMediaError` to refuse.

    ``want_frames`` is a request, not a promise: the ceiling in ``limits`` and the
    video's own length decide how many are actually pulled, and the manifest says
    which happened.
    """
    limits = limits or RemoteLimits()
    if not is_remote_url(url):
        raise RemoteMediaError(
            f"only http:// and https:// URLs can be opened as remote media (got {url!r}). "
            "A local file needs a path or a /memory/ URL.",
            kind="unsupported",
            url=str(url),
        )
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname:
        raise RemoteMediaError(f"that URL has no host: {url!r}", kind="unsupported", url=url)
    if not limits.allow_private and _is_private_host(parsed.hostname):
        # The tool already hands the model a filesystem; it should not also hand it
        # every HTTP service on the machine and on the LAN. Set
        # `REMOTE_MEDIA_ALLOW_PRIVATE=1` to opt in.
        raise RemoteMediaError(
            f"{parsed.hostname} is on this machine or the local network. Remote media is "
            "for public URLs; a local file needs a path or a /memory/ URL instead.",
            kind="unsupported",
            url=url,
        )

    provider, video_id, embed_url = classify_url(url)
    deadline = time.monotonic() + max(5.0, limits.total_timeout)
    if provider:
        return _resolve_embed(
            url,
            provider,
            video_id,
            embed_url,
            limits=limits,
            want_frames=max(0, int(want_frames)),
            deadline=deadline,
        )

    budget = ByteBudget(limits.max_bytes)
    try:
        return _resolve_direct(
            url,
            limits=limits,
            budget=budget,
            poster_wanted=bool(poster),
            frames_wanted=max(0, int(want_frames)),
            target_fps=float(target_fps or 1.0),
            deadline=deadline,
        )
    except RemoteMediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - a tool handler must not leak a traceback
        logger.exception("remote media probe failed for %s", url)
        raise RemoteMediaError(f"could not read {url}: {type(exc).__name__}: {exc}", kind="error", url=url)


def _is_private_host(host: str) -> bool:
    """Refuse loopback, link-local and RFC1918 addresses.

    Not a security boundary — a URL is chosen by the model, not the attacker — but
    ``http://127.0.0.1:8572/`` and ``http://localhost:11434/`` are the two URLs this
    tool would otherwise be pointed at all day, and neither is a video.
    """
    lowered = host.lower().strip("[]")
    if lowered in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    if lowered.endswith((".local", ".internal", ".localhost")):
        return True
    try:
        import ipaddress

        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local or address.is_reserved)


def wait_for_proxy(timeout: float = 5.0) -> str:
    """Start the proxy and wait until it answers. Tests use this; tools do not."""
    base = proxy_base()
    if not base:
        return ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with closing(urllib.request.urlopen(f"{base}/__ping", timeout=1)):  # noqa: S310
                return base
        except urllib.error.HTTPError:
            return base
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.02)
    return base
