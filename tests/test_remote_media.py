"""Tests for remote media: an ``https://`` URL turned into something displayable.

Almost all of this file runs without a network. :data:`server.remote_media.TRANSPORT`
is the single seam the module was built around, so every probe here is answered by a
fake that returns the same three things urllib returns — ``status``, ``headers`` and
``read`` — and the tests assert on the requests that were made as much as on the
result. That is the only way to prove the central claim: a video is *sampled*,
never downloaded, so the number of bytes charged to the budget for a 20-minute
1080p source stays in the kilobytes.

Two things are still real: the decoders (stubbed here, exercised end-to-end against
a real local clip in the last section) and the loopback playback proxy, which is
started for one test because a fake socket would prove nothing about whether a
browser can play the result.
"""

from __future__ import annotations

import functools
import http.server
import io
import json
import shutil
import threading
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepseek_client.tools import ToolRegistry
from server import remote_media
from server.media import MediaStore
from server.remote_media import (
    ByteBudget,
    RemoteLimits,
    RemoteMediaError,
    classify_url,
    is_remote_url,
    resolve_remote,
)
from server.settings import load_settings
from server.store import ConversationStore
from server.tools_builtin import build_media_tools

JPEG = b"\xff\xd8\xff" + b"fake-jpeg-payload" * 8
PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload" * 8

VIDEO_TYPE = "video/mp4"
HTML_TYPE = "text/html; charset=utf-8"


# ── the fake network ─────────────────────────────────────────────────────────


class FakeResponse:
    """The three attributes the module uses, and nothing else."""

    def __init__(self, body: bytes = b"", *, status: int = 200, headers: dict | None = None, url: str = ""):
        self.body = body
        self.status = status
        self.headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self.url = url
        self.closed = False
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            data, self.body = self.body, b""
            return data
        data, self.body = self.body[:size], self.body[size:]
        return data

    def close(self) -> None:
        self.closed = True


def media(
    body: bytes = b"",
    *,
    status: int = 200,
    ctype: str = VIDEO_TYPE,
    length: int | None = None,
    ranges: bool = True,
    url: str = "",
) -> FakeResponse:
    """A response that looks like a video host's HEAD."""
    headers = {"content-type": ctype}
    if length is not None:
        headers["content-length"] = str(length)
    if ranges:
        headers["accept-ranges"] = "bytes"
    return FakeResponse(body, status=status, headers=headers, url=url)


def failure(code: int, url: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url or "https://cdn.example.com/clip.mp4", code, str(code), None, io.BytesIO(b""))


class FakeTransport:
    """Routes URL -> response, exception or callable. An unrouted URL is a failure.

    Refusing unknown URLs is deliberate: several tests exist to assert that a probe
    makes *one* request, not a walk over a site, and a lenient fake would hide a
    regression that walked one.
    """

    def __init__(self, routes: dict | None = None):
        self.routes = dict(routes or {})
        self.calls: list[SimpleNamespace] = []

    def open(self, url: str, *, headers=None, method: str = "GET", timeout: float = 20.0):
        self.calls.append(SimpleNamespace(url=url, headers=dict(headers or {}), method=method, timeout=timeout))
        handler = self.routes.get(url)
        if handler is None:
            raise AssertionError(f"unexpected request: {method} {url}")
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            return handler(url, method, headers)
        return handler

    @property
    def urls(self) -> list[str]:
        return [call.url for call in self.calls]

    def methods(self) -> list[str]:
        return [call.method for call in self.calls]


@pytest.fixture
def net(monkeypatch):
    """Install a fake transport. Returns the installer, so each test picks its routes."""

    def install(routes: dict | None = None) -> FakeTransport:
        transport = FakeTransport(routes)
        monkeypatch.setattr(remote_media, "TRANSPORT", transport)
        return transport

    return install


class FakeDecoder:
    """Stands in for ffprobe and ffmpeg: metadata, plus a frame per seek.

    ``times`` is what makes seek-and-decode assertable — if the tool ever downloaded
    first, there would be no seeks to record.
    """

    def __init__(self, *, duration: float | None = 120.0, width: int = 1920, height: int = 1080,
                 codec: str = "h264", frame: bytes = JPEG):
        self.duration = duration
        self.width = width
        self.height = height
        self.codec = codec
        self.frame = frame
        self.times: list[float] = []
        self.sources: list[str] = []
        self.probe_timeouts: list[float | None] = []

    def meta(self) -> dict:
        return {
            "format": {"duration": str(self.duration) if self.duration is not None else ""},
            "streams": [
                {
                    "codec_type": "video",
                    "width": self.width,
                    "height": self.height,
                    "codec_name": self.codec,
                    "duration": str(self.duration) if self.duration is not None else "",
                }
            ],
        }

    def ffprobe(self, src, limits, *, timeout=None):
        self.sources.append(src)
        self.probe_timeouts.append(timeout)
        return self.meta()

    def decode(self, src, at, max_dim, limits, *, timeout=None):
        self.sources.append(src)
        self.times.append(at)
        # Distinct per seek, the way real frames are: a fake that returned the same
        # bytes every time would be deduplicated, and every count would read as one.
        return self.frame + bytes([len(self.times) % 251])


@pytest.fixture
def decoder(monkeypatch):
    """Install a fake decoder and hand it back, so a test can read its call log."""

    def install(**kwargs) -> FakeDecoder:
        fake = FakeDecoder(**kwargs)
        monkeypatch.setattr(remote_media, "_ffprobe", fake.ffprobe)
        monkeypatch.setattr(remote_media, "_decode_one", fake.decode)
        monkeypatch.setattr(remote_media, "_decode_with_opencv", lambda *a, **k: [])
        return fake

    return install


def limits(**overrides) -> RemoteLimits:
    """Probe limits with the proxy off, since a test has no browser to play into."""
    base = {"proxy": False}
    base.update(overrides)
    return replace(RemoteLimits(), **base)


def probe(url: str, *, want_frames: int = 0, poster: bool = True, target_fps: float = 1.0, **overrides):
    return resolve_remote(url, limits=limits(**overrides), want_frames=want_frames, poster=poster, target_fps=target_fps)


MP4 = "https://cdn.example.com/clip.mp4"


# ── what counts as a remote URL ──────────────────────────────────────────────


def test_http_and_https_are_remote_urls():
    assert is_remote_url("https://cdn.example.com/clip.mp4")
    assert is_remote_url("  HTTP://cdn.example.com/clip.mp4")
    assert is_remote_url("http://www.youtube.com/watch?v=abcdefghijk")


def test_a_local_path_is_not_a_remote_url():
    """The remote path must never capture a path — that is what `_resolve` is for."""
    assert not is_remote_url("/memory/conv1/clip.mp4")
    assert not is_remote_url("C:/videos/clip.mp4")
    assert not is_remote_url("file:///C:/videos/clip.mp4")
    assert not is_remote_url("clip.mp4")
    assert not is_remote_url(None)
    assert not is_remote_url(12)


def test_resolve_remote_refuses_something_that_is_not_a_url():
    with pytest.raises(RemoteMediaError) as caught:
        resolve_remote("/videos/clip.mp4", limits=limits())
    assert caught.value.kind == "unsupported"
    assert "path or a /memory/ URL" in str(caught.value)


# ── classification of the hosts we can only embed ────────────────────────────


@pytest.mark.parametrize(
    "url, video_id",
    [
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?list=PL1&v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_youtube_urls_yield_a_nocookie_embed(url, video_id):
    provider, found, embed = classify_url(url)
    assert (provider, found) == ("youtube", video_id)
    assert embed == f"https://www.youtube-nocookie.com/embed/{video_id}"


def test_vimeo_urls_yield_the_player_url():
    provider, video_id, embed = classify_url("https://vimeo.com/123456789")
    assert (provider, video_id) == ("vimeo", "123456789")
    assert embed == "https://player.vimeo.com/video/123456789"


def test_a_youtube_page_without_an_id_is_not_embeddable():
    assert classify_url("https://www.youtube.com/results?search_query=cats") == ("", "", "")
    assert classify_url("https://youtu.be/") == ("", "", "")


def test_a_random_host_is_not_embeddable():
    """Only a host with an oEmbed record and a stable player is embeddable."""
    assert classify_url(MP4) == ("", "", "")


# ── the arithmetic that keeps a probe cheap ──────────────────────────────────


def test_a_playback_only_request_samples_nothing():
    """'Show me' costs no tokens, which is the whole point of the manifest."""
    assert remote_media._sample_count(0, 1200.0, 1.0, RemoteLimits()) == 0
    assert remote_media._sample_count(-3, 1200.0, 1.0, RemoteLimits()) == 0


def test_the_sample_count_scales_with_duration_and_stops_at_the_ceiling():
    lim = RemoteLimits(frames_max=8)
    assert remote_media._sample_count(8, 10.0, 1.0, lim) == 8  # ten seconds, ten wanted
    assert remote_media._sample_count(8, 3.0, 1.0, lim) == 3  # only three seconds exist
    assert remote_media._sample_count(16, 1200.0, 1.0, lim) == 8  # a film is still eight
    assert remote_media._sample_count(4, None, 1.0, lim) == 4  # no duration: honour the ask


def test_frames_are_sampled_from_the_interior():
    """The first frame is a title card and the last is a fade; neither is the video."""
    times = remote_media._frame_times(10.0, 4)
    assert times == sorted(times)
    assert all(0.0 < at < 10.0 for at in times)
    assert remote_media._frame_times(10.0, 1) == [0.0]
    assert remote_media._frame_times(10.0, 0) == []


def test_a_known_duration_gives_interior_times_and_no_duration_gives_whole_seconds():
    assert remote_media._frame_times(None, 3) == [0.0, 1.0, 2.0]
    assert remote_media._frame_times(0.0, 3) == [0.0, 1.0, 2.0]


def test_identical_frames_are_collapsed():
    """A static shot sampled eight times is one picture, not eight."""
    out = remote_media._dedupe([JPEG, JPEG, PNG, JPEG, PNG])
    assert out == [JPEG, PNG]


def test_a_deadline_clamps_every_step_and_never_goes_non_positive():
    import time

    assert remote_media._left(None, 20.0) == 20.0
    assert remote_media._left(time.monotonic() + 120, 20.0) == 20.0
    assert remote_media._left(time.monotonic() - 120, 20.0) == 1.0
    assert remote_media._left(time.monotonic() + 0.2, 5.0) == 1.0


def test_expiry_needs_room_for_one_more_decode():
    import time

    assert not remote_media._expired(None)
    assert not remote_media._expired(time.monotonic() + 60)
    assert remote_media._expired(time.monotonic() - 1)
    assert remote_media._expired(time.monotonic() + 1.0, reserve=1.5)


def test_the_budget_stops_the_read_that_would_exceed_it():
    budget = ByteBudget(10)
    budget.charge(10)
    assert budget.left == 0
    with pytest.raises(RemoteMediaError) as caught:
        budget.charge(1)
    assert caught.value.kind == "too_big"
    assert "limit for a remote video" in str(caught.value)


# ── private addresses ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "10.1.2.3", "192.168.0.14", "172.16.5.5", "169.254.1.1", "::1", "box.local", "svc.internal"],
)
def test_private_hosts_are_recognised(host):
    assert remote_media._is_private_host(host)


@pytest.mark.parametrize("host", ["cdn.example.com", "8.8.8.8", "youtu.be"])
def test_public_hosts_are_not_private(host):
    assert not remote_media._is_private_host(host)


def test_a_localhost_url_is_refused_before_a_socket_is_opened(net):
    transport = net({MP4: media(ctype=VIDEO_TYPE)})
    with pytest.raises(RemoteMediaError) as caught:
        resolve_remote("http://127.0.0.1:8572/clip.mp4", limits=limits())
    assert caught.value.kind == "unsupported"
    assert "on this machine or the local network" in str(caught.value)
    assert transport.calls == []


def test_a_localhost_url_is_allowed_when_the_setting_says_so(net, decoder):
    decoder()
    transport = net({"http://127.0.0.1:8572/clip.mp4": media(ctype=VIDEO_TYPE, length=1024)})
    found = resolve_remote("http://127.0.0.1:8572/clip.mp4", limits=limits(allow_private=True), poster=False)
    assert found.kind == "video"
    assert transport.methods() == ["HEAD"]


# ── metadata from a HEAD, and nothing else ───────────────────────────────────


def test_a_video_url_is_probed_with_a_head_and_never_downloaded(net, decoder):
    fake = decoder(duration=1200.0)
    transport = net({MP4: media(ctype=VIDEO_TYPE, length=180_000_000)})

    found = probe(MP4, want_frames=0, poster=False)

    assert transport.methods() == ["HEAD"], "a manifest may only cost one request"
    assert transport.calls[0].headers["User-Agent"] == remote_media.USER_AGENT
    assert transport.calls[0].headers["Referer"] == "https://cdn.example.com/"
    assert transport.calls[0].headers["Accept-Encoding"] == "identity"
    assert fake.times == [], "asking for no picture must not cost a decode"
    assert found.kind == "video"
    assert found.length == 180_000_000
    assert found.duration == 1200.0
    assert (found.width, found.height) == (1920, 1080)
    assert found.codec == "h264"
    assert found.ranges is True
    assert found.frames == []
    assert found.poster is None
    assert found.bytes_read == 0, "the HEAD told us everything and read nothing"


def test_a_poster_is_one_seek_a_tenth_of_the_way_in(net, decoder):
    """One decode, not a scan: the poster is what the player shows before it plays."""
    fake = decoder(duration=1200.0)
    net({MP4: media(ctype=VIDEO_TYPE, length=180_000_000)})

    found = probe(MP4, want_frames=0)

    assert found.poster[: len(JPEG)] == JPEG
    assert fake.times == [120.0], "10% of 1200s, and only that"
    assert found.frames == []


def test_the_facts_of_a_twenty_minute_source_stay_inside_the_ceiling(net, decoder):
    """The acceptance criterion, measured: 20 minutes of 1080p costs a few hundred KB."""
    fake = decoder(duration=1200.0)
    transport = net({MP4: media(ctype=VIDEO_TYPE, length=180_000_000)})

    found = probe(MP4, want_frames=8, target_fps=1.0)

    assert transport.urls == [MP4] * len(transport.calls)
    assert len(fake.times) == 9, "eight frames and one poster"
    assert found.bytes_read <= RemoteLimits().max_bytes
    assert len(found.frames) == 8, "a 20-minute film is still eight samples"
    assert "identical" not in found.note, "nothing had to be explained away"


def test_a_host_that_refuses_head_is_probed_with_one_byte_of_body(net, decoder):
    """405 on HEAD is common. A one-byte range is the polite substitute."""
    fake = decoder(duration=60.0)
    body = media(b"", status=206, ctype=VIDEO_TYPE, url=MP4)
    body.headers["content-range"] = "bytes 0-0/1000000"

    def head_then_range(url, method, headers):
        if method == "HEAD":
            raise failure(405, url)
        return body

    transport = net({MP4: head_then_range})
    found = probe(MP4, want_frames=0)

    assert transport.methods() == ["HEAD", "GET"]
    assert transport.calls[1].headers["Range"] == "bytes=0-0"
    assert found.length == 1_000_000, "the total comes from Content-Range, not Content-Length"
    assert found.duration == 60.0


def test_the_name_comes_from_the_url_or_from_the_type():
    assert remote_media._name_for(MP4, VIDEO_TYPE, "video") == "clip.mp4"
    assert remote_media._name_for("https://cdn.example.com/stream", VIDEO_TYPE, "video") == "remote.mp4"
    assert remote_media._name_for("https://cdn.example.com/a.mp3", "audio/mpeg", "audio") == "a.mp3"
    assert remote_media._name_for("https://cdn.example.com/a.mp3", "audio/mpeg", "video") == "a.mp3"


# ── refusals, each named for what actually happened ──────────────────────────


@pytest.mark.parametrize(
    "code, kind, fragment",
    [
        (401, "blocked", "wants a login"),
        (403, "blocked", "refused the request"),
        (407, "blocked", "wants a login"),
        (451, "blocked", "legal reasons"),
        (429, "blocked", "rate limiting"),
        (404, "not_media", "has nothing at that URL"),
        (410, "not_media", "has nothing at that URL"),
        (416, "unsupported", "rejected a byte range"),
        (500, "network", "answered HTTP 500"),
    ],
)
def test_a_status_code_is_reported_as_itself(net, code, kind, fragment):
    net({MP4: failure(code, MP4)})
    with pytest.raises(RemoteMediaError) as caught:
        probe(MP4)
    assert caught.value.kind == kind
    assert caught.value.status == code
    assert fragment in str(caught.value)
    assert caught.value.url == MP4


def test_a_refusal_never_looks_like_emptiness(net):
    """403 must not read as 'no video here'; the fix is different for each."""
    net({MP4: failure(403, MP4)})
    with pytest.raises(RemoteMediaError) as refused:
        probe(MP4)
    net({MP4: media(ctype="text/html", length=200)})
    with pytest.raises(RemoteMediaError) as empty:
        probe(MP4)
    assert refused.value.kind == "blocked"
    assert empty.value.kind == "not_media"
    assert str(refused.value) != str(empty.value)
    assert "freshly copied URL" in str(refused.value)


def test_an_unreachable_host_is_a_network_error(net):
    net({MP4: urllib.error.URLError("no route to host")})
    with pytest.raises(RemoteMediaError) as caught:
        probe(MP4)
    assert caught.value.kind == "network"
    assert "could not reach cdn.example.com" in str(caught.value)


def test_a_timed_out_host_says_so(net):
    import socket

    net({MP4: urllib.error.URLError(socket.timeout("timed out"))})
    with pytest.raises(RemoteMediaError) as caught:
        probe(MP4)
    assert caught.value.kind == "network"
    assert "took too long to answer" in str(caught.value)


def test_a_content_type_that_is_not_media_is_named(net):
    """With no media extension to break the tie, the content type is the whole answer."""
    net({"https://cdn.example.com/download": media(b"PK\x03\x04zip", ctype="application/zip", length=100)})
    with pytest.raises(RemoteMediaError) as caught:
        probe("https://cdn.example.com/download")
    assert caught.value.kind == "not_media"
    assert "application/zip" in str(caught.value)
    assert "attach it" in str(caught.value)


def test_an_extension_breaks_the_tie_when_the_type_is_a_lie(net, decoder):
    """Hosts serve MP4 as octet-stream often enough that the header alone would refuse."""
    decoder(duration=30.0)
    net({MP4: media(ctype="application/octet-stream", length=1000)})
    assert probe(MP4, want_frames=0, poster=False).kind == "video"


def test_a_video_no_decoder_can_read_is_reported_as_a_decode_error(net, monkeypatch):
    monkeypatch.setattr(remote_media, "_ffprobe", lambda *a, **k: None)
    monkeypatch.setattr(remote_media, "_decode_one", lambda *a, **k: None)
    monkeypatch.setattr(remote_media, "_decode_with_opencv", lambda *a, **k: [])
    net({MP4: media(ctype="video/webm", length=5000)})

    with pytest.raises(RemoteMediaError) as caught:
        probe(MP4, want_frames=4)
    assert caught.value.kind == "decode"
    assert "no frame could be decoded" in str(caught.value)
    assert "ffmpeg was tried first, then OpenCV" in str(caught.value)


def test_a_read_is_capped_at_the_budget_rather_than_overflowing_it(net):
    """Every read is clamped to what is left, so the ceiling is never overflowed.

    The one place a caller can genuinely spend more than the ceiling is the playback
    proxy, which is what the next test covers.
    """
    body = media(PNG, ctype="image/png", length=len(PNG))
    net({"https://cdn.example.com/photo.png": body})
    found = probe("https://cdn.example.com/photo.png", max_bytes=len(PNG) + 4096)
    assert found.kind == "image"
    assert found.bytes_read == len(PNG), "exactly the image, and not a byte more"


# ── streams we cannot open ───────────────────────────────────────────────────


def test_a_plain_hls_playlist_is_unsupported_not_a_video(net, decoder):
    decoder()
    playlist = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:6.0,\nsegment0.ts\n"
    net({"https://cdn.example.com/live.m3u8": media(playlist, ctype="application/vnd.apple.mpegurl")})

    with pytest.raises(RemoteMediaError) as caught:
        probe("https://cdn.example.com/live.m3u8")
    assert caught.value.kind == "unsupported"
    assert "segmented stream" in str(caught.value)
    assert "download the file and attach it" in str(caught.value)


def test_a_fairplay_playlist_is_drm_with_an_actionable_message(net, decoder):
    decoder()
    playlist = (
        b"#EXTM3U\n"
        b'#EXT-X-KEY:METHOD=SAMPLE-AES,URI="skd://key",KEYFORMAT="com.apple.streamingkeydelivery"\n'
        b"#EXTINF:6.0,\nsegment0.ts\n"
    )
    net({"https://cdn.example.com/film.m3u8": media(playlist, ctype="application/vnd.apple.mpegurl")})

    with pytest.raises(RemoteMediaError) as caught:
        probe("https://cdn.example.com/film.m3u8")
    assert caught.value.kind == "drm"
    assert "sample-aes" in str(caught.value)
    assert "not DRM protected" in str(caught.value)


def test_a_widevine_dash_manifest_is_drm(net, decoder):
    decoder()
    manifest = b'<?xml version="1.0"?><MPD><ContentProtection schemeIdUri="urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"/></MPD>'
    net({"https://cdn.example.com/film.mpd": media(manifest, ctype="application/dash+xml")})

    with pytest.raises(RemoteMediaError) as caught:
        probe("https://cdn.example.com/film.mpd")
    assert caught.value.kind == "drm"


def test_a_manifest_that_cannot_be_read_is_still_unsupported(net, decoder):
    """An unreadable playlist is not a licence problem, and must not be reported as one."""
    decoder()
    net({"https://cdn.example.com/x.m3u8": media(b"", ctype="application/vnd.apple.mpegurl")})
    with pytest.raises(RemoteMediaError) as caught:
        probe("https://cdn.example.com/x.m3u8")
    assert caught.value.kind == "unsupported"


# ── the og:video hop out of a web page ───────────────────────────────────────


def test_a_page_that_declares_a_video_is_followed_once(net, decoder):
    fake = decoder(duration=90.0)
    page = b'<html><head><meta property="og:video:secure_url" content="https://cdn.example.com/real.mp4"></head></html>'
    transport = net(
        {
            "https://news.example.com/story": media(page, ctype=HTML_TYPE),
            "https://cdn.example.com/real.mp4": media(ctype=VIDEO_TYPE, length=9_000_000),
        }
    )

    found = probe("https://news.example.com/story", want_frames=0, poster=False)

    # The page is HEADed, then read, and only then is the video it names probed.
    assert transport.urls == [
        "https://news.example.com/story",
        "https://news.example.com/story",
        "https://cdn.example.com/real.mp4",
    ]
    assert transport.methods() == ["HEAD", "GET", "HEAD"]
    assert found.kind == "video"
    assert found.url == "https://cdn.example.com/real.mp4"
    assert "declares this video" in found.note
    assert fake.times == []


def test_a_page_with_only_an_og_image_shows_the_image(net):
    page = b'<html><head><meta name="twitter:image" content="https://cdn.example.com/hero.jpg"></head></html>'
    net(
        {
            "https://news.example.com/story": media(page, ctype=HTML_TYPE),
            "https://cdn.example.com/hero.jpg": media(JPEG, ctype="image/jpeg"),
        }
    )

    found = probe("https://news.example.com/story")

    assert found.kind == "image"
    assert found.poster == JPEG
    assert found.playable is False
    assert "no video in it" in found.note


def test_a_page_that_declares_nothing_says_so(net):
    page = b"<html><head><title>Just an article</title></head><body>words</body></html>"
    net({"https://news.example.com/story": media(page, ctype=HTML_TYPE)})

    with pytest.raises(RemoteMediaError) as caught:
        probe("https://news.example.com/story")
    assert caught.value.kind == "not_media"
    assert "is a web page, not a video" in str(caught.value)
    assert "web_fetch" in str(caught.value)


def test_an_og_video_that_is_not_http_is_ignored(net):
    page = b'<html><head><meta property="og:video" content="rtmp://example.com/live"></head></html>'
    net({"https://news.example.com/story": media(page, ctype=HTML_TYPE)})
    with pytest.raises(RemoteMediaError) as caught:
        probe("https://news.example.com/story")
    assert caught.value.kind == "not_media"


def test_page_following_can_be_turned_off(net):
    page = b'<html><head><meta property="og:video" content="https://cdn.example.com/real.mp4"></head></html>'
    net({"https://news.example.com/story": media(page, ctype=HTML_TYPE)})
    with pytest.raises(RemoteMediaError) as caught:
        probe("https://news.example.com/story", follow_pages=False)
    assert caught.value.kind == "not_media"


# ── images, which are the one thing copied locally ───────────────────────────


def test_an_image_url_comes_back_as_its_bytes(net):
    """`RemoteMedia.poster` holds bytes for an image; the tool layer saves them."""
    net({"https://cdn.example.com/photo.png": media(PNG, ctype="image/png", length=len(PNG))})
    found = probe("https://cdn.example.com/photo.png")
    assert found.kind == "image"
    assert found.poster == PNG
    assert found.name == "photo.png"
    assert found.playable is False
    assert found.bytes_read == len(PNG)


# ── the two tokens of one probe ──────────────────────────────────────────────


def test_a_successful_probe_keeps_a_playback_token_and_forgets_its_analysis_token(net, decoder):
    decoder()
    net({MP4: media(ctype=VIDEO_TYPE, length=1000)})
    before = set(remote_media.sources())

    found = probe(MP4, want_frames=2)

    added = {token: source for token, source in remote_media.sources().items() if token not in before}
    assert len(added) == 1, "the analysis token must not outlive the call"
    assert all(source.budget is None for source in added.values()), (
        "a budgeted URL left registered is a player that stalls mid-playback"
    )
    assert all(source.url == MP4 for source in added.values())
    assert found.frames

    for token in added:
        remote_media.forget(token)
    assert set(remote_media.sources()) == before


def test_a_failed_probe_leaves_nothing_registered(net, monkeypatch):
    monkeypatch.setattr(remote_media, "_ffprobe", lambda *a, **k: None)
    monkeypatch.setattr(remote_media, "_decode_one", lambda *a, **k: None)
    monkeypatch.setattr(remote_media, "_decode_with_opencv", lambda *a, **k: [])
    net({MP4: media(ctype=VIDEO_TYPE, length=1000)})
    before = set(remote_media.sources())

    with pytest.raises(RemoteMediaError):
        probe(MP4, want_frames=2)

    assert set(remote_media.sources()) == before


def test_a_refused_probe_registers_nothing_at_all(net):
    net({MP4: failure(403, MP4)})
    before = set(remote_media.sources())
    with pytest.raises(RemoteMediaError):
        probe(MP4)
    assert set(remote_media.sources()) == before


def test_the_streaming_proxy_serves_the_original_bytes_under_a_token(net):
    """The one test that uses a real socket, because the claim is about a browser."""
    base = remote_media.wait_for_proxy()
    if not base:
        pytest.skip("the loopback proxy could not bind on this machine")
    try:
        net(
            {
                MP4: lambda url, method, headers: media(
                    b"0123456789" * 8, ctype=VIDEO_TYPE, length=80, url=MP4
                )
            }
        )
        found = resolve_remote(MP4, limits=limits(proxy=True), want_frames=0, poster=False)
        assert found.stream_url.startswith(base + "/")
        assert found.url == MP4, "the manifest always names the original source"

        import urllib.request

        with urllib.request.urlopen(found.stream_url, timeout=5) as answer:
            assert answer.status == 200
            assert answer.headers["Content-Type"] == VIDEO_TYPE
            assert answer.headers["Cache-Control"] == "no-store"
            assert answer.read() == b"0123456789" * 8

        # The analysis token was dropped in the probe's `finally`, so its URL is gone.
        token = found.stream_url.rsplit("/", 1)[-1]
        assert remote_media.sources()[token].budget is None
        remote_media.forget(token)
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(found.stream_url, timeout=5)
        assert caught.value.code == 404, "an expired token must not read as an upstream error"
    finally:
        remote_media.shutdown_proxy()


def test_the_proxy_stops_a_stream_that_spends_the_budget(net):
    """The one place a source can honestly outrun the ceiling: a player streaming it.

    This is the acceptance criterion about total bytes, tested where it can actually
    be exceeded rather than where the read loop already clamps it.
    """
    base = remote_media.wait_for_proxy()
    if not base:
        pytest.skip("the loopback proxy could not bind on this machine")
    try:
        body = b"x" * 4096
        net({MP4: lambda url, method, headers: media(body, ctype=VIDEO_TYPE, length=len(body))})
        budget = remote_media.ByteBudget(64)
        token = remote_media._register(
            MP4, {}, name="clip.mp4", content_type=VIDEO_TYPE, timeout=5.0, budget=budget
        )
        url = f"{base}/{token}"

        import http.client
        import urllib.request

        delivered = b""
        try:
            with urllib.request.urlopen(url, timeout=5) as answer:
                delivered = answer.read()
        except (http.client.IncompleteRead, ConnectionError):
            # A promised Content-Length the proxy then refuses to honour: the client
            # sees a truncated body, which is the point.
            pass

        assert len(delivered) < len(body), "the player is cut off rather than fed the lot"
        assert budget.left == 0
        assert "limit for a remote video" in remote_media.sources()[token].error
        remote_media.forget(token)
    finally:
        remote_media.shutdown_proxy()


# ── the embed path ───────────────────────────────────────────────────────────


OEMBED = "https://www.youtube.com/oembed?url={url}&format=json"
YOUTUBE = "https://youtu.be/dQw4w9WgXcQ"
NOCCOOKIE = "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ"


def oembed_url(url: str) -> str:
    import urllib.parse

    return OEMBED.format(url=urllib.parse.quote(url, safe=""))


def test_a_youtube_url_becomes_an_embed_with_its_own_stills(net):
    """Nothing is downloaded: the player stays on YouTube's servers."""
    record = json.dumps(
        {
            "title": "A Video",
            "author_name": "Someone",
            "duration": 212,
            "width": 1920,
            "height": 1080,
            "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
        }
    ).encode()
    routes = {
        oembed_url(YOUTUBE): media(record, ctype="application/json"),
        "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg": media(JPEG, ctype="image/jpeg"),
        "https://i.ytimg.com/vi/dQw4w9WgXcQ/hq1.jpg": media(JPEG, ctype="image/jpeg"),
        "https://i.ytimg.com/vi/dQw4w9WgXcQ/hq2.jpg": media(JPEG, ctype="image/jpeg"),
        "https://i.ytimg.com/vi/dQw4w9WgXcQ/hq3.jpg": media(JPEG, ctype="image/jpeg"),
    }
    transport = net(routes)

    found = resolve_remote(YOUTUBE, limits=limits(), want_frames=3, poster=True)

    assert found.kind == "embed"
    assert found.provider == "youtube"
    assert found.video_id == "dQw4w9WgXcQ"
    assert found.embed_url == NOCCOOKIE
    assert found.title == "A Video"
    assert found.author == "Someone"
    assert (found.duration, found.width, found.height) == (212.0, 1920, 1080)
    assert found.poster == JPEG
    assert found.frames == [JPEG], "one still repeated four times is one still"
    assert "was not fetched" in found.note
    assert "2 of the 3 stills" in found.note, "and the duplicate is owned up to"
    assert found.url == YOUTUBE
    assert found.content_type == "text/html"
    assert set(transport.urls) <= set(routes)


def test_youtube_stills_are_deduplicated_when_they_differ(net):
    record = json.dumps({"title": "T"}).encode()
    stills = {
        "hqdefault.jpg": JPEG,
        "hq1.jpg": JPEG + b"1",
        "hq2.jpg": JPEG + b"2",
        "hq3.jpg": JPEG + b"3",
    }
    net(
        {
            oembed_url(YOUTUBE): media(record, ctype="application/json"),
            **{f"https://i.ytimg.com/vi/dQw4w9WgXcQ/{name}": media(data, ctype="image/jpeg")
               for name, data in stills.items()},
        }
    )

    found = resolve_remote(YOUTUBE, limits=limits(), want_frames=4, poster=True)

    assert found.frames == [JPEG + b"1", JPEG + b"2", JPEG + b"3"]
    assert "1/8, 3/8, 5/8 and 7/8" in found.note
    assert "identical" not in found.note, "nothing was dropped, so nothing to explain"


def test_a_private_or_blocked_video_says_which_it_might_be(net):
    net({oembed_url(YOUTUBE): failure(401, OEMBED)})
    with pytest.raises(RemoteMediaError) as caught:
        resolve_remote(YOUTUBE, limits=limits(), want_frames=0)
    assert caught.value.kind == "blocked"
    assert "private, deleted, age-restricted and region-blocked" in str(caught.value)


def test_an_embed_without_oembed_metadata_still_renders(net):
    """A 500 from oEmbed is a missing title, not a missing video."""
    net(
        {
            oembed_url(YOUTUBE): failure(500, OEMBED),
            "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg": media(JPEG, ctype="image/jpeg"),
        }
    )
    found = resolve_remote(YOUTUBE, limits=limits(), want_frames=0)
    assert found.kind == "embed"
    assert found.title == ""
    assert found.name == "youtube video dQw4w9WgXcQ"
    assert found.poster == JPEG


def test_an_embed_with_no_poster_says_the_player_will_show_its_own(net):
    net({oembed_url(YOUTUBE): failure(500, OEMBED)})
    found = resolve_remote(YOUTUBE, limits=limits(), want_frames=0)
    assert found.poster is None
    assert "its own placeholder" in found.note


def test_vimeo_has_no_frame_samples_and_says_so(net):
    vimeo = "https://vimeo.com/123456789"
    net({oembed_url(vimeo): failure(500, OEMBED)})

    found = resolve_remote(vimeo, limits=limits(), want_frames=4)

    assert found.kind == "embed"
    assert found.frames == []
    assert "publishes one still image for this video and no frame samples" in found.note
    assert found.embed_url == "https://player.vimeo.com/video/123456789"


# ── the manifest as a display block ──────────────────────────────────────────


def test_a_video_manifest_renders_as_a_block_with_a_stream_and_a_poster(net, decoder):
    decoder(duration=83.0)
    net({MP4: media(ctype=VIDEO_TYPE, length=4_000_000)})
    found = probe(MP4, want_frames=0)

    block = found.block(poster_url="/memory/conv1/remote_poster.jpg")

    assert block["type"] == "video"
    assert block["url"] == MP4
    assert block["poster"] == "/memory/conv1/remote_poster.jpg"
    assert block["source"] == "remote"
    assert block["duration"] == 83.0
    assert block["mime"] == VIDEO_TYPE
    assert "frames" not in block, "no frames were sampled, so none may be promised"
    assert "stream" not in block, "no proxy means the player uses the original URL"


def test_an_embed_manifest_renders_as_an_embed_block(net):
    net({oembed_url(YOUTUBE): failure(500, OEMBED)})
    found = resolve_remote(YOUTUBE, limits=limits(), want_frames=0)

    block = found.block()

    assert block["type"] == "embed"
    assert block["embed_url"] == NOCCOOKIE
    assert block["provider"] == "youtube"
    assert block["video_id"] == "dQw4w9WgXcQ"
    assert block["mime"] == "text/html"
    assert block["source"] == "remote"
    assert "stream" not in block


def test_the_facts_of_a_remote_video_name_their_origin(net, decoder):
    decoder(duration=83.0)
    net({MP4: media(ctype=VIDEO_TYPE, length=4_000_000)})
    facts = probe(MP4, want_frames=0).facts()

    assert facts["source"] == "remote"
    assert facts["url"] == MP4
    assert facts["kind"] == "video"
    assert facts["bytes"] == 4_000_000
    assert facts["bytes_fetched"] == 0
    assert facts["duration_text"] == "1m 23s"
    assert facts["width"] == 1920
    assert facts["codec"] == "h264"
    assert "range_requests" in facts, "a video says whether seeking in it is cheap"


# ── the tools, reached the way the model reaches them ────────────────────────


@pytest.fixture
def toolset(tmp_path):
    """The media tools, reachable the way the model reaches them.

    ``REMOTE_MEDIA_PROXY=false``, because a test has no browser to hand a stream to
    and starting a loopback server per test would test the proxy, not the tool.
    """
    store = ConversationStore(tmp_path / "memory")
    store.create(uuid="conv1")
    settings = load_settings(
        {
            "REMOTE_MEDIA_PROXY": "false",
            "REMOTE_MEDIA_ALLOW_PRIVATE": "true",
        },
        load_dotenv=False,
        memory_root=store.root,
    )
    media_store = MediaStore(store)
    registry = ToolRegistry()
    for tool in build_media_tools(store, media_store, lambda: "conv1", settings=settings):
        registry.add(tool)
    return SimpleNamespace(registry=registry, store=store, media=media_store, settings=settings, tools=registry)


async def call(toolset, name, **arguments):
    return await toolset.registry.call(name, arguments)


def text_of(result) -> str:
    content = result.content
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


def block_of(result) -> dict:
    """The media block a display tool returned, as the frontend would receive it."""
    if not isinstance(result.content, list):
        return {}
    for block in result.content:
        if isinstance(block, dict) and block.get("type") in ("image", "video", "audio", "embed"):
            return block
    return {}


async def test_display_media_shows_an_mp4_url_as_a_player(toolset, net, decoder):
    """Acceptance criterion 1, and no full-file download on the server."""
    fake = decoder(duration=83.0)
    transport = net({MP4: media(ctype=VIDEO_TYPE, length=180_000_000)})

    result = await call(toolset, "display_media", path=MP4)

    assert not result.is_error, text_of(result)
    block = block_of(result)
    assert block["type"] == "video"
    assert block["url"] == MP4
    assert block["source"] == "remote"
    assert block["duration"] == 83.0
    assert block["display"] is True
    assert result.meta["remote"] is True
    assert result.meta["bytes_fetched"] < RemoteLimits().max_bytes
    assert fake.times == [8.3], "showing a video costs one poster frame and no samples"
    assert transport.methods() == ["HEAD"]
    assert "not downloaded" in text_of(result)
    # The poster is the one thing written to disk, and it is a picture, not the film.
    saved = result.meta["poster"]
    assert saved.startswith("/memory/conv1/")
    assert toolset.media.path_for_url(saved).is_file()


async def test_display_media_saves_a_remote_image_locally(toolset, net):
    net({"https://cdn.example.com/photo.png": media(PNG, ctype="image/png", length=len(PNG))})

    result = await call(toolset, "display_media", path="https://cdn.example.com/photo.png", caption="a cat")

    assert not result.is_error, text_of(result)
    block = block_of(result)
    assert block["type"] == "image"
    assert block["source"] == "remote"
    assert block["url"].startswith("/memory/"), "the block must point at something we serve"
    assert block["caption"] == "a cat"
    assert "cdn.example.com" in text_of(result)


async def test_display_media_reports_a_blocked_source_as_an_error(toolset, net):
    """Acceptance criterion 3: a specific, actionable error."""
    net({MP4: failure(403, MP4)})

    result = await call(toolset, "display_media", path=MP4)

    assert result.is_error
    message = text_of(result)
    assert "refused the request (HTTP 403)" in message
    assert "freshly copied URL" in message


async def test_display_media_still_prefers_a_local_path(toolset, net):
    """A local file must keep working, and must not touch the network at all."""
    transport = net({})
    url = toolset.media.save_bytes("conv1", PNG, "image/png", name="chart.png")

    result = await call(toolset, "display_media", path=url)

    assert not result.is_error, text_of(result)
    assert transport.calls == [], "a local path is never a network request"
    block = block_of(result)
    assert block.get("source") != "remote", "a local file is not remote media"
    # It is still a durable copy under /memory/, which is what the local path does too.
    assert block["url"].startswith("/memory/conv1/")
    assert toolset.media.path_for_url(block["url"]).is_file()
    assert "chart" in block["url"] or block["name"].endswith(".png")


async def test_inspect_media_answers_about_a_url_without_sampling_it(toolset, net, decoder):
    """Acceptance criterion 2a: the facts, one HEAD, zero frames."""
    fake = decoder(duration=1220.0, width=1920, height=1080)
    transport = net({MP4: media(ctype=VIDEO_TYPE, length=200_000_000)})

    result = await call(toolset, "inspect_media", path=MP4)

    assert not result.is_error, text_of(result)
    info = json.loads(result.content)
    assert info["source"] == "remote"
    assert info["kind"] == "video"
    assert info["duration"] == 1220.0
    assert info["width"] == 1920
    assert info["bytes"] == 200_000_000
    assert "duration_text" in info
    assert result.meta["remote"] is True
    assert transport.methods() == ["HEAD"]
    assert fake.times == []


async def test_inspect_media_reports_a_drm_stream_as_drm(toolset, net):
    playlist = b'#EXTM3U\n#EXT-X-KEY:METHOD=SAMPLE-AES,KEYFORMAT="com.widevine.alpha"\n'
    net({"https://cdn.example.com/film.m3u8": media(playlist, ctype="application/vnd.apple.mpegurl")})

    result = await call(toolset, "inspect_media", path="https://cdn.example.com/film.m3u8")

    assert result.is_error
    assert "encrypted for a licensed player" in text_of(result)


async def test_reduce_video_frames_returns_frames_from_a_url(toolset, net, decoder):
    """Acceptance criterion 2: frames from the URL, with nothing persisted whole."""
    fake = decoder(duration=6.0)
    transport = net({MP4: media(ctype=VIDEO_TYPE, length=50_000_000)})

    result = await call(toolset, "reduce_video_frames", path=MP4, max_frames=3, target_fps=1)

    assert not result.is_error, text_of(result)
    frames = [block for block in result.content if isinstance(block, dict) and block.get("type") == "image"]
    assert len(frames) == 3, "three samples came back, so three pictures are shown"
    assert all(block["url"].startswith("/memory/conv1/") for block in frames)
    assert all(toolset.media.path_for_url(block["url"]).is_file() for block in frames)
    assert transport.methods() == ["HEAD"], "the only HTTP request is the probe"
    assert len(fake.times) == 4, "three seeks for the frames and one for the poster"
    header = result.content[0]["text"]
    assert "not downloaded" in header
    assert "longest edge" in header
    assert "tokens" in header


async def test_reduce_video_frames_refuses_audio_with_its_facts_instead(toolset, net, decoder):
    """An audio URL is not a failure: the answer is its details, said plainly."""
    decoder(duration=200.0)
    net({"https://cdn.example.com/song.mp3": media(ctype="audio/mpeg", length=8_000_000)})

    result = await call(toolset, "reduce_video_frames", path="https://cdn.example.com/song.mp3")

    assert not result.is_error
    assert "audio file, so there are no frames" in text_of(result)
    assert "3m 20s" in text_of(result), "the facts are the answer when there are no frames"
    assert block_of(result) == {}, "nothing is shown to the user by a frame tool"


async def test_reduce_video_frames_needs_frames_from_an_image_url(toolset, net):
    net({"https://cdn.example.com/photo.png": media(PNG, ctype="image/png", length=len(PNG))})
    result = await call(toolset, "reduce_video_frames", path="https://cdn.example.com/photo.png")
    assert result.is_error
    assert "not a video" in text_of(result).lower()


# ── end to end, with the real decoders ───────────────────────────────────────


@pytest.fixture
def clip_server(tmp_path):
    """Serve a directory over loopback so ffmpeg has a real URL to seek in."""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D102 - silence the test output
            pass

    handler = functools.partial(Quiet, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for a real decode")
def test_a_real_clip_over_http_is_sampled_by_seeking(make_clip, clip_server):
    """No fake decoder, no fake transport: the whole path, on a real file over HTTP."""
    clip = make_clip("colour.mp4", seconds=3)
    url = f"{clip_server}/{clip.name}"

    found = resolve_remote(
        url, limits=limits(allow_private=True), want_frames=3, poster=True, target_fps=1.0
    )

    assert found.kind == "video"
    assert found.poster is not None and found.poster[:3] == b"\xff\xd8\xff"
    assert len(found.frames) == 3, "three colours over three seconds"
    assert all(frame[:3] == b"\xff\xd8\xff" for frame in found.frames)
    # The clip is one solid colour per second, so three distinct frames can only come
    # from three seeks that landed in three different seconds. (This test server does
    # not advertise Accept-Ranges, so the header itself is asserted elsewhere.)
    assert len({bytes(frame) for frame in found.frames}) == 3, "the seek landed three times"
    assert found.duration and 2.5 <= found.duration <= 3.5
    assert found.bytes_read <= RemoteLimits().max_bytes
    assert found.length == clip.stat().st_size
    assert remote_media.sources(), "a playable source is registered for the player"


def test_a_page_of_unsupported_length_is_not_read_twice(net, decoder):
    """A page is read once, with a cap, and never walked."""
    decoder()
    body = b"<html>" + b"x" * (remote_media._PAGE_PREFIX * 3) + b"</html>"
    transport = net({"https://news.example.com/story": media(body, ctype=HTML_TYPE)})

    with pytest.raises(RemoteMediaError):
        probe("https://news.example.com/story")

    assert transport.urls == ["https://news.example.com/story"] * 2, "a HEAD and one capped GET"
    assert len(body) > remote_media._PAGE_PREFIX
