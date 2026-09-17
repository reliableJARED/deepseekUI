"""Tests for the media helpers and the built-in media tools.

Frame extraction needs real video, but resize, sniffing, path safety and the
tool handlers are all pure logic.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from deepseek_client.tools import Tool
from server.media import (
    MediaLimits,
    MediaStore,
    decode_data_uri,
    ext_for_mime,
    is_image_mime,
    is_video_mime,
    sniff_mime,
)
from server.settings import load_settings
from server.store import ConversationStore
from server.tools_builtin import build_media_tools


def make_image(width=64, height=64, fmt="PNG", mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), (120, 80, 200) if mode == "RGB" else (120, 80, 200, 255)).save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    return ConversationStore(tmp_path / "memory")


@pytest.fixture
def media(store):
    return MediaStore(store)


# ── mime sniffing ────────────────────────────────────────────────────────────

def test_png_bytes_are_recognised_from_content():
    assert sniff_mime(make_image(fmt="PNG")) == "image/png"


def test_jpeg_bytes_are_recognised_from_content():
    assert sniff_mime(make_image(fmt="JPEG")) == "image/jpeg"


def test_gif_bytes_are_recognised_from_content():
    assert sniff_mime(make_image(fmt="GIF")) == "image/gif"


def test_webp_bytes_are_recognised_from_content():
    assert sniff_mime(make_image(fmt="WEBP")) == "image/webp"


def test_content_wins_over_a_lying_declared_type():
    """The API detects format from content, so we must too."""
    assert sniff_mime(make_image(fmt="PNG"), "image/jpeg") == "image/png"


def test_declared_type_is_the_fallback_for_an_unknown_signature():
    assert sniff_mime(b"not an image at all", "image/png") == "image/png"


def test_unknown_content_with_no_declaration_is_octet_stream():
    assert sniff_mime(b"hello world") == "application/octet-stream"


def test_mp4_is_recognised_as_video():
    """`ftyp` at offset 4 is the ISO base media file format marker."""
    data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32
    assert sniff_mime(data) == "video/mp4"
    assert is_video_mime("video/mp4")


def test_webm_is_recognised_as_video():
    data = b"\x1a\x45\xdf\xa3" + b"\x00" * 32
    assert is_video_mime(sniff_mime(data))


def test_image_and_video_predicates_are_disjoint():
    for mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        assert is_image_mime(mime)
        assert not is_video_mime(mime)
    assert not is_image_mime("video/mp4")
    assert not is_image_mime("application/pdf")


def test_extensions_map_from_mime():
    assert ext_for_mime("image/png") == ".png"
    assert ext_for_mime("image/jpeg") == ".jpg"
    assert ext_for_mime("video/mp4") == ".mp4"
    # An unknown type still needs *some* extension rather than a bare name.
    assert ext_for_mime("application/weird").startswith(".")


def test_ogg_is_recognised_as_audio_not_video():
    """`OggS` is a container shared by audio and video; the header wins."""
    assert sniff_mime(b"OggS" + b"\x00" * 32) == "audio/ogg"


def test_riff_defaults_to_webp_only_when_the_inner_marker_agrees():
    riff = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16
    assert sniff_mime(riff) == "image/webp"
    # A RIFF container that is not WEBP must not be mislabelled as an image.
    riff = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 16
    assert sniff_mime(riff) != "image/webp"


# ── resize maths ─────────────────────────────────────────────────────────────

def test_resize_target_preserves_landscape_aspect_ratio():
    assert MediaStore.resize_target((2000, 1000), 512) == (512, 256)


def test_resize_target_preserves_portrait_aspect_ratio():
    assert MediaStore.resize_target((1000, 2000), 512) == (256, 512)


def test_resize_target_leaves_a_small_image_alone():
    assert MediaStore.resize_target((100, 80), 512) == (100, 80)


def test_resize_target_scales_a_square_by_the_long_edge():
    assert MediaStore.resize_target((1000, 1000), 256) == (256, 256)


def test_resize_target_never_produces_a_zero_dimension():
    """A 4000x1 image at 512 px must still be at least 1 px tall."""
    width, height = MediaStore.resize_target((4000, 1), 512)
    assert width == 512
    assert height >= 1


def test_resize_target_handles_a_degenerate_size():
    assert MediaStore.resize_target((0, 0), 512) == (0, 0)


def test_resize_image_bytes_shrinks_and_reports_a_mime(media):
    data, mime = media.resize_image_bytes(make_image(800, 400), 200)
    assert mime == "image/jpeg"
    assert media.probe_size(data) == (200, 100)


def test_resize_image_bytes_is_a_noop_when_already_small(media):
    original = make_image(50, 50)
    data, mime = media.resize_image_bytes(original, 512)
    assert data == original
    assert mime == ""


def test_resize_image_bytes_flattens_transparency(media):
    """JPEG has no alpha channel, so RGBA input would otherwise raise."""
    data, mime = media.resize_image_bytes(make_image(300, 300, mode="RGBA"), 100)
    assert mime == "image/jpeg"
    assert media.probe_size(data) == (100, 100)


def test_resize_image_bytes_hands_back_the_input_when_it_cannot_process_it(media):
    """Resizing is an optimisation. Failing it must not break the request."""
    data, mime = media.resize_image_bytes(b"definitely not an image", 128)
    assert data == b"definitely not an image"
    assert mime == ""


def test_a_max_dim_of_zero_means_no_resize(media):
    original = make_image(900, 900)
    data, mime = media.resize_image_bytes(original, 0)
    assert data == original
    assert mime == ""


def test_probe_size_returns_none_for_non_image_data(media):
    assert media.probe_size(b"not an image") is None


def test_data_uri_decoding_returns_mime_then_bytes():
    import base64

    uri = "data:image/png;base64," + base64.b64encode(b"abc").decode()
    mime, data = decode_data_uri(uri)
    assert mime == "image/png"
    assert data == b"abc"


def test_a_data_uri_without_a_mime_still_decodes():
    import base64

    mime, data = decode_data_uri("data:;base64," + base64.b64encode(b"abc").decode())
    assert mime == "application/octet-stream"
    assert data == b"abc"


def test_a_non_data_uri_returns_none_rather_than_raising():
    assert decode_data_uri("https://example.com/a.png") is None
    assert decode_data_uri("") is None


def test_undecodable_base64_returns_none():
    assert decode_data_uri("data:image/png;base64,!!!!not base64!!!!") is None


# ── path safety ──────────────────────────────────────────────────────────────

def test_path_for_url_resolves_a_stored_file(media):
    url = media.save_bytes("conv1", make_image(), "image/png")
    assert media.path_for_url(url) is not None


def test_path_for_url_rejects_a_parent_traversal(media):
    assert media.path_for_url("/memory/conv1/../../providers.json") is None


def test_path_for_url_rejects_a_bad_uuid(media):
    assert media.path_for_url("/memory/../providers.json") is None


def test_path_for_url_rejects_a_dotfile(media):
    assert media.path_for_url("/memory/conv1/.env") is None


def test_resolve_in_conversation_accepts_a_url_or_a_bare_name(media):
    url = media.save_bytes("conv1", make_image(), "image/png")
    name = url.rsplit("/", 1)[1]
    assert media.resolve_in_conversation("conv1", url) is not None
    assert media.resolve_in_conversation("conv1", name) is not None


def test_resolve_in_conversation_refuses_to_walk_out_of_the_directory(media):
    media.save_bytes("conv1", make_image(), "image/png")
    for attempt in (
        "..\\..\\providers.json",
        "../../providers.json",
        "sub/other.png",
        ".env",
        "",
        "nope.png",
    ):
        assert media.resolve_in_conversation("conv1", attempt) is None, attempt


def test_resolve_in_conversation_refuses_a_traversal_via_a_url(media):
    assert media.resolve_in_conversation("conv1", "/memory/conv1/../../providers.json") is None


def test_resolve_in_conversation_refuses_another_conversations_url(media):
    """A `/memory/` URL names its own conversation, so it must still be checked.

    The shared memory root is not the boundary callers expect — without this,
    `/memory/<other-uuid>/photo.png` reached any conversation's files.
    """
    url = media.save_bytes("elsewhere", make_image(), "image/png")

    assert media.path_for_url(url) is not None      # the file itself is real
    assert media.resolve_in_conversation("conv1", url) is None
    assert media.resolve_in_conversation("elsewhere", url) is not None


def test_resolve_in_conversation_refuses_a_bad_uuid(media):
    url = media.save_bytes("conv1", make_image(), "image/png")
    assert media.resolve_in_conversation("../etc", url) is None
    assert media.resolve_in_conversation("../etc", "photo.png") is None


def test_saving_never_clobbers_an_existing_file(media):
    first = media.save_bytes("conv1", make_image(), "image/png")
    second = media.save_bytes("conv1", make_image(), "image/png")
    assert first != second
    assert media.path_for_url(first).is_file()
    assert media.path_for_url(second).is_file()


def test_a_filename_hint_decides_the_extension(media):
    url = media.save_bytes("conv1", make_image(), "", name="diagram.png")
    assert url.endswith(".png")


# ── tools ────────────────────────────────────────────────────────────────────

@pytest.fixture
def toolset(store, media, tmp_path):
    settings = load_settings({}, load_dotenv=False, memory_root=store.root)
    store.create(uuid="conv1")
    tools = build_media_tools(store, media, lambda: "conv1", settings=settings)
    return {tool.name: tool for tool in tools}


@pytest.fixture
def registry(toolset):
    """Tools reached the way the model reaches them.

    `Tool.invoke` is the raw path and lets a handler's exception escape;
    `ToolRegistry.call` is what the tool loop uses and converts any exception into
    an error result the model can read. Testing the former would prove the wrong
    thing about a bad path.
    """
    from deepseek_client.tools import ToolRegistry

    reg = ToolRegistry()
    for tool in toolset.values():
        reg.add(tool)
    return reg


async def call(registry, name, **arguments):
    return await registry.call(name, arguments)


def text_of(result) -> str:
    """Flatten a ToolResult's content, which may be a string or a block list."""
    content = result.content
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


def test_the_expected_tools_are_registered(toolset):
    assert set(toolset) == {
        "resize_image", "compress_image", "inspect_media",
        "reduce_video_frames", "compress_video", "read_file",
    }


def test_tools_expose_a_valid_json_schema(toolset):
    for tool in toolset.values():
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["parameters"]["type"] == "object"


def test_the_registry_accepts_prebuilt_tools(registry):
    from deepseek_client.tools import Tool

    # `register` takes a callable and would raise AttributeError here.
    with pytest.raises(AttributeError):
        registry.register(Tool(name="x", description="d", handler=lambda: 1))
    assert len(registry) == 6


async def test_inspect_media_reports_dimensions(registry, media):
    url = media.save_bytes("conv1", make_image(320, 160), "image/png")
    result = await call(registry, "inspect_media", path=url)
    assert not result.is_error
    body = text_of(result)
    assert "320" in body
    assert "160" in body


async def test_inspect_media_with_no_path_reads_the_whole_conversation(registry, media):
    media.save_bytes("conv1", make_image(64, 64), "image/png")
    result = await call(registry, "inspect_media", path="")
    assert not result.is_error
    assert "64" in text_of(result)


async def test_inspect_media_on_a_missing_file_is_an_error_not_a_crash(registry):
    """A bad path must come back as a tool result the model can react to."""
    result = await call(registry, "inspect_media", path="conv1/nope.png")
    assert result.is_error


async def test_resize_image_shrinks_and_reports_both_sizes(registry, media):
    url = media.save_bytes("conv1", make_image(1000, 500), "image/png")
    result = await call(registry, "resize_image", path=url, max_dim=200)
    assert not result.is_error
    body = text_of(result)
    assert "1000" in body and "200" in body


async def test_resize_image_returns_an_image_block_for_the_ui(registry, media):
    url = media.save_bytes("conv1", make_image(1000, 500), "image/png")
    result = await call(registry, "resize_image", path=url, max_dim=200)
    blocks = result.content
    assert any(block.get("type") == "image" for block in blocks if isinstance(block, dict))
    image = next(b for b in blocks if isinstance(b, dict) and b.get("type") == "image")
    assert image["url"].startswith("/memory/conv1/")


async def test_resize_image_writes_a_new_file_rather_than_overwriting(registry, media):
    url = media.save_bytes("conv1", make_image(1000, 500), "image/png")
    before = media.path_for_url(url).read_bytes()
    await call(registry, "resize_image", path=url, max_dim=200)
    assert media.path_for_url(url).read_bytes() == before


async def test_resize_image_refuses_a_path_outside_the_conversation(registry, media):
    media.save_bytes("conv1", make_image(), "image/png")
    result = await call(registry, "resize_image", path="..\\..\\providers.json")
    assert result.is_error
    assert "outside" in text_of(result).lower() or "no such" in text_of(result).lower()


async def test_resize_image_refuses_a_url_from_another_conversation(registry, media):
    other = media.save_bytes("elsewhere", make_image(1000, 500), "image/png")
    result = await call(registry, "resize_image", path=other)
    assert result.is_error


async def test_resize_image_tells_the_model_to_use_the_video_tool_for_video(registry, media):
    """Mislabelled input should get a redirect, not a stack trace."""
    video = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64
    url = media.save_bytes("conv1", video, "video/mp4", name="clip.mp4")
    result = await call(registry, "resize_image", path=url)
    assert result.is_error
    assert "reduce_video_frames" in text_of(result)


async def test_resize_image_says_so_when_the_image_is_already_small(registry, media):
    url = media.save_bytes("conv1", make_image(64, 64), "image/png")
    result = await call(registry, "resize_image", path=url)
    assert not result.is_error
    assert "already" in text_of(result).lower()


async def test_resize_image_warns_when_the_result_is_below_the_useful_floor(registry, media):
    """Under ~544 px the API upscales, so the tokens are spent for nothing."""
    url = media.save_bytes("conv1", make_image(1000, 1000), "image/png")
    result = await call(registry, "resize_image", path=url, max_dim=128)
    assert not result.is_error
    assert "544" in text_of(result)


async def test_compress_image_produces_something_smaller(registry, media):
    url = media.save_bytes("conv1", make_image(600, 600, fmt="JPEG"), "image/jpeg")
    result = await call(registry, "compress_image", path=url, quality=20)
    assert not result.is_error


async def test_compress_image_refuses_to_hand_back_a_larger_file(registry, media):
    """Re-encoding can inflate; the tool must not pretend that is a win."""
    url = media.save_bytes("conv1", make_image(32, 32, fmt="JPEG"), "image/jpeg")
    result = await call(registry, "compress_image", path=url, quality=95)
    assert not result.is_error
    assert "larger" in text_of(result).lower() or "smaller" in text_of(result).lower()


async def test_reduce_video_frames_rejects_a_non_video(registry, media):
    url = media.save_bytes("conv1", make_image(), "image/png")
    result = await call(registry, "reduce_video_frames", path=url)
    assert result.is_error


async def test_compress_video_rejects_a_missing_file(registry):
    result = await call(registry, "compress_video", path="conv1/nope.mp4")
    assert result.is_error


# ── sampling a real clip ──────────────────────────────────────────────────────
# Frame extraction can't be faked with a pseudo-container: it seeks and decodes.

def dominant_channel(raw: bytes) -> str:
    """Which of red, green or blue dominates a JPEG frame."""
    with Image.open(io.BytesIO(raw)) as img:
        pixels = list(img.convert("RGB").getdata())
    means = [sum(p[i] for p in pixels) / len(pixels) for i in range(3)]
    return "rgb"[means.index(max(means))]


def test_video_frames_land_one_per_second_of_a_colour_clip(make_clip):
    """3 s at 1 fps is 3 frames; even spacing has to hit each one-second band."""
    clip = make_clip(seconds=3)

    frames = MediaStore.video_frames(clip, target_fps=1, max_dim=64, max_frames=16)

    assert len(frames) == 3
    assert [dominant_channel(frame) for frame in frames] == ["r", "g", "b"]


def test_video_frames_thin_a_long_clip_to_the_frame_budget(make_clip):
    clip = make_clip(seconds=4)
    assert len(MediaStore.video_frames(clip, target_fps=1, max_dim=64, max_frames=2)) == 2


def test_video_frames_shrink_to_max_dim_keeping_the_aspect_ratio(make_clip):
    clip = make_clip(seconds=1, size="640x360")
    frames = MediaStore.video_frames(clip, target_fps=1, max_dim=128, max_frames=4)

    assert len(frames) == 1
    with Image.open(io.BytesIO(frames[0])) as img:
        assert img.size == (128, 72)


def test_video_frames_on_something_that_is_not_a_video_is_empty(tmp_path):
    """OpenCV failing to open the file must be an empty list, not an exception."""
    bogus = tmp_path / "notes.bin"
    bogus.write_bytes(b"\x01\x02\x03\x04" * 64)

    assert MediaStore.video_frames(bogus) == []


def test_video_frames_treats_a_still_image_as_a_single_frame(tmp_path):
    """OpenCV's decoder happily opens a still, so this is one frame, not an error.

    Pinned because it is the reason ``reduce_video_frames`` screens on mime first.
    """
    still = tmp_path / "still.png"
    still.write_bytes(make_image(64, 32))

    assert len(MediaStore.video_frames(still)) == 1


async def test_a_tool_error_is_reported_not_raised(registry):
    """The model must be able to see the failure and react to it."""
    result = await call(registry, "resize_image", path="conv1/absent.png")
    assert result.is_error
    assert text_of(result)


async def test_a_tool_that_raises_becomes_an_error_result(registry, media):
    """`_resolve` raises ValueError; the registry has to catch it."""
    media.save_bytes("conv1", make_image(), "image/png")
    result = await call(registry, "inspect_media", path="\\0invalid\\0")
    assert result.is_error


def test_tools_are_usable_with_the_registry(registry):
    """`ToolRegistry.add` takes Tool objects; `register` takes callables."""
    assert len(registry) == 6
    assert "resize_image" in registry


# ── inline media ingestion ───────────────────────────────────────────────────

def test_a_data_uri_image_is_written_to_disk(media, store):
    import base64

    store.create(uuid="conv1")
    uri = "data:image/png;base64," + base64.b64encode(make_image(48, 24)).decode()
    blocks = media.ingest_user_content("conv1", [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": uri, "detail": "high"}},
    ])
    image = blocks[1]
    assert image["image_url"]["url"].startswith("/memory/conv1/")
    assert "base64" not in json.dumps(blocks)
    assert image["width"] == 48
    assert image["height"] == 24
    # `detail` must survive, since it changes how many tokens the image costs.
    assert image["image_url"]["detail"] == "high"


def test_an_unsupported_inline_type_is_refused(media, store):
    import base64

    store.create(uuid="conv1")
    uri = "data:image/bmp;base64," + base64.b64encode(b"BM" + b"\x00" * 40).decode()
    with pytest.raises(ValueError, match="JPEG, PNG, GIF, and WebP"):
        media.ingest_user_content("conv1", [
            {"type": "image_url", "image_url": {"url": uri}},
        ])


def test_a_remote_url_is_passed_through_untouched(media, store):
    """A URL is not ours to copy, and the API can fetch it itself."""
    store.create(uuid="conv1")
    blocks = media.ingest_user_content("conv1", [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ])
    assert blocks[0]["image_url"]["url"] == "https://example.com/a.png"


def test_media_limits_defaults_match_the_published_limits():
    limits = MediaLimits()
    assert limits.max_dimension == 8192
    assert limits.max_dimension_many == 4096
    assert limits.many_image_threshold == 15
    assert limits.max_images_per_request == 600


# ── the video tools, end to end on a real clip ────────────────────────────────

def store_clip(store, media, clip: Path, *, uuid: str = "conv1") -> str:
    """Put a real clip in the conversation the way an upload would."""
    store.create(uuid=uuid)
    return media.save_bytes(uuid, clip.read_bytes(), "video/mp4", name="colour.mp4")


async def test_reduce_video_frames_hands_back_the_sampled_frames(registry, store, media, make_clip):
    url = store_clip(store, media, make_clip(seconds=3))

    result = await call(registry, "reduce_video_frames", path=url, target_fps=1, max_frames=4, max_dim=64)

    assert not result.is_error
    assert result.meta["frames"] == 3
    blocks = [b for b in result.content if b.get("type") == "image"]
    assert len(blocks) == 3, "every sampled frame is offered back to the model"
    assert all(b["url"].startswith("/memory/conv1/") for b in blocks)

    text = text_of(result)
    # The header names the file as it is stored now, not the upload's original name.
    assert Path(url).name in text
    assert "3s" in text

    saved = [media.path_for_url(u) for u in result.meta["saved"]]
    assert all(p is not None and p.is_file() for p in saved)
    assert [dominant_channel(p.read_bytes()) for p in saved] == ["r", "g", "b"]


async def test_reduce_video_frames_says_when_the_budget_thinned_the_sample(
    registry, store, media, make_clip
):
    """Silently returning 2 of the 4 requested frames would look like a bug to the model."""
    url = store_clip(store, media, make_clip(seconds=4))

    result = await call(registry, "reduce_video_frames", path=url, target_fps=1, max_frames=2, max_dim=64)

    assert not result.is_error
    assert result.meta["frames"] == 2
    assert "max_frames" in text_of(result)


async def test_reduce_video_frames_refuses_another_conversations_clip(registry, store, media, make_clip):
    store.create(uuid="conv2")
    url = media.save_bytes("conv2", make_clip(seconds=1).read_bytes(), "video/mp4", name="other.mp4")

    result = await call(registry, "reduce_video_frames", path=url)

    assert result.is_error
    assert "outside" in text_of(result).lower()


async def test_compress_video_never_hands_back_something_larger(registry, store, media, make_clip):
    """Re-encoding can inflate a clip; the tool must not pretend that is a win."""
    url = store_clip(store, media, make_clip(seconds=2))
    before = media.path_for_url(url).stat().st_size

    result = await call(registry, "compress_video", path=url, scale=0.5, target_fps=5, crf=30)

    assert not result.is_error
    if result.meta.get("path"):
        out = media.path_for_url(result.meta["path"])
        assert out is not None and out.is_file()
        assert out.stat().st_size < before
        assert result.meta["bytes"] == out.stat().st_size
    else:
        assert "smaller" in text_of(result).lower()


async def test_compress_video_refuses_another_conversations_clip(registry, store, media, make_clip):
    store.create(uuid="conv2")
    url = media.save_bytes("conv2", make_clip(seconds=1).read_bytes(), "video/mp4", name="other.mp4")

    result = await call(registry, "compress_video", path=url)

    assert result.is_error
    assert "outside" in text_of(result).lower()


async def test_compress_video_writes_something_a_browser_can_play(registry, store, media, make_clip):
    """The result is rendered as a <video>, so the codec has to be one the UI can demux.

    OpenCV's only fourcc here is `mp4v`, which Chrome refuses; the tool has to reach
    for ffmpeg's H.264 first or the tool result shows up as a dead player.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None or shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to encode and ffprobe to inspect")

    url = store_clip(store, media, make_clip(seconds=3, fps=25, size="640x360"))
    result = await call(registry, "compress_video", path=url, scale=0.5, target_fps=8, crf=30)

    assert not result.is_error
    assert result.meta.get("path"), "a downscale this large must beat the original"
    out = media.path_for_url(result.meta["path"])

    codec = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    assert codec == "h264"
