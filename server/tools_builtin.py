"""Built-in tools for shaping media before it reaches the model.

These exist because of how the API bills images. Every image is resized to roughly
1300x1300 px and charged at up to 1024 tokens, with a floor around 544x544 — so a
tiny thumbnail is *upscaled* and billed nearly as much as a large photo, and a
6000 px screenshot is billed the same as a 1300 px one while being four times the
body size. Resizing before upload is therefore not just tidy, it is the difference
between a usable transcript and one that burns context.

Frame-count reduction matters for the same reason on video: a 90 second clip at
1 fps is 90 images, ~92k tokens, for a scene that three frames would describe.

Every path argument is resolved against the conversation directory and cannot
escape it.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from deepseek_client.tools import Tool, ToolResult, tool_schema

from .store import ConversationError

__all__ = ["build_media_tools"]

logger = logging.getLogger("deepseek_ui.tools")

#: A re-encode is a bounded piece of work; a wedged encoder must not hold a turn open.
ENCODE_TIMEOUT = 300

#: Below this, DeepSeek upscales the image anyway, so shrinking further wastes detail.
MIN_USEFUL_DIM = 544


def _url_for_attached_name(store, media, uuid: str, name: str) -> str | None:
    """Find the stored URL of a file the user attached under ``name``.

    An attachment keeps the name it was given in its ``_file`` block, but the bytes
    land under a generated, collision-free name. Without this, the continuation
    hints that ``rehydrate`` writes — which necessarily talk about the *attached*
    name — name a path that does not exist: a 40 KB ``routes.py`` arrives with
    "call read_file('routes.py', offset=40000)" and that call fails, so the model
    can read the first 40,000 characters of a file and never the rest.

    Newest match wins, since re-reading the most recent attachment is the intent.
    """
    if Path(name).name != name:
        return None                       # a path, not a bare name
    try:
        conversation = store.load(uuid)
    except (ConversationError, OSError):
        return None                       # unreadable conversation: nothing to match

    candidates: list[str] = []
    for message in conversation.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            url = block.get("url")
            if not isinstance(url, str) or not url.startswith("/memory/"):
                continue
            if block.get("name") == name or Path(url).name == name:
                candidates.append(url)

    for url in reversed(candidates):
        if media.path_for_url(url) is not None:
            return url
    return None


def _resolve(store, media, uuid: str, path: str) -> Path:
    """Resolve a tool's path argument inside the conversation directory.

    Accepts a ``/memory/<uuid>/<file>`` URL, a name relative to the conversation,
    an attachment's original name, or an absolute path that is already inside it.
    Anything else is refused.

    The containment check applies to *every* form. A ``/memory/`` URL used to be
    trusted because ``path_for_url`` guarantees it sits under ``memory/`` — but that
    is the root of every conversation, so ``/memory/<some-other-uuid>/photo.png``
    walked straight past the per-conversation boundary.
    """
    if not path:
        raise ValueError("path is required")

    allowed = store.dir(uuid).resolve()

    if path.startswith("/memory/"):
        resolved = media.path_for_url(path)
        if resolved is None:
            raise ValueError(f"no such media: {path}")
        if allowed not in resolved.resolve().parents:
            raise ValueError("path is outside the conversation directory")
        return resolved

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = store.dir(uuid) / candidate
    candidate = candidate.resolve()

    if candidate != allowed and allowed not in candidate.parents:
        raise ValueError("path is outside the conversation directory")
    if candidate.is_file():
        return candidate

    # Not on disk under that name — it may be the name it was attached as.
    url = _url_for_attached_name(store, media, uuid, path)
    if url is not None:
        resolved = media.path_for_url(url)
        if resolved is not None and allowed in resolved.resolve().parents:
            return resolved

    raise ValueError(f"no such file: {path}")


def _as_media_url(store, path: Path) -> str:
    return f"/memory/{path.parent.name}/{path.name}"


def _image_block(store, media, path: Path) -> dict[str, Any]:
    """An ``image`` block referencing a file already on disk."""
    return {"type": "image", "url": _as_media_url(store, path), "name": path.name}


def build_media_tools(
    store,
    media,
    uuid_provider: Callable[[], str | None],
    *,
    settings=None,
) -> list[Tool]:
    """Create the built-in media tools, bound to the active conversation."""

    def current() -> str:
        uuid = uuid_provider() if callable(uuid_provider) else None
        if not uuid:
            raise ValueError("no active conversation")
        return uuid

    # ── resize_image ──
    async def resize_image(path: str, max_dim: int = 1280, quality: int = 88) -> ToolResult:
        uuid = current()
        source = _resolve(store, media, uuid, path)
        if source.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv"):
            return ToolResult.error("resize_image expects an image; use reduce_video_frames for video")

        original = source.stat().st_size
        data = source.read_bytes()
        before = media.probe_size(data)
        resized, new_mime = media.resize_image_bytes(data, int(max_dim), quality=int(quality))

        if new_mime == "":
            note = "already within the target size" if before else "could not be processed"
            return ToolResult(
                content=f"{source.name}: {note} ({_fmt_bytes(original)})",
                meta={"path": _as_media_url(store, source)},
            )

        saved = media.save_bytes(uuid, resized, new_mime, prefix=f"{source.stem}_resized")
        target = media.path_for_url(saved)
        after = media.probe_size(resized)
        block = _image_block(store, media, target) if target else None

        summary = (
            f"Resized {source.name}: "
            f"{_dims(before)} ({_fmt_bytes(original)}) -> "
            f"{_dims(after)} ({_fmt_bytes(len(resized))})"
        )
        if after and max(after) < MIN_USEFUL_DIM:
            summary += (
                "\nNote: the result is below ~544 px, which the API will upscale back up "
                "while charging nearly full token cost. Prefer a larger max_dim."
            )
        if before and after and max(after) < max(before):
            summary += f"\nSaved to {saved}"

        parts: list[dict[str, Any]] = [{"type": "text", "text": summary}]
        if block:
            parts.append(block)
        return ToolResult(content=parts, meta={"path": saved})

    # ── compress_image ──
    async def compress_image(path: str, quality: int = 70, max_dim: int = 0) -> ToolResult:
        uuid = current()
        source = _resolve(store, media, uuid, path)
        data = source.read_bytes()
        original = len(data)

        try:
            from PIL import Image
        except ImportError:
            return ToolResult.error("Pillow is not installed; cannot compress images")

        import io

        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                if max_dim and max_dim > 0 and max(img.size) > max_dim:
                    img = img.resize(
                        media.resize_target(img.size, max_dim), Image.LANCZOS
                    )
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=int(quality), optimize=True, progressive=True)
                compressed = buffer.getvalue()
                dimensions = img.size
        except Exception as exc:
            return ToolResult.error(f"could not re-encode {source.name}: {exc}")

        if len(compressed) >= original:
            return ToolResult(
                content=(
                    f"{source.name} is already well compressed "
                    f"({_fmt_bytes(original)}); re-encoding would make it larger."
                ),
                meta={"path": _as_media_url(store, source)},
            )

        saved = media.save_bytes(uuid, compressed, "image/jpeg", prefix=f"{source.stem}_compressed")
        target = media.path_for_url(saved)
        ratio = 100 * (1 - len(compressed) / original)
        parts: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Compressed {source.name} at quality {quality}: "
                f"{_fmt_bytes(original)} -> {_fmt_bytes(len(compressed))} "
                f"({ratio:.0f}% smaller, {_dims(dimensions)})"
            ),
        }]
        if target:
            parts.append(_image_block(store, media, target))
        return ToolResult(content=parts, meta={"path": saved})

    # ── inspect_media ──
    async def inspect_media(path: str = "") -> ToolResult:
        uuid = current()
        directory = store.dir(uuid)

        if not path:
            entries = sorted(
                (p for p in directory.iterdir() if p.is_file() and p.name != "conversation.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not entries:
                return ToolResult(content="No media in this conversation yet.")

            # Report dimensions and cost, not just names: deciding whether to resize
            # is the whole reason to look at this list, and both of those numbers are
            # what the decision turns on.
            lines: list[str] = []
            total_tokens = 0
            for entry in entries[:60]:
                size = _fmt_bytes(entry.stat().st_size)
                data = entry.read_bytes()
                dims = media.probe_size(data)
                if dims:
                    from deepseek_client.messages import image_tokens

                    cost = image_tokens(*dims)
                    total_tokens += cost
                    lines.append(f"{entry.name} — {dims[0]}x{dims[1]}, {size}, ~{cost} tokens")
                else:
                    probe = _video_probe(entry)
                    if probe:
                        fps = settings.model_video_fps if settings else 1
                        cap = settings.model_video_max_frames if settings else 16
                        frames = max(1, min(cap, round(probe["duration"] * fps)))
                        total_tokens += frames * 1024
                        dims = (probe.get("width") or 0, probe.get("height") or 0)
                        lines.append(
                            f"{entry.name} — {_dims(dims) if all(dims) else 'video'}, "
                            f"{_fmt_duration(probe['duration'])}, {size}, "
                            f"~{frames} frames if sampled (~{frames * 1024} tokens)"
                        )
                    else:
                        document = media.read_text(entry, max_chars=200_000)
                        if document is not None:
                            lines.append(
                                f"{entry.name} — text, {size}, {document.lines:,} lines "
                                f"({document.encoding})"
                            )
                        else:
                            lines.append(f"{entry.name} — unreadable, {size}")

            if len(entries) > 60:
                lines.append(f"... and {len(entries) - 60} more")
            header = f"Media in this conversation ({len(entries)} files):"
            if total_tokens:
                header += f"\nRoughly {total_tokens} tokens if all of it were sent."
            return ToolResult(content=header + "\n" + "\n".join(lines))

        source = _resolve(store, media, uuid, path)
        data = source.read_bytes()
        info: dict[str, Any] = {
            "name": source.name,
            "url": _as_media_url(store, source),
            "bytes": len(data),
        }
        size = media.probe_size(data)
        if size:
            info["kind"] = "image"
            info["width"], info["height"] = size
            from deepseek_client.messages import image_tokens

            info["estimated_tokens"] = image_tokens(*size)
            info["note"] = (
                "Images are charged as if resized to ~1300x1300 px, so this is close to the "
                "cost of any image this size or larger."
            )
        else:
            probe = _video_probe(source)
            if probe:
                info.update(probe)
                info["kind"] = "video"
                frames = max(1, min(settings.model_video_max_frames if settings else 16,
                                    round(probe["duration"] * (settings.model_video_fps if settings else 1))))
                info["frames_if_sampled"] = frames
                info["estimated_tokens"] = frames * 1024
            else:
                # Not an image and not a video. It is very likely text — a `.md`, a
                # `.txt`, or something with no extension at all like `.gitignore` or
                # `Makefile` — and "unknown" told the model nothing it could act on.
                document = media.read_text(source)
                if document is None:
                    info["kind"] = "binary"
                    info["note"] = (
                        "Not text, and not an image or video this server can read. "
                        "Its contents cannot be shown to you."
                    )
                else:
                    info["kind"] = "text"
                    info["charset"] = document.encoding
                    info["characters"] = len(document.text)
                    info["lines"] = document.lines
                    if document.truncated:
                        info["truncated"] = True
                    info["note"] = (
                        "Read it with read_file(path=...) — it may be paged with "
                        "offset and limit if it is long."
                    )
                    if settings:
                        limit = getattr(settings, "model_text_max_chars", 40_000)
                        info["inlined_on_attach"] = min(len(document.text), int(limit))
        return ToolResult(content=json.dumps(info, indent=2), meta=info)

    # ── reduce_video_frames ──
    async def reduce_video_frames(
        path: str,
        target_fps: float = 1,
        max_frames: int = 16,
        max_dim: int = 512,
    ) -> ToolResult:
        uuid = current()
        source = _resolve(store, media, uuid, path)
        probe = _video_probe(source)
        if probe is None:
            return ToolResult.error(f"{source.name} is not a readable video")

        duration = probe["duration"]
        wanted = max(1, round(duration * float(target_fps)))
        effective = min(wanted, max(1, int(max_frames)))

        frames = media.video_frames(
            source,
            target_fps=float(target_fps),
            max_dim=int(max_dim),
            max_frames=int(max_frames),
        )
        if not frames:
            return ToolResult.error(
                "opencv-python is not installed or the video could not be decoded"
            )

        blocks: list[dict[str, Any]] = []
        saved: list[str] = []
        for index, frame in enumerate(frames):
            url = media.save_bytes(uuid, frame, "image/jpeg", prefix=f"{source.stem}_f{index:03d}")
            target = media.path_for_url(url)
            if target:
                blocks.append(_image_block(store, media, target))
                saved.append(url)

        header = (
            f"{source.name}: {_fmt_duration(duration)} at {probe['fps']:.1f} fps "
            f"({_dims((probe['width'], probe['height']))}).\n"
            f"Sampled {len(frames)} frames at ~{effective / max(duration, 0.001):.2f} fps, "
            f"longest edge {max_dim} px."
        )
        if wanted > effective:
            header += (
                f"\nNote: {wanted} frames at {target_fps} fps was reduced to {effective} to stay "
                f"within the frame budget. Raise max_frames if the detail matters."
            )
        header += f"\nEstimated cost: ~{len(frames) * 1024} tokens for the frames."
        return ToolResult(
            content=[{"type": "text", "text": header}, *blocks],
            meta={"frames": len(frames), "saved": saved},
        )

    # ── compress_video ──
    async def compress_video(
        path: str,
        scale: float = 0.5,
        target_fps: float = 8,
        crf: int = 30,
    ) -> ToolResult:
        """Re-encode a video smaller. Frames are what cost tokens; this is for storage."""
        uuid = current()
        source = _resolve(store, media, uuid, path)
        probe = _video_probe(source)
        if probe is None:
            return ToolResult.error(f"{source.name} is not a readable video")

        try:
            import cv2
        except ImportError:
            return ToolResult.error("opencv-python is not installed; cannot re-encode video")

        capture = cv2.VideoCapture(str(source))
        directory = store.dir(uuid)
        out_path = directory / f"{source.stem}_compressed_{int(crf)}.mp4"

        import os
        import tempfile

        # mkstemp hands back an already-open handle, and Windows refuses to delete a
        # file while one is alive — unlinking before closing raised WinError 32 every
        # time, so this tool never once succeeded on Windows. Close first, then clear
        # the path so VideoWriter starts from a clean slate.
        handle, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(directory))
        os.close(handle)
        tmp = Path(tmp_name)
        tmp.unlink(missing_ok=True)

        try:
            fps = float(target_fps) if target_fps else probe["fps"]
            out_size = (
                max(2, int(probe["width"] * float(scale)) // 2 * 2),
                max(2, int(probe["height"] * float(scale)) // 2 * 2),
            )
            # ffmpeg first: it writes H.264, which every browser plays. OpenCV can only
            # offer the ancient `mp4v` fourcc here (`avc1` wants an OpenH264 DLL that
            # does not ship with opencv-python), and the UI could not demux mp4v — so a
            # freshly compressed clip appeared in the transcript as a dead <video>.
            written = _encode_h264(source, tmp, out_size, fps, crf)
            if written is None:
                written = _encode_with_opencv(capture, tmp, out_size, fps, probe, cv2)
                if written is None:
                    return ToolResult.error("could not open a video writer (mp4v unavailable)")
        finally:
            capture.release()

        if not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return ToolResult.error("re-encode produced no output")

        original = source.stat().st_size
        new_size = tmp.stat().st_size
        os.replace(tmp, out_path)

        if new_size >= original:
            out_path.unlink(missing_ok=True)
            return ToolResult(
                content=f"{source.name} is already smaller ({_fmt_bytes(original)}) than any re-encode produced; kept the original."
            )

        ratio = 100 * (1 - new_size / original)
        return ToolResult(
            content=(
                f"Re-encoded {source.name}: {_fmt_bytes(original)} -> {_fmt_bytes(new_size)} "
                f"({ratio:.0f}% smaller), {written} frames at {fps:g} fps, {_dims(out_size)}."
            ),
            meta={"path": _as_media_url(store, out_path), "bytes": new_size},
        )

    # ── read_file ──
    async def read_file(path: str, offset: int = 0, limit: int = 20_000) -> ToolResult:
        """Read a text file, with line numbers so a window can be continued.

        This is what makes an attached `.md`, `.txt` or extension-less config file
        usable: the model can page through one rather than being told that "a file
        is attached" and left to guess at its contents.
        """
        uuid = current()
        source = _resolve(store, media, uuid, path)
        document = media.read_text(source)
        # Echo the name the caller used. It is the attachment's name when they followed
        # a hint, and the generated name when they passed a URL — either way it is what
        # they already have, rather than a third spelling to keep track of.
        label = Path(path).name or source.name
        if document is None:
            return ToolResult.error(
                f"{label} is not readable as text "
                f"({_fmt_bytes(source.stat().st_size)} of binary data)"
            )

        text = document.text
        start = max(0, int(offset or 0))
        if start and start >= len(text):
            return ToolResult.error(
                f"offset {start} is past the end of {label} ({len(text):,} characters)"
            )

        span = max(1, int(limit or 20_000))
        window = text[start:start + span]
        first_line = text.count("\n", 0, start) + 1
        numbered = "\n".join(
            f"{first_line + i:>5}  {line}" for i, line in enumerate(window.splitlines())
        )

        header = (
            f"{label} — {_fmt_bytes(document.bytes)}, {document.lines:,} lines, "
            f"{document.encoding}; showing characters {start:,}-{start + len(window):,} "
            f"(lines {first_line}-{first_line + window.count(chr(10))})"
        )
        parts: list[str] = [header, numbered]
        consumed = start + len(window)
        if consumed < len(text):
            parts.append(f"[more follows — call read_file(path={label!r}, offset={consumed})]")
        elif document.truncated:
            # read_text reads a bounded window from the start, so a file larger than
            # that is not fully reachable — say so instead of implying the end.
            parts.append(
                "[this file is larger than the 8 MiB text read window; only its beginning "
                "can be shown]"
            )
        return ToolResult(
            content="\n".join(parts),
            meta={
                "path": _as_media_url(store, source),
                "characters": len(text),
                "returned": len(window),
                "truncated": document.truncated,
            },
        )

    tools = [
        Tool(
            name="resize_image",
            description=(
                "Downscale an image so its longest edge is at most `max_dim` pixels. Images are "
                "charged up to 1024 tokens each regardless of size, so resizing saves request size "
                "without changing cost — it is worth doing for anything large. Returns the resized "
                "image so you can see the result."
            ),
            parameters=tool_schema(
                "resize_image",
                properties={
                    "path": {"type": "string", "description": "Media URL or filename in this conversation."},
                    "max_dim": {"type": "integer", "description": "Longest edge in pixels.", "default": 1280},
                    "quality": {"type": "integer", "description": "JPEG quality, 1-100.", "default": 88},
                },
                required=["path"],
            ),
            handler=resize_image,
        ),
        Tool(
            name="compress_image",
            description=(
                "Re-encode an image as JPEG at a lower quality to shrink it on disk. Use when a "
                "file is too large to attach or store, not to save model tokens."
            ),
            parameters=tool_schema(
                "compress_image",
                properties={
                    "path": {"type": "string", "description": "Media URL or filename in this conversation."},
                    "quality": {"type": "integer", "description": "JPEG quality, 1-100.", "default": 70},
                    "max_dim": {"type": "integer", "description": "Optional longest edge before encoding. 0 keeps the size.", "default": 0},
                },
                required=["path"],
            ),
            handler=compress_image,
        ),
        Tool(
            name="inspect_media",
            description=(
                "Report a file's kind, size, and estimated token cost — dimensions for an image, "
                "duration for a video, line count and charset for a text file. Call with no path "
                "to list every file in this conversation."
            ),
            parameters=tool_schema(
                "inspect_media",
                properties={
                    "path": {"type": "string", "description": "Media URL or filename. Omit to list everything."},
                },
                required=[],
            ),
            handler=inspect_media,
        ),
        Tool(
            name="reduce_video_frames",
            description=(
                "Sample frames from a video and return them as images so you can see its content. "
                "Frames are the expensive part — each costs up to 1024 tokens — so lower target_fps "
                "or max_frames for long clips. Use this instead of guessing what a video contains."
            ),
            parameters=tool_schema(
                "reduce_video_frames",
                properties={
                    "path": {"type": "string", "description": "Media URL or filename in this conversation."},
                    "target_fps": {"type": "number", "description": "Frames per second to sample.", "default": 1},
                    "max_frames": {"type": "integer", "description": "Hard cap on returned frames.", "default": 16},
                    "max_dim": {"type": "integer", "description": "Longest edge of each frame in pixels.", "default": 512},
                },
                required=["path"],
            ),
            handler=reduce_video_frames,
        ),
        Tool(
            name="compress_video",
            description=(
                "Re-encode a video at a lower resolution and frame rate to shrink the file. Useful "
                "for storage limits; it does not change the token cost of frames you later sample."
            ),
            parameters=tool_schema(
                "compress_video",
                properties={
                    "path": {"type": "string", "description": "Media URL or filename in this conversation."},
                    "scale": {"type": "number", "description": "Resolution multiplier, 0.5 halves each edge.", "default": 0.5},
                    "target_fps": {"type": "number", "description": "Output frame rate.", "default": 8},
                    "crf": {"type": "integer", "description": "Quality label; higher means smaller. Informational only.", "default": 30},
                },
                required=["path"],
            ),
            handler=compress_video,
        ),
        Tool(
            name="read_file",
            description=(
                "Read a text file from this conversation and return its contents with line numbers: "
                "source code, config, markdown, csv, json, logs, or a file such as `.gitignore` or "
                "`Makefile`. Use this whenever you need to see what a file actually says. Long files "
                "are returned in windows — the result says how to continue from where it stopped."
            ),
            parameters=tool_schema(
                "read_file",
                properties={
                    "path": {"type": "string", "description": "Media URL or filename in this conversation."},
                    "offset": {"type": "integer", "description": "Character offset to start from. Defaults to the beginning.", "default": 0},
                    "limit": {"type": "integer", "description": "Maximum characters to return.", "default": 20000},
                },
                required=["path"],
            ),
            handler=read_file,
        ),
    ]
    return tools


# ── small helpers ──

def _video_probe(path: Path) -> dict[str, Any] | None:
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total <= 0 or fps <= 0:
            return None
        return {
            "fps": fps,
            "frames": total,
            "duration": total / fps,
            "width": width,
            "height": height,
        }
    finally:
        capture.release()


def _encode_h264(
    source: Path, target: Path, size: tuple[int, int], fps: float, crf: int
) -> int | None:
    """Re-encode to H.264/MP4 with ffmpeg; return the frame count, or ``None``.

    ``None`` means "no usable ffmpeg", not "the clip is bad", so the caller falls
    back to OpenCV. H.264, ``yuv420p`` and ``+faststart`` are jointly what make the
    result play in a browser tab rather than download as dead bytes.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(source),
        "-vf", f"scale={size[0]}:{size[1]}",
        "-r", f"{fps:g}",
        "-c:v", "libx264",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        # Audio is never sent to the model, so there is nothing to re-encode.
        "-an",
        str(target),
    ]
    try:
        done = subprocess.run(command, capture_output=True, timeout=ENCODE_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffmpeg could not re-encode %s: %s", source.name, exc)
        return None

    if done.returncode != 0:
        logger.warning(
            "ffmpeg failed on %s: %s",
            source.name,
            done.stderr.decode("utf-8", "replace").strip()[:400],
        )
        return None
    if not target.is_file() or target.stat().st_size == 0:
        return None

    probe = _video_probe(target)
    return int(probe["frames"]) if probe else 0


def _encode_with_opencv(capture, target: Path, size, fps, probe, cv2) -> int | None:
    """Fallback encoder, used when ffmpeg is not installed.

    OpenCV only ships the ``mp4v`` fourcc here, which the browser cannot demux, so
    prefer :func:`_encode_h264`; this exists so the tool still works without ffmpeg.
    """
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        return None

    source_fps = probe["fps"] or 25.0
    step = max(1.0, source_fps / fps)
    index, written, next_emit = 0, 0, 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= next_emit:
                writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
                written += 1
                next_emit += step
            index += 1
    finally:
        writer.release()
    return written


def _fmt_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _dims(size: tuple[int, int] | None) -> str:
    return f"{size[0]}x{size[1]}" if size else "unknown size"


def _fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"
