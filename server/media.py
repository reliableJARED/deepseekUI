"""Media persistence and rehydration.

The rule this module exists to enforce: **base64 never lands in
``conversation.json``**. A 4 MB screenshot is ~5.5 MB of base64, and a transcript
full of those is unreadable, un-greppable, and slow to parse. Instead the bytes go
to disk next to the conversation and the message keeps a ``/memory/<uuid>/<file>``
URL. On the way *out* the URL is turned back into a data URI for the model.

The second rule is DeepSeek's: **images are only valid in ``user`` messages.**
Anywhere else the API answers 400. Tool results are therefore split — the text
stays in the ``tool`` message, and the images become a following ``user`` message.

The third rule is that a file is only useful to the model if its *characters*
reach the request. An extension is a poor guide to that — ``.gitignore`` has no
extension at all and ``Makefile`` has no suffix either — so what counts as text is
decided from content, and the decision is recorded on the stored block as its
``kind`` so nothing downstream has to guess again.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import io
import logging
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MediaStore",
    "MediaLimits",
    "TextDocument",
    "ext_for_mime",
    "sniff_mime",
    "is_image_mime",
    "is_video_mime",
    "is_text_mime",
    "looks_like_text",
    "looks_like_text_name",
    "decode_text",
    "text_document",
    "drop_partial_character",
    "classify_kind",
    "DISPLAY_KEY",
    "DISPLAY_SHOWN",
    "DISPLAY_INLINE",
    "is_display_block",
    "is_pinned_block",
    "is_inline_block",
    "split_display_blocks",
    "mark_display_blocks",
    "display_note",
]

logger = logging.getLogger("deepseek_ui.media")

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<params>;[^,]*)?,(?P<data>.*)$", re.DOTALL)

#: Formats DeepSeek accepts. Detection is by content, so this is only a hint.
IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
VIDEO_MIMES = frozenset({"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"})

#: Text is the third kind, and it is not a formality: a file whose characters never
#: reach the request is a file the model cannot read. These types are a hint only —
#: content decides (see :func:`looks_like_text`).
TEXT_MIMES = frozenset({
    "application/json", "application/xml", "application/javascript",
    "application/x-javascript", "application/x-sh", "application/x-python",
    "application/x-yaml", "application/yaml", "application/sql",
    "application/graphql", "application/toml", "application/x-httpd-php",
    "application/x-tex", "application/x-ndjson", "application/x-perl",
    "application/x-ruby", "application/x-php",
})

#: Whole filenames that are text and carry no usable suffix at all. Both
#: ``Path(".gitignore")`` and ``Path("Makefile")`` report an empty suffix, so a
#: suffix table alone can never classify them — which is exactly why content has the
#: final say and this is only ever consulted when there is nothing to judge.
TEXT_NAMES = frozenset({
    ".gitignore", ".gitattributes", ".gitmodules", ".dockerignore", ".npmrc",
    ".editorconfig", ".env", ".flake8", ".prettierrc", ".eslintrc", ".babelrc",
    "makefile", "gnumakefile", "dockerfile", "containerfile", "license",
    "licence", "readme", "changelog", "contributing", "authors", "notice",
    "copying", "todo", "version", "procfile", "gemfile", "rakefile",
    "justfile", "vagrantfile", "cmakelists.txt",
})

TEXT_SUFFIXES = frozenset({
    ".txt", ".text", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".config", ".properties", ".env", ".xml", ".xsd", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".js", ".jsx", ".mjs", ".cjs", ".ts",
    ".tsx", ".vue", ".svelte", ".py", ".pyi", ".rb", ".go", ".rs", ".java",
    ".kt", ".kts", ".c", ".h", ".cpp", ".cxx", ".hpp", ".cs", ".php", ".pl",
    ".lua", ".r", ".jl", ".swift", ".dart", ".scala", ".sh", ".bash", ".zsh",
    ".fish", ".ps1", ".bat", ".cmd", ".sql", ".graphql", ".gql", ".proto",
    ".tex", ".bib", ".diff", ".patch", ".svg", ".gitignore",
})

#: A BOM has to be tried before anything else, and UTF-32 before UTF-16 because the
#: UTF-32 BOM starts with the UTF-16 one.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

#: A content sniff only needs a small window.
MAX_TEXT_SNIFF = 8 * 1024
#: Upper bound on what one file may pull into memory. Text is read in order to be
#: counted, and a stray 2 GB log file must not become a 2 GB read.
MAX_TEXT_READ = 8 * 1024 * 1024
#: Below this fraction of printable characters the bytes are not text. Decoding is
#: not enough on its own: a run of 0x01 bytes is perfectly valid UTF-8.
MIN_PRINTABLE_RATIO = 0.85

#: Magic-number prefixes, checked before trusting any declared MIME type. The API
#: detects formats from content, so if we pass something else through it is the
#: request that fails, not ours — better to catch it here with a clear message.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),          # refined below (RIFF....WEBP)
    (b"\x1a\x45\xdf\xa3", "video/webm"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
    (b"%PDF", "application/pdf"),
)

_EXT_FALLBACK = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
    "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/mp4": ".m4a",
    "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
    # A text file with no usable suffix of its own (`.gitignore`) still deserves an
    # extension that says what it is. `mimetypes` reads the Windows registry for
    # these, so they are only reachable when that lookup finds nothing.
    "text/plain": ".txt", "text/markdown": ".md", "text/csv": ".csv",
    "application/json": ".json", "application/xml": ".xml", "application/yaml": ".yaml",
    "text/yaml": ".yaml", "text/x-python": ".py", "text/x-sh": ".sh",
}


def ext_for_mime(mime: str) -> str:
    guessed = mimetypes.guess_extension((mime or "").split(";")[0].strip())
    if guessed:
        return guessed.replace(".jpe", ".jpg")
    return _EXT_FALLBACK.get((mime or "").lower(), ".bin")


def is_image_mime(mime: str) -> bool:
    return (mime or "").lower() in IMAGE_MIMES


def is_video_mime(mime: str) -> bool:
    return (mime or "").lower() in VIDEO_MIMES


def is_text_mime(mime: str) -> bool:
    base = (mime or "").split(";")[0].strip().lower()
    return base.startswith("text/") or base in TEXT_MIMES


def looks_like_text_name(name: str) -> bool:
    """Whether a filename *suggests* text. A hint, never the decision."""
    leaf = Path(name or "").name.lower()
    return leaf in TEXT_NAMES or Path(leaf).suffix in TEXT_SUFFIXES


def decode_text(data: bytes) -> tuple[str, str] | None:
    """Decode bytes that claim to be text, returning ``(text, encoding)``.

    ``None`` means "not text". A BOM is consumed rather than kept: Notepad and
    PowerShell 5.1 both write one, and a leading U+FEFF would otherwise be glued to
    the first line the model reads. With no BOM the bytes must be valid UTF-8, or
    Windows-1252 as the one fallback — a strict decode of that fails only on bytes
    that are not text at all, and a NUL in the head rules it out first.
    """
    if not data:
        return "", "utf-8"
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            try:
                return data.decode(encoding), encoding
            except (UnicodeDecodeError, ValueError):
                return None
    try:
        return data.decode("utf-8"), "utf-8"
    except (UnicodeDecodeError, ValueError):
        pass
    if b"\x00" not in data[:MAX_TEXT_SNIFF]:
        try:
            return data.decode("cp1252"), "cp1252"
        except (UnicodeDecodeError, ValueError):
            pass
    return None


def _printable_ratio(text: str) -> float:
    """Fraction of characters that belong in a text file.

    Tabs and newlines are text; a run of 0x01 bytes is not, and both decode happily
    as UTF-8 — so a decode alone cannot be the test.
    """
    sample = text[:MAX_TEXT_SNIFF]
    if not sample:
        return 1.0
    printable = sum(1 for char in sample if char.isprintable() or char in "\t\r\n\f\v")
    return printable / len(sample)


def text_document(data: bytes) -> tuple[str, str] | None:
    """``(text, encoding)`` when ``data`` really is text, else ``None``.

    One decision in one place, so the upload path, the tools and the rehydrator
    cannot disagree about what a text file is.
    """
    decoded = decode_text(data)
    if decoded is None:
        return None
    text, encoding = decoded
    if text and _printable_ratio(text) < MIN_PRINTABLE_RATIO:
        return None
    return text, encoding


def looks_like_text(data: bytes, name: str = "") -> bool:
    """Whether these bytes are safe to hand the model as text.

    Content decides, not the extension — that is the whole point, because
    ``.gitignore`` and ``Makefile`` have no extension to be judged by. The name is
    consulted only when the file is empty and there is nothing else to go on.
    """
    if not data:
        return looks_like_text_name(name)
    return text_document(data) is not None


def drop_partial_character(data: bytes) -> bytes:
    """Drop a character sliced in half by a bounded read.

    Reading at most N bytes can stop between the two bytes of a UTF-8 character.
    Left alone, that stray lead byte makes the strict UTF-8 decode fail and the
    whole file falls into the cp1252 fallback — where every remaining byte becomes
    its own wrong glyph. One byte of overrun turns a readable file into mojibake, so
    the incomplete tail goes instead.
    """
    if not data:
        return data
    try:
        data.decode("utf-8")
        return data  # nothing was cut mid-character
    except (UnicodeDecodeError, ValueError):
        pass
    for cut in range(1, 5):
        if cut >= len(data):
            break
        decoded = text_document(data[: len(data) - cut])
        if decoded is not None and decoded[1].startswith("utf-8"):
            return data[: len(data) - cut]
    return data


def text_facts(data: bytes, *, max_bytes: int = MAX_TEXT_READ) -> dict[str, int]:
    """Size and line count for a text file, for a chip caption in the UI.

    ``bytes`` is always the true size, and is returned even for bytes that turn out
    not to be text. The line count is measured over at most ``max_bytes`` of the file:
    a 200 MB log is text like any other, and decoding the whole of it to produce a
    number in a caption is not worth the time. When the sample had to be cut the count
    would be a lower bound rather than a fact, so it is left out instead of guessed.
    """
    if not data:
        return {"bytes": 0, "lines": 0}
    if len(data) > max_bytes:
        return {"bytes": len(data)}
    decoded = text_document(data)
    if decoded is None:
        return {"bytes": len(data)}
    return {"bytes": len(data), "lines": len(decoded[0].splitlines())}


def sniff_mime(data: bytes, declared: str = "") -> str:
    """Best-effort content sniff, falling back to the declared type.

    The API identifies formats from the bytes rather than the label, so a
    mislabelled upload would fail there — this catches it earlier and lets us
    store the file with a correct extension.
    """
    head = bytes(data[:16])
    for prefix, mime in _MAGIC:
        if head.startswith(prefix):
            if prefix == b"RIFF":
                return "image/webp" if head[8:12] == b"WEBP" else (declared or "application/octet-stream")
            return mime
    if head[4:12] in (b"ftypisom", b"ftypmp42", b"ftypMSNV", b"ftypqt  "):
        return "video/quicktime" if head[8:12] == b"qt  " else "video/mp4"
    return (declared or "").split(";")[0].strip() or "application/octet-stream"


def classify_kind(data: bytes, mime: str = "", name: str = "") -> str:
    """One of ``image``, ``video``, ``audio``, ``text``, ``binary``.

    The order is the point: a magic number outranks both the declared type and the
    extension, and text is decided from content, so an extensionless file
    (``Makefile``, ``.gitignore``) is readable instead of "unknown".
    """
    sniffed = sniff_mime(data, mime)
    if is_image_mime(sniffed):
        return "image"
    if is_video_mime(sniffed):
        return "video"
    if sniffed.startswith("audio/"):
        return "audio"
    if sniffed == "application/pdf":
        # A PDF header is ASCII and everything after it is compressed streams, so
        # without this it would pass as text and print as mojibake.
        return "binary"
    if looks_like_text(data, name):
        return "text"
    return "binary"


def decode_data_uri(value: str) -> tuple[str, bytes] | None:
    """Split a ``data:`` URI into ``(mime, bytes)``. Returns None if not one."""
    match = _DATA_URI_RE.match(value or "")
    if not match:
        return None
    mime = (match.group("mime") or "application/octet-stream").lower()
    payload = match.group("data")
    if "base64" in (match.group("params") or ""):
        try:
            return mime, base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError):
            return None
    from urllib.parse import unquote_to_bytes

    return mime, unquote_to_bytes(payload)


@dataclass(frozen=True, slots=True)
class MediaLimits:
    """DeepSeek's documented image constraints, mirrored so we fail locally."""

    #: Request body cap is 48 MiB; base64 inflates by ~4/3.
    max_inline_bytes: int = 32 * 1024 * 1024
    #: Longest edge, per side.
    max_dimension: int = 8192
    #: Longest edge once a request carries 15 or more images.
    max_dimension_many: int = 4096
    #: Above this many images in one request the lower dimension cap applies.
    many_image_threshold: int = 15
    max_images_per_request: int = 600


@dataclass(frozen=True, slots=True)
class TextDocument:
    """A text file read off disk, with enough detail to be honest about it."""

    text: str
    encoding: str
    #: Size on disk, which is not the same as ``len(text)`` once anything is capped.
    bytes: int
    #: Totals for the whole file, measured before any character cap was applied.
    characters: int
    lines: int
    #: True when ``text`` is shorter than the file.
    truncated: bool


class MediaStore:
    """Reads and writes the media that belongs to one conversation."""

    def __init__(self, store, limits: MediaLimits | None = None) -> None:
        self.store = store
        self.limits = limits or MediaLimits()

    # ── writing ──

    def save_bytes(
        self,
        uuid: str,
        data: bytes,
        mime: str = "",
        *,
        prefix: str = "file",
        name: str = "",
    ) -> str:
        """Write ``data`` into the conversation directory; return its URL path."""
        mime = sniff_mime(data, mime)
        directory = self.store.dir(uuid)
        directory.mkdir(parents=True, exist_ok=True)
        ext = Path(name).suffix if name and Path(name).suffix else ext_for_mime(mime)
        filename = f"{prefix}_{int(time.time() * 1000)}{ext}"
        target = directory / filename
        # Never clobber: two uploads in the same millisecond are plausible.
        counter = 1
        while target.exists():
            target = directory / f"{prefix}_{int(time.time() * 1000)}_{counter}{ext}"
            counter += 1
        target.write_bytes(data)
        return f"/memory/{uuid}/{target.name}"

    def save_file(self, uuid: str, source: Path, *, prefix: str = "file") -> str:
        return self.save_bytes(
            uuid, Path(source).read_bytes(), "", prefix=prefix, name=Path(source).name
        )

    def save_b64(self, uuid: str, b64: str, mime: str = "", *, prefix: str = "file") -> str:
        try:
            data = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"could not decode base64 media: {exc}") from exc
        return self.save_bytes(uuid, data, mime, prefix=prefix)

    def path_for_url(self, url: str) -> Path | None:
        """Resolve a ``/memory/<uuid>/<file>`` URL to an on-disk path, safely."""
        if not url or not url.startswith("/memory/"):
            return None
        parts = url[len("/memory/"):].split("/")
        if len(parts) != 2:
            return None
        uuid, filename = parts
        try:
            self.store.validate(uuid)
        except Exception:
            return None
        if filename != Path(filename).name or filename.startswith("."):
            return None                       # no traversal, no dotfiles
        candidate = (self.store.root / uuid / filename).resolve()
        root = self.store.root.resolve()
        if root not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    def resolve_in_conversation(self, uuid: str, name: str) -> Path | None:
        """Resolve a `/memory/...` URL *or* a bare filename inside one conversation.

        Callers accept either: a URL copied out of a message block, or a filename
        typed into a form. Both have to survive the same containment check, because a
        bare name is the easier one to weaponise — `..\\..\\providers.json` reads the
        API key straight out of the project directory.

        The boundary checked here is the *conversation*, not the shared memory root.
        A `/memory/` URL carries its own uuid, so taking it at face value let
        `/memory/<some-other-uuid>/photo.png` reach any conversation's files. That is
        the same hole ``tools_builtin._resolve`` closes for tool arguments.
        """
        if not name:
            return None
        try:
            directory = self.store.dir(uuid).resolve()
        except Exception:
            return None

        if name.startswith("/memory/"):
            resolved = self.path_for_url(name)
            candidate = resolved.resolve() if resolved is not None else None
        else:
            # Reject anything that is not a plain leaf name in the conversation
            # directory: no separators, no `..`, no dotfiles. `Path(...).name`
            # collapses both slash styles on Windows, so comparing against it
            # catches traversal either way.
            if name != Path(name).name or name.startswith("."):
                return None
            candidate = (directory / name).resolve()

        if candidate is None or directory not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    # ── reading text back ──

    def read_text(
        self,
        path: Path,
        *,
        max_chars: int | None = None,
        max_bytes: int = MAX_TEXT_READ,
    ) -> TextDocument | None:
        """Read a text file, or return ``None`` when the file is not text.

        ``max_bytes`` bounds the *read*, not the judgement: a stray 2 GB log must not
        be pulled into memory to be counted, so an over-long file comes back truncated
        and says so. ``max_chars`` is the caller's budget — the model's context, or a
        preview — and applies on top of it.
        """
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                data = handle.read(max(1, int(max_bytes)))
        except OSError:
            return None

        if size > len(data):
            # The read stopped at the cap, so the final character may be half present.
            # Trimming it has to happen *before* the decode is judged: cp1252 accepts
            # almost any byte sequence, so a sliced character would otherwise win the
            # fallback and hand back mojibake instead of the file.
            data = drop_partial_character(data)

        decoded = text_document(data)
        if decoded is None:
            return None

        text, encoding = decoded
        total_chars, total_lines = len(text), len(text.splitlines())
        truncated = size > len(data)
        if max_chars is not None and total_chars > int(max_chars):
            text = text[: int(max_chars)]
            truncated = True
        return TextDocument(
            text=text,
            encoding=encoding,
            bytes=size,
            characters=total_chars,
            lines=total_lines,
            truncated=truncated,
        )

    # ── image utilities ──

    @staticmethod
    def probe_size(data: bytes) -> tuple[int, int] | None:
        """``(width, height)`` for an image, or None when Pillow is unavailable."""
        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            with Image.open(io.BytesIO(data)) as img:
                return img.size
        except Exception:
            return None

    @staticmethod
    def resize_target(size: tuple[int, int], max_dim: int) -> tuple[int, int]:
        """The aspect-preserving size an image should take to fit ``max_dim``."""
        width, height = size
        if max_dim <= 0 or max(width, height) <= max_dim:
            return size
        if width >= height:
            return (max_dim, max(1, round(height * max_dim / width)))
        return (max(1, round(width * max_dim / height)), max_dim)

    @staticmethod
    def resize_image_bytes(data: bytes, max_dim: int, *, quality: int = 85) -> tuple[bytes, str]:
        """Downscale so the longest edge is at most ``max_dim``.

        Returns ``(bytes, mime)``. Falls back to the input unchanged when Pillow is
        missing or the image is already small enough — never a hard failure, since
        image processing is an optimisation, not a requirement.
        """
        try:
            from PIL import Image
        except ImportError:
            return data, ""
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                width, height = img.size
                if max_dim <= 0 or max(width, height) <= max_dim:
                    return data, ""
                if width >= height:
                    new_size = (max_dim, max(1, round(height * max_dim / width)))
                else:
                    new_size = (max(1, round(width * max_dim / height)), max_dim)
                resized = img.resize(new_size, Image.LANCZOS)

                # JPEG has no alpha; flatten rather than emit a broken file.
                if resized.mode in ("RGBA", "LA", "P") and "transparency" in resized.info:
                    background = Image.new("RGB", resized.size, (255, 255, 255))
                    background.paste(resized.convert("RGBA"), mask=resized.convert("RGBA").split()[-1])
                    resized = background
                elif resized.mode not in ("RGB", "L"):
                    resized = resized.convert("RGB")

                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=quality, optimize=True)
                return buffer.getvalue(), "image/jpeg"
        except Exception as exc:                      # pragma: no cover - defensive
            logger.warning("image resize failed, sending original: %s", exc)
            return data, ""

    # ── video ──

    @staticmethod
    def video_frames(
        path: Path,
        *,
        target_fps: int = 1,
        max_dim: int = 512,
        max_frames: int = 16,
        quality: int = 85,
    ) -> list[bytes]:
        """Sample frames from a video as JPEGs, newest-first order preserved.

        This is *frame-count reduction* in the sense that matters: a 60 s clip at
        1 fps is 60 frames, which is 60 images and far past the useful budget, so
        the sample rate is thinned until the count fits ``max_frames``.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("opencv-python is not installed; cannot sample video frames")
            return []

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return []
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            if total <= 0:
                return []
            duration = total / source_fps
            count = max(1, round(duration * target_fps))
            # Thin the sample so a long clip still fits the frame budget.
            count = min(count, max(1, max_frames))
            indices = [int(i * total / count) for i in range(count)]

            frames: list[bytes] = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                if max(width, height) > max_dim:
                    if width >= height:
                        new_size = (max_dim, max(1, round(height * max_dim / width)))
                    else:
                        new_size = (max(1, round(width * max_dim / height)), max_dim)
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok:
                    frames.append(buffer.tobytes())
            return frames
        finally:
            capture.release()

    # ── inbound: strip base64 out of client-supplied content ──

    def ingest_user_content(self, uuid: str, content: Any) -> Any:
        """Persist any inline media in a user message, replacing it with a URL.

        Handles four shapes the frontend may send:

        * ``image_url`` with a ``data:`` URI — the standard OpenAI vision block;
        * a ``_file`` block, our own display-only shape for video, audio and files;
        * a ``text`` block, which passes through untouched;
        * anything else, copied as-is.

        A ``_file`` block is classified from its bytes on the way in, so ``kind``
        says ``text`` for a ``.md`` or a ``.gitignore`` and the rehydrator never has
        to guess what it is holding.
        """
        if isinstance(content, str) or not isinstance(content, list):
            return content

        out: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                out.append(block)
                continue
            btype = block.get("type")

            if btype in ("image_url", "input_image"):
                out.append(self._ingest_image_block(uuid, block))

            elif btype in ("_file", "file"):
                out.append(self._ingest_file_block(uuid, block))

            else:
                out.append(dict(block))
        return out

    def _ingest_image_block(self, uuid: str, block: dict[str, Any]) -> dict[str, Any]:
        inner = dict(block.get("image_url") or {})
        url = str(inner.get("url") or "")
        if not url.startswith("data:"):
            # Already a URL (external or /memory/...) — nothing to persist.
            return dict(block)

        decoded = decode_data_uri(url)
        if decoded is None:
            logger.warning("dropping unreadable data URI image block")
            return {"type": "text", "text": "[image could not be decoded]"}
        declared, data = decoded
        mime = sniff_mime(data, declared)
        if not is_image_mime(mime):
            raise ValueError(
                f"unsupported image type {mime!r} — DeepSeek accepts JPEG, PNG, GIF, and WebP only"
            )

        saved = self.save_bytes(uuid, data, mime, prefix="user_img")
        out: dict[str, Any] = {"type": "image_url", "image_url": {"url": saved}}
        if block.get("image_url", {}).get("detail"):
            out["image_url"]["detail"] = block["image_url"]["detail"]

        size = self.probe_size(data)
        if size:
            out["width"], out["height"] = size
            out["mime"] = mime
        return out

    def _ingest_file_block(self, uuid: str, block: dict[str, Any]) -> dict[str, Any]:
        kind = str(block.get("kind") or "file")
        name = str(block.get("name") or f"{kind}")
        declared = str(block.get("mime") or "")
        data: bytes | None = None

        raw = block.get("b64") or block.get("data")
        if isinstance(raw, str) and raw:
            try:
                data = base64.b64decode(raw, validate=False)
            except (binascii.Error, ValueError):
                data = None
        elif block.get("url", "").startswith("data:"):
            decoded = decode_data_uri(block["url"])
            if decoded:
                declared, data = decoded[0] or declared, decoded[1]

        if data is None:
            # Already persisted (or external) — keep the reference, drop any b64.
            # There are no bytes to classify, but the *name* is enough to call a
            # `.md` or a `Makefile` text, and that is what makes it readable later.
            cleaned = {k: v for k, v in block.items() if k not in ("b64", "data")}
            cleaned.setdefault("type", "_file")
            if str(cleaned.get("kind") or "") in ("", "file", "unknown") and looks_like_text_name(name):
                cleaned["kind"] = "text"
            cleaned.setdefault("kind", kind)
            return cleaned

        mime = sniff_mime(data, declared)
        detected = classify_kind(data, declared, name)
        if detected != "binary":
            # Content has the final say over the label the client sent, the same way
            # it does over an extension: a `.txt` holding JPEG bytes is an image,
            # and a `.gitignore` is text whatever the frontend called it.
            kind = detected
        if detected == "text" and not is_text_mime(mime):
            mime = "text/plain"        # so a name with no suffix still gets `.txt`

        prefix = {"video": "user_video", "audio": "user_audio", "image": "user_image"}.get(kind, "user_file")
        saved = self.save_bytes(uuid, data, mime, prefix=prefix, name=name)

        cleaned = {k: v for k, v in block.items() if k not in ("b64", "data")}
        cleaned.update({"type": "_file", "kind": kind, "name": name, "mime": mime, "url": saved})
        if kind == "text":
            # So the UI can caption the chip without fetching the file to count it.
            cleaned.update(text_facts(data))
        return cleaned

    def ingest_tool_blocks(
        self, uuid: str, blocks: Sequence[Any], tool_name: str
    ) -> list[dict[str, Any]]:
        """Persist inline media returned by a tool, keyed by tool name."""
        out: list[dict[str, Any]] = []
        counters: dict[str, int] = {}

        for block in blocks:
            if not isinstance(block, dict):
                out.append({"type": "text", "text": str(block)})
                continue

            btype = str(block.get("type") or "text")
            raw = block.get("data") or block.get("b64")
            mime = str(block.get("mimeType") or block.get("mime") or "")

            if btype in ("image", "video", "audio") and isinstance(raw, str) and raw:
                index = counters.get(btype, 0)
                counters[btype] = index + 1
                try:
                    data = base64.b64decode(raw, validate=False)
                except (binascii.Error, ValueError):
                    out.append({"type": "text", "text": f"[{btype} from {tool_name} was not valid base64]"})
                    continue
                saved = self.save_bytes(uuid, data, mime, prefix=f"{tool_name}_{btype}_{index}")
                cleaned = {k: v for k, v in block.items() if k not in ("data", "b64")}
                cleaned["url"] = saved
                cleaned["type"] = btype
                if btype == "image":
                    size = self.probe_size(data)
                    if size:
                        cleaned.setdefault("width", size[0])
                        cleaned.setdefault("height", size[1])
                out.append(cleaned)

            elif btype in ("image", "video") and block.get("url"):
                out.append(dict(block))

            elif btype == "text":
                out.append({"type": "text", "text": str(block.get("text") or "")})

            else:
                out.append(dict(block))
        return out


# ── media meant for the person, not for the model ─────────────────────────────

#: Set on a tool's media block when the file is meant for the *user's* eyes.
#:
#: The two audiences are genuinely different, and which one a block serves has to
#: ride on the block itself: by the time a result is being stored, nothing else
#: knows.
#:
#: * an unmarked block feeds the model's vision — ``reduce_video_frames`` samples a
#:   clip precisely so the model can describe it — so it stays in the tool message
#:   and :mod:`server.rehydrate` inlines it into the next request;
#: * a marked block exists to be *shown* rather than seen, and is deliberately never
#:   sent upstream — the model can reach for ``inspect_media`` or ``resize_image``
#:   with the path it is given if it actually needs to look.
#:
#: There are two reasons a block ends up marked, and they do not want the same
#: treatment, so the marker carries which one it was.
DISPLAY_KEY = "display"

#: The user *asked* to see this, and on the strength of that alone: ``display_media``
#: was handed a file, a caption, and a request. It is hung on the assistant's turn so
#: it renders above the answer — the picture is the point, and it should be the first
#: thing on screen rather than a card that has to be opened.
DISPLAY_SHOWN = "shown"

#: The media merely *arrived* with a result. ``web_fetch`` downloads the images on the
#: page it read because they are part of that page, not because anyone asked for them,
#: and an MCP server has no way to distinguish the two by the time it answers. Pinning
#: these to the top of the reply is what pushed five page posters between the reasoning
#: and the answer for the life of the transcript — and they are not the user's to look
#: at anyway; they are the same page whose text is already sitting in the tool card.
#: So the block stays *in* that result and renders inside the card that carried it,
#: which is collapsed by default. Still marked, so it is still never replayed upstream.
DISPLAY_INLINE = "inline"

#: A block the user watches inside an `<iframe>` rather than reads from disk: a
#: YouTube or Vimeo page, where the file is on someone else's server and will stay
#: there. It carries no media to send upstream, so ``display`` is the only thing that
#: decides where it goes — but it is a third type, so every test for "is this media"
#: has to name it.
EMBED_TYPE = "embed"

#: Block types that carry something a person can look at or listen to.
MEDIA_BLOCK_TYPES = ("image", "video", "audio", EMBED_TYPE)


#: Set on the block of a source that was never downloaded and never will be.
#: `server/remote_media.py` puts it there; the frontend reads it to label the player.
REMOTE_KEY = "source"
REMOTE_VALUE = "remote"


def is_remote_block(block: Any) -> bool:
    """True for a block describing a file that stayed on its own server."""
    return isinstance(block, dict) and str(block.get(REMOTE_KEY) or "") == REMOTE_VALUE


def is_display_block(block: Any) -> bool:
    """True for a media block that is for the user rather than for the model."""
    if not isinstance(block, dict) or not block.get(DISPLAY_KEY):
        return False
    block_type = str(block.get("type") or "")
    if block_type == EMBED_TYPE:
        # An embed has a page URL, not a file URL, so the `url` check below would be
        # the wrong question to ask it.
        return bool(block.get("embed_url") or block.get("url"))
    return block_type in ("image", "video", "audio") and bool(block.get("url"))


def is_inline_block(block: Any) -> bool:
    """True for media that came in with a result rather than by request.

    Marked, so still the user's and still never replayed upstream — but it belongs
    with the result it arrived in, inside the collapsed card, not pinned above the
    reply. See ``DISPLAY_INLINE``.
    """
    return is_display_block(block) and block.get(DISPLAY_KEY) == DISPLAY_INLINE


def is_pinned_block(block: Any) -> bool:
    """True for media that goes above the reply: the user asked to be shown it.

    ``True`` is the original spelling of the marker and counts as pinned, which is
    what keeps a transcript written before the two kinds existed rendering the way it
    did when it was written.
    """
    return is_display_block(block) and not is_inline_block(block)


def split_display_blocks(blocks: Sequence[Any]) -> tuple[list[dict], list[Any]]:
    """Partition a tool result into ``(pinned above the reply, kept in the result)``.

    Only what the user asked to see is taken out. Media that merely came in with the
    result is *kept*, because where it is is where it belongs — the card that carried
    it — and it is still marked, so keeping it in the message does not put it back in
    front of the model (:mod:`server.rehydrate` turns it into a path instead).
    """
    shown: list[dict] = []
    kept: list[Any] = []
    for block in blocks:
        (shown if is_pinned_block(block) else kept).append(block)
    return shown, kept


def mark_display_blocks(blocks: Sequence[Any]) -> list[Any]:
    """Copy ``blocks`` with the marker set on every media block an MCP tool returned.

    For a source that is user-facing by nature rather than by intent: ``web_fetch``
    downloads the images on the page it read because a page has images, so there is no
    per-block decision to make and no place in the tool to make it. That is exactly
    why the marker these get is ``DISPLAY_INLINE`` — the block is the user's, not the
    model's, but nobody asked for it, so it stays in the result rather than being
    lifted onto the reply.

    A block that already carries a marker is left alone, which is how a remote server
    that genuinely knows it is showing something says so.
    """
    out: list[Any] = []
    for block in blocks:
        if isinstance(block, dict) and str(block.get("type") or "") in MEDIA_BLOCK_TYPES:
            if (block.get("url") or block.get("embed_url")) and not block.get(DISPLAY_KEY):
                block = {**block, DISPLAY_KEY: DISPLAY_INLINE}
        out.append(block)
    return out


def display_note(block: Mapping[str, Any]) -> str:
    """The one line left in a tool result where media was cut out of it.

    A pointer, not a summary: what the model has to end up with is the path, since
    the bytes are on disk either way and every tool that can reach them takes a path.
    """
    url = str(block.get("url") or "")
    kind = str(block.get("type") or "file")
    name = str(block.get("name") or "") or url.rsplit("/", 1)[-1]
    if kind == EMBED_TYPE:
        title = str(block.get("title") or name)
        provider = str(block.get("provider") or "an embed provider")
        return (
            f"[{provider} player for {title!r} was shown to the user. The video itself "
            f"stays on {provider}'s servers and is not attached here. Its URL is {url} — "
            "pass that URL to inspect_media to read its metadata.]"
        )
    if is_remote_block(block):
        return (
            f"[{kind} {name} shown to the user, streamed from its own server and not "
            f"downloaded. Its URL is {url} — pass that URL to inspect_media or "
            "reduce_video_frames to look at it.]"
        )
    return (
        f"[{kind} {name} was shown to the user and is not attached here. "
        f"Its path is {url} — pass that path to inspect_media or read_file to look at it.]"
    )

