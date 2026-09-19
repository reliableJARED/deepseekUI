"""Server-side configuration, read from the environment.

Kept separate from :mod:`deepseek_client.config` on purpose: that package answers
"how do I talk to the model", this one answers "how does *this deployment* behave".
The two are deliberately not coupled, so the wrapper stays reusable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from deepseek_client.config import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("deepseek_ui.settings")

#: The one credential this deployment needs. `providers.json` references it as
#: ``${DEEPSEEK_API_KEY}``, which is why writing the file is not enough — the
#: value also has to reach the process (see :func:`load_environ`).
API_KEY_VAR = "DEEPSEEK_API_KEY"

#: Where the UI sends someone who has no key yet.
API_KEY_SIGNUP_URL = "https://platform.deepseek.com/sign_up"

#: Values shipped in `.env.example` — or commonly pasted from a tutorial — that are
#: *not* a key. A placeholder resolves to a non-empty string, so without this a
#: copied example would look configured right up to the first 401.
PLACEHOLDER_KEYS = frozenset({
    "my_deepseek_api_key",
    "your_deepseek_api_key",
    "your-api-key",
    "your_api_key",
    "changeme",
    "sk-xxx",
    "sk-your-key-here",
})

#: One `NAME=value` line, tolerating `export` and surrounding whitespace.
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")

#: The template a first run's ``mcp.json`` is seeded from. Committed; the file it is
#: copied to is not, because it holds API keys in plain text. Kept next to its
#: destination rather than at a fixed path, so pointing ``MCP_CONFIG`` at a temp
#: directory (as the tests do) finds no template and therefore creates nothing.
MCP_EXAMPLE_NAME = "mcp.example.json"

#: The tool-loop ceiling, which the settings panel can change while the server runs.
MAX_TOOL_STEPS_VAR = "MAX_TOOL_STEPS"
#: Bounds the panel enforces. Every step is a full round trip that re-sends the whole
#: conversation, so a stray zero is a real bill rather than just a slow turn.
MIN_TOOL_STEPS = 1
MAX_TOOL_STEPS_LIMIT = 100

#: Extra directories the media-display tool may read from, ``;``-separated. The
#: project directory is always allowed (minus the media tree), so a tool that drops
#: a file beside the code — MCP's ``web_media/`` is the one that exists today — is
#: reachable with no configuration at all. This is for media that lands somewhere
#: else entirely: a renders folder, a scratch directory, another drive.
MEDIA_DISPLAY_ROOTS_VAR = "MEDIA_DISPLAY_ROOTS"

__all__ = [
    "Settings",
    "load_settings",
    "load_environ",
    "PROJECT_ROOT",
    "API_KEY_VAR",
    "API_KEY_SIGNUP_URL",
    "MAX_TOOL_STEPS_VAR",
    "MIN_TOOL_STEPS",
    "MAX_TOOL_STEPS_LIMIT",
    "MEDIA_DISPLAY_ROOTS_VAR",
    "is_usable_key",
    "mask_key",
    "read_env_var",
    "upsert_env_var",
    "ensure_env_file",
    "MCP_EXAMPLE_NAME",
    "ensure_mcp_file",
    "set_api_key",
    "api_key_status",
    "parse_tool_steps",
    "set_tool_steps",
]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _as_paths(value: Any) -> tuple[Path, ...]:
    """Parse a ``;``-separated list of directories from one environment value.

    ``;`` rather than ``os.pathsep``, even on Windows where they are the same thing:
    the alternative separator is ``:``, which is a drive letter, so a value following
    the platform would be unparseable on the platform most likely to need it. A
    relative entry resolves against the project root, which is the only directory it
    could sensibly mean.
    """
    roots: list[Path] = []
    for part in str(value or "").split(";"):
        cleaned = part.strip().strip('"').strip("'")
        if not cleaned:
            continue
        path = Path(cleaned).expanduser()
        roots.append(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
    return tuple(roots)


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the server needs that is not specific to a single request."""

    host: str = "127.0.0.1"
    port: int = 5000
    #: Refuse to bind to anything but loopback unless this is explicitly set, so
    #: an API key can never leak onto a LAN by accident.
    allow_lan: bool = False

    memory_root: Path = PROJECT_ROOT / "memory"
    providers_path: Path = PROJECT_ROOT / "providers.json"
    env_file: Path = PROJECT_ROOT / ".env"
    frontend_dir: Path = PROJECT_ROOT / "frontend" / "dist"

    #: Default model. Empty means "whatever providers.json lists first".
    model: str = ""

    #: Upper bound on media rehydrated into one request. DeepSeek's body limit is
    #: 48 MiB, and base64 inflates by ~4/3, so this stays comfortably under it.
    media_char_budget: int = 6_000_000

    #: Longest edge for images sent *to the model*. The API resizes to roughly
    #: 1300x1300 (about 1024 tokens) anyway, so sending much more is wasted
    #: upload bandwidth and much less throws away detail.
    model_image_max_dim: int = 1280
    #: Frame extraction for video that reaches the model.
    model_video_max_dim: int = 512
    model_video_fps: int = 1
    model_video_max_frames: int = 16
    #: Characters of one attached text file that may be inlined into a request, and
    #: the total across all of them. A README is a few thousand characters; a log
    #: file is not, and inlining all of it would spend the window on one attachment.
    #: Anything past the cap is reported to the model, which can page through the
    #: rest with `read_file`.
    model_text_max_chars: int = 40_000
    text_char_budget: int = 200_000
    #: Once a request carries this many images the API drops the per-side limit
    #: from 8192 px to 4096 px, so stay under it.
    model_max_images: int = 600

    #: How much of the context window may be filled before old turns are dropped.
    context_safety_ratio: float = 0.92

    #: Tool loop ceiling. Each step is one round trip to the API.
    max_tool_steps: int = 8

    #: Request timeout for a single upstream call, in seconds.
    request_timeout: float = 300.0

    #: MCP servers, as loaded from `mcp.json` (see `server/mcp.py`).
    mcp_config_path: Path = PROJECT_ROOT / "mcp.json"

    #: Extra directories the media-display tool may read from, on top of the project
    #: directory. See :meth:`display_roots` and ``MEDIA_DISPLAY_ROOTS_VAR``.
    media_display_roots: tuple[Path, ...] = ()

    #: Values changed at runtime from the settings panel.
    #:
    #: One `Settings` instance is shared by the engine, the route closures and the
    #: built-in tools, and the dataclass is frozen — so a live change cannot rebind a
    #: field. Replacing the object would leave every existing holder pointing at the
    #: old one, which is indistinguishable from a save that silently did nothing. The
    #: override is recorded here instead, and :meth:`effective` is the read path for
    #: anything the panel can change. Excluded from equality and hashing so an
    #: unhashable dict field cannot break the frozen dataclass.
    runtime: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def effective(self, name: str, default: Any = None) -> Any:
        """The value of ``name`` with any runtime override applied."""
        if name in self.runtime:
            return self.runtime[name]
        return getattr(self, name, default)

    def override(self, **values: Any) -> None:
        """Record live overrides for the running process. See :attr:`runtime`."""
        self.runtime.update(values)

    def display_roots(self) -> tuple[Path, ...]:
        """Directories a display tool may read from, in order.

        The conversation's own directory is deliberately *absent*: it is per-request,
        and the tool adds it. This is the part that is the same for every
        conversation — the project directory, so a tool that drops a file beside the
        code (MCP's ``web_media/`` is the one that exists today) is reachable with no
        configuration, plus anything ``MEDIA_DISPLAY_ROOTS`` adds.

        The media tree is not filtered out here, it cannot be: it is a *subdirectory*
        of a root, not a root. Excluding it is the caller's job, and the tool does it,
        because ``memory/`` holds every conversation's files — allowing it would undo
        the per-conversation boundary the built-in tools exist to enforce.
        """
        seen: set[Path] = set()
        roots: list[Path] = []
        for candidate in (PROJECT_ROOT, *self.media_display_roots):
            try:
                root = Path(candidate).expanduser().resolve()
            except (OSError, ValueError):                # pragma: no cover - defensive
                continue
            if root not in seen and root.is_dir():
                seen.add(root)
                roots.append(root)
        return tuple(roots)

    def media_root_for(self, uuid: str) -> Path:
        return self.memory_root / uuid


def load_environ(
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv: bool = True,
) -> dict[str, str]:
    """The process environment with ``.env`` merged over it, as a new dict.

    Returned rather than applied, so a `.env` value can never leak into
    ``os.environ`` and change the behaviour of a later import. Shared with
    :func:`load_mcp_config` so a token referenced in ``mcp.json`` resolves against
    the same environment the rest of the server sees.

    The file's values *are* also applied to ``os.environ``, because the settings
    panel writes that file while the server is running: a value that only ever
    reached a local copy would be ignored by anything reading the process
    environment directly, including the next call to
    :meth:`ProvidersConfig.load_or_env`. An empty line never clobbers a value the
    shell set, so `DEEPSEEK_API_KEY=sk-…` exported in a container still wins over
    a blank entry left in the file.
    """
    env: dict[str, str] = dict(os.environ)
    if environ:
        env.update({k: str(v) for k, v in environ.items()})

    path = Path(env_file) if env_file else Path(env.get("ENV_FILE") or PROJECT_ROOT / ".env")
    if load_dotenv and path.exists():
        # load_env_file writes into the mapping it is given; a fresh copy keeps
        # `os.environ` untouched until the explicit pass below.
        parsed = load_env_file(path, environ=env)
        if environ is None:
            # Only when the caller injected nothing: an explicit mapping is how
            # tests stay deterministic, and it must never escape into the process.
            for key, value in parsed.items():
                if not value and os.environ.get(key):
                    continue
                os.environ[key] = value
    return env


# ── the API key in `.env` ─────────────────────────────────────────────────────

def is_usable_key(value: Any) -> bool:
    """True when ``value`` looks like a key rather than an unfilled placeholder."""
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("${"):        # a `${VAR}` reference that never resolved
        return False
    return text.lower() not in PLACEHOLDER_KEYS


def mask_key(value: Any) -> str:
    """How a key may be *displayed*. Never returns more than the last four chars."""
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 4:
        return "*" * len(text)
    return "••••••••" + text[-4:]


def _env_literal(value: str) -> str:
    """Serialise a value so :func:`deepseek_client.config.load_env_file` reads it back."""
    if value == "":
        return ""
    if re.fullmatch(r"[A-Za-z0-9_@%+./:,\-]+", value):
        return value
    # Anything else (spaces, `#`, quotes) has to be quoted or the parser would
    # treat a `#` as the start of a comment.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def read_env_var(path: str | Path, name: str) -> str:
    """The raw value of ``name`` in ``.env``, or ``""``. BOM-tolerant."""
    file = Path(path)
    if not file.is_file():
        return ""
    try:
        text = file.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for raw in text.splitlines():
        match = _ENV_LINE.match(raw)
        if match and match.group(1) == name:
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                return value[1:-1]
            return re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return ""


def upsert_env_var(path: str | Path, name: str, value: str) -> Path:
    """Set ``name=value`` in ``.env``, creating the file when it does not exist.

    Only that one line is touched: `.env` is a hand-editable file, so comments,
    ordering and blank lines are all preserved rather than rewritten from a model
    of what the file should contain. Written atomically (``.tmp`` + ``os.replace``)
    and without a BOM, which is what makes it safe to do this while the server runs.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    raw = file.read_bytes() if file.is_file() else b""
    # Keep the file's own line endings. `Path.read_text` would normalise them away
    # and quietly convert a CRLF file on the first save.
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n") if raw else ""

    line = f"{name}={_env_literal(value)}"
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    lines = text.split("\n") if text else []
    if lines and lines[-1] == "":
        # The file's own final newline. Dropped here and re-added at the end, so the
        # rewrite cannot leave the last line without one.
        lines.pop()

    for index, existing in enumerate(lines):
        if pattern.match(existing):
            lines[index] = line
            break
    else:
        # A blank separator line, but never a leading blank one in an empty file.
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(line)

    tmp = file.with_name(file.name + ".tmp")
    tmp.write_bytes((newline.join(lines) + newline).encode("utf-8"))
    os.replace(tmp, file)
    return file


def ensure_env_file(settings) -> Path | None:
    """Give a first run a `.env` to paste a key into, seeded from `.env.example`.

    Returns the path, or ``None`` when there is no example to seed from — in that
    case nothing is created, because an `.env` with no comments and no values is
    worse than no file at all (and the panel creates it the moment there is a key
    to write).

    An unfilled placeholder is also normalised to an empty value, so a copied
    example is honestly reported as "not configured".
    """
    path = Path(settings.env_file)
    if not path.is_file():
        example = path.parent / ".env.example"
        if not example.is_file():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(example.read_bytes())
        logger.info("created %s from %s", path, example)

    declared = read_env_var(path, API_KEY_VAR)
    if not is_usable_key(declared) and not is_usable_key(os.environ.get(API_KEY_VAR)):
        upsert_env_var(path, API_KEY_VAR, "")
    return path


def ensure_mcp_file(settings) -> Path | None:
    """Give a first run an ``mcp.json``, seeded from ``mcp.example.json``.

    The sibling of :func:`ensure_env_file`, for the same reason: the destination is
    git-ignored (it stores API keys as plain ``x-api-key`` headers), so a fresh clone
    has no ``mcp.json`` at all. Without this the settings panel would open on an
    empty list with nothing to suggest a documented template was ever shipped.

    Returns the path, or ``None`` when there is no template to seed from. An
    existing ``mcp.json`` is never touched — not even an empty, corrupt, or
    directory-shaped one. That restraint matters more here than for ``.env``:
    ``MCPDocument.save`` refuses to overwrite a config it could not parse, so a
    clobbered file here would leave neither program able to explain what happened.

    The bytes are copied verbatim rather than re-serialised through ``json``, which
    is what preserves the template's ``_comment`` documentation and its formatting.
    """
    if not settings.mcp_config_path:
        return None
    path = Path(settings.mcp_config_path)
    # `exists`, not `is_file`: a directory in the way must stop this too, since
    # writing over it would raise and creating "around" it makes no sense.
    if path.exists():
        return path

    example = path.parent / MCP_EXAMPLE_NAME
    if not example.is_file():
        logger.info("no MCP template at %s; not creating %s", example, path)
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(example.read_bytes())
    except OSError as exc:
        # Non-fatal, exactly like ensure_env_file: a read-only checkout still runs,
        # it simply has nowhere to save a server from the UI.
        logger.warning("could not create %s from %s (%s)", path, example, exc)
        return None
    logger.info("created %s from %s", path, example)
    return path


def set_api_key(settings, key: str) -> Path:
    """Persist ``key`` to `.env` **and** make it live in this process.

    Both halves matter: `ProvidersConfig.load_or_env` merges `.env` into a copy of
    ``os.environ`` and never overrides an entry that is already there, so writing
    the file alone would leave every later request on the previous key.
    """
    path = ensure_env_file(settings) or Path(settings.env_file)
    upsert_env_var(path, API_KEY_VAR, key)
    os.environ[API_KEY_VAR] = key
    logger.info("stored a new %s in %s", API_KEY_VAR, path)
    return path


def parse_tool_steps(value: Any) -> int:
    """Validate a ``max_tool_steps`` value from the settings panel.

    Raises ``ValueError`` carrying a message the panel can show as-is. The ceiling is
    the point: each step re-sends the whole conversation, so an accidental extra zero
    is a real bill, and a panel that silently accepted it would be a trap.
    """
    try:
        steps = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("tool steps must be a whole number") from None
    if not MIN_TOOL_STEPS <= steps <= MAX_TOOL_STEPS_LIMIT:
        raise ValueError(
            f"tool steps must be between {MIN_TOOL_STEPS} and {MAX_TOOL_STEPS_LIMIT}"
        )
    return steps


def set_tool_steps(settings, value: Any) -> int:
    """Persist and apply ``MAX_TOOL_STEPS`` without a restart.

    Three things, for the same reasons :func:`set_api_key` needs three: the value has
    to reach the file, it has to reach the process environment (``load_env_file``
    never overrides an entry that is already there, so a stale ``os.environ`` would
    win on the next boot), and it has to reach the one ``Settings`` instance the
    engine, the route closures and the built-in tools all already point at.
    """
    steps = parse_tool_steps(value)
    upsert_env_var(settings.env_file, MAX_TOOL_STEPS_VAR, str(steps))
    os.environ[MAX_TOOL_STEPS_VAR] = str(steps)
    settings.override(max_tool_steps=steps)
    logger.info("stored %s=%d in %s", MAX_TOOL_STEPS_VAR, steps, settings.env_file)
    return steps


def api_key_status(settings) -> dict[str, Any]:
    """What the UI is allowed to know about the key. Never the key itself."""
    value = os.environ.get(API_KEY_VAR) or read_env_var(settings.env_file, API_KEY_VAR)
    usable = is_usable_key(value)
    return {
        "present": usable,
        "masked": mask_key(value) if usable else "",
        "file": str(settings.env_file),
        "file_exists": Path(settings.env_file).is_file(),
        "signup_url": API_KEY_SIGNUP_URL,
    }


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv: bool = True,
    **overrides: Any,
) -> Settings:
    """Build :class:`Settings` from the environment, then apply ``overrides``.

    ``load_dotenv`` merges ``.env`` into a *copy* of the environment, so nothing
    leaks back into the process and tests stay deterministic.
    """
    env = load_environ(overrides.get("env_file"), environ, load_dotenv=load_dotenv)
    env_file = Path(
        overrides.get("env_file") or env.get("ENV_FILE") or PROJECT_ROOT / ".env"
    )

    def var(name: str, default: str = "") -> str:
        return str(env.get(name, default) or default)

    memory_raw = var("MEDIA_ROOT", "./memory")
    memory_root = Path(memory_raw)
    if not memory_root.is_absolute():
        memory_root = (PROJECT_ROOT / memory_root).resolve()

    settings = Settings(
        host=var("HOST", "127.0.0.1"),
        port=_as_int(var("PORT"), 5000),
        allow_lan=_as_bool(var("ALLOW_LAN"), False),
        memory_root=memory_root,
        providers_path=Path(var("PROVIDERS_PATH") or PROJECT_ROOT / "providers.json"),
        env_file=env_file,
        frontend_dir=Path(var("FRONTEND_DIR") or PROJECT_ROOT / "frontend" / "dist"),
        model=var("DEEPSEEK_MODEL"),
        media_char_budget=_as_int(var("MEDIA_CHAR_BUDGET"), 6_000_000),
        model_image_max_dim=_as_int(var("MODEL_IMAGE_MAX_DIM"), 1280),
        model_video_max_dim=_as_int(var("MODEL_VIDEO_MAX_DIM"), 512),
        model_video_fps=_as_int(var("MODEL_VIDEO_FPS"), 1),
        model_video_max_frames=_as_int(var("MODEL_VIDEO_MAX_FRAMES"), 16),
        model_text_max_chars=_as_int(var("MODEL_TEXT_MAX_CHARS"), 40_000),
        text_char_budget=_as_int(var("TEXT_CHAR_BUDGET"), 200_000),
        model_max_images=_as_int(var("MODEL_MAX_IMAGES"), 600),
        context_safety_ratio=_as_float(var("CONTEXT_SAFETY_RATIO"), 0.92),
        # Clamped to the same ceiling the panel enforces, so ``max_tool_steps`` is
        # always a value the panel could also have saved. `_as_int` already rejects a
        # non-positive value by falling back to the default.
        max_tool_steps=min(_as_int(var(MAX_TOOL_STEPS_VAR), 8), MAX_TOOL_STEPS_LIMIT),
        request_timeout=_as_float(var("REQUEST_TIMEOUT"), 300.0),
        mcp_config_path=Path(var("MCP_CONFIG") or PROJECT_ROOT / "mcp.json"),
        media_display_roots=_as_paths(var(MEDIA_DISPLAY_ROOTS_VAR)),
    )

    if overrides:
        settings = replace(settings, **{k: v for k, v in overrides.items() if k != "env_file"})
    return settings
