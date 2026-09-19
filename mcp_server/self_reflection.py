"""Self-reflection MCP server: let the model read and reason about this project.

This is a SEPARATE MCP server from ``mcp_server/mcpserver.py`` (which does web
search / fetch). It exposes three tools whose only subject is the deepseekUI
source tree itself, so the model can diagnose, debug, and plan improvements
against the real code instead of a guess:

  ``introspect``    a directory map plus structural facts, and a written summary
                    of how the pieces fit together.
  ``explain_file``  a detailed summary of one file: what it contains and how it
                    works, with every class/function it defines.
  ``read_source``   the file itself, numbered so the model can cite lines.

Design rules, deliberately kept strict:

* **Read-only.** No tool writes, moves, renames, or executes anything. This
  process is a window, not a hand. The one exception is at startup, before any
  request is served: a missing ``mcp.json`` is seeded from ``mcp.example.json``
  (see ``ensure_mcp_config``) so that a fresh clone finds the MCP template where
  the app expects it. No tool can reach that code, the read-only commands
  (``--map``/``--explain``/``--read``) do not trigger it, and nothing else in the
  tree is ever written.
* **Confined to the project root.** Every path is resolved and then checked
  against ``PROJECT_ROOT``; anything outside is refused, including symlink
  escapes. ``memory/`` (other people's conversations) and any dotenv or key
  material are refused by name even inside the root.
* **Standalone on purpose.** A debugging tool must not depend on the thing it
  debugs, so this module never imports ``server/`` or ``app.py``. It only
  borrows ``deepseek_client`` for the summariser, and even that is optional:
  if the API is missing or unreachable the deterministic map/symbols are still
  returned, clearly labelled as "no summary" rather than silently empty.
* **Secrets are masked on the way out.** Files are scrubbed before they are
  shown *or* sent to the summariser, so a stray key never reaches the model or
  the transcript.

Run it::

    python mcp_server/self_reflection.py            # serve on 127.0.0.1:8590
    python mcp_server/self_reflection.py --map      # print the map and exit
    python mcp_server/self_reflection.py --map --include 'server/*'
    python mcp_server/self_reflection.py --explain server/media.py
    python mcp_server/self_reflection.py --read run.py --limit 4000
    python mcp_server/self_reflection.py --no-llm   # deterministic output only
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import copy
import fnmatch
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get("INTROSPECTION_ROOT") or Path(__file__).resolve().parent.parent
).resolve()

SERVER_PORT = int(os.environ.get("INTROSPECTION_PORT") or 8590)
HEARTBEAT_INTERVAL = 30        # seconds between "." keep-alive bytes
SUMMARY_TIMEOUT = float(os.environ.get("INTROSPECTION_TIMEOUT") or 180)

MAX_READ_CHARS = 80_000        # per read_source page
MAX_SYMBOLS = 250              # outline entries per file
MAX_OUTLINE_SIG = 160          # characters of a signature
MAX_DOC_LINE = 150             # characters of a one-line docstring
MAX_SUMMARY_INPUT = 60_000     # characters handed to the summariser
MAX_SUMMARY_TOKENS = 1_600

# The only thing this server ever writes, and only at startup. Committed template in;
# git-ignored ``mcp.json`` out. The name sits beside its destination rather than at a
# fixed path so that pointing INTROSPECTION_ROOT at a source tree without a template
# (which is what the tests do) quietly does nothing.
MCP_CONFIG_NAME = "mcp.json"
MCP_EXAMPLE_NAME = "mcp.example.json"


def ensure_mcp_config(root: Path | None = None) -> Path | None:
    """Give a first run an ``mcp.json``, seeded from the committed example.

    Returns the path it created, or ``None`` when there was nothing to do — an
    existing ``mcp.json``, or no template to copy from. Never raises: this process
    exists to inspect a project, so being unable to write must not stop it serving.
    """
    root = Path(root) if root else PROJECT_ROOT
    config = root / MCP_CONFIG_NAME
    if config.exists():
        return None
    example = root / MCP_EXAMPLE_NAME
    if not example.is_file():
        return None
    try:
        config.write_bytes(example.read_bytes())
    except OSError as exc:
        print(f"could not create {config} from {example} ({exc})")
        return None
    print(f"Created {config} from {example}")
    return config

# Directories never walked. Not a security boundary (that is _resolve_path),
# just noise control — plus memory/, which is other people's conversations.
# NOTE: frontend/dist is NOT here on purpose. In this project the files under
# frontend/dist/assets/ are the hand-written frontend, not build output, so
# excluding "dist" would hide the entire UI from introspection.
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", ".idea", ".vscode",
    "build", "htmlcov", ".eggs", ".tox", "memory",
})
IGNORED_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin", ".zip", ".gz", ".tar",
    ".whl", ".sqlite", ".db", ".db-wal", ".db-shm", ".log", ".ttf", ".woff",
    ".woff2", ".eot", ".ico", ".icns", ".pdf", ".mp4", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".ogg", ".zipx", ".lock",
})

# Off limits even though they live inside the project root.
REFUSED_TOP_LEVEL = {
    "memory": "it holds other conversations rather than source",
    ".git": "version-control internals",
}
REFUSED_NAMES = frozenset({
    ".env", ".env.local", ".env.production", ".env.development", "id_rsa",
    "id_ed25519", "id_ecdsa", ".netrc", ".pypirc", "credentials",
    "credentials.json", "secrets.json", "token.json",
})
REFUSED_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"})

LANGUAGES = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".json": "json", ".jsonl": "json",
    ".md": "markdown", ".markdown": "markdown", ".rst": "rst",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".properties": "ini",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
    ".bat": "batch", ".cmd": "batch",
    ".sql": "sql", ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".lua": "lua", ".r": "r",
    ".csv": "csv", ".tsv": "tsv", ".txt": "text",
}
NAMED_LANGUAGES = {
    "dockerfile": "dockerfile", "makefile": "makefile", "procfile": "text",
    ".gitignore": "ini", ".dockerignore": "ini", ".editorconfig": "ini",
    ".gitattributes": "ini",
}

# ─────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────
def _fmt_bytes(size: int) -> str:
    if size < 1024:
        return f"{size:,} B"
    value = float(size)
    for unit in ("KB", "MB", "GB"):
        value /= 1024.0
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}"
    return f"{value:,.1f} GB"


def _one_line(text: str, limit: int = MAX_DOC_LINE) -> str:
    """Collapse a docstring/description to a single line, never containing a
    backtick — inline code spans in the transcript are delimited by backticks
    and a stray one would break the rendering of everything after it."""
    flat = " ".join(str(text).replace("\x00", "").split())
    flat = flat.replace("`", "")
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "\u2026"
    return flat


def language_of(path: Path) -> str:
    return LANGUAGES.get(path.suffix.lower()) or NAMED_LANGUAGES.get(path.name.lower(), "text")


class Refused(RuntimeError):
    """A request that is not allowed (bad path, not text, missing)."""


class SummaryUnavailable(RuntimeError):
    """The summariser could not be reached or is not configured."""


# ─────────────────────────────────────────────
# Path safety — everything is confined to PROJECT_ROOT
# ─────────────────────────────────────────────
def _resolve_path(raw: str) -> Path:
    """Resolve a caller-supplied path and confine it to the project root.

    Mirrors ``server/tools_builtin._resolve``: resolve first (so ``..`` and
    symlinks are already collapsed), then require the result to be the root or
    below it. Checking the *resolved* path is what stops ``link -> /etc`` from
    escaping.
    """
    if raw is None or not str(raw).strip():
        raise Refused("no path given")

    text = str(raw).strip().strip("\"'")
    if "\x00" in text:
        raise Refused("path contains a NUL byte")

    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise Refused(f"cannot resolve {text!r}: {exc}") from None

    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise Refused(
            f"{text!r} is outside the project root ({root}); only files inside "
            f"{root.name}/ can be inspected."
        )

    rel = resolved.relative_to(root)
    if rel.parts and rel.parts[0] in REFUSED_TOP_LEVEL:
        raise Refused(f"{rel.as_posix()!r} is off limits: {REFUSED_TOP_LEVEL[rel.parts[0]]}")
    if resolved.name.lower() in REFUSED_NAMES or resolved.suffix.lower() in REFUSED_SUFFIXES:
        raise Refused(f"{rel.as_posix()!r} is off limits: it can hold credentials")

    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _is_refused_file(path: Path) -> bool:
    """Same refusals ``_resolve_path`` applies, so the map never even admits
    that a credentials file exists."""
    if path.name.lower() in REFUSED_NAMES or path.suffix.lower() in REFUSED_SUFFIXES:
        return True
    try:
        parts = path.resolve().relative_to(PROJECT_ROOT.resolve()).parts
    except (OSError, ValueError):
        return True
    return bool(parts) and parts[0] in REFUSED_TOP_LEVEL


def _iter_files(include: str | None = None) -> list[str]:
    """Every text-ish source file under the root, relative, sorted, posix."""
    found: list[str] = []
    root = PROJECT_ROOT
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info")
        )
        for name in filenames:
            if Path(name).suffix.lower() in IGNORED_SUFFIXES:
                continue
            full = Path(dirpath) / name
            if _is_refused_file(full):
                continue
            try:
                rel = full.relative_to(root).as_posix()
            except ValueError:      # pragma: no cover - os.walk is rooted here
                continue
            if include and not fnmatch.fnmatch(rel, include):
                continue
            found.append(rel)
    return sorted(found)


# ─────────────────────────────────────────────
# Reading text, safely
# ─────────────────────────────────────────────
def _decode(data: bytes) -> tuple[str, str] | None:
    """Decode file bytes, or None if this is not text.

    The same rule uploads use (``server/media.py:text_document``): a NUL byte or
    a low ratio of printable characters means binary. It is re-implemented here
    rather than imported so this server stays runnable when the app itself is
    broken — the whole point of a self-inspection tool.

    ``utf-8-sig`` first so a hand-edited file's BOM (Windows editors add one)
    neither shows up as a character nor breaks ``ast.parse``.
    """
    if not data:
        return "", "utf-8"
    if data[:5] == b"%PDF-":
        return None
    if b"\x00" in data[:8192]:
        return None

    sample = data[:8192]
    decoded_sample: str | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        for drop in range(4):           # a bounded read can cut a character in half
            chunk = sample if drop == 0 else sample[:-drop]
            try:
                decoded_sample = chunk.decode(encoding)
            except UnicodeDecodeError:
                continue
            break
        if decoded_sample is not None:
            break
    if decoded_sample is None:
        return None

    printable = sum(1 for ch in decoded_sample if ch.isprintable() or ch in "\t\n\r")
    if not decoded_sample or printable / len(decoded_sample) < 0.85:
        return None

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8 (with replacements)"


def _read_text_file(path: Path) -> tuple[str, str]:
    """(text, encoding) for a file that must be text, else ``Refused``."""
    if not path.exists():
        raise Refused(f"{_relative(path)} does not exist")
    if path.is_dir():
        raise Refused(f"{_relative(path)} is a directory, not a file")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise Refused(f"cannot read {_relative(path)}: {exc}") from None
    decoded = _decode(data)
    if decoded is None:
        raise Refused(f"{_relative(path)} is not text ({_fmt_bytes(len(data))} of binary data)")
    return decoded


# ─────────────────────────────────────────────
# Secret masking — applied before anything leaves this process
# ─────────────────────────────────────────────
_REDACTORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"), "<redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"), "Bearer <redacted>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}"), "<redacted-jwt>"),
    (
        re.compile(
            r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|apikey|secret|token|password|passwd"
            r"|credential|privatekey|private[_-]?key)[A-Za-z0-9_]*)(\s*[:=]\s*)([\"']?)"
            r"([A-Za-z0-9_\-./+=]{8,})([\"']?)"
        ),
        r"\1\2\3<redacted>\5",
    ),
)


def _redact(text: str) -> tuple[str, int]:
    """(scrubbed text, number of substitutions). Never returns a key."""
    total = 0
    for pattern, replacement in _REDACTORS:
        text, count = pattern.subn(replacement, text)
        total += count
    return text, total


def _redaction_note(count: int) -> str:
    if not count:
        return ""
    return f"\n_({count} secret-looking value{'s' if count != 1 else ''} masked.)_"


# ─────────────────────────────────────────────
# Structural facts: import graph and symbol outline
# ─────────────────────────────────────────────
def _third_party_roots() -> set[str]:
    """Top-level names declared in requirements.txt, used to classify imports."""
    names: set[str] = set()
    req = PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        return names
    for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower().replace("-", "_"))
    return names


def _signature(node: ast.AST) -> str:
    """A one-line signature: the declaration, minus its body and decorators.

    Parsing the real source beats regex here — decorators, defaults, ``*args``,
    annotations, and multi-line parameter lists all come out right.
    """
    clone = copy.copy(node)
    try:
        clone.body = [ast.Pass()]
        clone.decorator_list = []
        text = ast.unparse(clone).strip()
    except Exception:
        name = getattr(node, "name", "?")
        keyword = "class" if isinstance(node, ast.ClassDef) else "def"
        return f"{keyword} {name}(...)"
    head = text.split("\n    pass")[0]
    head = re.sub(r"\s*\n\s*", " ", head).strip()
    if len(head) > MAX_OUTLINE_SIG:
        head = head[: MAX_OUTLINE_SIG - 1].rstrip() + "\u2026"
    return head


def _decorators(node: ast.AST) -> list[str]:
    out: list[str] = []
    for dec in getattr(node, "decorator_list", []) or []:
        try:
            out.append(ast.unparse(dec).strip())
        except Exception:
            out.append("?")
    return out


def _doc_line(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node, clean=True) or ""
    except Exception:
        doc = ""
    if not doc:
        return ""
    first = doc.strip().split("\n\n", 1)[0]
    return _one_line(first)


@dataclass
class Symbol:
    kind: str
    name: str
    line: int
    signature: str = ""
    doc: str = ""
    decorators: list[str] | None = None


def _count_nested_definitions(children: Sequence[ast.AST], inside_function: bool = False) -> int:
    """Functions and lambdas defined *inside* a function.

    Class bodies are deliberately opaque: a method is a method, already listed
    on its own line, not a "nested definition".
    """
    total = 0
    for node in children:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total += 1 if inside_function else 0
            total += _count_nested_definitions(list(ast.iter_child_nodes(node)), True)
        elif isinstance(node, ast.ClassDef):
            continue
        elif isinstance(node, ast.Lambda):
            total += 1 if inside_function else 0
        else:
            total += _count_nested_definitions(list(ast.iter_child_nodes(node)),
                                               inside_function)
    return total


def _python_outline(text: str) -> tuple[list[Symbol], dict[str, Any]]:
    """Top-level symbols, methods, and import facts for a Python module."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], {"error": f"{exc.msg} (line {exc.lineno})"}

    symbols: list[Symbol] = []
    module_doc = ast.get_docstring(tree, clean=True) or ""

    # Module constants and assignments worth naming: ALL_CAPS or long strings.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                    target.id.isupper() or isinstance(node.value, ast.Constant)
                ):
                    symbols.append(Symbol(
                        kind="constant" if target.id.isupper() else "assignment",
                        name=target.id,
                        line=node.lineno,
                        signature=_one_line(ast.get_source_segment(text, node) or target.id, 90),
                    ))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(Symbol(
                kind="assignment",
                name=node.target.id,
                line=node.lineno,
                signature=_one_line(ast.get_source_segment(text, node) or node.target.id, 90),
            ))

    def add(node: ast.AST, kind: str) -> None:
        symbols.append(Symbol(
            kind=kind,
            name=getattr(node, "name", "?"),
            line=node.lineno,
            signature=_signature(node),
            doc=_doc_line(node),
            decorators=_decorators(node),
        ))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            add(node, "class")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    base = "method" if not isinstance(child, ast.AsyncFunctionDef) else "async method"
                    if any(d == "staticmethod" or d == "classmethod" for d in _decorators(child)):
                        base = "classmethod" if "classmethod" in _decorators(child) else "staticmethod"
                    add(child, base)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, "async function" if isinstance(node, ast.AsyncFunctionDef) else "function")

    # Imports, categorised so the model can see how the module sits in the stack.
    stdlib: set[str] = set()
    third_party: set[str] = set()
    local: set[str] = set()
    declared = _third_party_roots()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                (local if root in {"server", "deepseek_client", "mcp_server"}
                 else third_party if root.lower() in declared
                 else stdlib).add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # `from .media import ...`
                local.add(f"{'.' * node.level}{node.module or ''}".rstrip("."))
            elif node.module:
                root = node.module.split(".")[0]
                (local if root in {"server", "deepseek_client", "mcp_server"}
                 else third_party if root.lower() in declared or root in declared
                 else stdlib).add(node.module)

    facts = {
        "imports": {
            "stdlib": sorted(stdlib),
            "third_party": sorted(third_party),
            "local": sorted(local),
        },
        "nested_definitions": _count_nested_definitions(list(ast.iter_child_nodes(tree))),
        "doc": _one_line(module_doc, 300) if module_doc else "",
    }
    symbols.sort(key=lambda s: s.line)
    return symbols, facts


def _markdown_outline(text: str) -> list[Symbol]:
    out: list[Symbol] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            out.append(Symbol(kind=f"heading {hashes}", name=_one_line(line.lstrip("# ").strip(), 90),
                              line=number))
    return out


def _plain_outline(text: str, limit: int = 40) -> list[Symbol]:
    """Section-ish landmarks for config/ini/toml-shaped files."""
    out: list[Symbol] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            out.append(Symbol(kind="section", name=stripped, line=number))
        elif re.match(r"^[A-Za-z_][\w.\-]*\s*[:=]", stripped) and len(out) < limit:
            out.append(Symbol(kind="key", name=_one_line(stripped.split("=")[0].split(":")[0], 60),
                              line=number))
        if len(out) >= limit:
            break
    return out


def _outline(text: str, path: Path) -> tuple[list[Symbol], dict[str, Any]]:
    language = language_of(path)
    if language == "python":
        return _python_outline(text)
    if language in ("markdown", "rst"):
        return _markdown_outline(text), {}
    return _plain_outline(text), {}


def _render_outline(symbols: Sequence[Symbol], limit: int = MAX_SYMBOLS) -> str:
    """Outline as markdown that survives the local renderer: no fences, and no
    backticks inside the inline spans (see ``_one_line``).

    Indentation carries the nesting, so a method reads as belonging to the
    class above it rather than floating free.
    """
    if not symbols:
        return "_(no classes, functions, or sections found)_"
    method_kinds = ("method", "async method", "classmethod", "staticmethod")
    lines: list[str] = []
    for symbol in symbols[:limit]:
        is_method = symbol.kind in method_kinds
        indent = "  " if is_method else ""
        label = f"{symbol.kind} " if is_method else ""
        decorators = "".join(f"`@{d}` " for d in symbol.decorators or [])
        detail = symbol.signature or symbol.name or "?"
        lines.append(f"{indent}- line {symbol.line}: {label}{decorators}`{detail}`")
        if symbol.doc:
            lines.append(f"{indent}  - {symbol.doc}")
    if len(symbols) > limit:
        lines.append(f"- _(\u2026 and {len(symbols) - limit} more)_")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Directory map
# ─────────────────────────────────────────────
@dataclass
class FileFacts:
    path: str
    size: int
    lines: int
    language: str
    blurb: str = ""
    error: str = ""


def _facts_for(rel: str, full: Path) -> FileFacts:
    """Cheap, deterministic facts about one file. Never raises."""
    try:
        size = full.stat().st_size
    except OSError as exc:                                    # pragma: no cover
        return FileFacts(rel, 0, 0, language_of(full), error=str(exc))
    language = language_of(full)
    decoded = None
    try:
        data = full.read_bytes()
        decoded = _decode(data)
        size = len(data)
    except OSError as exc:
        return FileFacts(rel, size, 0, language, error=str(exc))
    if decoded is None:
        return FileFacts(rel, size, 0, language, error="binary")
    text, _ = decoded
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    facts = FileFacts(rel, size, lines, language)
    if language == "python":
        symbols, info = _python_outline(text)
        if "error" in info:
            facts.error = f"syntax error: {info['error']}"
        classes = sum(1 for s in symbols if s.kind == "class")
        functions = sum(1 for s in symbols if s.kind in ("function", "async function"))
        methods = sum(1 for s in symbols if s.kind in ("method", "async method", "classmethod",
                                                      "staticmethod"))
        bits = [f"{classes} class{'es' if classes != 1 else ''}"] if classes else []
        bits.append(f"{functions} function{'s' if functions != 1 else ''}")
        if methods:
            bits.append(f"{methods} method{'s' if methods != 1 else ''}")
        facts.blurb = ", ".join(bits)
        if info.get("doc"):
            facts.blurb += f" \u2014 {_one_line(info['doc'], 110)}"
    elif language == "markdown":
        headings = _markdown_outline(text)
        facts.blurb = f"{len(headings)} headings"
        first = next((s.name for s in headings if s.kind == "heading 1"), None)
        if first:
            facts.blurb += f" \u2014 {first}"
    return facts


_FACTS_CACHE: dict[str, FileFacts] = {}


def _project_facts(include: str | None = None, refresh: bool = False) -> list[FileFacts]:
    """Facts for every file in the project, cached until the tree changes.

    ``introspect`` is meant to be called repeatedly while debugging, so the
    parse is memoised. A stat-based signature (which files exist, their size and
    mtime) invalidates it the moment anything is edited.
    """
    rels = _iter_files(include)
    signature = []
    for rel in rels:
        try:
            stat = (PROJECT_ROOT / rel).stat()
            signature.append((rel, stat.st_size, int(stat.st_mtime)))
        except OSError:
            signature.append((rel, -1, -1))
    key = json.dumps([include, signature])
    cached = _FACTS_CACHE.get("key")
    if cached == key and not refresh:
        return _FACTS_CACHE["facts"]           # type: ignore[return-value]

    facts = [_facts_for(rel, PROJECT_ROOT / rel) for rel in rels]
    _FACTS_CACHE["key"] = key
    _FACTS_CACHE["facts"] = facts
    return facts


def _render_tree(facts: Sequence[FileFacts]) -> str:
    tree: dict[str, Any] = {}
    for item in facts:
        node = tree
        parts = item.path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = item

    def totals(node: dict[str, Any]) -> tuple[int, int]:
        files = 0
        lines = 0
        for value in node.values():
            if isinstance(value, FileFacts):
                files += 1
                lines += value.lines
            else:
                sub_files, sub_lines = totals(value)
                files += sub_files
                lines += sub_lines
        return files, lines

    out: list[str] = []

    def walk(node: dict[str, Any], prefix: str) -> None:
        directories = sorted((k for k, v in node.items() if not isinstance(v, FileFacts)),
                             key=str.lower)
        files = sorted((k for k, v in node.items() if isinstance(v, FileFacts)), key=str.lower)
        items: list[tuple[str, bool]] = [(d, True) for d in directories] + [(f, False) for f in files]
        for index, (name, is_dir) in enumerate(items):
            last = index == len(items) - 1
            branch = "\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 "
            if is_dir:
                child = node[name]
                count, lines = totals(child)
                out.append(f"{prefix}{branch}{name}/  ({count} file"
                           f"{'s' if count != 1 else ''}, {lines:,} lines)")
                walk(child, prefix + ("    " if last else "\u2502   "))
            else:
                item = node[name]
                detail = f"{item.lines:,} lines"
                if item.blurb:
                    detail += f" \u00b7 {item.blurb}"
                if item.error:
                    detail += f" \u00b7 {item.error}"
                out.append(f"{prefix}{branch}{name}  ({detail})")

    walk(tree, "")
    return "\n".join(out)


# ─────────────────────────────────────────────
# Summariser — the app's own API, via deepseek_client
# ─────────────────────────────────────────────
def _client_class():
    """Import ``deepseek_client``, adding the project root to sys.path if the
    server was started from somewhere else (it is a standalone process)."""
    try:
        from deepseek_client import DeepSeekClient
    except ImportError:
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from deepseek_client import DeepSeekClient
    return DeepSeekClient


class Summarizer:
    """Optional LLM pass over the deterministic facts.

    Uses the same ``providers.json`` and ``.env`` the app uses, so the model
    answering is whatever the user configured. When it cannot be reached every
    caller degrades to the map/symbols and says so — a missing summary must
    never be presented to the model as "there is nothing here".
    """

    def __init__(self, providers: str, env_file: str | None = None, model: str = "",
                 timeout: float = SUMMARY_TIMEOUT, disabled: bool = False):
        self.providers = providers
        self.env_file = env_file
        self.model = model
        self.timeout = timeout
        self.disabled = disabled or os.environ.get("INTROSPECTION_LLM", "").lower() in (
            "0", "off", "false", "no",
        )
        self._client = None
        self._error: str | None = None

    # -- availability ------------------------------------------------
    def _ensure(self) -> None:
        if self._client is not None or self._error is not None:
            return
        if self.disabled:
            self._error = "summary generation is switched off (INTROSPECTION_LLM)"
            return
        try:
            client_cls = _client_class()
            self._client = client_cls.from_providers(
                self.providers,
                model=self.model or None,
                env_file=self.env_file,
                max_retries=1,
                timeout=self.timeout,
            )
        except Exception as exc:
            # Messages from config errors can quote the environment; scrub them.
            self._error = _redact(f"{type(exc).__name__}: {exc}")[0]

    @property
    def available(self) -> bool:
        self._ensure()
        return self._client is not None

    @property
    def reason(self) -> str:
        self._ensure()
        return self._error or ""

    @property
    def model_name(self) -> str:
        self._ensure()
        if self._client is None:
            return ""
        try:
            return self._client.model
        except Exception:                                    # pragma: no cover
            return self.model or "configured model"

    # -- the call ----------------------------------------------------
    def summarize(self, system: str, user: str, max_tokens: int = MAX_SUMMARY_TOKENS) -> str:
        """Blocking on purpose: the MCP tool path calls tools from a worker
        thread, so bridging the async wrapper is just ``asyncio.run`` there.

        The client is closed after every call rather than kept open — the tool
        path may hop threads, and a fresh client per summary costs one TLS
        handshake on a call that is already seconds long.
        """
        self._ensure()
        if self._client is None:
            raise SummaryUnavailable(self._error or "no model configured")
        client = self._client

        async def run() -> str:
            try:
                return await client.complete(
                    user, system=system, thinking=False, max_tokens=max_tokens,
                )
            finally:
                closer = getattr(client, "aclose", None)
                if closer is not None:
                    try:
                        await closer()
                    except Exception:                        # pragma: no cover
                        pass

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run())
        # Already inside a loop (only reachable if a caller wires this into an
        # async route): asyncio.run needs a thread with no running loop.
        with ThreadPoolExecutor(max_workers=1) as pool:       # pragma: no cover
            return pool.submit(lambda: asyncio.run(run())).result()


SUMMARY_UNAVAILABLE_NOTE = (
    "Summary unavailable: {reason}. The structural facts above are complete and "
    "were produced locally — they are not affected by this."
)

OVERVIEW_SYSTEM = (
    "You are documenting a Python web application for another engineer who is "
    "about to change its code. You are given a deterministic directory map: file "
    "names with line counts and, for Python modules, their class/function counts "
    "and first docstring line. Write a tight orientation, in this order:\n"
    "1. What this project is, in one sentence.\n"
    "2. How it is layered — which modules depend on which, following the imports "
    "you can infer, and where the entry points are.\n"
    "3. The three or four most important modules and what each owns.\n"
    "4. Where the risky or subtle parts are, and what a newcomer would get wrong.\n"
    "Be concrete and name files. Do not invent files, functions, or behaviour that "
    "the map does not show; if the map is silent on something, say the map does "
    "not show it. No code fences, no preamble."
)

FILE_SYSTEM = (
    "You are explaining a single source file to another engineer who must modify "
    "it. You are given the file's real contents plus an extracted outline of its "
    "classes and functions. Explain, in this order:\n"
    "1. What this file is responsible for, in one sentence.\n"
    "2. Its public surface — the names another module would use — and what each "
    "does.\n"
    "3. How it works internally: the main flow through the functions, and any "
    "invariants, state, ordering constraints, or error handling that a change "
    "could break.\n"
    "4. Anything surprising, fragile, or worth improving.\n"
    "Ground every claim in the code shown. If the excerpt is partial, say what "
    "you could not see. No code fences, no preamble."
)


def _clip_for_model(text: str, budget: int = MAX_SUMMARY_INPUT) -> tuple[str, bool]:
    """Head+tail clip on line boundaries so the model sees the shape of a file."""
    if len(text) <= budget:
        return text, False
    head_budget = int(budget * 0.75)
    tail_budget = budget - head_budget
    head = text[:head_budget]
    tail = text[-tail_budget:]
    head = head[: head.rfind("\n") + 1] if "\n" in head else head
    tail = tail[tail.find("\n") + 1:] if "\n" in tail else tail
    omitted = len(text) - len(head) - len(tail)
    joined = (
        f"{head}\n"
        f"[... {omitted:,} characters in the middle of this file were not included "
        f"because the summary budget is {budget:,} characters ...]\n"
        f"{tail}"
    )
    return joined, True


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────
def tool_project_overview(arguments: dict, summarizer: Summarizer | None = None) -> list[dict]:
    include = (arguments.get("include") or "").strip() or None
    facts = _project_facts(include)
    included, masked = _redact(_render_tree(facts))

    modules = [f for f in facts if f.language == "python"]
    packages = sorted({
        f.path.split("/")[0] for f in facts if "/" in f.path
    })
    total_lines = sum(f.lines for f in facts)

    head = [
        f"# {PROJECT_ROOT.name} — project map",
        f"Root: {PROJECT_ROOT}",
        f"{len(facts)} text files, {total_lines:,} lines. "
        f"Packages/dirs: {', '.join(packages) if packages else '(flat)'}.",
    ]
    if include:
        head.append(f"Filtered to `{include}`.")
    body = [
        "## Directory tree",
        "```",
        included.strip("\n"),
        "```",
        "",
        "## Python modules",
    ]
    for item in sorted(modules, key=lambda f: f.path):
        line = f"- `{item.path}` — {_fmt_bytes(item.size)}, {item.lines:,} lines"
        if item.blurb:
            line += f"; {item.blurb}"
        if item.error:
            line += f"; {item.error}"
        body.append(line)
    excluded = sorted(d for d in IGNORED_DIRS)
    body += [
        "",
        f"Not walked: {', '.join(excluded)}.",
        "`memory/` holds other conversations and is deliberately unreadable.",
    ]

    if summarizer is not None:
        body += ["", "## How it fits together"]
        if not summarizer.available:
            body.append(SUMMARY_UNAVAILABLE_NOTE.format(reason=summarizer.reason))
        else:
            prompt = _clip_for_model(included)[0]
            try:
                prose = summarizer.summarize(OVERVIEW_SYSTEM, prompt)
            except SummaryUnavailable as exc:
                body.append(SUMMARY_UNAVAILABLE_NOTE.format(reason=exc))
            except Exception as exc:
                body.append(SUMMARY_UNAVAILABLE_NOTE.format(reason=_redact(str(exc))[0]))
            else:
                body.append(prose.strip())
                body.append(f"_Summarised by {summarizer.model_name}._")

    text = "\n".join(head + [""] + body) + _redaction_note(masked)
    return [{"type": "text", "text": text}]


def tool_explain_file(arguments: dict, summarizer: Summarizer | None = None) -> list[dict]:
    raw = arguments.get("path") or ""
    path = _resolve_path(raw)
    text, encoding = _read_text_file(path)
    rel = _relative(path)
    symbols, facts = _outline(text, path)

    without_lines, masked = _redact(text)
    lines = without_lines.count("\n") + (1 if without_lines and not without_lines.endswith("\n") else 0)

    head = [
        f"# {rel}",
        f"{_fmt_bytes(path.stat().st_size)} on disk, {lines:,} lines, "
        f"{language_of(path)}, {encoding}.",
    ]
    if facts.get("error"):
        head.append(f"**This file does not parse: {facts['error']}**")

    body: list[str] = []
    imports = facts.get("imports") or {}
    if imports:
        for label in ("local", "third_party", "stdlib"):
            if imports.get(label):
                body.append(f"- **{label} imports:** {', '.join(imports[label])}")
    if facts.get("nested_definitions"):
        body.append(f"- **Nested definitions:** {facts['nested_definitions']} "
                    "(not listed individually below)")
    if symbols:
        body += ["", "## What it contains", _render_outline(symbols)]

    body += ["", "## How it works"]
    if summarizer is None or not summarizer.available:
        reason = summarizer.reason if summarizer is not None else "summariser disabled"
        body.append(SUMMARY_UNAVAILABLE_NOTE.format(reason=reason))
    else:
        clipped, truncated = _clip_for_model(without_lines)
        if truncated:
            body.append("_(The file was longer than the summary budget, so the ends "
                        "were sent and the middle was skipped.)_")
        prompt = f"File: {rel} ({language_of(path)})\n\n{clipped}"
        try:
            prose = summarizer.summarize(FILE_SYSTEM, prompt)
        except SummaryUnavailable as exc:
            body.append(SUMMARY_UNAVAILABLE_NOTE.format(reason=exc))
        except Exception as exc:
            body.append(SUMMARY_UNAVAILABLE_NOTE.format(reason=_redact(str(exc))[0]))
        else:
            body.append(prose.strip())
            body.append(f"_Summarised by {summarizer.model_name}._")

    body += [
        "",
        f"Read it verbatim with `read_source(path=\"{rel}\")` — `offset`/`limit` "
        "are character offsets for paging.",
    ]
    result = "\n".join(head + [""] + body) + _redaction_note(masked)
    return [{"type": "text", "text": result}]


def tool_read_source(arguments: dict, summarizer: Summarizer | None = None) -> list[dict]:
    raw = arguments.get("path") or ""
    path = _resolve_path(raw)
    rel = _relative(path)
    text, encoding = _read_text_file(path)
    # Mask the whole file first, then page it: that keeps the byte and line
    # offsets in the footer consistent with the offsets of the next call.
    text, masked = _redact(text)
    total_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)

    try:
        offset = max(0, int(arguments.get("offset") or 0))
    except (TypeError, ValueError):
        raise Refused("offset must be a whole number of characters") from None
    try:
        limit = int(arguments.get("limit") or MAX_READ_CHARS)
    except (TypeError, ValueError):
        raise Refused("limit must be a whole number of characters") from None
    limit = max(1, min(limit, MAX_READ_CHARS))

    start = min(offset, len(text))
    end = min(start + limit, len(text))
    page = text[start:end]
    if not page:
        if not text:
            return [{"type": "text", "text": f"{rel} — empty file (0 characters, 0 lines)"}]
        return [{"type": "text", "text": (
            f"{rel} — offset {start:,} is past the end of the file "
            f"({len(text):,} characters, {total_lines:,} lines). Start again at offset 0."
        )}]

    # Split on "\n" only, so the numbering cannot drift from the "\n" count used
    # to find first_line. A file ending with a newline must not gain a phantom
    # final line, and the "\r" of CRLF belongs to the terminator, not the line.
    page_lines = page.split("\n")
    if page_lines[-1] == "":
        page_lines.pop()
    page_lines = [line[:-1] if line.endswith("\r") else line for line in page_lines]

    first_line = text.count("\n", 0, start) + 1
    last_line = first_line + len(page_lines) - 1

    gutter = "\n".join(
        f"{first_line + index:>5}  {line}" for index, line in enumerate(page_lines)
    )
    header = (
        f"{rel} — {_fmt_bytes(path.stat().st_size)}, {total_lines:,} lines, "
        f"{language_of(path)}, {encoding}; showing characters {start:,}-{end:,} "
        f"(lines {first_line}-{last_line})"
    )

    # If the page stopped a character or two short of a line break, resume after
    # the break. Starting the next page inside a CRLF would show a blank line
    # that is not in the file.
    nxt = end
    if not page.endswith("\n"):
        while nxt < len(text) and text[nxt] in "\r\n":
            nxt += 1

    body = [header, gutter]
    if nxt < len(text):
        body.append(
            f"[more follows — call read_source(path=\"{rel}\", offset={nxt})]"
        )
    if start > 0:
        body.append("[showing a page; the file starts at offset 0]")
    note = _redaction_note(masked).strip()
    if note:
        body.append(note)
    return [{"type": "text", "text": "\n".join(body)}]


# ─────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────
TOOLS = [
    {
        "name": "introspect",
        "description": (
            "Map of the deepseekUI project plus a written summary of how it fits "
            "together. Returns the directory tree (line counts, and for Python "
            "modules their class/function counts and docstring line), the module "
            "list, and an orientation covering layering, entry points, and the "
            "subtle parts. Use this FIRST when asked to explain, diagnose, debug, or "
            "improve this codebase, then drill into specific files with "
            "explain_file or read_source. Read-only; confined to the project root. "
            "If it reports that the summary is unavailable, the map itself is still "
            "accurate — that is a model-access problem, not an empty project."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "string",
                    "description": (
                        "Optional glob to limit the map, matched against relative "
                        "paths, e.g. `server/*` or `mcp_server/**`. Omit for everything."
                    ),
                },
            },
        },
    },
    {
        "name": "explain_file",
        "description": (
            "Detailed explanation of ONE file in the deepseekUI project: what it is "
            "responsible for, its classes and functions with signatures, line "
            "numbers and docstrings, its imports grouped into stdlib / third-party / "
            "local, and how its code works internally. Use it to understand a module "
            "before editing it. Read-only. Use read_source instead when you need the "
            "exact text rather than an explanation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to explain, relative to the project root "
                                   "(e.g. `server/media.py`) or absolute inside it.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_source",
        "description": (
            "Read ONE file of the deepseekUI project verbatim, with a line-number "
            "gutter so you can cite and quote exact lines. Nothing is summarised or "
            "reordered. Large files are paged: pass the `offset` from the "
            "[more follows] footer to continue. Read-only; confined to the project "
            "root; `.env`, keys, and the `memory/` transcripts are refused. Secret-"
            "looking values are masked before the text is returned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the project root or "
                                   "absolute inside it.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start from. Default 0.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum characters to return (max {MAX_READ_CHARS:,}).",
                },
            },
            "required": ["path"],
        },
    },
]


# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────
_DEFAULT_SUMMARIZER: Summarizer | None = None


def _summarizer() -> Summarizer:
    global _DEFAULT_SUMMARIZER
    if _DEFAULT_SUMMARIZER is None:
        _DEFAULT_SUMMARIZER = Summarizer(
            providers=str(PROJECT_ROOT / "providers.json"),
            env_file=str(PROJECT_ROOT / ".env"),
            model=os.environ.get("INTROSPECTION_MODEL", ""),
        )
    return _DEFAULT_SUMMARIZER


def call_tool(name: str, arguments: dict | None = None,
              summarizer: Summarizer | None = None) -> list[dict]:
    """Run one tool. Always returns MCP content blocks, never raises."""
    arguments = arguments or {}
    print(f"[tool call] name={name} arguments={arguments}", flush=True)
    engine = summarizer if summarizer is not None else _summarizer()
    try:
        if name == "introspect":
            return tool_project_overview(arguments, engine)
        if name == "explain_file":
            return tool_explain_file(arguments, engine)
        if name == "read_source":
            return tool_read_source(arguments, engine)
        return [{"type": "text", "text": f"Unknown tool: {name}"}]
    except Refused as exc:
        return [{"type": "text", "text": f"Refused: {exc}"}]
    except Exception as exc:                                  # pragma: no cover
        return [{"type": "text", "text": f"Tool error: {type(exc).__name__}: {exc}"}]


# ─────────────────────────────────────────────
# Heartbeat-wrapped tool call
# ─────────────────────────────────────────────
async def _tool_call_with_heartbeat(req_id, name: str, arguments: dict) -> StreamingResponse:
    """Blocking tool in a worker thread, "." keep-alive bytes while it runs.

    NOTE: the body is `....{"jsonrpc": ...}` — dots then JSON, not pure JSON.
    That is deliberate (same contract as mcp_server/mcpserver.py): the client
    must strip anything before the first '{' before json-parsing.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, call_tool, name, arguments)

    async def gen():
        task = asyncio.ensure_future(future)
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_INTERVAL)
            if not done:
                print("[heartbeat] .", flush=True)
                yield b"."
        try:
            content = task.result()
        except Exception as exc:                             # pragma: no cover
            content = [{"type": "text", "text": f"Tool error: {exc}"}]
        payload = {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": content, "isError": False},
        }
        yield json.dumps(payload).encode()

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

    params = body.get("params", {}) or {}
    req_id = body.get("id")
    method = body.get("method")

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "serverInfo": {
                    "name": "self-reflection-mcp",
                    "version": "1.0.0",
                    "instructions": (
                        "Three read-only tools that let you inspect the deepseekUI "
                        "project you are running inside. introspect() returns the "
                        "directory map and an orientation of the whole codebase. "
                        "explain_file(path) explains one file in detail, with its "
                        "classes and functions. read_source(path) returns a file "
                        "verbatim with line numbers. Nothing here writes or executes "
                        "code, and every path is confined to the project root. If a "
                        "tool says a summary is unavailable, the deterministic map or "
                        "outline it returned is still correct — do not report the "
                        "project as empty."
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
        return await _tool_call_with_heartbeat(
            req_id, params.get("name"), params.get("arguments") or {},
        )

    return JSONResponse(
        {"jsonrpc": "2.0", "id": req_id,
         "error": {"code": -32601, "message": f"Method not found: {method}"}},
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deepseekUI self-reflection MCP server")
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the summariser entirely; deterministic output only")
    parser.add_argument("--map", action="store_true", help="print the project map and exit")
    parser.add_argument("--include", default="", help="glob filter for --map")
    parser.add_argument("--explain", metavar="PATH", help="explain one file and exit")
    parser.add_argument("--read", metavar="PATH", help="read one file and exit")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=MAX_READ_CHARS)
    parsed = parser.parse_args(argv)

    # The tree is drawn with box-drawing characters, which a redirected cp1252
    # console stream cannot encode. The served output is UTF-8 JSON and never
    # hits this; it only matters for the convenience CLI.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                        # pragma: no cover
        pass

    engine = Summarizer(
        providers=str(PROJECT_ROOT / "providers.json"),
        env_file=str(PROJECT_ROOT / ".env"),
        model=os.environ.get("INTROSPECTION_MODEL", ""),
        disabled=parsed.no_llm,
    )

    if parsed.map or parsed.explain or parsed.read:
        if parsed.explain:
            blocks = call_tool("explain_file", {"path": parsed.explain}, engine)
        elif parsed.read:
            blocks = call_tool("read_source", {"path": parsed.read, "offset": parsed.offset,
                                               "limit": parsed.limit}, engine)
        else:
            blocks = call_tool("introspect", {"include": parsed.include}, engine)
        print("\n".join(block["text"] for block in blocks))
        return 0

    # Serving, so it owns mcp.json's first run. Kept out of the branch above on
    # purpose: `--map` and friends are read-only commands, and leaving a new file in
    # a project someone asked you to inspect is exactly the kind of surprise this
    # module was written to avoid.
    ensure_mcp_config()

    print(f"Self-reflection MCP server on http://127.0.0.1:{parsed.port}/mcp")
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Tools               : {', '.join(t['name'] for t in TOOLS)}")
    if engine.available:
        print(f"Summariser          : {engine.model_name} (provider config "
              f"{Path(engine.providers).name})")
    else:
        print("Summariser          : UNAVAILABLE — "
              f"{engine.reason}\n                      introspect/explain_file will "
              "still return the map,\n                      symbols, and imports.")
    print(f"Heartbeat           : '.' every {HEARTBEAT_INTERVAL}s")
    uvicorn.run(app, host="127.0.0.1", port=parsed.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
