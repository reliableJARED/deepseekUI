"""Configuration loading: a ``.env`` file, a ``providers.json`` file, or both.

Secrets never live in ``providers.json``.  Instead the JSON references
environment variables with ``${VAR}`` placeholders and this module resolves them
after the ``.env`` file has been read::

    {
      "providers": [
        {
          "name": "DeepSeek",
          "apiKey": "${DEEPSEEK_API_KEY}",
          "apiType": "chat-completions",
          "models": [{"id": "deepseek-flash", "url": "https://api.deepseek.com", ...}]
        }
      ]
    }

Both key spellings are accepted (``apiKey``/``api_key``, ``maxInputTokens``/
``max_input_tokens``) so a config written for a JS frontend drops in unchanged.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

from .errors import ConfigError

__all__ = [
    "DEFAULT_API_TYPE",
    "DEFAULT_API_PATH",
    "DEFAULT_BASE_URL",
    "BETA_BASE_URL",
    "REASONING_EFFORTS",
    "THINKING_TOP_P_FLOOR",
    "DEPRECATED_PARAMS",
    "ModelSpec",
    "Provider",
    "ProvidersConfig",
    "load_env_file",
    "expand_env_refs",
    "find_config_file",
]

DEFAULT_API_TYPE = "chat-completions"
#: DeepSeek's documented path.  The API also accepts a ``/v1`` prefix for OpenAI
#: SDK compatibility; ``_dedupe_base_url`` guarantees we never emit both.
DEFAULT_API_PATH = "/chat/completions"
DEFAULT_BASE_URL = "https://api.deepseek.com"

#: Beta endpoint required for ``strict`` tool schemas.
BETA_BASE_URL = "https://api.deepseek.com/beta"

#: Reasoning effort levels accepted by ``reasoning_effort``.  The server also
#: accepts ``minimal`` (→low), ``medium``/``xhigh`` (→high), and ``ultra`` (→max),
#: but those are normalised away here so callers only deal with four values.
REASONING_EFFORTS = ("none", "low", "high", "max")

#: In thinking mode ``top_p`` is raised to this floor; in non-thinking mode it is
#: ignored entirely and fixed at 1.0.
THINKING_TOP_P_FLOOR = 0.95

#: Parameters the API accepts but silently ignores (documented as deprecated).
DEPRECATED_PARAMS = frozenset({"frequency_penalty", "presence_penalty"})

# The one API shape this wrapper speaks. Anything else is a config error.
SUPPORTED_API_TYPES = frozenset({"chat-completions", "chat_completions", "openai"})

#: Request-shaping flavours.  ``deepseek`` is the historical behaviour and stays
#: the default, so an existing config produces a byte-identical request;
#: ``openai`` drops every field only DeepSeek understands, so the same client can
#: target any plain ``chat-completions`` endpoint (llama.cpp, vLLM, OpenAI).
API_FLAVORS = frozenset({"deepseek", "openai"})
DEFAULT_API_FLAVOR = "deepseek"

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
#: A placeholder that was never filled in — treated as "not configured".
_UNRESOLVED = re.compile(r"^\s*(PULL FROM ENV FILE.*)?\s*$", re.IGNORECASE)


# ── .env parsing ──────────────────────────────────────────────────────────────

def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].lstrip()
    key, sep, value = line.partition("=")
    if not sep:
        return None
    key = key.strip()
    if not key:
        return None
    value = value.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            # Only double quotes honour escapes, matching shell behaviour.
            value = (
                value.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
    else:
        # Unquoted: strip a trailing inline comment introduced by whitespace-#.
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()

    return key, value


def _merge_environ(environ: Mapping[str, str] | None) -> dict[str, str]:
    """A mutable lookup for env resolution: the process environment plus overrides.

    Always returns a *copy*, so loading a ``.env`` file never mutates the real
    process environment as a side effect — which matters for tests, for multiple
    configs loaded with different keys, and for a server that reloads config.
    """
    env: dict[str, str] = dict(os.environ)
    if environ:
        env.update(environ)
    return env


def load_env_file(
    path: str | Path = ".env",
    *,
    override: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Read a ``.env`` file into ``environ`` (the process environment by default).

    Returns the values that were parsed (regardless of whether they were applied).
    Existing entries win unless ``override`` is true, so a shell export or a
    container secret can always supersede the file.

    Supports ``export`` prefixes, single/double quoting, ``#`` comments, and
    ``${OTHER_VAR}`` references to earlier or already-set variables.
    """
    path = Path(path)
    if not path.exists():
        return {}

    env = environ if environ is not None else os.environ
    parsed: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        item = _parse_env_line(raw_line)
        if item is None:
            continue
        key, value = item
        # Resolve ${VAR} against earlier keys in this file, then the environment.
        if _ENV_REF.search(value):
            lookup = {**env, **parsed}
            value = expand_env_refs(value, lookup, strict=False)
        parsed[key] = value

    for key, value in parsed.items():
        if override or key not in env:
            env[key] = value

    return parsed


# ── ${VAR} interpolation ──────────────────────────────────────────────────────

def expand_env_refs(
    value: Any,
    environ: Mapping[str, str] | None = None,
    *,
    strict: bool = True,
) -> Any:
    """Recursively replace ``${VAR}`` placeholders inside any JSON-like value.

    With ``strict=True`` an unset variable raises :class:`ConfigError`, which is
    what you want at startup: a missing API key should fail loudly and early
    rather than produce a confusing 401 on the first request.
    """
    env = environ if environ is not None else os.environ

    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in env:
                if strict:
                    raise ConfigError(
                        f"Config references ${{{name}}} but that variable is not set. "
                        f"Add it to your .env file or export it."
                    )
                return match.group(0)
            return env[name]

        return _ENV_REF.sub(_sub, value)

    if isinstance(value, dict):
        return {k: expand_env_refs(v, env, strict=strict) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_refs(v, env, strict=strict) for v in value]
    return value


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick(data: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first present, non-null value among ``names``."""
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _normalise_api_path(value: Any) -> str:
    path = str(value or DEFAULT_API_PATH).strip()
    if not path:
        return DEFAULT_API_PATH
    return path if path.startswith("/") else f"/{path}"


def _normalise_api_flavor(value: Any, name: str) -> str:
    """Validate ``apiFlavor`` / ``<PREFIX>_API_FLAVOR``, mirroring ``apiType``.

    Case-insensitive, and *unlike* ``apiType`` the value is kept rather than
    reset to a constant — ``openai`` is a real choice, not a synonym.
    """
    flavor = str(value or DEFAULT_API_FLAVOR).strip().lower()
    if flavor not in API_FLAVORS:
        raise ConfigError(
            f"Provider {name!r} declares apiFlavor {flavor!r}, but the supported "
            f"flavours are: {', '.join(sorted(API_FLAVORS))}"
        )
    return flavor


def _dedupe_base_url(base_url: str, api_path: str) -> str:
    """Avoid producing ``https://host/v1/v1/chat/completions``.

    Some configs put the version segment in the base URL, others in the path.
    Either is valid; only the combination is wrong.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1") and api_path.startswith("/v1/"):
        return base + api_path[3:]
    return base + api_path


# ── model / provider models ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model entry: its endpoint, capabilities, and context budget."""

    id: str
    name: str = ""
    url: str = ""
    tool_calling: bool = False
    vision: bool = False
    max_input_tokens: int = 1_000_000
    max_output_tokens: int = 393_216
    #: Whether the model can emit a chain of thought (``reasoning_content``).
    thinking: bool = True
    #: Effort used when a call does not specify one.  Thinking is on by default
    #: server-side, so this mirrors that rather than silently disabling it.
    default_reasoning_effort: str = "high"
    #: Whether ``strict`` JSON-schema tool enforcement is available (Beta).
    strict_tools: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def display_name(self) -> str:
        return self.name or self.id

    def endpoint(self, api_path: str = DEFAULT_API_PATH, provider_url: str = "") -> str:
        """Full URL for this model, preferring its own ``url`` over the provider's."""
        return _dedupe_base_url(self.url or provider_url or DEFAULT_BASE_URL, api_path)

    @property
    def max_output_ceiling(self) -> int:
        """The hard ``max_tokens`` ceiling the API enforces for this model."""
        return 393_216  # 384K, per the DeepSeek Chat Completions API reference

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        model_id = _pick(data, "id", "name", "model")
        if not model_id:
            raise ConfigError(f"Model entry is missing an 'id': {dict(data)!r}")

        known = {
            "id", "name", "model", "url", "baseUrl", "base_url",
            "toolCalling", "tool_calling", "tools",
            "vision", "image", "images",
            "maxInputTokens", "max_input_tokens", "contextWindow",
            "maxOutputTokens", "max_output_tokens", "maxTokens",
            "thinking", "reasoning", "reasoningEffort", "reasoning_effort",
            "strictTools", "strict_tools",
        }
        effort = str(
            _pick(data, "reasoningEffort", "reasoning_effort", default="high") or "high"
        ).strip().lower()
        return cls(
            id=str(model_id),
            name=str(_pick(data, "name", default="") or ""),
            url=str(_pick(data, "url", "baseUrl", "base_url", default="") or ""),
            tool_calling=_as_bool(_pick(data, "toolCalling", "tool_calling", "tools"), False),
            vision=_as_bool(_pick(data, "vision", "image", "images"), False),
            max_input_tokens=_as_int(
                _pick(data, "maxInputTokens", "max_input_tokens", "contextWindow"), 1_000_000
            ),
            max_output_tokens=_as_int(
                _pick(data, "maxOutputTokens", "max_output_tokens", "maxTokens"), 393_216
            ),
            thinking=_as_bool(_pick(data, "thinking", "reasoning"), True),
            default_reasoning_effort=_normalise_effort(effort),
            strict_tools=_as_bool(_pick(data, "strictTools", "strict_tools"), False),
            extra={k: v for k, v in data.items() if k not in known},
        )


def _normalise_effort(value: Any) -> str:
    """Map the server's full effort vocabulary onto our four canonical levels."""
    text = str(value or "high").strip().lower()
    return {
        "minimal": "low",
        "none": "none",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "high",
        "max": "max",
        "ultra": "max",
    }.get(text, "high")


@dataclass(frozen=True, slots=True)
class Provider:
    """A named OpenAI-compatible endpoint and the models it serves."""

    name: str
    api_key: str = ""
    vendor: str = "customendpoint"
    api_type: str = DEFAULT_API_TYPE
    #: ``deepseek`` (default) or ``openai`` — which request fields may be sent.
    api_flavor: str = DEFAULT_API_FLAVOR
    url: str = ""
    api_path: str = DEFAULT_API_PATH
    models: tuple[ModelSpec, ...] = ()
    default_model: str = ""
    #: Extra request body fields merged into every call (e.g. ``{"top_k": 40}``).
    defaults: Mapping[str, Any] = field(default_factory=dict, repr=False)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # ── accessors ──

    @property
    def is_configured(self) -> bool:
        """False when the API key placeholder was never filled in."""
        return bool(self.api_key and not _UNRESOLVED.match(self.api_key))

    @property
    def sends_reasoning_fields(self) -> bool:
        """True when ``thinking``/``reasoning_effort`` may go on the wire.

        Only DeepSeek understands them; a strict OpenAI-compatible server
        rejects the request outright.
        """
        return self.api_flavor == "deepseek"

    @property
    def sends_user_id(self) -> bool:
        """True when the DeepSeek-only ``user_id`` field may go on the wire."""
        return self.api_flavor == "deepseek"

    @property
    def replays_reasoning(self) -> bool:
        """True when ``reasoning_content`` must be echoed back on assistant turns.

        DeepSeek requires it whenever ``tools`` are present; other endpoints
        ignore it, so it is stripped on the way out for them.
        """
        return self.api_flavor == "deepseek"

    def model(self, model_id: str | None = None) -> ModelSpec:
        """Look up a model by id, falling back to the provider's default."""
        wanted = model_id or self.default_model
        if wanted:
            for spec in self.models:
                if spec.id == wanted or spec.name == wanted:
                    return spec
        if self.models:
            return self.models[0]
        raise ConfigError(
            f"Provider {self.name!r} has no models configured"
            + (f" and {wanted!r} was not found" if wanted else "")
        )

    def has_model(self, model_id: str) -> bool:
        return any(spec.id == model_id or spec.name == model_id for spec in self.models)

    def endpoint(self, model_id: str | None = None) -> str:
        spec = self.model(model_id)
        return spec.endpoint(self.api_path, self.url)

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self.models)

    # ── construction ──

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provider":
        name = _pick(data, "name", "id", "provider", default="default")

        api_type = str(_pick(data, "apiType", "api_type", default=DEFAULT_API_TYPE)).strip()
        if api_type.lower() not in SUPPORTED_API_TYPES:
            raise ConfigError(
                f"Provider {name!r} declares apiType {api_type!r}, but this wrapper only "
                f"speaks: {', '.join(sorted(SUPPORTED_API_TYPES))}"
            )

        api_flavor = _normalise_api_flavor(
            _pick(data, "apiFlavor", "api_flavor"), name
        )

        raw_models = _pick(data, "models", default=[]) or []
        if isinstance(raw_models, Mapping):
            # {"models": {"deepseek-flash": {...}}} — tolerate the dict form.
            raw_models = [{"id": k, **(v or {})} for k, v in raw_models.items()]
        models = tuple(ModelSpec.from_dict(m) for m in raw_models)

        known = {
            "name", "id", "provider", "vendor", "apiKey", "api_key", "key",
            "apiType", "api_type", "url", "baseUrl", "base_url", "apiPath", "api_path",
            "models", "defaultModel", "default_model", "model", "defaults",
            "apiFlavor", "api_flavor",
        }
        return cls(
            name=str(name),
            api_key=str(_pick(data, "apiKey", "api_key", "key", default="") or "").strip(),
            vendor=str(_pick(data, "vendor", default="customendpoint") or "customendpoint"),
            api_type=DEFAULT_API_TYPE,
            api_flavor=api_flavor,
            url=str(_pick(data, "url", "baseUrl", "base_url", default="") or "").rstrip("/"),
            api_path=_normalise_api_path(_pick(data, "apiPath", "api_path")),
            models=models,
            default_model=str(_pick(data, "defaultModel", "default_model", "model", default="") or ""),
            defaults=dict(_pick(data, "defaults", default={}) or {}),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def with_overrides(self, **changes: Any) -> "Provider":
        return replace(self, **changes)


# ── config container ──────────────────────────────────────────────────────────

def find_config_file(start: str | Path | None = None, name: str = "providers.json") -> Path | None:
    """Walk up from ``start`` looking for ``providers.json``."""
    current = Path(start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class ProvidersConfig:
    """The whole provider list, plus lookup and env-only fallback."""

    providers: tuple[Provider, ...] = ()
    source: str = ""

    # ── lookup ──

    def get(self, name: str | None = None) -> Provider:
        if not self.providers:
            raise ConfigError("No providers are configured.")
        if name is None:
            return self.providers[0]
        lowered = name.lower()
        for provider in self.providers:
            if provider.name.lower() == lowered or provider.vendor.lower() == lowered:
                return provider
        available = ", ".join(p.name for p in self.providers)
        raise ConfigError(f"Unknown provider {name!r}. Available: {available}")

    def default(self) -> Provider:
        return self.get(None)

    def __iter__(self) -> Iterator[Provider]:
        return iter(self.providers)

    def __len__(self) -> int:
        return len(self.providers)

    def __bool__(self) -> bool:
        return bool(self.providers)

    # ── construction ──

    @classmethod
    def from_dict(cls, data: Any, *, source: str = "<dict>") -> "ProvidersConfig":
        """Accept a single provider, a list of them, or ``{"providers": [...]}``."""
        if isinstance(data, Mapping):
            if "providers" in data:
                entries = data["providers"]
                if not isinstance(entries, list):
                    raise ConfigError(f"'providers' must be a list in {source}")
            elif any(k in data for k in ("apiKey", "api_key", "models", "apiType")):
                entries = [data]  # a bare single-provider object
            else:
                raise ConfigError(f"{source} does not look like a provider config")
        elif isinstance(data, list):
            entries = data
        else:
            raise ConfigError(f"{source} must contain an object or a list of providers")

        return cls(
            providers=tuple(Provider.from_dict(entry) for entry in entries),
            source=source,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        env_file: str | Path | None = ".env",
        environ: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> "ProvidersConfig":
        """Read ``providers.json``, loading ``.env`` first so ``${VAR}`` resolves.

        With ``strict=False`` an unset ``${VAR}`` is left in place instead of
        raising. That is only useful for *display* metadata such as the model
        catalogue — a config loaded that way still holds unresolved ``apiKey``
        placeholders and must never be used to make requests.
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Provider config not found: {path}")

        env = _merge_environ(environ)
        if env_file:
            env_path = Path(env_file)
            if not env_path.is_absolute():
                # Resolve a relative .env next to the config file.
                candidate = path.parent / env_path
                env_path = candidate if candidate.is_file() else env_path
            load_env_file(env_path, environ=env)

        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc

        resolved = expand_env_refs(raw, env, strict=strict)
        return cls.from_dict(resolved, source=str(path))

    @classmethod
    def from_env(
        cls,
        prefix: str = "DEEPSEEK",
        *,
        environ: Mapping[str, str] | None = None,
        load_dotenv: bool = True,
        env_file: str | Path = ".env",
    ) -> "ProvidersConfig":
        """Build a one-provider config purely from environment variables.

        Reads ``<PREFIX>_API_KEY``, ``_BASE_URL``, ``_MODEL``, ``_API_PATH``,
        ``_API_FLAVOR``, and optionally ``_MAX_INPUT_TOKENS`` /
        ``_MAX_OUTPUT_TOKENS`` / ``_VISION`` / ``_TOOL_CALLING``.
        """
        if load_dotenv:
            env = _merge_environ(environ)
            load_env_file(env_file, environ=env)
        else:
            env = _merge_environ(environ)

        def var(suffix: str, default: str = "") -> str:
            return str(env.get(f"{prefix}_{suffix}", default) or default)

        api_key = var("API_KEY")
        model_id = var("MODEL", "deepseek-flash")
        base_url = var("BASE_URL", DEFAULT_BASE_URL)
        api_flavor = _normalise_api_flavor(var("API_FLAVOR"), var("PROVIDER_NAME", "DeepSeek"))

        if not api_key:
            raise ConfigError(
                f"Environment variable {prefix}_API_KEY is not set. "
                f"Copy .env.example to .env and fill it in."
            )

        spec = ModelSpec(
            id=model_id,
            name=var("MODEL_NAME", model_id),
            url=base_url,
            tool_calling=_as_bool(var("TOOL_CALLING", "true"), True),
            vision=_as_bool(var("VISION", "true"), True),
            max_input_tokens=_as_int(var("MAX_INPUT_TOKENS"), 1_000_000),
            max_output_tokens=_as_int(var("MAX_OUTPUT_TOKENS"), 393_216),
            thinking=_as_bool(var("THINKING", "true"), True),
            default_reasoning_effort=_normalise_effort(var("REASONING_EFFORT", "high")),
            strict_tools=_as_bool(var("STRICT_TOOLS", "false"), False),
        )
        provider = Provider(
            name=var("PROVIDER_NAME", "DeepSeek"),
            api_key=api_key,
            vendor="customendpoint",
            api_flavor=api_flavor,
            url=base_url,
            api_path=_normalise_api_path(var("API_PATH", DEFAULT_API_PATH)),
            models=(spec,),
            default_model=model_id,
        )
        return cls(providers=(provider,), source=f"env:{prefix}")

    @classmethod
    def load_or_env(
        cls,
        config_path: str | Path | None = None,
        *,
        prefix: str = "DEEPSEEK",
        env_file: str | Path = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> "ProvidersConfig":
        """Prefer ``providers.json`` if it exists; otherwise fall back to ``.env``.

        This is the entry point the server uses, so either configuration style
        works without a code change.
        """
        env = _merge_environ(environ)
        if env_file:
            load_env_file(env_file, environ=env)

        path = Path(config_path) if config_path else find_config_file()
        if path and path.is_file():
            # env_file=None: the .env was already merged into `env` above.
            return cls.load(path, env_file=None, environ=env)
        return cls.from_env(prefix, environ=env, load_dotenv=False, env_file=env_file)
