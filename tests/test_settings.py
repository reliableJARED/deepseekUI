"""Tests for the DeepSeek API key: the `.env` helpers and `/api/settings/api-key`.

The interesting failure here is not a wrong value but a *stale* one, and there are
three places a stale one can hide:

``providers.json``   references the key as ``${DEEPSEEK_API_KEY}``.
``load_env_file``    never overrides an entry that is already in the environment, so
                     a value the process booted with silently wins over the file.
``make_client_factory`` / ``LazyClient``
                     both cache a client built with that old value for the life of
                     the process.

A save has to defeat all three, or it looks like it worked while still sending the
old key. ``test_a_saved_key_is_the_one_the_next_request_will_send`` and
``test_set_api_key_updates_the_process_environment_as_well_as_the_file`` are the
tests that pin that behaviour down.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from deepseek_client import ProvidersConfig
from deepseek_client.config import load_env_file
from server.app import create_app
from server.settings import (
    API_KEY_SIGNUP_URL,
    API_KEY_VAR,
    api_key_status,
    ensure_env_file,
    is_usable_key,
    load_settings,
    mask_key,
    read_env_var,
    set_api_key,
    upsert_env_var,
)

OLD_KEY = "sk-old-key-0001"
NEW_KEY = "sk-new-key-0002"
MASK = "\u2022" * 8


@pytest.fixture(autouse=True)
def clean_key_environment(monkeypatch):
    """Pin the key environment.

    ``set_api_key`` and ``load_environ`` both write to the real ``os.environ``, so
    without this a test would pick up whatever the previous one left behind, and the
    developer's own shell would decide whether the "no key" tests pass.
    """
    original = os.environ.get(API_KEY_VAR)
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    monkeypatch.delenv("ENV_FILE", raising=False)
    yield
    # `delenv` records no undo when the name was already absent, so restore by hand
    # as well: without this the key a test saved would leak into the rest of the run.
    if original is None:
        os.environ.pop(API_KEY_VAR, None)
    else:
        os.environ[API_KEY_VAR] = original


def make_settings(tmp_path: Path, env_file: Path | None = None):
    """Settings rooted in ``tmp_path``, never reading the project's real `.env`."""
    return load_settings(
        environ={},                       # an empty mapping keeps os.environ untouched
        env_file=str(env_file or tmp_path / ".env"),
        memory_root=tmp_path / "memory",
        providers_path=tmp_path / "providers.json",
    )


def write_providers(tmp_path: Path, api_key: str = "${" + API_KEY_VAR + "}") -> Path:
    """A minimal ``providers.json``. The default apiKey is the shape the real one uses."""
    providers = tmp_path / "providers.json"
    providers.write_text(json.dumps({
        "providers": [{
            "name": "DeepSeek",
            "vendor": "customendpoint",
            "apiKey": api_key,
            "apiType": "chat-completions",
            "models": [{
                "id": "deepseek-flash", "name": "deepseek-flash",
                "url": "https://api.deepseek.com",
                "toolCalling": True, "vision": True,
                "maxInputTokens": 1000000, "maxOutputTokens": 393216,
            }],
        }]
    }), encoding="utf-8")
    return providers


def build_app(tmp_path: Path, *, env_file: Path | None = None, api_key: str = "${" + API_KEY_VAR + "}"):
    return create_app(
        memory_root=str(tmp_path / "memory"),
        providers_path=str(write_providers(tmp_path, api_key)),
        env_file=str(env_file or tmp_path / ".env"),
        frontend_dir=str(tmp_path / "frontend"),
        mcp_config_path=str(tmp_path / "no-mcp.json"),
    )


@pytest.fixture
def key_app(tmp_path):
    """An app whose `.env` already holds a key, plus that file."""
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_VAR}={OLD_KEY}\n", encoding="utf-8")
    return build_app(tmp_path, env_file=env_file), env_file


@pytest.fixture
async def key_client(key_app):
    app, _ = key_app
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ── reading `.env` ────────────────────────────────────────────────────────────

def test_read_env_var_tolerates_a_bom_export_quotes_and_comments(tmp_path):
    env_file = tmp_path / ".env"
    # Exactly what a hand-edited Windows file looks like: BOM, `export`, quotes,
    # a trailing comment.
    env_file.write_bytes(
        "\ufeff# a comment\n"
        f'export {API_KEY_VAR}="sk-quoted-0001"\n'
        "TRAILING=abc  # why\n"
        "SINGLE='sk-single'\n"
        "OTHER=1\n".encode("utf-8")
    )

    assert read_env_var(env_file, API_KEY_VAR) == "sk-quoted-0001"
    assert read_env_var(env_file, "TRAILING") == "abc"
    assert read_env_var(env_file, "SINGLE") == "sk-single"
    assert read_env_var(env_file, "OTHER") == "1"
    assert read_env_var(env_file, "ABSENT") == ""
    assert read_env_var(tmp_path / "nothing.env", API_KEY_VAR) == ""


# ── writing `.env` ────────────────────────────────────────────────────────────

def test_upsert_replaces_one_line_and_leaves_the_rest_byte_for_byte(tmp_path):
    """`.env` is hand-editable, so only the one line may change."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# leading comment\n"
        "HOST=127.0.0.1\n"
        "\n"
        f"{API_KEY_VAR}=stale\n"
        "# trailing comment\n"
        "PORT=5000\n",
        encoding="utf-8",
    )

    upsert_env_var(env_file, API_KEY_VAR, NEW_KEY)

    assert env_file.read_text(encoding="utf-8") == (
        "# leading comment\n"
        "HOST=127.0.0.1\n"
        "\n"
        f"{API_KEY_VAR}={NEW_KEY}\n"
        "# trailing comment\n"
        "PORT=5000\n"
    )
    assert read_env_var(env_file, API_KEY_VAR) == NEW_KEY


def test_upsert_appends_a_blank_line_when_the_name_is_absent(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HOST=127.0.0.1\n", encoding="utf-8")

    upsert_env_var(env_file, API_KEY_VAR, NEW_KEY)

    assert env_file.read_text(encoding="utf-8") == f"HOST=127.0.0.1\n\n{API_KEY_VAR}={NEW_KEY}\n"


def test_upsert_creates_the_file_when_there_is_none(tmp_path):
    env_file = tmp_path / "nested" / ".env"

    upsert_env_var(env_file, API_KEY_VAR, NEW_KEY)

    assert env_file.read_text(encoding="utf-8") == f"{API_KEY_VAR}={NEW_KEY}\n"


def test_upsert_keeps_crlf_and_writes_no_bom(tmp_path):
    """A BOM would break the very parsers that read this file back."""
    env_file = tmp_path / ".env"
    env_file.write_bytes(
        f"\ufeffHOST=127.0.0.1\r\n{API_KEY_VAR}=stale\r\n".encode("utf-8")
    )

    upsert_env_var(env_file, API_KEY_VAR, NEW_KEY)

    raw = env_file.read_bytes()
    assert raw == f"HOST=127.0.0.1\r\n{API_KEY_VAR}={NEW_KEY}\r\n".encode("utf-8")


@pytest.mark.parametrize("value,expected", [
    (NEW_KEY, NEW_KEY),
    ("", ""),
    ("has spaces", '"has spaces"'),
    ("hash#inside", '"hash#inside"'),
])
def test_upsert_quotes_only_what_needs_quoting(tmp_path, value, expected):
    env_file = tmp_path / ".env"

    upsert_env_var(env_file, API_KEY_VAR, value)

    assert env_file.read_text(encoding="utf-8") == f"{API_KEY_VAR}={expected}\n"


@pytest.mark.parametrize("value", [
    NEW_KEY,
    "",
    "has spaces",
    "hash#inside",
    'quote"inside',
])
def test_every_written_value_reads_back_unchanged(tmp_path, value):
    """The file is read back by the wrapper's own parser, not by ours."""
    env_file = tmp_path / ".env"

    upsert_env_var(env_file, API_KEY_VAR, value)

    assert load_env_file(env_file, environ={})[API_KEY_VAR] == value


# ── what counts as a key ──────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    "", "   ", None,
    "${" + API_KEY_VAR + "}",         # a reference that never resolved
    "MY_DEEPSEEK_API_KEY",            # straight out of .env.example
    "YOUR_API_KEY",
    "sk-xxx",
])
def test_a_placeholder_is_not_a_key(value):
    assert is_usable_key(value) is False


def test_a_real_looking_key_is_a_key():
    assert is_usable_key(NEW_KEY) is True


def test_mask_never_returns_more_than_the_last_four_characters():
    assert mask_key("") == ""
    assert mask_key("abcd") == "****"
    assert mask_key(OLD_KEY) == MASK + "0001"
    assert mask_key(NEW_KEY)[-4:] == "0002"


# ── first run: creating `.env` ────────────────────────────────────────────────

def test_ensure_env_file_seeds_a_first_run_from_the_example(tmp_path):
    settings = make_settings(tmp_path)
    (tmp_path / ".env.example").write_text(
        "# every setting, documented\n"
        f"{API_KEY_VAR}=MY_DEEPSEEK_API_KEY\n",
        encoding="utf-8",
    )
    assert not settings.env_file.exists()

    assert ensure_env_file(settings) == settings.env_file

    text = settings.env_file.read_text(encoding="utf-8")
    assert "# every setting, documented" in text
    # The unfilled placeholder is normalised to empty: an example that shipped a
    # non-empty placeholder would otherwise look configured right up to the 401.
    assert read_env_var(settings.env_file, API_KEY_VAR) == ""


def test_ensure_env_file_creates_nothing_without_an_example(tmp_path):
    """An `.env` with no comments and no values is worse than no file at all."""
    settings = make_settings(tmp_path)

    assert ensure_env_file(settings) is None
    assert not settings.env_file.exists()


def test_ensure_env_file_leaves_a_real_key_alone(tmp_path):
    settings = make_settings(tmp_path)
    settings.env_file.write_text(f"{API_KEY_VAR}={OLD_KEY}\n", encoding="utf-8")

    ensure_env_file(settings)

    assert read_env_var(settings.env_file, API_KEY_VAR) == OLD_KEY


# ── making a saved key live ───────────────────────────────────────────────────

def test_set_api_key_updates_the_process_environment_as_well_as_the_file(tmp_path, monkeypatch):
    """Writing the file is not enough, and this is why.

    The server booted with the old key in ``os.environ``. ``load_env_file`` does not
    override an entry that already exists, so every later read of the key would keep
    returning the old value from the *file* that says otherwise.
    """
    settings = make_settings(tmp_path, env_file=tmp_path / ".env")
    settings.env_file.write_text(f"{API_KEY_VAR}={OLD_KEY}\n", encoding="utf-8")
    providers = write_providers(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, OLD_KEY)

    set_api_key(settings, NEW_KEY)

    assert read_env_var(settings.env_file, API_KEY_VAR) == NEW_KEY
    assert os.environ[API_KEY_VAR] == NEW_KEY

    # The thing that actually builds requests, resolved the way a request would.
    provider = ProvidersConfig.load_or_env(providers, env_file=settings.env_file).get("DeepSeek")
    assert provider.api_key == NEW_KEY


# ── api_key_status ────────────────────────────────────────────────────────────

def test_api_key_status_calls_a_placeholder_absent(tmp_path):
    settings = make_settings(tmp_path)
    settings.env_file.write_text(f"{API_KEY_VAR}=MY_DEEPSEEK_API_KEY\n", encoding="utf-8")

    status = api_key_status(settings)

    assert status["present"] is False
    assert status["masked"] == ""
    assert status["file_exists"] is True
    assert status["file"] == str(settings.env_file)
    assert status["signup_url"] == API_KEY_SIGNUP_URL


def test_api_key_status_masks_a_real_key(tmp_path):
    settings = make_settings(tmp_path)
    settings.env_file.write_text(f"{API_KEY_VAR}=sk-live-0123456789abcd\n", encoding="utf-8")

    status = api_key_status(settings)

    assert status["present"] is True
    assert status["masked"] == MASK + "abcd"
    assert "0123456789" not in json.dumps(status)


# ── the route ─────────────────────────────────────────────────────────────────

async def test_get_api_key_returns_a_mask_and_never_the_key(key_client):
    response = await key_client.get("/api/settings/api-key")

    assert response.status_code == 200
    body = response.json()["api_key"]
    assert body["present"] is True
    assert body["masked"] == MASK + OLD_KEY[-4:]
    assert OLD_KEY not in response.text


async def test_post_api_key_refuses_an_empty_value(key_client, key_app):
    _, env_file = key_app

    missing = await key_client.post("/api/settings/api-key", json={})
    blank = await key_client.post("/api/settings/api-key", json={"api_key": "   "})

    assert missing.status_code == 400
    assert blank.status_code == 400
    # A rejected save must not blank the key that was already working.
    assert read_env_var(env_file, API_KEY_VAR) == OLD_KEY


async def test_post_api_key_strips_a_bearer_prefix(key_client, key_app):
    """The docs page hands out `sk-...`; tooling around it often hands out `Bearer sk-...`."""
    _, env_file = key_app

    response = await key_client.post(
        "/api/settings/api-key", json={"api_key": f"Bearer {NEW_KEY}"}
    )

    assert response.status_code == 200
    assert read_env_var(env_file, API_KEY_VAR) == NEW_KEY


async def test_a_saved_key_is_the_one_the_next_request_will_send(key_client, key_app):
    """The whole point: a save has to invalidate *both* caches.

    The factory caches a client for the process and `LazyClient` caches the one it
    resolved. Miss either and the old key keeps being sent, which is indistinguishable
    from a save that silently did nothing.
    """
    app, env_file = key_app

    # Resolve the lazy client now, the way a first message would have.
    before = app.state.app_state.engine.client.resolve()
    assert before.provider.api_key == OLD_KEY

    response = await key_client.post("/api/settings/api-key", json={"api_key": NEW_KEY})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["client_ready"] is True, body
    assert body["api_key"]["masked"] == MASK + NEW_KEY[-4:]
    assert NEW_KEY not in response.text

    assert read_env_var(env_file, API_KEY_VAR) == NEW_KEY

    after = app.state.app_state.engine.client.resolve()
    assert after is not before                    # dropped, not mutated
    assert after.provider.api_key == NEW_KEY


async def test_saving_a_key_brings_a_keyless_server_up_without_a_restart(tmp_path):
    """The flow the settings panel is for: first run, no `.env` at all."""
    app = build_app(tmp_path)                     # no .env anywhere yet

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            started = (await c.get("/api/props")).json()
            assert started["configured"] is False
            assert started["api_key"]["present"] is False
            assert started["api_key"]["masked"] == ""
            assert "Settings" in started["hint"]

            saved = await c.post("/api/settings/api-key", json={"api_key": NEW_KEY})

            assert saved.status_code == 200, saved.text
            assert saved.json()["client_ready"] is True
            assert read_env_var(tmp_path / ".env", API_KEY_VAR) == NEW_KEY

            # No restart: the very next read of /api/props is configured.
            live = (await c.get("/api/props")).json()
            assert live["configured"] is True
            assert live["api_key"]["present"] is True


async def test_props_exposes_the_key_status_to_the_ui(key_client):
    body = (await key_client.get("/api/props")).json()

    assert body["configured"] is True
    assert body["api_key"]["present"] is True
    assert body["api_key"]["masked"] == MASK + OLD_KEY[-4:]
    assert body["api_key"]["signup_url"] == API_KEY_SIGNUP_URL
