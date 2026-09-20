"""Application wiring.

Assembles the pieces into one Starlette app and owns their lifetimes. The tool
registry is rebuilt rather than mutated, because MCP servers can be reloaded at
runtime and a half-updated registry would let the model call a tool that no longer
exists.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette

from deepseek_client.config import REASONING_EFFORTS

from . import __version__
from .llm import ChatEngine, make_client_factory
from .mcp import MCPManager, load_mcp_config
from .media import MediaStore, mark_display_blocks
from .routes import create_routes, make_middleware
from .settings import ensure_env_file, ensure_mcp_file, load_environ, load_settings
from .store import ConversationStore
from .tools_builtin import build_media_tools

logger = logging.getLogger("deepseek_ui.app")


class LazyClient:
    """A stand-in that resolves the real client on first use.

    The server has to boot without an API key — otherwise you cannot develop or even
    open the frontend offline. Every attribute access triggers resolution, so the
    failure lands inside the request handler that tried to use the model, where
    ``/api/props`` can turn it into ``configured: false`` with the real message.
    """

    def __init__(self, factory):
        self._factory = factory
        self._client = None
        self._error: Exception | None = None

    def resolve(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def invalidate(self) -> None:
        """Forget the resolved client so the next access builds it again.

        Necessary *as well as* dropping the factory's cache: this object holds its
        own reference, and a new key that only reached the factory would still be
        shadowed by a client resolved a moment earlier.
        """
        self._client = None
        self._error = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        return "<LazyClient resolved>" if self._client else "<LazyClient pending>"


@dataclass
class AppState:
    """Shared, long-lived objects the route handlers close over."""

    settings: Any
    store: ConversationStore
    media: MediaStore
    engine: ChatEngine
    mcp: MCPManager | None = None
    version: str = __version__
    reasoning_efforts: tuple[str, ...] = REASONING_EFFORTS
    refresh_tools: Any = None
    client_factory: Any = None
    _tool_names: list[str] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    def rebuild_tools(self) -> None:
        """Recreate the registry from the current MCP connections plus built-ins."""
        from deepseek_client.tools import ToolRegistry

        registry = ToolRegistry()

        builtin = build_media_tools(
            self.store,
            self.media,
            self.engine.active_uuid,
            settings=self.settings,
        )
        for tool in builtin:
            registry.add(tool)

        if self.mcp is not None:
            self.mcp.set_ingest(self._ingest_tool_media)
            self.mcp.register_into(registry, uuid_provider=self.engine.active_uuid)

        self.engine.tool_registry = registry
        names = sorted(entry["function"]["name"] for entry in registry.schemas())
        self._tool_names = names
        logger.info("tool registry rebuilt: %d tools", len(names))
        return registry

    def _ingest_tool_media(self, uuid: str, blocks, tool_name: str):
        """Callback MCP tools use to persist the media they return.

        Everything an MCP tool returns is marked user-facing, but as *incidental* media:
        a remote tool hands back media because of what it was asked to do rather than
        because anyone asked to see a picture — ``web_fetch`` downloads the images on
        the page it read, because a page has images. Nobody chose those, so they are
        marked ``inline``: the block stays in the result and renders inside the tool
        card that carried it, collapsed, instead of being pinned above the reply for
        the life of the transcript.

        The model still gets each file's path in place of the media — ``inline`` is a
        mark, not an exemption, so :mod:`server.rehydrate` swaps it for a path exactly
        as it does for a block ``display_media`` showed — and can pull one back into
        its own vision with ``resize_image`` or ``reduce_video_frames`` if it genuinely
        needs to look.
        """
        blocks = self.media.ingest_tool_blocks(uuid, blocks, tool_name)
        return mark_display_blocks(blocks)

    async def refresh_client(self) -> None:
        """Drop the cached model client so the next request re-reads `.env`.

        **Both** caches have to go. The factory caches one client for the whole
        process and `LazyClient` caches the one it resolved; invalidating only one
        of them leaves the old key in use, which looks exactly like a save that
        silently did nothing.
        """
        client = getattr(self.engine, "client", None)
        forget = getattr(client, "invalidate", None)
        if callable(forget):
            forget()
        if self.client_factory is not None:
            drop = getattr(self.client_factory, "invalidate", None)
            if callable(drop):
                await drop()


def build_state(settings=None, **overrides) -> AppState:
    settings = settings or load_settings(**overrides)

    # A first run has no `.env` at all, and "paste your key into .env" is not advice
    # anyone can follow against a file that does not exist. Seeding it from
    # `.env.example` keeps the documentation (the comments explaining every setting)
    # and gives the settings panel somewhere to write. Non-fatal: a read-only
    # checkout still runs, it just cannot save a key from the UI.
    try:
        ensure_env_file(settings)
    except OSError as exc:
        logger.warning("could not prepare %s (%s)", settings.env_file, exc)

    # The same argument for `mcp.json`, which matters more because its destination is
    # git-ignored rather than merely usually-absent: a fresh clone has no file, so the
    # settings panel opens on an empty server list with no sign that a template was
    # ever meant to be there. Must run BEFORE the read below, or the first boot would
    # seed a file it then failed to use. Also non-fatal.
    try:
        ensure_mcp_file(settings)
    except OSError as exc:
        logger.warning("could not prepare %s (%s)", settings.mcp_config_path, exc)

    store = ConversationStore(settings.memory_root)
    media = MediaStore(store)

    # Same environment the rest of the server resolved against, so a token written as
    # `${MY_TOKEN}` in mcp.json can come from .env.
    environ = load_environ(settings.env_file)
    specs = load_mcp_config(settings.mcp_config_path, environ=environ)
    # The manager is built even with no servers configured. It used to be left as
    # None, but the settings panel can now add the first server at runtime, and
    # "no manager" would mean "no way to ever have one" without a restart. An empty
    # manager exposes no tools and costs one object.
    mcp = MCPManager(specs, config_path=settings.mcp_config_path, environ=environ)
    if not specs:
        logger.info("no MCP servers configured in %s", settings.mcp_config_path)

    factory = make_client_factory(settings)

    # Try once up front so a bad config is reported at startup rather than on the
    # first message, but do not make it fatal — the client stays lazy either way.
    try:
        factory()
    except Exception as exc:
        logger.warning(
            "the model is not configured yet (%s: %s). The UI will start and "
            "GET /api/props will report configured=false.",
            type(exc).__name__,
            exc,
        )

    engine = ChatEngine(
        settings, store, media, LazyClient(factory), tool_registry=None, mcp_manager=mcp
    )

    state = AppState(settings=settings, store=store, media=media, engine=engine, mcp=mcp)
    state.client_factory = factory            # closed on shutdown
    state.refresh_tools = state.rebuild_tools
    state.rebuild_tools()
    return state


def create_app(settings=None, **overrides) -> Starlette:
    """Build the ASGI application."""
    state = build_state(settings, **overrides)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        if state.mcp is not None:
            await state.mcp.start()
            state.rebuild_tools()

        try:
            model = state.engine.client.spec_for(state.settings.model or None).id
        except Exception:
            model = "(not configured)"

        logging.getLogger("deepseek_ui").info(
            "ready on http://%s:%s  |  model=%s  |  media=%s  |  tools=%d",
            state.settings.host,
            state.settings.port,
            model,
            state.settings.memory_root,
            len(state.tool_names()),
        )
        try:
            yield
        finally:
            if state.mcp is not None:
                await state.mcp.stop()
            if state.client_factory is not None:
                try:
                    await state.client_factory.aclose()
                except Exception:                 # pragma: no cover - shutdown noise
                    logger.debug("error closing the HTTP client", exc_info=True)

    app = Starlette(
        debug=False,
        routes=create_routes(state),
        middleware=make_middleware(state.settings),
        lifespan=lifespan,
    )
    # Starlette's own `app.state` is a plain attribute bag; expose ours under a name
    # that cannot collide so tests and debugging can reach the wired objects.
    app.state.app_state = state
    return app


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request at INFO, which buries our own output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
