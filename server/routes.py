"""HTTP surface.

Starlette ``Route`` objects rather than FastAPI decorators, following the reference
implementation — the handlers need raw ``Request`` access for streaming and multipart
bodies, and there is no request-model validation worth the indirection here.

Every route that takes a conversation id resolves it through the store, which
validates it as a single safe path segment. ``/memory/...`` is the only place a
filename comes from the URL, and it is checked against the resolved conversation
directory so ``..`` cannot escape.
"""

from __future__ import annotations

import copy
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Mapping

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from deepseek_client.config import ProvidersConfig, expand_env_refs
from deepseek_client.messages import ensure_tool_pairing

from .llm import compose_system, sse
from .mcp import DEFAULT_CALL_TIMEOUT, MCPDocument, MCPServerSpec, probe_server, read_mcp_document
from .media import classify_kind, is_text_mime, sniff_mime, text_facts
from .settings import (
    MAX_TOOL_STEPS_LIMIT,
    MIN_TOOL_STEPS,
    api_key_status,
    load_environ,
    set_api_key,
    set_tool_steps,
)

logger = logging.getLogger("deepseek_ui.routes")


# ── helpers ───────────────────────────────────────────────────────────────────

def served_mime(path: Path) -> str:
    """The content type to serve a stored media file with.

    Content is checked before the extension because the extension is only whatever
    the upload happened to be called. ``mimetypes`` has no ``.webp`` entry on Windows,
    which served genuine WebP images as ``text/plain`` and left them unrenderable.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(32)
    except OSError:                                       # pragma: no cover
        head = b""
    sniffed = sniff_mime(head)
    if sniffed != "application/octet-stream":
        return sniffed
    # Nothing recognisable in the bytes, so the extension is the better guess.
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def spec_payload(spec: Any) -> dict:
    """The model metadata the frontend uses to render its picker and warnings."""
    return {
        "id": spec.id,
        "name": spec.display_name,
        "tool_calling": spec.tool_calling,
        "vision": spec.vision,
        "thinking": spec.thinking,
        "default_reasoning_effort": spec.default_reasoning_effort,
        "max_input_tokens": spec.max_input_tokens,
        "max_output_tokens": spec.max_output_tokens,
        "max_output_ceiling": spec.max_output_ceiling,
    }


def ok(data: Any) -> JSONResponse:
    return JSONResponse(data)


def fail(message: str, status: int = 400, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


async def read_json(request: Request) -> dict[str, Any]:
    """Parse a JSON body, returning ``{}`` when it is absent or malformed."""
    try:
        body = await request.body()
    except Exception:
        return {}
    if not body:
        return {}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def convo_payload(conv) -> dict[str, Any]:
    """The full conversation as the frontend wants it, media URLs included."""
    return {
        "uuid": conv.uuid,
        "title": conv.title,
        "model": conv.model,
        "created": conv.created,
        "updated": conv.updated,
        "sys_base": conv.sys_base,
        "sys_todo": conv.sys_todo,
        "system_prompt": compose_system(conv.sys_base or "", conv.sys_todo or ""),
        "messages": conv.messages,
        "extra": conv.extra,
    }


# ── app factory ───────────────────────────────────────────────────────────────

def create_routes(state) -> list[Route]:
    """Build the route table. ``state`` carries the wired-up dependencies."""

    settings = state.settings
    store = state.store
    media = state.media
    engine = state.engine
    mcp = state.mcp

    # ── frontend ──

    async def index(_: Request) -> Response:
        directory = Path(settings.frontend_dir)
        for name in ("index.html", "dist/index.html"):
            candidate = directory / name
            if candidate.is_file():
                return FileResponse(candidate, media_type="text/html")
        return HTMLResponse(
            "<h1>deepseekUI</h1>"
            "<p>No frontend build found. Set <code>FRONTEND_DIR</code> to the built "
            "frontend, or use the JSON API directly.</p>"
            f"<p style='color:#888'>Looked in <code>{directory}</code></p>",
            status_code=200,
        )

    async def assets(request: Request) -> Response:
        """Serve `/assets/<x>` from whichever search root actually holds `x`.

        The URL prefix is the route, not a directory name, so it is stripped by the
        matcher: `/assets/app.js` arrives here as `app.js`. Bundlers emit that file
        as `dist/assets/app.js`, so the search has to include the `assets`
        subdirectory as well as the build root.
        """
        relative = request.path_params.get("path", "")
        if not relative or relative.startswith(".") or ".." in Path(relative).parts:
            return fail("bad asset path", 400)
        root = Path(settings.frontend_dir)
        for base in (root / "assets", root / "dist" / "assets", root, root / "dist"):
            base = base.resolve()
            candidate = (base / relative).resolve()
            # A `..` hidden inside a symlink would otherwise escape the tree.
            if base not in candidate.parents and candidate != base:
                continue
            if candidate.is_file():
                return FileResponse(candidate)
        return fail("not found", 404)

    # ── media files ──

    async def memory_file(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        filename = request.path_params["filename"]
        try:
            store.validate(uuid)
        except Exception:
            return fail("bad conversation id", 400)

        candidate = (store.root / uuid / filename).resolve()
        root = (store.root / uuid).resolve()
        if root not in candidate.parents or not candidate.is_file():
            return fail("not found", 404)

        return FileResponse(
            candidate,
            media_type=served_mime(candidate),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    async def download_file(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        filename = request.path_params["filename"]
        try:
            store.validate(uuid)
        except Exception:
            return fail("bad conversation id", 400)

        candidate = (store.root / uuid / filename).resolve()
        root = (store.root / uuid).resolve()
        if root not in candidate.parents or not candidate.is_file():
            return fail("not found", 404)

        return FileResponse(candidate, media_type=served_mime(candidate), filename=filename)

    # ── props ──

    def _limits() -> dict:
        """Limits the UI displays, plus the bounds of the one the panel can change.

        ``max_tool_steps`` is read through ``settings.effective`` rather than the
        attribute so a value changed at runtime is reported back, instead of the one
        the process booted with.
        """
        return {
            "model_image_max_dim": settings.model_image_max_dim,
            "model_video_max_dim": settings.model_video_max_dim,
            "model_video_fps": settings.model_video_fps,
            "model_video_max_frames": settings.model_video_max_frames,
            "model_max_images": settings.model_max_images,
            "context_safety_ratio": settings.context_safety_ratio,
            "max_tool_steps": settings.effective("max_tool_steps"),
            "min_tool_steps": MIN_TOOL_STEPS,
            "max_tool_steps_limit": MAX_TOOL_STEPS_LIMIT,
        }

    def _static_props() -> dict:
        """The parts of the props payload that never depend on the model being up."""
        return {
            "version": state.version,
            "reasoning_efforts": list(state.reasoning_efforts),
            "limits": _limits(),
            # Both MCP helpers are defined further down in this same closure. They are
            # bound long before the first request is served, and sharing them keeps
            # /api/props and /api/mcp from drifting into two different shapes.
            "mcp": _mcp_payload()["servers"],
            "mcp_config": _mcp_summary(),
            "tools": state.tool_names(),
            # Masked only. The UI needs to distinguish "a key is set" from "no key",
            # and it must never be able to read the key back out of the API.
            "api_key": api_key_status(settings),
        }

    def _catalogue() -> tuple[list[dict], str]:
        """Model metadata for display, parsed without resolving credentials.

        ``providers.json`` model entries carry no secrets — only ids, capability
        flags and token limits. Re-reading them leniently means the picker still
        lists the models, and still shows which ones accept images, on a first run
        where the API key has not been set yet. That is precisely when the user
        needs to see the catalogue, and an empty dropdown would be a dead end.
        """
        config = ProvidersConfig.load(
            settings.providers_path, env_file=settings.env_file, strict=False
        )
        provider = config.get(getattr(settings, "provider", None) or None)
        specs = provider.models
        return ([spec_payload(spec) for spec in specs], specs[0].id if specs else "")

    async def props(_: Request) -> Response:
        """Everything the UI needs to render itself before the first request."""
        try:
            spec_default = engine.client.spec_for(settings.model or None)
            models = [spec_payload(spec) for spec in engine.client.models()]
            endpoint = engine.client.endpoint(spec_default.id)
            provider = engine.client.provider.name if engine.client.provider else ""
        except Exception as exc:
            logger.error("could not read provider config: %s", exc)
            # The UI still has to render so the user can browse conversations and
            # attach files while the key is missing, so degrade instead of failing.
            try:
                models, default_model = _catalogue()
            except Exception as cat_exc:
                logger.debug("model catalogue unavailable: %s", cat_exc)
                models, default_model = [], settings.model or ""
            return ok({
                "configured": False,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": (
                    "Paste a key into the DeepSeek API key field in Settings, or set "
                    "DEEPSEEK_API_KEY in .env. Fix providers.json if that is the problem."
                ),
                "models": models,
                "default_model": default_model,
                **_static_props(),
            })

        return ok({
            "configured": True,
            "provider": provider,
            "endpoint": endpoint,
            "default_model": spec_default.id,
            "models": models,
            **_static_props(),
        })

    async def settings_api_key(request: Request) -> Response:
        """Read or replace the DeepSeek API key.

        ``GET`` reports a *masked* summary — never the key — so the settings panel can
        show that one is already stored instead of an empty box that invites a
        needless overwrite.

        ``POST`` writes `.env` and then drops the cached model clients, which is the
        whole point: a written file changes nothing on its own, because the client
        built at startup keeps its old ``Authorization`` header for the life of the
        process.
        """
        if request.method == "GET":
            return ok({"api_key": api_key_status(settings)})

        body = await read_json(request)
        raw = body.get("api_key", body.get("key"))
        if raw is None:
            return fail("api_key is required")

        key = str(raw).strip()
        # The docs page hands out `sk-…`; the OpenAI-compatible tooling around it
        # often hands out `Bearer sk-…`. Accept both rather than storing a header.
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        if not key:
            return fail("api_key must not be empty")

        try:
            path = set_api_key(settings, key)
        except OSError as exc:
            logger.error("could not write %s: %s", settings.env_file, exc)
            return fail(f"could not write {settings.env_file}: {exc}", 500)

        await state.refresh_client()

        # Report whether the *client* can be built now. This is not a validity check
        # against DeepSeek — that would need a billable round trip — but it does catch
        # a broken providers.json or a key that resolves to nothing.
        try:
            engine.client.spec_for(settings.model or None)
            ready, error = True, ""
        except Exception as exc:
            ready, error = False, f"{type(exc).__name__}: {exc}"
            logger.warning("key stored but the client is still unusable: %s", error)

        return ok({
            "api_key": api_key_status(settings),
            "saved": str(path),
            "client_ready": ready,
            "error": error,
        })

    async def settings_limits(request: Request) -> Response:
        """Read or change the tool-loop ceiling.

        ``max_tool_steps`` is the one limit here that can be changed while the server
        runs. Writing `.env` is not enough on its own, for the same reason it is not
        enough for the API key: ``load_env_file`` never overrides an entry that is
        already in the process environment, so the value the process booted with would
        keep winning. :func:`set_tool_steps` writes the file, updates ``os.environ``
        and records a live override on the shared ``Settings`` instance the engine and
        the built-in tools already point at — so the next message uses the new ceiling
        with nothing to restart and nothing to invalidate.
        """
        if request.method == "GET":
            return ok({"limits": _limits()})

        body = await read_json(request)
        if "max_tool_steps" not in body:
            return fail("max_tool_steps is required")

        try:
            steps = set_tool_steps(settings, body["max_tool_steps"])
        except ValueError as exc:
            # A rejected save must leave the working value alone, so nothing is
            # written before validation succeeds.
            return fail(str(exc))
        except OSError as exc:
            logger.error("could not write %s: %s", settings.env_file, exc)
            return fail(f"could not write {settings.env_file}: {exc}", 500)

        return ok({
            "limits": _limits(),
            "max_tool_steps": steps,
            "saved": str(settings.env_file),
        })

    async def health(_: Request) -> Response:
        return ok({"status": "ok", "version": state.version})

    # ── conversations ──

    async def list_conversations(_: Request) -> Response:
        return ok({"conversations": store.summaries()})

    async def create_conversation(request: Request) -> Response:
        body = await read_json(request)
        conv = store.create(
            title=str(body.get("title") or ""),
            model=str(body.get("model") or settings.model or ""),
            sys_base=str(body.get("sys_base") or body.get("system_prompt") or ""),
        )
        return ok(convo_payload(conv))

    async def get_conversation(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        conv = store.load_or_none(uuid)
        if conv is None:
            return fail("no such conversation", 404)
        return ok({**convo_payload(conv), "context": engine.context_status(conv)})

    async def delete_conversation(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        if not store.delete(uuid):
            return fail("no such conversation", 404)
        return ok({"deleted": uuid})

    async def rename_conversation(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        body = await read_json(request)
        title = str(body.get("title") or "").strip()
        if not title:
            return fail("title is required")

        try:
            conv = await store.mutate(uuid, lambda c: setattr(c, "title", title[:200]))
        except Exception as exc:
            return fail(str(exc), 404)
        return ok({"uuid": uuid, "title": conv.title})

    async def set_system_prompt(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        body = await read_json(request)
        base = body.get("sys_base", body.get("system_prompt"))
        todo = body.get("sys_todo")

        def _apply(conv):
            if base is not None:
                conv.sys_base = str(base)
            if todo is not None:
                conv.sys_todo = str(todo)

        try:
            conv = await store.mutate(uuid, _apply)
        except Exception as exc:
            return fail(str(exc), 404)

        # `messages[0]` is rebuilt on every request, so the stored transcript does
        # not need rewriting — only the two prompt fields change.
        return ok({
            "uuid": uuid,
            "sys_base": conv.sys_base,
            "sys_todo": conv.sys_todo,
            "system_prompt": compose_system(conv.sys_base, conv.sys_todo),
        })

    async def set_model(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        body = await read_json(request)
        model = str(body.get("model") or "").strip()
        if not model:
            return fail("model is required")
        try:
            spec = engine.client.spec_for(model)          # resolves the provider
            known = {s.id for s in engine.client.models()} | {s.name for s in engine.client.models()}
        except Exception as exc:
            return fail(f"cannot check models — {exc}", 400)

        # `spec_for` falls back to the default model rather than raising, so the
        # membership check above is what actually rejects a typo.
        if model not in known:
            return fail(f"unknown model {model!r}. Available: {', '.join(sorted(known))}")

        conv = await store.mutate(uuid, lambda c: setattr(c, "model", model))
        return ok({"uuid": uuid, "model": conv.model})

    async def append_message(request: Request) -> Response:
        """Append a user turn, persisting any inline media to disk first."""
        uuid = request.path_params["uuid"]
        body = await read_json(request)
        content = body.get("content")
        if content is None:
            return fail("content is required")

        try:
            store.validate(uuid)
            if not store.exists(uuid):
                # The client asked to post into a conversation that is not on disk.
                # Create it at the id it named, so a client that generates its own
                # ids can keep talking to the id it already used.
                store.create(
                    title=str(body.get("title") or ""),
                    model=str(body.get("model") or settings.model or ""),
                    uuid=uuid,
                )
        except Exception as exc:
            return fail(str(exc), 400)

        try:
            content = media.ingest_user_content(uuid, content)
        except Exception as exc:
            return fail(f"could not store media: {exc}", 400)

        message = {"role": str(body.get("role") or "user"), "content": content}
        try:
            conv = await store.append(uuid, message)
        except Exception as exc:
            return fail(str(exc), 404)

        return ok({
            "uuid": uuid,
            "message": message,
            "message_count": len(conv.messages),
            "context": engine.context_status(conv),
        })

    async def upload_media(request: Request) -> Response:
        """Multipart upload for anything too big to inline as base64.

        Video in particular should come through here: base64 inflates by a third and
        a 200 MB clip would be 270 MB of JSON.
        """
        uuid = request.path_params["uuid"]
        try:
            store.validate(uuid)
        except Exception as exc:
            return fail(str(exc), 400)

        try:
            form = await request.form()
        except Exception as exc:
            return fail(f"could not read upload: {exc}", 400)

        upload = form.get("file") or form.get("media")
        if upload is None or not hasattr(upload, "read"):
            return fail("no file in the request")

        data = await upload.read()
        if not data:
            return fail("the uploaded file was empty")

        limit = 512 * 1024 * 1024
        if len(data) > limit:
            return fail(f"file is too large ({len(data)} bytes); limit is {limit}")

        declared = str(getattr(upload, "content_type", "") or "")
        name = str(getattr(upload, "filename", "") or "upload")

        mime = sniff_mime(data, declared)
        # Content decides the kind, so a `.md` or a `.gitignore` is stored as text and
        # is readable later. When the bytes say nothing (`binary`) the client's label
        # still wins, which keeps odd-but-supported formats working as before.
        kind = classify_kind(data, declared, name)
        if kind == "binary":
            kind = str(form.get("kind") or "") or "file"
        if kind == "text" and not is_text_mime(mime):
            mime = "text/plain"      # so a name with no suffix still lands on `.txt`

        prefix = {"video": "user_video", "audio": "user_audio", "image": "user_image"}.get(
            kind, "user_file"
        )
        try:
            url = media.save_bytes(uuid, data, mime, prefix=prefix, name=name)
        except Exception as exc:
            return fail(f"could not store the upload: {exc}", 500)

        block: dict[str, Any] = {"type": "_file", "kind": kind or "file", "name": name, "mime": mime, "url": url}
        if kind == "text":
            # Size and line count travel with the block so the UI can caption the chip
            # without fetching the file back just to count its lines.
            block.update(text_facts(data))
        if kind == "image" or mime.startswith("image/"):
            block["kind"] = "image"
            size = media.probe_size(data)
            if size:
                block["width"], block["height"] = size
            block["type"] = "image_url"
            block["image_url"] = {"url": url}

        return ok({"url": url, "mime": mime, "bytes": len(data), "block": block})

    async def upload_to_conversation(request: Request) -> Response:
        """Upload a file and append it as a user turn in one step."""
        uuid = request.path_params["uuid"]
        try:
            store.validate(uuid)
        except Exception as exc:
            return fail(str(exc), 400)

        try:
            form = await request.form()
        except Exception as exc:
            return fail(f"could not read upload: {exc}", 400)

        upload = form.get("file") or form.get("media")
        if upload is None or not hasattr(upload, "read"):
            return fail("no file in the request")

        data = await upload.read()
        declared = str(getattr(upload, "content_type", "") or "")
        name = str(getattr(upload, "filename", "") or "upload")
        mime = sniff_mime(data, declared)
        kind = classify_kind(data, declared, name)
        if kind == "binary":
            kind = str(form.get("kind") or "") or (
                "image" if mime.startswith("image/")
                else "video" if mime.startswith("video/")
                else "file"
            )
        if kind == "text" and not is_text_mime(mime):
            mime = "text/plain"
        note = str(form.get("text") or "")

        prefix = {"video": "user_video", "audio": "user_audio", "image": "user_image"}.get(kind, "user_file")
        url = media.save_bytes(uuid, data, mime, prefix=prefix, name=name)

        blocks: list[dict[str, Any]] = []
        if kind == "image" or mime.startswith("image/"):
            block: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
            size = media.probe_size(data)
            if size:
                block["width"], block["height"] = size
        else:
            # Displayed in the UI and sampled into frames at request time.
            block = {"type": "_file", "kind": kind, "name": name, "mime": mime, "url": url}
            if kind == "text":
                block.update(text_facts(data))
        blocks.append(block)
        if note:
            blocks.append({"type": "text", "text": note})

        message = {"role": "user", "content": blocks}
        conv = await store.append(uuid, message)
        return ok({
            "uuid": uuid,
            "message": message,
            "url": url,
            "message_count": len(conv.messages),
        })

    async def truncate_conversation(request: Request) -> Response:
        """Drop trailing messages. Used by "edit and resend" and by regenerate."""
        uuid = request.path_params["uuid"]
        body = await read_json(request)
        keep = body.get("keep")
        drop_tail = int(body.get("drop_tail") or 0)

        conv = store.load_or_none(uuid)
        if conv is None:
            return fail("no such conversation", 404)

        if keep is None:
            # A raw count, so a caller asking for one fewer message than it means
            # to can slice a tool group in half. `store.truncate` snaps the index
            # back to a group boundary; a caller that wants to keep *up to* a turn
            # should send `keep` rather than a tail count, which cannot express
            # "the boundary before this group".
            keep = max(0, len(conv.messages) - drop_tail)
        keep = int(keep)

        conv = await store.truncate(uuid, keep)
        return ok({"uuid": uuid, "message_count": len(conv.messages)})

    async def replace_messages(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        body = await read_json(request)
        messages = body.get("messages")
        if not isinstance(messages, list):
            return fail("messages must be a list")

        # The client supplies this list whole, so it is the one write door where the
        # tool-call invariant can be broken from outside. Repair rather than reject:
        # a history edit is exactly the moment a half-finished group is visible to
        # the user, and refusing the edit would leave them unable to fix it.
        conv = await store.replace_messages(
            uuid, ensure_tool_pairing(messages, context=f"replace={uuid}"),
        )
        return ok({"uuid": uuid, "message_count": len(conv.messages)})

    async def conversation_context(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        conv = store.load_or_none(uuid)
        if conv is None:
            return fail("no such conversation", 404)
        return ok(engine.context_status(conv, model=conv.model or None))

    async def export_conversation(request: Request) -> Response:
        uuid = request.path_params["uuid"]
        conv = store.load_or_none(uuid)
        if conv is None:
            return fail("no such conversation", 404)
        return JSONResponse(
            {**convo_payload(conv), "format": "deepseek-ui/v1"},
            headers={"Content-Disposition": f'attachment; filename="{uuid}.json"'},
        )

    async def import_conversation(request: Request) -> Response:
        """Create a conversation from an exported file. A new id is always issued."""
        body = await read_json(request)
        payload = body.get("conversation") if isinstance(body.get("conversation"), dict) else body
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return fail("the import has no messages list")

        # An exported file can be hand-edited before it is re-imported, and one
        # exported *during* an interrupted tool loop carries the half-group with
        # it. Importing is the second of the two doors a broken transcript can
        # come through, so it gets the same repair as every outbound request.
        repaired = ensure_tool_pairing(
            [m for m in messages if isinstance(m, dict)], context="import",
        )

        conv = store.create(
            title=str(payload.get("title") or "Imported conversation"),
            model=str(payload.get("model") or settings.model or ""),
            sys_base=str(payload.get("sys_base") or payload.get("system_prompt") or ""),
        )

        def _fill(entry, source=payload, filled=repaired):
            if source.get("sys_todo"):
                entry.sys_todo = str(source["sys_todo"])
            entry.messages = list(filled)

        conv = await store.mutate(conv.uuid, _fill)
        return ok(convo_payload(conv))

    # ── chat ──

    async def chat(request: Request) -> Response:
        """Stream a turn as SSE."""
        uuid = request.path_params["uuid"]
        body = await read_json(request)

        if store.load_or_none(uuid) is None:
            return fail("no such conversation", 404)

        # An optional user turn, appended before the model is called so the two
        # cannot interleave with a concurrent append.
        incoming = body.get("content")
        if incoming is not None:
            try:
                content = media.ingest_user_content(uuid, incoming)
                await store.append(uuid, {"role": "user", "content": content})
            except Exception as exc:
                return fail(f"could not append the message: {exc}", 400)

        generator = engine.stream_turn(
            uuid,
            model=body.get("model") or None,
            system_override=body.get("system_prompt") if body.get("override_system") else None,
            reasoning_effort=body.get("reasoning_effort") or None,
            thinking=body.get("thinking") if isinstance(body.get("thinking"), bool) else None,
            tools_enabled=bool(body.get("tools", True)),
            max_steps=int(body["max_steps"]) if str(body.get("max_steps", "")).isdigit() else None,
            temperature=float(body["temperature"]) if isinstance(body.get("temperature"), (int, float)) else None,
        )

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                # Proxies that buffer would defeat streaming entirely.
                "X-Accel-Buffering": "no",
            },
        )

    # ── media tools, callable directly from the UI ──

    async def resize_media(request: Request) -> Response:
        """Resize an uploaded image or an existing file in a conversation."""
        try:
            form = await request.form()
        except Exception as exc:
            return fail(f"could not read the request: {exc}", 400)

        max_dim = int(str(form.get("max_dim") or settings.model_image_max_dim) or 1280)
        quality = int(str(form.get("quality") or 88))
        uuid = str(form.get("uuid") or "")
        target = str(form.get("url") or form.get("path") or "")

        if target:
            try:
                store.validate(uuid)
            except Exception as exc:
                return fail(str(exc), 400)
            source = media.resolve_in_conversation(uuid, target)
            if source is None:
                return fail("no such media", 404)
            data = source.read_bytes()
            name = source.stem
        else:
            upload = form.get("file") or form.get("media")
            if upload is None or not hasattr(upload, "read"):
                return fail("provide either a file upload or a url")
            data = await upload.read()
            name = Path(str(getattr(upload, "filename", "upload"))).stem

        before = media.probe_size(data)
        resized, mime = media.resize_image_bytes(data, max_dim, quality=quality)
        after = media.probe_size(resized)

        if not uuid:
            import base64

            return ok({
                "data_uri": f"data:{mime or 'image/jpeg'};base64,{base64.b64encode(resized).decode('ascii')}",
                "bytes": len(resized),
                "before": {"width": before[0], "height": before[1]} if before else None,
                "after": {"width": after[0], "height": after[1]} if after else None,
            })

        url = media.save_bytes(uuid, resized, mime or "image/jpeg", prefix=f"{name}_resized")
        return ok({
            "url": url,
            "bytes": len(resized),
            "before": {"width": before[0], "height": before[1]} if before else None,
            "after": {"width": after[0], "height": after[1]} if after else None,
            "image_block": {"type": "image_url", "image_url": {"url": url}},
        })

    async def video_frames(request: Request) -> Response:
        """Sample frames from a video so the UI can preview what the model will see."""
        body = await read_json(request)
        uuid = str(body.get("uuid") or "")
        url = str(body.get("url") or "")

        try:
            store.validate(uuid)
        except Exception as exc:
            return fail(str(exc), 400)

        source = media.resolve_in_conversation(uuid, url)
        if source is None:
            return fail("no such video", 404)

        frames = media.video_frames(
            source,
            target_fps=int(body.get("fps") or settings.model_video_fps),
            max_dim=int(body.get("max_dim") or settings.model_video_max_dim),
            max_frames=int(body.get("max_frames") or settings.model_video_max_frames),
        )
        urls = [
            media.save_bytes(uuid, frame, "image/jpeg", prefix=f"{source.stem}_preview_{i:03d}")
            for i, frame in enumerate(frames)
        ]
        return ok({"frames": urls, "count": len(urls)})

    # ── MCP ──
    #
    # The settings panel writes `mcp.json`, so these handlers carry two extra
    # obligations. First, never invent a file the user cannot read back: everything
    # goes through MCPDocument, which preserves the file's shape, its comments and
    # its top-level extras. Second, never leave the live connections out of step with
    # what is on disk, or the panel's status would be describing the previous save.

    def _mcp_document() -> MCPDocument:
        return read_mcp_document(settings.mcp_config_path)

    def _mcp_environ() -> dict[str, str]:
        """The environment `${VAR}` references resolve against: `.env`, then the OS."""
        try:
            return load_environ(settings.env_file)
        except Exception as exc:                        # pragma: no cover - defensive
            logger.warning("could not read %s: %s", settings.env_file, exc)
            return {}

    def _mcp_summary() -> dict[str, Any]:
        document = _mcp_document()
        return {
            "path": str(document.path) if document.path else "",
            "exists": document.exists,
            "error": document.error,
            "editable": not document.error,
        }

    def _mcp_entry_from_body(body: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        """Validate a submitted server definition. Returns the entry and a problem.

        Deliberately not ``MCPServerSpec.from_dict``: that helper drops a header still
        holding ``${...}``, which is right when *connecting* but wrong when *saving* —
        it would quietly delete a placeholder the user had just typed.
        """
        name = str(body.get("name") or "").strip()
        if not name:
            return {}, "name is required"
        url = str(body.get("url") or "").strip()
        if not url:
            return {}, "url is required"
        if not url.lower().startswith(("http://", "https://")):
            # Only Streamable HTTP is supported. Saying so here gives a message the
            # user can act on, instead of failing later with talk about transports.
            return {}, "url must start with http:// or https://"

        headers = body.get("headers") or {}
        if not isinstance(headers, Mapping):
            return {}, "headers must be an object of name/value pairs"

        allowed = body.get("allowedTools") or body.get("allowed_tools") or []
        if isinstance(allowed, str):
            allowed = [part.strip() for part in allowed.replace(",", "\n").splitlines() if part.strip()]

        try:
            timeout = float(body.get("timeout") or DEFAULT_CALL_TIMEOUT)
        except (TypeError, ValueError):
            return {}, "timeout must be a number of seconds"

        entry: dict[str, Any] = {"name": name, "url": url}
        if headers:
            entry["headers"] = {str(k): str(v) for k, v in headers.items()}
        if not bool(body.get("enabled", True)):
            entry["enabled"] = False
        prefix = str(body.get("prefix") or "").strip()
        if prefix:
            entry["prefix"] = prefix
        if allowed:
            entry["allowedTools"] = [str(t) for t in allowed]
        if timeout != DEFAULT_CALL_TIMEOUT:
            entry["timeout"] = timeout
        return entry, ""

    def _mcp_usable_spec(entry: Mapping[str, Any]) -> MCPServerSpec:
        """A spec as the client would build it: refs resolved, unset headers dropped."""
        try:
            resolved = expand_env_refs(copy.deepcopy(dict(entry)), _mcp_environ(), strict=False)
        except Exception:                               # pragma: no cover - defensive
            resolved = dict(entry)
        return MCPServerSpec.from_dict(resolved if isinstance(resolved, dict) else dict(entry))

    def _mcp_payload() -> dict[str, Any]:
        """Every configured server — disabled ones included — with its live status.

        Built from the file rather than from the manager, because the manager only
        knows about enabled servers and the panel still has to render a switch for the
        ones that are off. Header *values* are sent because the panel edits them, and
        in practice they are usually `${VAR}` references rather than secrets.
        """
        live = {row["name"]: row for row in (mcp.status() if mcp is not None else [])}
        document = _mcp_document()
        servers: list[dict[str, Any]] = []

        for index, entry in enumerate(document.entries):
            if not isinstance(entry, dict):
                continue
            spec = MCPServerSpec.from_dict(entry, index=index)
            row = live.get(spec.name, {
                "name": spec.name,
                "url": spec.url,
                "enabled": spec.enabled,
                "connected": False,
                "error": "" if spec.enabled else "",
                "prefix": spec.prefix or spec.slug,
                "timeout": spec.timeout,
                "tools": [],
                "log": [],
            })
            row.update({
                "name": spec.name,
                "url": entry.get("url") or entry.get("endpoint") or "",
                # From the file, not the connection: a disabled server has no
                # connection to ask, and the switch must show what is on disk.
                "enabled": spec.enabled,
                "headers": {str(k): str(v) for k, v in (entry.get("headers") or {}).items()}
                if isinstance(entry.get("headers"), Mapping) else {},
                # `prefix` is what the file says; `toolPrefix` is what the tools will
                # actually be called. They differ when the prefix is left empty and the
                # slug is used instead, and the card should show the real one.
                "prefix": str(entry.get("prefix") or ""),
                "toolPrefix": spec.prefix or spec.slug,
                "allowedTools": list(spec.allowed_tools),
            })
            servers.append(row)

        return {
            "servers": servers,
            "tools": state.tool_names(),
            "config": _mcp_summary(),
        }

    async def _mcp_commit(document: MCPDocument) -> Response:
        """Write the file, reconnect, and rebuild the tool registry.

        Reconnecting on every save rather than lazily keeps the panel honest — it
        reports what the servers actually do now, not what they did the last time
        something happened to ask.
        """
        try:
            document.save()
        except OSError as exc:
            logger.error("could not write %s: %s", document.path, exc)
            return fail(f"could not write {document.path}: {exc}", 500)

        if mcp is not None:
            await mcp.reload(document.specs(_mcp_environ()))
        if state.refresh_tools is not None:
            state.refresh_tools()
        return ok(_mcp_payload())

    def _mcp_unreadable(document: MCPDocument) -> Response | None:
        """A response when the config exists but cannot be parsed, else ``None``."""
        if not document.error:
            return None
        return fail(
            f"refusing to change {document.path}: it cannot be read ({document.error}). "
            "Repair or delete the file first.",
            409,
        )

    async def mcp_status(_: Request) -> Response:
        return ok(_mcp_payload())

    async def mcp_add_server(request: Request) -> Response:
        body = await read_json(request)
        document = _mcp_document()
        blocked = _mcp_unreadable(document)
        if blocked is not None:
            return blocked

        entry, problem = _mcp_entry_from_body(body)
        if problem:
            return fail(problem)
        if document.find(entry["name"]) is not None:
            return fail(f"a server called {entry['name']!r} already exists", 409)

        document.upsert(entry)
        return await _mcp_commit(document)

    async def mcp_update_server(request: Request) -> Response:
        name = request.path_params["name"]
        body = await read_json(request)
        document = _mcp_document()
        blocked = _mcp_unreadable(document)
        if blocked is not None:
            return blocked
        if document.find(name) is None:
            return fail(f"no server called {name!r}", 404)

        entry, problem = _mcp_entry_from_body({**body, "name": body.get("name") or name})
        if problem:
            return fail(problem)
        if entry["name"] != name and document.find(entry["name"]) is not None:
            return fail(f"a server called {entry['name']!r} already exists", 409)

        # Keyed by the old name, so a rename is a single edit rather than a delete
        # followed by an add.
        document.upsert(entry, replacing=name)
        return await _mcp_commit(document)

    async def mcp_set_enabled(request: Request) -> Response:
        """Toggle a server on or off without resubmitting the whole definition."""
        name = request.path_params["name"]
        body = await read_json(request)
        document = _mcp_document()
        blocked = _mcp_unreadable(document)
        if blocked is not None:
            return blocked

        entry = document.find(name)
        if entry is None:
            return fail(f"no server called {name!r}", 404)
        if "enabled" not in body:
            return fail("enabled is required")

        replaced = dict(entry)
        if bool(body["enabled"]):
            replaced.pop("enabled", None)
        else:
            replaced["enabled"] = False

        document.upsert(replaced, replacing=name)
        return await _mcp_commit(document)

    async def mcp_delete_server(request: Request) -> Response:
        name = request.path_params["name"]
        document = _mcp_document()
        blocked = _mcp_unreadable(document)
        if blocked is not None:
            return blocked
        if not document.remove(name):
            return fail(f"no server called {name!r}", 404)
        return await _mcp_commit(document)

    async def mcp_test(request: Request) -> Response:
        """Connect to one server and report what it offers, without saving anything.

        Accepts either a full definition in the body — so a URL can be tried before it
        is committed to the file — or just a name, which tests what is already stored.
        """
        body = await read_json(request)
        if str(body.get("url") or "").strip():
            entry: Mapping[str, Any] | None = body
        else:
            name = str(body.get("name") or request.query_params.get("name") or "")
            entry = _mcp_document().find(name)
            if entry is None:
                return fail(f"no server called {name!r}", 404)

        spec = _mcp_usable_spec(entry)
        if not spec.url:
            return fail("url is required")
        if not spec.url.lower().startswith(("http://", "https://")):
            return fail("url must start with http:// or https://")

        return ok(await probe_server(spec))

    async def mcp_reload(_: Request) -> Response:
        """Re-read the file from disk and reconnect.

        This is the path a hand-edited `mcp.json` takes: the pre-existing way of
        configuring MCP still works, and still only needs this one button.
        """
        if mcp is not None:
            await mcp.reload()
        if state.refresh_tools is not None:
            state.refresh_tools()
        return ok(_mcp_payload())

    return [
        Route("/", index),
        Route("/assets/{path:path}", assets),
        Route("/memory/{uuid}/{filename}", memory_file),
        Route("/download/{uuid}/{filename}", download_file),

        Route("/api/health", health),
        Route("/api/props", props),
        Route("/api/settings/api-key", settings_api_key, methods=["GET", "POST"]),
        Route("/api/settings/limits", settings_limits, methods=["GET", "POST"]),

        Route("/api/conversations", list_conversations, methods=["GET"]),
        Route("/api/conversations", create_conversation, methods=["POST"]),
        # Declared before the `{uuid}` patterns so it can never be read as an id.
        Route("/api/conversations/import", import_conversation, methods=["POST"]),
        Route("/api/conversations/{uuid}", get_conversation, methods=["GET"]),
        Route("/api/conversations/{uuid}", delete_conversation, methods=["DELETE"]),
        Route("/api/conversations/{uuid}/title", rename_conversation, methods=["POST"]),
        Route("/api/conversations/{uuid}/system", set_system_prompt, methods=["POST"]),
        Route("/api/conversations/{uuid}/model", set_model, methods=["POST"]),
        Route("/api/conversations/{uuid}/messages", append_message, methods=["POST"]),
        Route("/api/conversations/{uuid}/messages", replace_messages, methods=["PUT"]),
        Route("/api/conversations/{uuid}/upload", upload_media, methods=["POST"]),
        Route("/api/conversations/{uuid}/attach", upload_to_conversation, methods=["POST"]),
        Route("/api/conversations/{uuid}/truncate", truncate_conversation, methods=["POST"]),
        Route("/api/conversations/{uuid}/context", conversation_context, methods=["GET"]),
        Route("/api/conversations/{uuid}/export", export_conversation, methods=["GET"]),

        Route("/api/chat/{uuid}", chat, methods=["POST"]),

        Route("/api/media/resize", resize_media, methods=["POST"]),
        Route("/api/media/frames", video_frames, methods=["POST"]),

        Route("/api/mcp", mcp_status, methods=["GET"]),
        Route("/api/mcp/reload", mcp_reload, methods=["POST"]),
        Route("/api/mcp/test", mcp_test, methods=["POST"]),
        Route("/api/mcp/servers", mcp_add_server, methods=["POST"]),
        Route("/api/mcp/servers/{name}", mcp_update_server, methods=["PUT"]),
        Route("/api/mcp/servers/{name}", mcp_delete_server, methods=["DELETE"]),
        Route("/api/mcp/servers/{name}/enabled", mcp_set_enabled, methods=["POST"]),
    ]


def make_middleware(settings) -> list[Middleware]:
    """CORS for a Vite dev server.

    Only loopback origins are allowed, and only because the packaged app serves the
    frontend from the same origin and needs no CORS at all.
    """
    return [
        Middleware(
            CORSMiddleware,
            allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        ),
    ]
