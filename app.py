"""
Async relay server for llama.cpp chat UI  (Starlette + uvicorn + httpx)

Drop-in replacement for the Flask version.  Same endpoints, same client
contract — but built on ASGI so streamed SSE chunks are flushed to the
browser the instant they arrive from llama-server.

Responsibilities:
  - Serve index.html
  - Proxy /v1/chat/completions to llama-server (streaming, with timings)
  - Expose llama-server /props (model name, context size) via /server/props
  - Manage conversation storage under ./memory/<uuid>/
      conversation.json  — full message history
      <files>            — any media saved from tool results or user uploads
  - Strip base64 data from tool results AND user uploads before saving;
    replace with local URLs that get served from /memory/<uuid>/<filename>
  - Rehydrate stored media (images + video frames) back into the model's
    context window when sending to llama.cpp

Client contract:
  POST  /conv/load           { "uuid": "..." }
  POST  /conv/append_user    { "uuid": "...", "message": {...} }
  POST  /v1/chat/completions (streaming SSE proxy)
  GET   /memory/<uuid>/<filename>
  POST  /conv/new            { "uuid": "...", "name": "..." }
  DELETE /conv/<uuid>
  POST  /conv/tool_result    { "uuid": "...", ... }
  GET   /conv/list           — list all conversations (recovery from server)
  GET   /server/props        — llama.cpp /props passthrough (cached)
  Any other /v1/* → passthrough to llama-server

MCP Server
    POST /conv/edit_system_prompt    { "uuid": "...", "text": "...", "mode": "todo" }

Install:
  pip install starlette uvicorn httpx
  (optional, for video frame extraction) pip install opencv-python

Run:
  python app.py
  (or: uvicorn app:app --host 0.0.0.0 --port 5000)
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import time
import uuid as uuidlib
from pathlib import Path
import asyncio
from enum import Enum
import httpx

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")
model_logger = logging.getLogger("app.model")  # focused logger for model I/O
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

# ── Config ────────────────────────────────────────────────────────────────────
HF_HUB_CACHE = r"X:\huggingface\hub" # r"C:\Users\jared\.cache\huggingface\hub" C is faster ssd  # set to "" to skip / auto-detect
print("="*50)
print(f"\nUsing Hugging Face cache at: {HF_HUB_CACHE}\n")
print("="*50)
LLAMA_BASE  = "http://localhost:8080"
MCP_BASE    = "http://localhost:8585"   # MCP server address injected into index.html
MEMORY_ROOT = Path(__file__).parent / "memory"
MEMORY_ROOT.mkdir(exist_ok=True)

# Path to the batch file that starts llama.cpp server
LLAMA_BAT = Path(__file__).parent / "run_qwen35_8b_ablit_llama.bat"
#LLAMA_BAT = Path(__file__).parent / "run_qwen36_27b_heretic_llama.bat"
#LLAMA_BAT = Path(__file__).parent / "run_gemma4_31b_heretic_llama.bat"


# Shared async HTTP client — reused across requests (connection pooling)
http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

# In-process cache for /props (refreshed every 30s — model name/ctx don't change)
_props_cache = {"data": None, "ts": 0.0}
_PROPS_TTL = 30.0

# ── Windows asyncio Proactor workaround ──────────────────────────────────────
def _suppress_connection_reset(loop, context):
    """
    Suppress spurious ConnectionResetError (WinError 10054) that Windows'
    Proactor event loop raises when a client (e.g. browser) closes a streaming
    or range-request connection before the server finishes writing.  This is
    harmless — the client simply got what it needed and hung up.
    """
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return
    loop.default_exception_handler(context)


# ── llama.cpp subprocess management ───────────────────────────────────────────
_llama_process: subprocess.Popen | None = None
_llama_lock = asyncio.Lock()


def _cleanup_vram():
    """
    Attempt to free GPU VRAM after killing processes.
    Tries torch first (if available), then falls back to nvidia-smi.
    """
    # Try torch.cuda.empty_cache() if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared VRAM via torch.cuda.empty_cache()")
            return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"torch cleanup error: {e}")

    # Fallback: use nvidia-smi to reset GPU (more aggressive)
    try:
        # Just query to ensure driver is responsive after process kill
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            free_mb = result.stdout.strip().split('\n')
            logger.debug(f"GPU free memory: {free_mb} MB")
            return True
    except Exception as e:
        logger.debug(f"nvidia-smi query error: {e}")

    return False


def _kill_llama_process():
    """Kill the llama.cpp server process and any child processes."""
    global _llama_process
    if _llama_process is None:
        return False

    pid = _llama_process.pid
    logger.info(f"Killing llama process tree (PID {pid})...")

    try:
        # On Windows, use taskkill to kill the entire process tree
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, timeout=30
        )
    except Exception as e:
        logger.error(f"taskkill error: {e}")
        # Fallback: try to terminate directly
        try:
            _llama_process.terminate()
            _llama_process.wait(timeout=10)
        except Exception as e2:
            logger.error(f"terminate error: {e2}")
            try:
                _llama_process.kill()
            except:
                pass

    _llama_process = None
    logger.info("Llama process killed")
    return True


def _start_llama_process() -> bool:
    """Start the llama.cpp server using the batch file."""
    global _llama_process, HF_HUB_CACHE
    if HF_HUB_CACHE:
        logger.info(f"Using custom Hugging Face Hub cache directory: {HF_HUB_CACHE}")
        os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
        os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE   # older alias, set just to cover
        os.environ["HF_HOME"] = str(Path(HF_HUB_CACHE).parent)
    else:
        logger.info("Using default Hugging Face Hub cache directory")

    if _llama_process is not None:
        # Check if still running
        if _llama_process.poll() is None:
            logger.debug("Server already running")
            return True
        _llama_process = None

    if not LLAMA_BAT.exists():
        logger.error(f"Batch file not found: {LLAMA_BAT}")
        return False

    logger.info(f"Starting llama server via {LLAMA_BAT.name}...")
    try:
        # Start the batch file in a new console window (Windows)
        # CREATE_NEW_CONSOLE = 0x00000010
        _llama_process = subprocess.Popen(
            ["cmd", "/c", str(LLAMA_BAT)],
            cwd=str(LLAMA_BAT.parent),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            #stdout=subprocess.DEVNULL,
            #stderr=subprocess.DEVNULL,
        )
        logger.info(f"Started llama server with PID {_llama_process.pid}")
        return True
    except Exception as e:
        logger.error(f"Failed to start llama server: {e}")
        return False


async def _wait_for_llama_ready(timeout: float = 120.0) -> bool:
    """Wait for llama.cpp server to become responsive."""
    start = time.time()
    while (time.time() - start) < timeout:
        try:
            r = await http_client.get(f"{LLAMA_BASE}/health", timeout=5.0)
            if r.status_code == 200:
                logger.info("Llama server is ready")
                return True
        except Exception:
            pass
        await asyncio.sleep(2.0)
    logger.warning(f"Llama server not ready after {timeout}s")
    return False


async def _is_llama_running() -> bool:
    """Check if llama.cpp server is responsive."""
    try:
        r = await http_client.get(f"{LLAMA_BASE}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


# ── Conversation file helpers ─────────────────────────────────────────────────

def conv_dir(uuid: str) -> Path:
    return MEMORY_ROOT / uuid

def conv_file(uuid: str) -> Path:
    return conv_dir(uuid) / "conversation.json"

def load_conv(uuid: str) -> dict | None:
    p = conv_file(uuid)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def save_conv(uuid: str, data: dict):
    # CHANGED: write atomically (temp file + os.replace) so an overlapping write
    # or a crash can never leave a half-written conversation.json on disk.
    p = conv_file(uuid)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)  # atomic on the same filesystem


# ── NEW: Per-conversation async lock ─────────────────────────────────────────
# Every mutation of conversation.json (append user / tool / assistant message,
# edit system prompt, summarize) MUST hold this lock for the given uuid. Without
# it, concurrent async tasks each did load_conv -> mutate -> save_conv with
# `await` points in between, lost-updating one another and silently dropping
# messages — the root cause of "the model doesn't have the full history".
_conv_locks: dict[str, asyncio.Lock] = {}

def _conv_lock(uuid: str) -> asyncio.Lock:
    lock = _conv_locks.get(uuid)
    if lock is None:
        lock = asyncio.Lock()
        _conv_locks[uuid] = lock
    return lock

async def _append_message(uuid: str, msg: dict) -> bool:
    """Atomically append one message to a conversation under its per-uuid lock."""
    async with _conv_lock(uuid):
        data = load_conv(uuid)
        if data is None:
            return False
        data["messages"].append(msg)
        save_conv(uuid, data)
    return True


# ── System-prompt / todo composition ────────────────────────────────────
# The todo tool syncs the task tracker into the system prompt. To stop the
# prompt growing on every call, the base prompt and the current todo block are
# stored SEPARATELY on the conversation (data["sys_base"] / data["sys_todo"]),
# and messages[0] is ALWAYS rebuilt from those two fields. We never read the
# existing (possibly bloated) system message back as input, so updates are
# idempotent and any already-inflated conversation self-heals on the next sync.


def _compose_system(base: str, todo: str) -> str:
    todo_begin = "\n\n===== YOUR CURRENT TASK TRACKER (use your todo tool to manage tasks) =====\n"
    todo_end   = "\n===== END TASK TRACKER ====="
    base = (base or "").rstrip()
    todo = (todo or "").strip()
    return f"{base}{todo_begin}{todo}{todo_end}" if todo else base


# ── Truncation helper for large data (images, base64, etc.) ──────────────────

def _truncate_data(data, prefix_len=50, suffix_len=50):
    """Return a truncated representation of data showing first N and last N chars/bytes."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", errors="replace")
    if len(data) <= prefix_len + suffix_len:
        return data
    return f"{data[:prefix_len]} ... ({len(data)} chars) ... {data[-suffix_len:]}"


# ── Media helpers ─────────────────────────────────────────────────────────────

def _extract_video_frames(path: Path, target_fps: int = 3, max_dim: int = 640) -> list[str]:
    """
    Extract frames from a video at target_fps, resized so the longest dimension
    is at most max_dim (for model context — does NOT affect the saved file).
    """
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return []
    duration_s = total / source_fps
    n = max(1, round(duration_s * target_fps))
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(w, h) > max_dim:
            if w >= h:
                new_w, new_h = max_dim, round(h * max_dim / w)
            else:
                new_w, new_h = round(w * max_dim / h), max_dim
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(base64.b64encode(buf.tobytes()).decode())
    cap.release()
    return frames


def _resize_b64_image(b64_data: str, mime: str, max_dim: int = 640) -> str:
    """
    Resize a base64-encoded image so its longest dimension is at most max_dim,
    preserving aspect ratio.  Returns original data unchanged if Pillow is not
    installed or the image is already small enough.
    Only used when building model context — never touches saved files or the UI.
    """
    try:
        from PIL import Image
        import io as _io
    except ImportError:
        return b64_data
    raw = base64.b64decode(b64_data)
    img = Image.open(_io.BytesIO(raw))
    w, h = img.size
    if max(w, h) <= max_dim:
        return b64_data
    if w >= h:
        new_w, new_h = max_dim, round(h * max_dim / w)
    else:
        new_w, new_h = round(w * max_dim / h), max_dim
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = _io.BytesIO()
    fmt = "JPEG" if mime in ("image/jpeg", "image/jpg", "image/pjpeg") else "PNG"
    save_kwargs = {"quality": 85} if fmt == "JPEG" else {}
    img.save(buf, format=fmt, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode()


def ext_for_mime(mime: str) -> str:
    """Return a file extension for a given MIME type."""
    guessed = mimetypes.guess_extension(mime)
    if guessed:
        return guessed.replace(".jpe", ".jpg")
    mime_to_ext = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
        "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/mp4": ".m4a",
        "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
    }
    return mime_to_ext.get(mime, ".bin")


def save_b64_file(uuid: str, b64_data: str, mime: str, prefix: str = "file") -> str:
    d = conv_dir(uuid)
    d.mkdir(parents=True, exist_ok=True)
    ext = ext_for_mime(mime)
    filename = f"{prefix}_{int(time.time()*1000)}{ext}"
    (d / filename).write_bytes(base64.b64decode(b64_data))
    return f"/memory/{uuid}/{filename}"


def process_tool_result_blocks(uuid: str, blocks: list, tool_name: str) -> list:
    """Strip base64 from tool result media; save to disk and replace with URL."""
    out = []
    for i, block in enumerate(blocks):
        b = dict(block)
        if b.get("data") and b.get("type") in ("image", "audio", "video"):
            mime = b.get("mimeType", "application/octet-stream")
            url  = save_b64_file(uuid, b["data"], mime, prefix=f"{tool_name}_{i}")
            del b["data"]
            b["url"] = url
        out.append(b)
    return out


def process_user_content(uuid: str, content) -> list | str:
    """
    Process a user message's content:
      - image_url blocks with base64 data URIs → save file, swap URL
      - _file blocks (video/audio uploads) with embedded b64 data:
          → save to disk, strip the b64, keep a 'url' field for replay
    Returns the rewritten content (no base64 ever lands in conversation.json).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        b = dict(block)

        # User-attached images (vision)
        if b.get("type") == "image_url":
            url = b.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                if m:
                    mime, b64 = m.group(1), m.group(2)
                    saved = save_b64_file(uuid, b64, mime, prefix="user_img")
                    b["image_url"] = {"url": saved}
                    local_path = str((MEMORY_ROOT / saved[len("/memory/"):]).resolve())
                    out.append(b)
                    out.append({"type": "text", "text": f"[Uploaded image saved at local path: {local_path}]"})
                    continue

        # User-attached video / audio (custom display-only blocks)
        elif b.get("type") == "_file":
            # Client sends {type:'_file', kind:'video'|'audio', name, mime, b64}
            b64  = b.pop("b64", None)
            mime = b.get("mime", "application/octet-stream")
            kind = b.get("kind", "file")
            if b64:
                saved = save_b64_file(uuid, b64, mime, prefix=f"user_{kind}")
                b["url"] = saved
                local_path = str((MEMORY_ROOT / saved[len("/memory/"):]).resolve())
                out.append(b)
                out.append({"type": "text", "text": f"[Uploaded {kind} saved at local path: {local_path}]"})
                continue

        out.append(b)
    return out


# ── SSE accumulation (server-side, for saving to memory) ─────────────────────

def _accumulate(line: str, assembled: dict):
    """Parse one SSE line and accumulate delta into assembled dict."""
    if not line.startswith("data: "):
        return
    raw = line[6:].strip()
    if raw == "[DONE]":
        return
    try:
        chunk = json.loads(raw)
    except json.JSONDecodeError:
        return

    # Stash the latest timings + usage so we can persist them on the assistant msg
    if "timings" in chunk:
        assembled["timings"] = chunk["timings"]
    if "usage" in chunk and chunk["usage"]:
        assembled["usage"] = chunk["usage"]

    choices = chunk.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta", {})
    if not delta:
        return

    if delta.get("reasoning_content"):
        assembled["reasoning"] = (assembled.get("reasoning") or "") + delta["reasoning_content"]

    if delta.get("content"):
        assembled["content"] = (assembled.get("content") or "") + delta["content"]

    for dtc in delta.get("tool_calls", []):
        idx = dtc.get("index", 0)
        while len(assembled["tool_calls"]) <= idx:
            assembled["tool_calls"].append({"id": "", "type": "function",
                                             "function": {"name": "", "arguments": ""}})
        tc = assembled["tool_calls"][idx]
        if dtc.get("id"):
            tc["id"] = dtc["id"]
        if dtc.get("function", {}).get("name"):
            tc["function"]["name"] += dtc["function"]["name"]
        if dtc.get("function", {}).get("arguments"):
            tc["function"]["arguments"] += dtc["function"]["arguments"]


# CHANGED: now async so it can take the per-uuid lock via _append_message,
# preventing this assistant-message save from racing a concurrent tool-result
# save or a background summarize (which previously clobbered each other).
async def _finalise_and_save(uuid: str, assembled: dict):
    """Clean up assembled message and append to conversation.json."""
    msg = {"role": "assistant"}
    if assembled.get("content"):
        msg["content"] = assembled["content"]
    if assembled.get("reasoning"):
        msg["reasoning"] = assembled["reasoning"]
    if assembled.get("tool_calls"):
        valid_tcs = []
        for tc in assembled["tool_calls"]:
            args = (tc.get("function") or {}).get("arguments", "")
            try:
                if args:
                    json.loads(args)
                valid_tcs.append(tc)
            except json.JSONDecodeError:
                name = (tc.get("function") or {}).get("name", "?")
                logger.warning(
                    f"Conv {uuid[:8]}: dropping tool_call '{name}' — "
                    f"malformed JSON args (truncated stream?): {args[:80]!r}"
                )
        if valid_tcs:
            msg["tool_calls"] = valid_tcs
    # Persist perf for this turn — surfaced in UI on reload
    if assembled.get("timings"):
        msg["_timings"] = assembled["timings"]
    if assembled.get("usage"):
        msg["_usage"] = assembled["usage"]

    # Don't save empty assistant messages — llama.cpp rejects them with 400
    if not msg.get("content") and not msg.get("tool_calls"):
        logger.warning(f"Conv {uuid[:8]}: skipping empty assistant message (no content or tool_calls)")
        return

    await _append_message(uuid, msg)  # CHANGED: atomic, locked append


# ── Route handlers ────────────────────────────────────────────────────────────

async def index(request: Request):
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    # Inject the server-configured MCP address into the input field's default value
    logger.debug(f"Injecting MCP address into index.html: {MCP_BASE}")
    #TODO: this is a hacky way of injecting, was done as a quick fix. should just use proper templating at some point.
    html = html.replace(
        'id="mcp-url" type="text" placeholder="http://localhost:8585" value="http://localhost:8585"',
        f'id="mcp-url" type="text" placeholder="{MCP_BASE}" value="{MCP_BASE}"',
    )
    return HTMLResponse(html)


async def serve_media(request: Request):
    uuid = request.path_params["uuid"]
    filename = request.path_params["filename"]
    p = conv_dir(uuid) / filename
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p)


async def download_media(request: Request):
    uuid = request.path_params["uuid"]
    filename = request.path_params["filename"]
    p = conv_dir(uuid) / filename
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p, filename=filename, headers={
        "Content-Disposition": f'attachment; filename="{filename}"'
    })


async def conv_new(request: Request):
    body = await request.json()
    uid  = body.get("uuid") or str(uuidlib.uuid4())
    name = body.get("name", "New conversation")
    d = conv_dir(uid)
    d.mkdir(parents=True, exist_ok=True)
    data = {"uuid": uid, "name": name, "created": time.time(), "messages": []}
    save_conv(uid, data)
    return JSONResponse({"ok": True, "uuid": uid})


async def conv_load(request: Request):
    body = await request.json()
    uid  = body.get("uuid", "")
    data = load_conv(uid)
    if data is None:
        return Response(status_code=404)
    return JSONResponse(data)


async def conv_list(request: Request):
    """List all stored conversations (lightweight: uuid + name + created + msg count)."""
    out = []
    for d in sorted(MEMORY_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        cf = d / "conversation.json"
        if not cf.exists():
            continue
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
            out.append({
                "uuid": data.get("uuid", d.name),
                "name": data.get("name", "Untitled"),
                "created": data.get("created", d.stat().st_mtime),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    return JSONResponse({"conversations": out})


async def conv_append_user(request: Request):
    body = await request.json()
    uid  = body["uuid"]
    msg  = body["message"]

    # CHANGED: the whole load -> process -> append -> auto-name -> save sequence
    # now runs inside the per-uuid lock as one critical section, so it can't
    # lost-update against a tool-result save or a background summarize.
    async with _conv_lock(uid):
        data = load_conv(uid)
        if data is None:
            return Response(status_code=404)

        # Save any base64 media to disk BEFORE writing the message into history
        msg["content"] = process_user_content(uid, msg["content"])
        data["messages"].append(msg)

        # Auto-name the conversation from the first user message
        user_msgs = [m for m in data["messages"] if m["role"] == "user"]
        if len(user_msgs) == 1:
            if isinstance(msg["content"], str):
                txt = msg["content"]
            else:
                txt = " ".join(b.get("text", "") for b in msg["content"]
                               if isinstance(b, dict) and b.get("type") == "text")
            data["name"] = (txt[:50] or "Conversation").strip()

        save_conv(uid, data)

    return JSONResponse({
        "ok": True,
        "name": data["name"],
        "message": msg,   # echo back, so client sees the rewritten URLs
    })


async def conv_delete(request: Request):
    uid = request.path_params["uid"]
    d = conv_dir(uid)
    if d.exists():
        shutil.rmtree(d)
    return JSONResponse({"ok": True})


async def conv_edit_system_prompt(request: Request):
    """
    Maintain a conversation's system prompt as TWO stored parts so repeated
    todo syncs can never inflate it:
      data["sys_base"] — base/original system prompt text
      data["sys_todo"] — current task-tracker block (REPLACED, never appended)

    messages[0] is always rebuilt from these via _compose_system(), and we never
    read the existing (possibly bloated) system message back as input — so any
    conversation already inflated by the old append path self-heals on the next
    call.

    Payload:
      {
        "uuid": "...",
        "text": "...",
        "mode": "todo" | "replace" | "append"   (default "append")
      }
      - todo:    set sys_todo = text   (used by the mcp todo-tool sync)
      - replace: set sys_base = text
      - append:  sys_base += "\n\n" + text   (general, NON-todo use only)
    """
    body = await request.json()
    uid  = body.get("uuid", "")
    text = body.get("text", "").strip()
    mode = body.get("mode", "append")  # "todo" | "replace" | "append"

    # read-modify-write under the per-uuid lock (this endpoint is
    # called by the mcp server's todo tool and can race the chat completion's
    # saves / the summarizer).
    async with _conv_lock(uid):
        data = load_conv(uid)
        if data is None:
            return JSONResponse({"ok": False, "error": "Conversation not found"}, status_code=404)

        sys_base = data.get("sys_base", "")
        sys_todo = data.get("sys_todo", "")

        if mode == "todo":
            sys_todo = text                                   
        elif mode == "replace":
            sys_base = text
        else:  # append — general use only, NOT the todo path
            sys_base = f"{sys_base}\n\n{text}".strip() if sys_base else text

        data["sys_base"] = sys_base
        data["sys_todo"] = sys_todo
        new_content = _compose_system(sys_base, sys_todo)

        
        # No longer mirror into messages[0] — the todo block is injected
        # server-side in chat_completions. Strip any system message a previous
        # version of this endpoint already wrote, so old conversations
        # self-heal instead of sending a stale duplicate forever.
        messages = data.get("messages", [])
        data["messages"] = [m for m in messages if m.get("role") != "system"]
        save_conv(uid, data)

    print(f"Current system prompt for conversation {uid}:\n\n {new_content}\n\n")
    return JSONResponse({
        "ok": True,
        "action": "sorted",
        "mode": mode,
        "system_prompt": new_content,
    })


async def conv_tool_result(request: Request):
    body     = await request.json()
    uuid     = body["uuid"]
    tool_id  = body.get("tool_call_id", "")
    name     = body.get("tool_name", "tool")
    blocks   = body.get("blocks", [])

    data = load_conv(uuid)
    if data is None:
        return Response(status_code=404)

    clean_blocks = process_tool_result_blocks(uuid, blocks, name)

    tool_msg = {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps(clean_blocks),
    }
    # CHANGED: atomic, locked append (was an unlocked load->append->save that
    # raced the assistant-message save and the summarizer).
    if not await _append_message(uuid, tool_msg):
        return Response(status_code=404)

    # Trigger background summarization if history is getting long
    _fire_summarize(uuid)

    return JSONResponse({"ok": True, "blocks": clean_blocks, "tool_message": tool_msg})


# ── llama.cpp server info (cached) ────────────────────────────────────────────

async def server_props(request: Request):
    """
    Return llama-server /props + a flattened, easy-to-display summary.
    Cached for _PROPS_TTL seconds because model & ctx never change at runtime.
    """
    now = time.time()
    if _props_cache["data"] is None or (now - _props_cache["ts"]) > _PROPS_TTL:
        try:
            r = await http_client.get(f"{LLAMA_BASE}/props", timeout=10.0)
            r.raise_for_status()
            props = r.json()
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": str(e), "summary": {}},
                status_code=200,
            )

        # Field names changed across llama.cpp versions — probe a few locations.
        dgs = props.get("default_generation_settings", {}) or {}
        model_info  = (props.get("model_path")
                       or dgs.get("model")
                       or props.get("model_alias")
                       or "unknown")
        model_short = Path(str(model_info)).name if model_info else "unknown"
        n_ctx       = (props.get("n_ctx")
                       or dgs.get("n_ctx")
                       or 0)
        chat_template = props.get("chat_template", "")

        summary = {
            "model": model_short,
            "model_full": str(model_info),
            "n_ctx": int(n_ctx) if n_ctx else 0,
            "chat_template_present": bool(chat_template),
            "total_slots": props.get("total_slots", 1),
        }
        _props_cache["data"] = {"ok": True, "props": props, "summary": summary}
        _props_cache["ts"]   = now

    return JSONResponse(_props_cache["data"])

# ── VRAM lifecycle state ───────────────────────────────────────────────────


class LLMState(Enum):
    UNLOADED    = "unloaded"
    LOADING     = "loading"
    LOADED      = "loaded"
    SUMMARIZING = "summarizing"

_llm_state = LLMState.UNLOADED
_summary_pending: set[str] = set()   # uuids needing summary when llm next loads
_summary_done_event   = asyncio.Event()
_summary_done_event.set()            # starts in "not summarizing" state


def _set_llm_state(state: LLMState):
    global _llm_state
    _llm_state = state
    logger.info(f"LLM state → {state.value}")


async def _wait_for_summary_complete(timeout: float = 120.0) -> bool:
    """Block until no summary is running. Returns False if timeout."""
    try:
        await asyncio.wait_for(_summary_done_event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for summary to complete")
        return False
    
# ── Server lifecycle control (for VRAM-intensive tools) ───────────────────────
async def server_stop(request: Request):
    async with _llama_lock:
        # Wait for any in-progress summary before unloading
        if _llm_state == LLMState.SUMMARIZING:
            logger.info("Waiting for summary to finish before unloading LLM...")
            await _wait_for_summary_complete(timeout=120.0)

        was_running = await _is_llama_running()
        if was_running or _llama_process is not None:
            _set_llm_state(LLMState.UNLOADED)
            _kill_llama_process()
            await asyncio.sleep(1.0)
            _cleanup_vram()
            _props_cache["data"] = None
            return JSONResponse({
                "ok": True,
                "message": "Server stopped and VRAM cleared",
                "was_running": was_running,
            })
        _set_llm_state(LLMState.UNLOADED)
        return JSONResponse({
            "ok": True, 
            "message": "Server was not running",
            "was_running": False,
        })

async def server_start(request: Request):
    async with _llama_lock:
        if await _is_llama_running():
            _set_llm_state(LLMState.LOADED)
            return JSONResponse({
                "ok": True,
                "message": "Server already running",
                "started": False,
            })

        _set_llm_state(LLMState.LOADING)
        started = _start_llama_process()
        if not started:
            _set_llm_state(LLMState.UNLOADED)
            return JSONResponse({
                "ok": False, "error": "Failed to start server process"
            }, status_code=500)

        ready = await _wait_for_llama_ready(timeout=300.0)
        if not ready:
            _set_llm_state(LLMState.UNLOADED)
            return JSONResponse({
                "ok": False, "error": "Server started but not responding in time"
            }, status_code=500)

        _set_llm_state(LLMState.LOADED)

        # ── Run any pending summaries NOW before the model takes new work ──
        if _summary_pending:
            pending = set(_summary_pending)  # snapshot
            logger.info(f"LLM reloaded — running {len(pending)} pending summaries...")
            # Don't await here — server_start needs to return so mcp_server.py
            # gets its "ok" response. Schedule as a task; chat_completions will
            # wait on _summary_done_event before proceeding.
            asyncio.create_task(_run_pending_summaries(pending))

        return JSONResponse({
            "ok": True,
            "message": "Server started and ready",
            "started": True,
        })
    
async def server_status(request: Request):
    """Check if the llama.cpp server is running and responsive."""
    running = await _is_llama_running()
    process_alive = _llama_process is not None and _llama_process.poll() is None
    return JSONResponse({
        "ok": True,
        "running": running,
        "process_alive": process_alive,
        "process_pid": _llama_process.pid if _llama_process else None,
    })

async def _run_pending_summaries(uuids: set[str]):
    """Run all pending summaries. Blocks chat_completions via _summary_done_event."""
    _summary_done_event.clear()
    _set_llm_state(LLMState.SUMMARIZING)
    try:
        for uuid in uuids:
            _summary_pending.discard(uuid)
            await _maybe_summarize(uuid, force=True)
    finally:
        _set_llm_state(LLMState.LOADED)
        _summary_done_event.set()
        logger.info("All pending summaries complete.")

# ── Context management config ──────────────────────────────────────────────
# Fraction of n_ctx to use as the trigger threshold (0.65 = summarize at 65% full)
_SUMMARY_THRESHOLD   = 0.65
# How many recent messages to always preserve intact (keep the active tool chain visible)
_PRESERVE_TAIL       = 6
# Rough chars-per-token estimate for budget calculations
_CHARS_PER_TOKEN     = 3.5

# Track in-progress summaries so we don't double-trigger
_summarizing: set[str] = set()


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate from raw message content (no base64 — already stripped)."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c) / _CHARS_PER_TOKEN
        elif isinstance(c, list):
            for block in c:
                total += len(block.get("text", "")) / _CHARS_PER_TOKEN
    logger.debug(f"Estimated tokens in conversation history: {int(total)}")
    return int(total)


# NEW: best-effort REAL context window from llama.cpp props. The old code used a
# hard-coded 40000 fallback, so with a cold props cache it summarized at
# 0.65*40000≈26k tokens even when the model's true window was far larger — the
# "history truncated well below max context" symptom. We now refuse to summarize
# on a guessed size (unless explicitly forced).
async def _get_n_ctx() -> int:
    """Return the real n_ctx from llama.cpp props, or 0 if unknown."""
    if _props_cache["data"]:
        n = _props_cache["data"].get("summary", {}).get("n_ctx", 0) or 0
        if n:
            return int(n)
    try:
        r = await http_client.get(f"{LLAMA_BASE}/props", timeout=10.0)
        r.raise_for_status()
        props = r.json()
        dgs = props.get("default_generation_settings", {}) or {}
        return int(props.get("n_ctx") or dgs.get("n_ctx") or 0)
    except Exception:
        return 0


async def _summarize_segment(messages: list) -> str | None:
    """
    Ask llama.cpp to summarize a segment of conversation history.
    Returns the summary string, or None on failure.
    """
    # Build a clean text representation — no base64, no tool schemas
    logger.debug(f"Summarizing segment with {len(messages)} messages")
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        
        if isinstance(content, str):
            # Tool messages are JSON-encoded block lists
            try:
                blocks = json.loads(content)
                if isinstance(blocks, list):
                    text_only = " ".join(
                        b.get("text", "") for b in blocks 
                        if b.get("type") == "text"
                    )
                    content = text_only or content
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(content, list):
            content = " ".join(
                b.get("text", "") or b.get("image_url", {}).get("url", "")[:50]
                for b in content if isinstance(b, dict)
            )
        
        if content.strip():
            lines.append(f"{role.upper()}: {content.strip()[:500]}")  # cap per-message

    segment_text = "\n".join(lines)
    
    prompt = (
        "Summarize the following conversation segment concisely. "
        "Focus on: what was requested, what tools were called, what was produced (file paths, descriptions), "
        "and any decisions made. Preserve all file paths exactly. Be dense — this summary replaces the original.\n\n"
        f"{segment_text}\n\nSUMMARY:"
    )
    
    try:
        r = await http_client.post(
            f"{LLAMA_BASE}/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "stream": False,
                "temperature": 0.1,   # deterministic — this is compression not creativity
            },
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Failed to summarize segment: {e}")
        return None


async def _maybe_summarize(uuid: str, force: bool = False):
    """
    Check if conversation history is getting long; if so, summarize the middle.
    Called after each tool result is saved (fire-and-forget) and on reload.

    CHANGED — race-safety + correct context size:
      * Uses the REAL n_ctx from llama.cpp props; refuses to summarize on a
        guessed size unless explicitly forced (kills premature truncation).
      * The summarize decision + segment capture happen UNDER the lock with NO
        network call held.
      * The summarization request runs OUTSIDE the lock.
      * The splice re-reads the conversation FRESH under the lock and applies the
        summary positionally, so any messages appended during the (slow) network
        call survive instead of being clobbered by a stale snapshot.
    """
    if uuid in _summarizing:
        return

    n_ctx = await _get_n_ctx()
    if n_ctx <= 0:
        # Unknown real context size — do NOT summarize on a guess.
        if not force:
            logger.debug(f"Conv {uuid[:8]}: n_ctx unknown, skipping summary")
            return
        n_ctx = 40000  # only fall back for an explicitly forced pass
    threshold_tokens = int(n_ctx * _SUMMARY_THRESHOLD)

    # ── Decide + capture the segment UNDER LOCK (no network here) ──
    async with _conv_lock(uuid):
        data = load_conv(uuid)
        if data is None:
            return
        messages = data.get("messages", [])
        if len(messages) < 8:
            return

        estimated = _estimate_tokens(messages)
        if estimated < threshold_tokens and not force:
            logger.debug(f"Conversation {uuid[:8]} has ~{estimated} tokens, below threshold "
                         f"of {threshold_tokens}. No summary needed.")
            return

        # If the LLM isn't loaded (e.g. unloaded for a VRAM-heavy tool), enqueue
        # this uuid so server_start's _run_pending_summaries handles it later.
        if _llm_state != LLMState.LOADED and not force:
            logger.info(f"LLM not available, queuing summary for {uuid[:8]}")
            _summary_pending.add(uuid)
            return

        has_system     = messages[0].get("role") == "system"
        protected_head = 1 if has_system else 0
        if len(messages) - protected_head <= _PRESERVE_TAIL:
            return  # nothing to compress

        original_len = len(messages)
        middle_end   = original_len - _PRESERVE_TAIL          # exclusive
        real_middle  = messages[protected_head:middle_end]
        if not real_middle:
            return

        # Fold an existing summary node into this pass to keep it rolling
        seg_for_summary  = list(real_middle)
        existing_summary = ""
        if seg_for_summary and seg_for_summary[0].get("_is_summary"):
            existing_summary = seg_for_summary[0].get("content", "")
            seg_for_summary  = seg_for_summary[1:]
        if existing_summary:
            seg_for_summary.insert(0, {
                "role": "user",
                "content": f"[Previous summary: {existing_summary}]",
            })

        _summarizing.add(uuid)

    # ── Network summarization OUTSIDE the lock ──
    try:
        summary_text = await _summarize_segment(seg_for_summary)
        if not summary_text:
            logger.warning(f"Summarization failed for {uuid[:8]}, keeping original")
            return

        summary_msg = {
            "role": "user",   # neutral role llama.cpp won't choke on
            "content": f"[CONVERSATION SUMMARY — replaces earlier history]\n{summary_text}",
            "_is_summary": True,   # marker so next pass can fold it in
        }

        # ── Re-read FRESH and splice UNDER LOCK ──
        async with _conv_lock(uuid):
            data = load_conv(uuid)
            if data is None:
                return
            fresh = data.get("messages", [])

            # Only splice if the head/middle region is structurally unchanged.
            # Pure appends (suffix growth) during the network call are safe and
            # preserved by slicing fresh[middle_end:]. A concurrent summarize, a
            # deletion, or a newly-inserted system message → skip this pass.
            fresh_has_system = bool(fresh) and fresh[0].get("role") == "system"
            if (fresh_has_system != has_system
                    or len(fresh) < original_len
                    or middle_end > len(fresh)):
                logger.info(f"Conv {uuid[:8]}: history shifted during summary, skipping splice")
                return

            new_messages = fresh[:protected_head] + [summary_msg] + fresh[middle_end:]
            data["messages"] = new_messages
            save_conv(uuid, data)
            logger.info(f"Conv {uuid[:8]}: compressed {middle_end - protected_head} messages "
                        f"→ 1 summary node. History {original_len} → {len(new_messages)} messages.")
    finally:
        _summarizing.discard(uuid)


def _fire_summarize(uuid: str):
    """Fire-and-forget summarization from a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_maybe_summarize(uuid))
    except Exception as e:
        logger.error(f"Could not schedule summarization: {e}")


# ── Tool-result rehydration ───────────────────────────────────────────────────

def _rehydrate_user_msg(msg: dict) -> dict:
    """
    Convert stored user message back into something llama.cpp can ingest:
      - image_url with /memory/... → inline base64 data URI
      - _file (video) → frame-extract and inline as image_url blocks
      - _file (audio) → text placeholder (vision model can't hear)
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    new_content = []
    for block in content:
        btype = block.get("type", "")

        if btype == "text":
            new_content.append({"type": "text", "text": block.get("text", "")})

        elif btype == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("/memory/"):
                fp = MEMORY_ROOT / url[len("/memory/"):]
                if fp.exists():
                    mime, _ = mimetypes.guess_type(str(fp))
                    mime = mime or "image/jpeg"
                    b64 = base64.b64encode(fp.read_bytes()).decode()
                    b64 = _resize_b64_image(b64, mime)  # reduce for model context
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                else:
                    new_content.append({"type": "text",
                                        "text": f"[Image missing: {url}]"})
            elif url.startswith("data:"):
                # already a data URI — resize for model context
                import re as _re
                m = _re.match(r"data:([^;]+);base64,(.+)", url, _re.DOTALL)
                if m:
                    mime, b64 = m.group(1), m.group(2)
                    b64 = _resize_b64_image(b64, mime)
                    new_content.append({"type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
                else:
                    new_content.append(block)
            else:
                # external URL — pass through
                new_content.append(block)

        elif btype == "_file":
            kind = block.get("kind", "")
            url  = block.get("url", "")
            name = block.get("name", "")
            if kind == "video" and url.startswith("/memory/"):
                fp = MEMORY_ROOT / url[len("/memory/"):]
                if fp.exists():
                    frames = _extract_video_frames(fp)
                    if frames:
                        new_content.append({"type": "text",
                                            "text": f"<|video_start|> (user video: {name})"})
                        for fb64 in frames:
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
                            })
                        new_content.append({"type": "text", "text": "<|video_end|>"})
                    else:
                        new_content.append({"type": "text",
                                            "text": f"[Video '{name}' could not be decoded]"})
                else:
                    new_content.append({"type": "text",
                                        "text": f"[Video '{name}' missing on disk]"})
            elif kind == "audio":
                new_content.append({"type": "text",
                                    "text": f"[Audio '{name}' — not supported by vision model]"})
            else:
                new_content.append({"type": "text",
                                    "text": f"[Attached: {name}]"})

        else:
            new_content.append(block)

    new = dict(msg)
    new["content"] = new_content
    logger.debug(f"Rehydrated user message with {len(new_content)} blocks")
    return new


# ── Rehydration config ─────────────────────────────────────────────────────
#save on context window by resizing images and limiting video frames. UI shows full versions from disk, these are just for model context.
_MODEL_IMG_MAX_DIM  = 640   # longest edge for images sent to model
_MODEL_VID_MAX_DIM  = 512   # longest edge for video frames sent to model  
_MODEL_VID_MAX_FRAMES = 20   # max frames extracted from a tool-result video - 20 second video at 1 fps
# Rough char budget for ALL rehydrated media across the whole message list.
# At ~4 chars/token, 40K tokens ≈ 160K chars — leaves headroom for text history.
#help context truncation symptom. The model finished all 5 tool calls successfully, but when it went to generate the final summary response, the conversation had grown large enough (5 images + video frames rehydrated inline) that llama.cpp
_MEDIA_CHAR_BUDGET = 160_000

def _rehydrate_messages(messages: list, rehydrate_all: bool = False) -> list:
    # First pass: find all tool messages that have media, newest-first
    media_tool_indices = []
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            try:
                blocks = json.loads(c)
                if isinstance(blocks, list) and \
                   any(b.get("type") in ("image", "video", "audio") for b in blocks):
                    media_tool_indices.append(i)
            except (json.JSONDecodeError, ValueError):
                pass

    # Greedily assign budget newest-first: most recent scene always gets media,
    # older scenes get media only if budget allows, otherwise text placeholder
    indices_to_rehydrate = set()
    if not rehydrate_all:
        remaining_budget = _MEDIA_CHAR_BUDGET
        for i in media_tool_indices:
            # Estimate cost: count image/video blocks
            try:
                blocks = json.loads(messages[i].get("content", "[]"))
                img_count = sum(1 for b in blocks if b.get("type") == "image")
                vid_count = sum(1 for b in blocks if b.get("type") == "video")
            except Exception:
                img_count, vid_count = 0, 0
            # Rough estimate: image ≈ 50K chars base64 at 640px, video ≈ 6*40K frames
            estimated_cost = (img_count * 50_000) + (vid_count * _MODEL_VID_MAX_FRAMES * 40_000)
            if estimated_cost <= remaining_budget:
                indices_to_rehydrate.add(i)
                remaining_budget -= estimated_cost
            else:
                break  # budget exhausted — older scenes get placeholders
    else:
        indices_to_rehydrate = set(range(len(messages)))

    out = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")

        if role == "system":
            # System prompt is plain text — pass through unchanged
            out.append(msg)
            continue
        if role == "user":
            out.append(_rehydrate_user_msg(msg))
            continue
        if role == "assistant":
            clean = {k: v for k, v in msg.items() if not k.startswith("_")}
            out.append(clean)
            continue
        if role != "tool":
            out.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            out.append(msg)
            continue
        try:
            blocks = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            out.append(msg)
            continue
        if not isinstance(blocks, list):
            out.append(msg)
            continue

        should_rehydrate = idx in indices_to_rehydrate
        text_parts = []
        image_blocks = []

        for block in blocks:
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image":
                if not should_rehydrate:
                    # Still include the file path text so model knows it exists
                    url = block.get("url", "")
                    text_parts.append(f"[Image already shown earlier — omitted to save context: {url}]")
                    continue
                url  = block.get("url", "")
                mime = block.get("mimeType", "image/jpeg")
                if url.startswith("/memory/"):
                    fp = MEMORY_ROOT / url[len("/memory/"):]
                    if fp.exists() and fp.is_file():
                        b64 = base64.b64encode(fp.read_bytes()).decode()
                        b64 = _resize_b64_image(b64, mime, max_dim=_MODEL_IMG_MAX_DIM)
                        image_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}
                        })
                    else:
                        text_parts.append(f"[Media file not found: {url}]")
                else:
                    image_blocks.append({"type": "image_url", "image_url": {"url": url}})
            elif btype == "video":
                if not should_rehydrate:
                    url = block.get("url", "")
                    text_parts.append(f"[Video already shown earlier — omitted to save context: {url}]")
                    continue
                url = block.get("url", "")
                if url.startswith("/memory/"):
                    fp = MEMORY_ROOT / url[len("/memory/"):]
                    if fp.exists() and fp.is_file():
                        frames = _extract_video_frames(fp, target_fps=1, max_dim=_MODEL_VID_MAX_DIM)
                        frames = frames[:_MODEL_VID_MAX_FRAMES]
                        if frames:
                            text_parts.append(f"[Video — {len(frames)} preview frames below]")
                            for fb64 in frames:
                                image_blocks.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}
                                })
                        else:
                            text_parts.append("[Video generated — could not extract preview frames]")
                    else:
                        text_parts.append(f"[Video file not found: {url}]")
                else:
                    text_parts.append("[Video generated]")
            elif btype == "audio":
                text_parts.append("[Audio content — not supported by this vision model]")

        new_msg = dict(msg)
        new_msg["content"] = "\n".join(text_parts)
        out.append(new_msg)

        if image_blocks:
            image_blocks.insert(0, {"type": "text", "text": "Tool result images:"})
            out.append({"role": "user", "content": image_blocks})

    return out

def _sanitize_messages_for_llm(msgs: list, ctx: str = "") -> list:
    """
    Scrub messages before forwarding to llama.cpp to prevent 400/500 errors:
      - Empty assistant messages (no content, no tool_calls)
      - Assistant tool_calls with invalid/truncated JSON arguments
      - Orphaned tool results whose matching tool_call was removed above
    """
    valid_tc_ids: set[str] = set()
    out: list[dict] = []

    for m in msgs:
        role = m.get("role")

        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                good_tcs = []
                for tc in tcs:
                    args = (tc.get("function") or {}).get("arguments", "")
                    try:
                        if args:
                            json.loads(args)
                        good_tcs.append(tc)
                        if tc.get("id"):
                            valid_tc_ids.add(tc["id"])
                    except json.JSONDecodeError:
                        name = (tc.get("function") or {}).get("name", "?")
                        logger.warning(
                            f"{ctx}: stripped malformed tool_call '{name}' "
                            f"— invalid JSON args (truncated stream?)"
                        )

                if good_tcs:
                    out.append({**m, "tool_calls": good_tcs})
                elif m.get("content"):
                    # All tool_calls were bad but there is text content — keep without tool_calls
                    out.append({k: v for k, v in m.items() if k != "tool_calls"})
                else:
                    logger.warning(
                        f"{ctx}: dropped assistant message — "
                        f"all tool_calls malformed and no content"
                    )
            elif m.get("content"):
                out.append(m)
            else:
                logger.warning(f"{ctx}: dropped empty assistant message")

        elif role == "tool":
            tc_id = m.get("tool_call_id", "")
            if tc_id and tc_id not in valid_tc_ids:
                logger.warning(
                    f"{ctx}: dropped orphaned tool result "
                    f"(tool_call_id={tc_id!r} has no matching tool_call)"
                )
            else:
                out.append(m)

        else:
            out.append(m)

    return out


# ── Main streaming proxy ──────────────────────────────────────────────────────
async def chat_completions(request: Request):
    body = await request.json()
    conv_uuid = body.pop("_uuid", None)

    # ── Wait for LLM server if it's restarting after a VRAM-intensive tool ──
    # The MCP server now kicks the restart into a background thread and returns
    # the tool result immediately.  The next chat request may arrive while the
    # model is still loading (mcp cleanup + model load can take 4-5 min for large
    # models).  We wait here transparently so the browser never sees a 503.
    if not await _is_llama_running():
        logger.info("LLM server not ready; waiting up to 300 s for restart...")
        ready = await _wait_for_llama_ready(timeout=300.0)
        if not ready:
            return JSONResponse(
                {"error": "LLM server did not become ready in time — please retry"},
                status_code=503,
            )
        logger.info("LLM server ready, continuing with completion request.")

    # ── Gate: if a summary is running, wait for it before sending completion ──
    # This prevents the user's new message from racing a mid-summary context rewrite.
    if not _summary_done_event.is_set():
        logger.info(f"Summary in progress, holding completion for {conv_uuid}...")
        await _wait_for_summary_complete(timeout=120.0)
        logger.info("Summary done, proceeding with completion.")

    # Force streaming, and ask llama.cpp to emit `timings` + final usage
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    # llama-server-specific: emit a `timings` field on streamed chunks
    body.setdefault("timings_per_token", True)

    # Re-attach media from saved tool results so the model can actually see it as well as merge system todo updates
    if conv_uuid and "messages" in body:
        data = load_conv(conv_uuid)
        todo = (data or {}).get("sys_todo", "")
        msgs = body["messages"]

        print("\n\nMESSAGES:\n\n")
        for m in msgs:
            print(m)

        # drop any extra system msgs the client picked up from a reload
        first_sys = next((m for m in msgs if m.get("role") == "system"), None)
        msgs = [m for m in msgs if m.get("role") != "system"]
        base = first_sys["content"] if first_sys else ""
        msgs.insert(0, {"role": "system", "content": _compose_system(base, todo)})

        # Sanitize messages — remove empty/malformed assistant turns and
        # orphaned tool results so llama.cpp never sees invalid history.
        msgs = _sanitize_messages_for_llm(
            msgs, ctx=f"Conv {(conv_uuid or '?')[:8]}"
        )

        body["messages"] = msgs

    assembled = {
        "role": "assistant",
        "content": "",
        "reasoning": "",
        "tool_calls": [],
        "timings": None,
        "usage": None,
    }

    async def generate():
        raw_payload = bytearray()

        async with http_client.stream(
            "POST",
            f"{LLAMA_BASE}/v1/chat/completions",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Accept-Encoding": "identity",
            },
        ) as upstream:
            async for chunk in upstream.aiter_bytes():
                if conv_uuid:
                    raw_payload.extend(chunk)
                yield chunk

        # Stream has fully finished — now parse the text to save to memory
        if conv_uuid:
            text = raw_payload.decode("utf-8", errors="replace")
            for line in text.splitlines():
                _accumulate(line, assembled)
            await _finalise_and_save(conv_uuid, assembled)  # CHANGED: now awaited

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",         # critical for browser streaming
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Other /v1/* passthrough ───────────────────────────────────────────────────

async def relay(request: Request):
    subpath = request.path_params["subpath"]

    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
            },
        )

    url = f"{LLAMA_BASE}/v1/{subpath}"
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    body = await request.body()

    resp = await http_client.request(
        request.method, url, headers=fwd_headers, content=body
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={
            "Content-Type": resp.headers.get("content-type", "application/json"),
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── App assembly ──────────────────────────────────────────────────────────────

app = Starlette(
    routes=[
        Route("/", index),
        Route("/memory/{uuid}/{filename}", serve_media),
        Route("/download/{uuid}/{filename}", download_media),
        Route("/conv/new", conv_new, methods=["POST"]),
        Route("/conv/load", conv_load, methods=["POST"]),
        Route("/conv/list", conv_list, methods=["GET"]),
        Route("/conv/append_user", conv_append_user, methods=["POST"]),
        Route("/conv/{uid}", conv_delete, methods=["DELETE"]),
        Route("/conv/tool_result", conv_tool_result, methods=["POST"]),
        Route("/conv/edit_system_prompt", conv_edit_system_prompt, methods=["POST"]),
        Route("/server/props", server_props, methods=["GET"]),
        Route("/server/stop", server_stop, methods=["POST"]),
        Route("/server/start", server_start, methods=["POST"]),
        Route("/server/status", server_status, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/{subpath:path}", relay, methods=["GET", "POST", "OPTIONS"]),
    ],
)

@app.on_event("startup")
async def on_startup():
    """Start the llama.cpp server when the app starts."""
    # Suppress spurious Windows Proactor ConnectionResetError on client disconnects
    asyncio.get_event_loop().set_exception_handler(_suppress_connection_reset)

    print("[app] Starting llama.cpp server...")
    success = _start_llama_process()
    if not success:
        print("[app] ERROR: Failed to start llama.cpp server.")
        raise RuntimeError("Failed to start llama.cpp server")
    # Give it time to load the model
    print("[app] Waiting for llama.cpp to become ready (this may take a minute)...")
    ready = await _wait_for_llama_ready(timeout=180.0)
    if ready:
        print("[app] llama.cpp server is ready!")
    else:
        print("[app] WARNING: llama.cpp server not responding - you may need to start it manually")


@app.on_event("shutdown")
async def on_shutdown():
    """Stop the llama.cpp server when the app shuts down."""
    print("[app] Shutting down llama.cpp server...")
    _kill_llama_process()
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    port = 5000
    print(f"llama.cpp UI relay      →  http://localhost:{port}")
    print(f"Upstream llama-server   →  {LLAMA_BASE}")
    print(f"Memory store            →  {MEMORY_ROOT}")
    print(f"llama.cpp batch file    →  {LLAMA_BAT}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")