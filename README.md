# deepseekUI

A local, privacy-first chat interface for a DeepSeek endpoint.

Everything stays on your machine. Conversations, images, videos and extracted frames are
written to a local `memory/` directory as plain files you can read, copy, back up or delete
with Explorer. Nothing is sent anywhere except the messages you choose to send to the model
API. There is no account, no telemetry and no cloud sync.

---

## Quick start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
#    then edit .env and set DEEPSEEK_API_KEY

# 3. Run
python run.py
```

Open <http://127.0.0.1:5000>.

The server starts happily without an API key — the UI loads and explains what is missing —
so you can look around before signing up for anything.

### Command line

```
python run.py [--host 127.0.0.1] [--port 5000] [--lan] [--reload] [--log-level INFO]
```

---

## Safety model

The server holds your API key in memory and spends your credits. It therefore **refuses to
bind to anything but loopback** unless you say otherwise, twice:

```powershell
# Set ALLOW_LAN=true in .env FIRST, then:
python run.py --lan
```

Without `--lan` it exits with an explanation rather than quietly exposing your key to the
network. Do not do this on an untrusted network: there is no authentication, because there
was never supposed to be a second user.

---

## How it works

```
run.py                    argument parsing and the loopback gate
server/routes.py          HTTP surface; Starlette routes, no decorators
server/app.py             wiring: settings -> store -> media -> engine -> app
server/llm.py             the turn: compose, rehydrate, stream, run tools, persist
server/tools_builtin.py   resize / compress / inspect / frame-reduce / read (six tools)
server/mcp.py             MCP servers over Streamable HTTP
server/rehydrate.py       put stored media and text back into the request, under a budget
server/media.py           sniffing, probing, resizing, frame extraction, text detection
server/store.py           one directory per conversation, atomic writes
server/settings.py        configuration, read from the environment
deepseek_client/          the API wrapper — reusable, imports nothing from server/
```

`deepseek_client/` is a self-contained async client for any OpenAI chat-completions
compatible endpoint. It is deliberately independent: nothing in it imports from `server/`,
so you can lift the package into another project unchanged.

---

## Storage

```
memory/
└── <uuid>/
    ├── conversation.json        title, model, system prompt, message blocks
    ├── upload_1739...png        media you attached, exactly as uploaded
    ├── upload_1739..._resized.png
    ├── user_file_1739...py      a text or code file you attached
    └── clip_preview_000.jpg     frames sampled from a video
```

Design decisions worth knowing:

- **Atomic writes.** Every save goes to `conversation.json.tmp` and is then `os.replace`d, so
  a crash mid-write cannot leave a truncated conversation behind.
- **Base64 is never stored.** Inline images arriving from the model or the browser are decoded
  once and written as real files. `conversation.json` holds only paths. A conversation stays
  readable in a text editor and diffable.
- **Neither is inlined text.** An attached `.md`, `.py` or `.gitignore` is stored as bytes
  beside the conversation and read back on the way out, under budget. The characters the model
  sees are never written to `conversation.json`, for the same reason base64 is not: a 40 KB
  source file inlined into a transcript would make every later turn resend it.
- **Media lives beside its conversation**, so deleting a conversation deletes its images. There
  is no orphaned-media cleanup to get wrong.
- **Idempotent uploads.** A file is never overwritten; a collision gets a `_1` suffix.

Deleting `memory/` is a complete reset. Copying it is a complete backup.

### Media and previews

Files are streamed from `/memory/<uuid>/<file>` with the content type taken from the file's
own bytes rather than its extension, so a `.webp` or an extension-less file still previews
correctly. Responses carry byte ranges, which is what lets the player scrub a video.

Whether a clip actually plays depends on the *browser's* decoders, not on this server. H.264 in
an MP4 is universally supported. VP9 in a WebM is served byte-for-byte correctly — verified by
hash — but VS Code's integrated browser cannot demux it and reports `DEMUXER_ERROR_COULD_NOT_OPEN`.
If a video thumbnail stays blank while the file is plainly on disk, try the same URL in a normal
browser before suspecting the backend.

That decoder gap is also why `compress_video` shells out to `ffmpeg` for H.264 when it is on
`PATH`, and only falls back to OpenCV's writer when it is not: OpenCV offers just the older
`mp4v` fourcc here, which browsers refuse, and a compressed clip that cannot be previewed is
worse than the original.

---

## Configuration

See `.env.example` for the annotated list. The important ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | — | required; referenced from `providers.json` as `${DEEPSEEK_API_KEY}` |
| `DEEPSEEK_MODEL` | `deepseek-flash` | default model; blank means "first in providers.json" |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | where to listen |
| `ALLOW_LAN` | `false` | must be `true` *and* `--lan` to bind off-loopback |
| `MEDIA_ROOT` | `./memory` | where conversations are written |
| `MCP_CONFIG` | `./mcp.json` | MCP server list; seeded from `mcp.example.json` on a first run |
| `MODEL_TEXT_MAX_CHARS` | `40000` | how much of one attached text file is inlined |
| `TEXT_CHAR_BUDGET` | `200000` | total attached text per request, newest first |
| `MAX_TOOL_STEPS` | `8` | tool-call ceiling per turn; editable from Settings |
| `CONTEXT_SAFETY_RATIO` | `0.92` | how full the context may get before old turns are dropped |

### `providers.json`

Model definitions live here rather than in the environment, because a model is more than a
name. `apiKey` supports `${VAR}` interpolation so a key never has to be written into a file
that might be shared or committed.

```json
{
  "providers": [{
    "name": "DeepSeek",
    "vendor": "customendpoint",
    "apiKey": "${DEEPSEEK_API_KEY}",
    "apiType": "chat-completions",
    "url": "https://api.deepseek.com",
    "apiPath": "/chat/completions",
    "models": [
      { "id": "deepseek-flash",  "toolCalling": true, "vision": true,
        "maxInputTokens": 1000000, "maxOutputTokens": 393216 },
      { "id": "deepseek-v4-pro", "toolCalling": true, "vision": false,
        "maxInputTokens": 1000000, "maxOutputTokens": 393216 }
    ]
  }]
}
```

`vision` is not cosmetic. The UI reads it and warns before you attach an image to a model that
will reject it, rather than letting you discover that with a 400.

#### `apiFlavor` — talking to a plain OpenAI endpoint

DeepSeek accepts a few request fields that most OpenAI-compatible servers do not. `apiFlavor`
decides which of them go on the wire, so the same server can target llama.cpp, vLLM, or
anything else that speaks `/chat/completions` without a code change.

| Value | Meaning |
| --- | --- |
| `"deepseek"` | **Default.** Byte-identical to the historical request shape. |
| `"openai"` | Strict OpenAI only: the DeepSeek-specific fields are dropped. |

| Field | `deepseek` (default) | `openai` |
| --- | --- | --- |
| `thinking` | sent when an effort is in play | never sent |
| `reasoning_effort` | sent, normalised to `none`/`low`/`high`/`max` | never sent |
| `user_id` | sent after charset sanitising | never sent |
| replayed `reasoning_content` | kept (mandatory whenever `tools` are present) | stripped from outgoing messages |
| `stream_options.include_usage` | sent when streaming | sent when streaming |

The flavour is authoritative over the config: a provider whose `defaults` block still carries
`thinking` or `reasoning_effort` will not leak those keys to an `"openai"` endpoint. If a
non-DeepSeek server does understand an extra field, `extra_body` on the request is the
deliberate escape hatch — it is passed through untouched in either flavour.

To point the app at a local model:

1. Set `url` (and, if the server does not serve `/chat/completions`, `apiPath`) to the local
   address, e.g. `"url": "http://127.0.0.1:8080"`.
2. Set `"apiFlavor": "openai"`.
3. Keep an `apiKey`: local servers usually ignore it, but the client refuses to construct
   itself with an empty key, so use a dummy such as `"apiKey": "local"`.
4. Set `maxInputTokens` / `maxOutputTokens` to the server's real limits — the defaults are
   sized for DeepSeek, and a wrong `maxInputTokens` only shows up as a late, confusing error.

### `mcp.json`

MCP servers are reached over **Streamable HTTP**. There is no stdio transport: this server is
itself a local process and will not spawn children on your behalf. For a stdio-only server,
bridge it (e.g. `mcp-proxy`) and point `url` at the bridge.

```json
{
  "servers": [{
    "name": "filesystem",
    "url": "http://127.0.0.1:8931/mcp",
    "prefix": "fs",
    "timeout": 120,
    "allowedTools": ["read_file", "list_directory"]
  }]
}
```

Each tool is exposed to the model as `<prefix>__<tool>`. Headers support `${VAR}` too, so a
bearer token belongs in `.env` rather than in this file; a header whose variable is unset is
dropped with a warning instead of being sent literally.

`allowedTools` is an **allow-list of remote tool names, and it is a filter, not a hint** — a
tool that is not listed is never registered, so the model can neither see nor call it. Omit
the key, or leave the list empty, to expose everything the server offers. A sentence like
*"the server has this tool but the model never uses it"* usually means it is missing here.
The names are the **remote** ones: write `read_file`, not `fs__read_file`.

A missing `mcp.json` is normal on a first run — it is created from the template for you, see
[First run: where `mcp.json` comes from](#first-run-where-mcpjson-comes-from). Only if the
template is gone too is MCP simply off, which the startup log says out loud.

#### Editing servers from the UI

**Settings → MCP servers** lists every entry with its live status, and can add, edit, toggle,
test and delete servers. Every change is written straight back to `mcp.json` and the
connections are reopened, so there is nothing to restart.

There is no separate store: the panel edits *your* file. Three consequences worth knowing.

* **Hand editing still works.** The panel is one way to change the file, not the only way.
  Edit `mcp.json` in your editor as before and press **Reload** to pick up the change.
* **The file's shape is preserved.** Whether you wrote `{"servers": [...]}` or
  `{"mcpServers": {...}}`, saves go back in the same shape, and top-level keys that are not
  the server list — a `_comment`, say — survive untouched. Values are edited as written, so a
  header of `Bearer ${MY_TOKEN}` is not expanded and overwritten with the resolved secret.
* **An unreadable file is not overwritten.** If `mcp.json` cannot be parsed the panel says so,
  disables editing, and every write route returns `409` rather than replacing your file with a
  half-understood version of it. Saves are atomic (write to a sibling, then `os.replace`), so a
  crash mid-write cannot truncate it.

A UTF-8 BOM is tolerated, because Notepad and PowerShell's `Set-Content -Encoding utf8` both
write one; saves are written back without it. **Test** probes a server without saving it, which
is the quickest way to check a URL is reachable before committing it to the file.

#### First run: where `mcp.json` comes from

This file holds API keys in plain text as `x-api-key` headers, so it is **git-ignored** — the
committed `mcp.example.json` is the template. On a fresh clone there is no `mcp.json` at all,
and "paste your server into `mcp.json`" is not advice you can follow against a file that does
not exist. So whichever starts first — the app, or either server in `mcp_server/` — copies the
template to `mcp.json` once, byte for byte, comments included.

It is a copy and nothing else:

* **An existing `mcp.json` is never touched.** Not empty, not unparseable, not even if it is a
  directory. A half-finished hand edit is not replaced by the template, and an unreadable file
  stays unreadable rather than being silently rewritten into something the panel can read.
* **No template, no file.** Pointing `MCP_CONFIG` at a path whose directory has no
  `mcp.example.json` creates nothing, which is what keeps the tests hermetic.
* **Non-fatal.** A read-only checkout still runs; it just has nowhere to save a server.
* **`mcp.json` is what is read afterwards.** Editing `mcp.example.json` changes nothing once the
  copy exists — edit `mcp.json`, or use the panel.

Delete both files and MCP is simply off, with the startup log saying so. Both bundled servers
ship **disabled** in the template: a first run should not open two connections that are certain
to fail because those processes are not started yet.

#### The bundled example server

`mcp_server/mcpserver.py` is a small self-contained MCP server (Streamable HTTP, port `8572`)
that exists to exercise the MCP path end to end:

```powershell
python mcp_server/mcpserver.py
```

Starting it also copies `mcp.example.json` to `mcp.json` if that file is missing, so a fresh
clone that starts this server first still ends up with the template where the app expects it.
It never reads that file itself — the app does, and passes the key in as a header.

It exposes two tools:

* **`web_search(query)`** — one grounded Gemini call. Gemini runs the search, reads the pages
  and writes the summary; the source URLs it was grounded on come back underneath, so the
  answer can be checked and any one of them can be opened with `web_fetch`.
* **`web_fetch(url)`** — opens a single page and returns its readable text plus content images.
  It needs no search, so it is the right tool whenever you already have an address. A bare
  domain such as `example.com` is accepted and assumed to be `https`. There is **no model in
  this path**: the page comes back as plain text with its HTML tags removed, and the calling
  model does the reading and summarising. It can therefore still carry page furniture
  (navigation, cookie notices); the tool description tells the model to read past it.

Search needs a Google AI Studio API key — free, no billing account:

1. Get one at <https://aistudio.google.com/apikey>.
2. In the app, **MCP servers → Edit** on the `manual` entry, and paste it into **API key**.
   That is stored in `mcp.json` as an `x-api-key` header on that server's entry and sent with
   every call, so there is nothing to restart after changing it.
3. Running the server by hand instead? Set `GEMINI_API_KEY` in its environment.

Nothing scrapes a search engine any more. Every keyless engine sits behind a bot filter that
answers with a `202` challenge page or a `429` rather than an error, and none of those raise —
so a *throttled* search used to be indistinguishable from an *empty* one and got reported as
"no results". That is worse than a failure, because it tells you a topic does not exist when
the truth was that the request was refused. Search now either works or says why it could not,
and the tool description tells the model to treat "unavailable" as a setup problem rather than
an absence of coverage.

| Variable | Effect |
| --- | --- |
| `GEMINI_SEARCH_MODEL` | The model to ground with. Defaults to `gemini-2.5-flash`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Fallback key, when no header reaches the server. |

`gemini-2.5-flash` is named rather than left to the client because it is the model whose
Google-Search grounding is free (500 grounded requests/day, shared with `gemini-2.5-flash-lite`);
a model outside that tier would quietly bill the key's project. **The free tier uses what you
send and receive to improve Google's products** — do not put anything confidential through it.
Grounding errors (`API key not valid`, `RESOURCE_EXHAUSTED` for an exhausted quota) are reported
in plain words instead of as a stack trace.

`web_fetch` needs no key and no second model. It downloads the page, strips the markup, and
hands back the text and the page's content images as image blocks.

That is a deliberate simplification, not an omission. `web_fetch` used to strip boilerplate
with the local llama.cpp instance (a 4-way parallel map over chunks, then a reduce), which took
~135 s on a large page. The MCP client's per-call timeout is 120 s, and a call that outlives
that budget dooms the session it was sent on — the abandoned request's late response lands on
the shared connection and the transport reads it as end-of-stream. Every fetch after the first
large page therefore failed with `Connection closed`, including a bare `example.com` that could
not fail for any content reason. Fetching is now a download and a tag strip, so it finishes in
the time the network takes, and reading the page is the calling model's job — which is what it
is for.

The response to a tool call is **pure JSON** — a single JSON-RPC payload with nothing in front
of it. (It used to be a run of `.` heartbeat bytes followed by the JSON, which the official MCP
client cannot parse: it validates the whole body at once, so any call slower than the heartbeat
failed as if the tool had never answered.)

#### Introspection: the model reading its own source

`mcp_server/self_reflection.py` is a second, separate MCP server (port `8590`) whose only job
is to let the model look at this repository — its own code — so that *"why does the media
parser fail on this file?"* can be answered from the source instead of from a guess.

```powershell
python mcp_server/self_reflection.py
```

| Tool | Use it for |
| --- | --- |
| `introspect(include?)` | The directory map (files, line counts, and per-module class/function counts) plus an orientation written by the model: layering, entry points, and the parts that are easy to get wrong. Start here. |
| `explain_file(path)` | One file in detail: responsibilities, classes and functions with signatures and line numbers, imports split into stdlib / third-party / local, and how the code works inside. |
| `read_source(path, offset?, limit?)` | The file itself, verbatim, with a line-number gutter so lines can be quoted. Nothing is summarised or reordered. |

**Why a second server rather than two more tools in the one above.** A tool that diagnoses this
project must not be built out of this project: if an import in `server/` breaks, the tool that
would tell you so has to still start. So this file shares no code with `app.py` or the `server/`
package — it borrows only `deepseek_client`, and duplicates the JSON-RPC plumbing on purpose.

**What it cannot do.** It is read-only: nothing writes, edits, or executes anything, and every
path argument is resolved and then checked against the project root, so `../`, an absolute path
outside the tree, and a symlink or junction pointing out of it are all refused. `.env`, key and
certificate files, and `memory/` are refused by name — the model can neither read another
conversation's transcripts nor enumerate them. Values that look like credentials (`api_key`,
`sk-…`, `Bearer …`, JWTs) are masked before any text leaves the process, including text that is
about to be sent to the API for summarising.

Summaries come from the same DeepSeek API the app itself uses, through `providers.json`, so
they need `DEEPSEEK_API_KEY` like everything else. `INTROSPECTION_MODEL` picks a different
model; `--no-llm` (or `INTROSPECTION_LLM=off`) skips the model entirely.

**When the model cannot be reached, the tool says so.** The map, the symbol outline and the
imports are computed locally and are still returned and still accurate; only the prose is
missing, and it is labelled *"Summary unavailable"* with the reason. This distinction is the
point: *"could not ask the model"* and *"there is nothing here"* look the same in a transcript,
and a model that reads the second will repeat it to you as fact.

| Variable | Effect |
| --- | --- |
| `INTROSPECTION_ROOT` | Repository to expose. Defaults to the parent of `mcp_server/`. |
| `INTROSPECTION_PORT` | Listen port, default `8590`. |
| `INTROSPECTION_MODEL` | Model for summaries, default the provider's own (`deepseek-flash`). |
| `INTROSPECTION_LLM` | `off`, `0`, `false` or `no` disables summaries everywhere. |
| `INTROSPECTION_TIMEOUT` | Seconds to wait for one summary, default `180`. |

The same three tools are available from a shell, which is often quicker than going through the
model: `--map [--include GLOB]`, `--explain PATH`, `--read PATH [--offset N] [--limit N]`,
`--no-llm`, `--port N`.

To expose it to the model, add it to `mcp.json` — **without** `allowedTools`, which is a filter
and would hide the tools you just added:

```json
{
  "mcpServers": {
    "self": { "url": "http://127.0.0.1:8590/mcp", "prefix": "self" }
  }
}
```

The tools then reach the model as `self__introspect`, `self__explain_file` and
`self__read_source`. The server is not started for you; run it yourself, as above.

Starting it also copies `mcp.example.json` to `mcp.json` if that file is missing — the same
first-run seeding the app does, because on a fresh clone this is often the first thing you
start. It uses `INTROSPECTION_ROOT`, so it writes the template into the tree it was pointed at
and not into whatever directory you happened to launch from. The read-only commands (`--map`,
`--explain`, `--read`) do **not** seed: inspecting a project should never leave a file behind
in it.

One implementation detail, in case you write your own client: a slow tool call streams a
heartbeat, so *this* server's response body is a run of `.` characters followed by the JSON —
strip everything before the first `{` before parsing. (`mcp_server/mcpserver.py` does **not**
do this; its replies are pure JSON, and the official MCP client — which validates the whole
body at once — depends on that.)

---

## Tools

Six built-in tools shape media and files before they reach the model:

| Tool | Why it exists |
| --- | --- |
| `inspect_media` | reports dimensions, duration and estimated token cost, or a text file's lines and charset |
| `resize_image` | downscale before upload |
| `compress_image` | re-encode at lower quality |
| `reduce_video_frames` | sample a clip down to a handful of stills |
| `compress_video` | re-encode a clip smaller, as H.264 when `ffmpeg` is on `PATH` |
| `read_file` | page through an attached text file by line (`offset`, `limit`) |

These are not conveniences. DeepSeek resizes every image to roughly 1300x1300 px and charges
up to 1024 tokens for it, and it **upscales** anything smaller than ~544 px — so a 200 px
thumbnail costs nearly as much as a 1300 px photo, and a 6000 px screenshot costs the same
while being four times the upload. On video the numbers are starker: a 90-second clip at 1 fps
is 90 images, around 92,000 tokens, to describe a scene that three frames would cover.

The practical consequence is that resizing before upload is not tidiness, it is the difference
between a usable transcript and one that burns its context window on a single screenshot.

Every tool path argument is resolved inside the active conversation and cannot escape it —
neither by traversal nor by referencing another conversation's media.

Two of these depend on binaries that may not be installed. Without `opencv-python` the video
tools report that they cannot decode rather than raising, and without `ffmpeg` on `PATH`
`compress_video` still works but produces `mp4v`, which browsers will not play. Everything else
is pure Python.

---

## Text files

An attached `.md`, `.txt`, `.py`, `.json`, `.gitignore` — or a `Makefile`, which has no
extension at all — is read and inlined into the request, so the model sees the *contents*
rather than a note that something was attached.

What counts as text is decided by **content**, not by the name. The extension is only consulted
when a file is empty and there is nothing else to go on. This matters more than it sounds:
`Path(".gitignore").suffix` and `Path("Makefile").suffix` are both empty, so no extension table
can ever classify the files people actually attach. It also means the decision is safe in both
directions — a `.txt` full of JPEG bytes is treated as an image, and a suffix-less file full of
English is treated as text.

Detection is: a BOM if there is one (`utf-8-sig`, then UTF-32 before UTF-16, since the UTF-32
BOM starts with the UTF-16 one), otherwise strict UTF-8, otherwise Windows-1252 as the single
fallback and only when there is no NUL in the head. A successful decode is not enough on its
own — `bytes(range(1, 32)) * 40` is perfectly valid UTF-8 — so at least 85% of the sample has
to be printable. `application/pdf` is excluded explicitly: its header is ASCII and the rest is
compressed streams, so it would otherwise pass and print as mojibake.

Text and media share one budget, filled newest-first. A file that does not fit is either shown
partly (with a `[truncated: showing N of at least M characters — call read_file(...)]` footer,
including the offset to continue from) or, if less than 2,000 characters remain, left out with
a note telling the model to `read_file` it. Anything genuinely unreadable says so — *"256 bytes
of binary data, which cannot be read as text"* — because a placeholder that only says "a file is
attached" leaves the model guessing about content it can never reach.

Two settings bound this: `MODEL_TEXT_MAX_CHARS` (default 40,000) per file and
`TEXT_CHAR_BUDGET` (default 200,000) per request. Files still live on disk and are still
referenced by URL — base64 never lands in `conversation.json`, and neither does the text.

`read_file` accepts a `/memory/...` URL, a path inside the conversation, **or the name the file
was attached as**. That last form is load-bearing: an attachment keeps the name you gave it, but
its bytes land under a generated name (`user_file_1739...py`) so two files of the same name
cannot collide — and the continuation footer necessarily names the file the way *you* attached
it. Without the lookup, `call read_file('routes.py', offset=40000)` would point at a path that
does not exist, and a 43 KB source file could be read to the 40,000-character mark and no
further. Inlining is character-exact, including CRLF; the numbered view normalises line endings
to `\n` the way `splitlines()` does, since a line number is what makes a window continuable.

A single read is bounded to 8 MiB from the start of the file. A file larger than that is read
from the beginning and `read_file` says so rather than implying it reached the end.

---

## Code in the transcript

Code is highlighted in **both** directions, and both directions get the same container — a chip:
a labelled frame with a badge, a name, a size and line count, `copy`, and (for an attachment) a
download link.

- **Input.** An attached file that reads as text becomes a chip. The badge is its extension and
  its language is resolved from the *name* (`.py` is Python, `.gitignore` is `ini` config), so
  the caption says what the file is before anything is read.
- **Output.** A fenced block in a reply is highlighted. A tool result is highlighted when it
  either *is* a `read_file` listing, or detection is confident enough — `json`, `diff`, `sql`,
  `html` and `css` are auto-converted, and YAML is deliberately not, because `Key: value` is a
  sentence pattern as well as a config format. Everything else stays prose, which is why a
  three-line summary of a directory listing still renders as a paragraph.

Highlighting is **auto-detected from content** — the extension is never the decider, for the same
reason `Makefile` and `.gitignore` broke the extension table on the way in. Detection is scored
and only fires above a threshold: a wrong guess is worse than no guess, so plain prose stays
plain. A fenced block that *names* a language, though, is always believed; an unknown name
(`ocaml`) is kept verbatim rather than quietly relabelled `text`.

**The engine is hand-written** (`frontend/dist/assets/highlight.js`, ~1000 lines) rather than a
vendored library or a CDN build. There is no build step here and no Node on the machine, so
~120 KB of generated code that cannot be executed or tested would be a liability rather than a
shortcut. It maps ~90 file names and extensions onto 27 grammars — `.py`, `.pyi` and `Makefile`
all land somewhere — and those 27 cover what actually shows up in a coding chat. Nothing else is
faked: a `.csv` is a chip with no colour, because colouring data would only invent structure.
Three invariants hold for every input, and each is asserted rather than intended:

1. **Lossless.** Stripping the tags and decoding the entities returns the source exactly, modulo
   a deliberate CRLF → LF normalisation (so the losslessness claim is true for Windows files
   too, rather than nearly true).
2. **Balanced per line.** Every `<span>` is closed before the end of its line, which is what
   makes a gutter possible at all.
3. **Only spans.** Text is escaped *before* it is wrapped, so the only markup the tokenizer can
   emit is `<span class="t-…">` — a file containing `<span class="t-kw">` or `</script>` renders
   as characters.

Chips show the numbers that travel with the block (`bytes`, `lines`), computed once at ingest so
the UI never fetches a file back just to count its lines. What differs between input and output
is only *when* a chip opens: text already in hand is the answer the reader asked for, so it opens
immediately, while an attachment is a reference and opens on demand, with a ranged request capped
at 512 KB (and refused outright above 16 MB). Numbered views keep the digits in `<span class="ln">`
**outside** the highlighted code so they are never tokenised, and `copy` gives you the code
without the gutter.

`read_file` listings are recognised *structurally* — a header naming the file, then a five-wide
right-aligned gutter — and strictly: every line after the header must be numbered, or it is left
alone as prose. The listing keeps its own first line number, so a window starting at line 4000 is
numbered from 4000.

---

## Architecture notes

**The wrapper is reusable; the server is policy.** `deepseek_client/` knows how to talk to a
chat-completions endpoint and nothing else. `server/` decides what this deployment does:
where files go, which tools exist, how much media to rehydrate. The dependency arrow points
one way only.

**Images are user-message-only on this API.** An `image_url` block anywhere else is a hard 400.
That single constraint explains a lot of otherwise odd-looking code: tool results that return
an image replay it as a *following user message*, and the rehydrator re-inserts stored media
at the one position the API accepts.

**Thinking mode is on by default.** The reasoning stream is shown in the UI and
`reasoning_content` is echoed back on every assistant turn — mandatory once tools are in play,
or the next request is rejected.

**Errors belong in the stream.** An SSE response commits HTTP 200 before its first frame, so
anything raised later cannot become a status code. Configuration and model errors are therefore
emitted as `event: error` frames the UI renders, rather than crashing the connection.

---

## Development

```powershell
python -m pytest tests -q      # 611 tests, no network required
```

The suite covers the wrapper (offline, via `httpx.MockTransport` and a byte-stream that splits
SSE frames mid-event), the HTTP surface (via `httpx.ASGITransport`), MCP config translation and
round-tripping, the media helpers, and the web-search MCP server. Nothing is mocked beyond the
network boundary itself: the store writes real files to a temporary directory, images are real
PNGs, and the search tests replace only `genai.Client` — the canned replies are built through the
SDK's own `types`, so the attribute names the server reads are asserted against the real ones.

The frontend has no test runner — it is hand-written ES modules loaded straight from
`frontend/dist/assets/`, so there is nothing to build and nothing to install. Instead there is
one page, `assets/__selftest.html`, which is deliberately **not** linked from the app: run the
server, open `http://127.0.0.1:5000/assets/__selftest.html`, and check that it says `ALL PASS`
(`window.__harness` carries the detail). It holds the highlighter to its invariants over a corpus
of samples — including ones that try to break out with `<span>` and `&amp;` in the source — and
asserts the file-name resolver, the language detector, the listing recogniser, the code block
rendering, and the MCP header/API-key rules in `assets/mcpheaders.js`. That last module is
separate from `app.js` precisely so it can be executed here: a header that fails to survive the
editor's round trip does not throw, it produces a server that connects and then refuses every
call, which is indistinguishable from a broken server. It earned its place the first time it ran,
by finding nine bugs in code that had been reviewed and looked clean: **frontend JavaScript has to
be executed, not inspected.**

---

## Licence

Not yet chosen.
