"""Tests for the tool-call pairing invariant.

The bug these exist to prevent: DeepSeek validates the *whole* transcript on every
request, so an assistant turn whose ``tool_calls`` are not each answered by exactly
one ``tool`` message is a 400 — and it is a 400 forever, because the same history is
replayed on every later request. Only the broken request looks like the problem.

A tool loop persists its assistant turn *before* the tools run and appends results
one at a time, so an interruption — a closed tab, the Stop button, ``CancelledError``
during a shutdown — can land between the two and write a half-finished group. Three
things are pinned here: the repair itself, the doors a broken transcript can come
through, and the interrupted loop that is the reachable trigger for all of it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from deepseek_client.messages import (
    INTERRUPTED_NOTE,
    ensure_tool_pairing,
    sanitize_messages,
)
from deepseek_client.tools import ToolRegistry
from deepseek_client.types import ChatMessage, ChatResponse, ToolCall
from server.llm import ChatEngine
from server.media import MediaStore
from server.rehydrate import rehydrate
from server.settings import load_settings
from server.store import ConversationStore, safe_keep


# ── builders ──────────────────────────────────────────────────────────────────

def call(call_id: str, name: str = "lookup", arguments: str = "{}") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant(*call_ids: str, text: str = "") -> dict:
    message: dict = {"role": "assistant", "content": text}
    if call_ids:
        message["tool_calls"] = [call(c) for c in call_ids]
    return message


def result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": "lookup", "content": content}


def user(text: str = "hi") -> dict:
    return {"role": "user", "content": text}


def roles(messages) -> list:
    return [m.get("role") for m in messages if isinstance(m, dict)]


def stub_of(messages, call_id: str):
    """The synthesised result answering ``call_id``, or None."""
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            return message
    return None


def assert_paired(messages) -> None:
    """Independently check the invariant the API enforces.

    Deliberately written out longhand rather than reusing the production walk, so a
    bug in the repair cannot make its own test pass.
    """
    outstanding: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            assert not outstanding, f"a new tool group opened over {outstanding}"
            outstanding = [c.get("id") for c in message["tool_calls"]]
        elif role == "tool":
            call_id = message.get("tool_call_id")
            assert call_id, "a tool result with no tool_call_id answers nothing"
            assert call_id in outstanding, f"{call_id!r} answers no open call"
            outstanding.remove(call_id)
        else:
            assert not outstanding, f"{outstanding} left unanswered by a {role!r} turn"
    assert not outstanding, f"{outstanding} were never answered"


# ── the repair ────────────────────────────────────────────────────────────────

def test_a_healthy_transcript_is_untouched():
    messages = [user("go"), assistant("c1", "c2"), result("c1"), result("c2")]

    assert ensure_tool_pairing(messages) == messages
    assert_paired(messages)


def test_an_unanswered_call_gets_a_stub_result():
    messages = [user("go"), assistant("c1"), result("c1"), assistant("c2")]

    repaired = ensure_tool_pairing(messages)

    assert roles(repaired) == ["user", "assistant", "tool", "assistant", "tool"]
    assert repaired[-1]["content"] == INTERRUPTED_NOTE
    assert_paired(repaired)


def test_parallel_calls_with_one_result_get_stubs_for_the_rest():
    messages = [assistant("c1", "c2", "c3"), result("c2")]

    repaired = ensure_tool_pairing(messages)

    # Results follow *call* order, not the order they happened to be recorded in.
    assert [m["tool_call_id"] for m in repaired[1:]] == ["c1", "c2", "c3"]
    assert repaired[1]["content"] == INTERRUPTED_NOTE
    assert repaired[2]["content"] == "ok"
    assert repaired[3]["content"] == INTERRUPTED_NOTE
    assert_paired(repaired)


def test_a_group_with_no_results_at_all_is_closed():
    repaired = ensure_tool_pairing([assistant("c1")])

    assert roles(repaired) == ["assistant", "tool"]
    assert_paired(repaired)


def test_a_result_before_any_call_is_dropped():
    messages = [result("c1"), user("go")]

    repaired = ensure_tool_pairing(messages)

    assert repaired == [user("go")]
    assert_paired(repaired)


def test_a_result_whose_call_is_gone_is_dropped():
    messages = [assistant("c1"), result("c1"), result("c9")]

    repaired = ensure_tool_pairing(messages)

    assert [m.get("tool_call_id") for m in repaired if m["role"] == "tool"] == ["c1"]
    assert_paired(repaired)


def test_a_duplicate_result_answers_only_once():
    messages = [assistant("c1"), result("c1", "first"), result("c1", "second")]

    repaired = ensure_tool_pairing(messages)

    assert [m["content"] for m in repaired if m["role"] == "tool"] == ["first"]
    assert_paired(repaired)


def test_a_result_with_no_id_is_dropped():
    messages = [assistant("c1"), {"role": "tool", "content": "orphan"}, result("c1")]

    repaired = ensure_tool_pairing(messages)

    # The nameless result is skipped; the call is answered by the real result that
    # follows it rather than by a stub.
    assert roles(repaired) == ["assistant", "tool"]
    assert repaired[1]["content"] == "ok"
    assert_paired(repaired)


def test_a_nameless_result_does_not_answer_the_call_it_follows():
    messages = [assistant("c1"), {"role": "tool", "content": "orphan"}]

    repaired = ensure_tool_pairing(messages)

    assert roles(repaired) == ["assistant", "tool"]
    assert repaired[1]["content"] == INTERRUPTED_NOTE
    assert_paired(repaired)


def test_a_call_with_no_id_is_dropped_but_the_prose_survives():
    nameless = {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    messages = [{"role": "assistant", "content": "thinking out loud", "tool_calls": [nameless]}]

    repaired = ensure_tool_pairing(messages)

    assert repaired == [{"role": "assistant", "content": "thinking out loud"}]


def test_a_turn_whose_only_call_is_unanswerable_goes_entirely():
    nameless = {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    assert ensure_tool_pairing([{"role": "assistant", "tool_calls": [nameless]}]) == []


def test_a_user_turn_inside_a_group_does_not_swallow_the_rest():
    """What a concurrent turn (or an old rehydrate bug) could leave behind."""
    messages = [assistant("c1", "c2"), result("c1"), user("by the way"), result("c2")]

    repaired = ensure_tool_pairing(messages)

    # c1 is answered, the user turn survives, and c2's late result is orphaned —
    # it cannot be reattached to a group that a user turn has already ended — so
    # the call itself is closed with a stub instead.
    assert roles(repaired) == ["assistant", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in repaired[1:3]] == ["c1", "c2"]
    assert repaired[2]["content"] == INTERRUPTED_NOTE
    assert_paired(repaired)


def test_pairing_twice_changes_nothing():
    broken = [assistant("c1", "c2"), result("c2"), result("ghost"), assistant("c3")]
    once = ensure_tool_pairing(broken)

    assert ensure_tool_pairing(once) == once


def test_sanitize_drops_a_tool_result_with_no_id():
    """The other half: a result that names no call never reaches `tool` handling."""
    messages = [assistant("c1"), {"role": "tool", "content": "nameless"}, result("c1")]

    kept = sanitize_messages(messages, keep_reasoning=False)

    assert [m.get("tool_call_id") for m in kept if m["role"] == "tool"] == ["c1"]


# ── truncation boundaries ─────────────────────────────────────────────────────

def test_safe_keep_snaps_back_off_a_trailing_tool_group():
    messages = [user("go"), assistant("c1"), result("c1"), assistant("c2")]

    # Keeping the trailing assistant turn would keep a call with nothing after it.
    assert safe_keep(messages, 4) == 3
    # Ending on a result whose group is complete is valid, so no snap is needed.
    assert safe_keep(messages, 3) == 3
    assert safe_keep(messages, 2) == 1
    assert safe_keep(messages, 1) == 1


def test_safe_keep_snaps_back_off_a_parallel_run():
    messages = [user("go"), assistant("c1", "c2"), result("c1"), result("c2")]

    # Cutting after the first result drops the second, so the whole group goes.
    assert safe_keep(messages, 3) == 1
    assert safe_keep(messages, 4) == 4


def test_safe_keep_is_a_no_op_on_plain_messages():
    messages = [user("one"), user("two"), user("three")]

    assert safe_keep(messages, 2) == 2
    assert safe_keep(messages, 99) == 3
    assert safe_keep(messages, -5) == 0


async def test_truncate_refuses_to_leave_a_half_group(tmp_path):
    store = ConversationStore(tmp_path / "memory")
    uuid = store.create(title="t").uuid
    await store.extend(uuid, [user("go"), assistant("c1", "c2"), result("c1"), result("c2")])

    # One too far: drops the second result but keeps the call that wants it.
    conv = await store.truncate(uuid, 3)

    assert roles(conv.messages) == ["user"]
    assert_paired(conv.messages)


# ── the engine: repair on the way out ─────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return ConversationStore(tmp_path / "memory")


@pytest.fixture
def media(store):
    return MediaStore(store)


@pytest.fixture
def settings(store):
    return load_settings({}, load_dotenv=False, memory_root=store.root)


def test_prepare_messages_heals_a_transcript_that_is_already_broken(store, media, settings):
    """A conversation poisoned before this fix must keep working — for free.

    Nothing rewrites the file: the repair is applied to the copy that goes out, so
    the stored history is never silently edited, and the same healing happens again
    on every later request.
    """
    uuid = store.create(title="t", sys_base="be brief").uuid
    broken = [user("go"), assistant("c1", "c2"), result("c1")]
    poisoned = store.load(uuid)
    poisoned.messages = list(broken)
    store.save(poisoned)

    engine = ChatEngine(settings, store, media, SimpleNamespace(replays_reasoning=True))
    prepared = engine.prepare_messages(store.load(uuid))

    assert roles(prepared) == ["system", "user", "assistant", "tool", "tool"]
    assert_paired(prepared)
    assert prepared[-1]["content"] == INTERRUPTED_NOTE

    # The transcript on disk is still exactly what it was.
    assert store.load(uuid).messages == broken


# ── the engine: the interrupted loop ──────────────────────────────────────────

class FakeStream:
    """As much of ``ChatStream`` as one step of the loop touches."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def __aiter__(self):
        for event in ():                # nothing streams in these tests
            yield event


class FakeClient:
    """Hands out canned responses and records what it was asked."""

    replays_reasoning = True

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def spec_for(self, model=None):
        return SimpleNamespace(
            id=model or "deepseek-flash", vision=True, thinking=True,
            default_reasoning_effort="high", max_input_tokens=1_000_000,
        )

    def stream(self, messages, **kwargs):
        self.requests.append([dict(m) for m in messages])
        assert self.responses, "the client was called more often than the test provides"
        return FakeStream(self.responses.pop(0))


def assistant_turn(*calls, text: str = "") -> ChatResponse:
    return ChatResponse(message=ChatMessage(
        role="assistant",
        content=text,
        reasoning="weighing it up",
        tool_calls=[ToolCall(id=cid, name=name, arguments=args) for cid, name, args in calls],
    ))


def final_turn(text: str = "all done") -> ChatResponse:
    return ChatResponse(message=ChatMessage(role="assistant", content=text, reasoning="done"))


async def test_a_normal_tool_loop_is_left_alone(store, media, settings):
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("look it up"))

    async def lookup(**kwargs):
        return "found it"

    registry = ToolRegistry()
    registry.register(
        lookup, name="lookup", description="looks things up",
        parameters={"type": "object", "properties": {}},
    )
    client = FakeClient(assistant_turn(("c1", "lookup", "{}")), final_turn())
    engine = ChatEngine(settings, store, media, client, tool_registry=registry)

    frames = [frame async for frame in engine.stream_turn(uuid)]

    written = store.load(uuid).messages
    assert roles(written) == ["user", "assistant", "tool", "assistant"]
    assert written[2]["content"] == "found it"
    assert_paired(written)

    # The second step saw its own tool result, and nothing invented a third call.
    assert roles(client.requests[1]) == ["user", "assistant", "tool"]
    assert client.requests[1][-1]["content"] == "found it"
    assert_paired(client.requests[1])
    assert any("done" in frame for frame in frames)


async def test_a_cancelled_tool_leaves_a_closed_group(store, media, settings):
    """The reachable trigger: ``ToolRegistry.call`` does not catch CancelledError.

    The assistant turn is already in the buffer by then, so this used to persist an
    unanswered call — and every later request for that conversation replayed it and
    got a 400 back.
    """
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("look these up"))

    async def maybe(**kwargs):
        if kwargs.get("fail"):
            raise asyncio.CancelledError()
        return "the first one worked"

    registry = ToolRegistry()
    registry.register(
        maybe, name="maybe", description="sometimes gives up",
        parameters={"type": "object", "properties": {"fail": {"type": "boolean"}}},
    )
    client = FakeClient(assistant_turn(
        ("c1", "maybe", "{}"),
        ("c2", "maybe", json.dumps({"fail": True})),
    ))
    engine = ChatEngine(settings, store, media, client, tool_registry=registry)

    with pytest.raises(asyncio.CancelledError):
        async for _frame in engine.stream_turn(uuid):
            pass

    written = store.load(uuid).messages
    assert roles(written) == ["user", "assistant", "tool", "tool"]
    assert_paired(written)
    # The call that finished keeps its real result; the one that did not is
    # recorded as unanswered rather than deleted from the model's history.
    assert written[2]["tool_call_id"] == "c1"
    assert written[2]["content"] == "the first one worked"
    assert written[3]["tool_call_id"] == "c2"
    assert written[3]["content"] == INTERRUPTED_NOTE

    # And the next turn can be served: the transcript it replays is valid.
    engine.prepare_messages(store.load(uuid))


# ── rehydrate: images cannot split a group ────────────────────────────────────

def make_png(width: int = 8, height: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def tool_message_with_image(call_id: str) -> dict:
    uri = "data:image/png;base64," + base64.b64encode(make_png()).decode()
    blocks = [
        {"type": "text", "text": "here is the screenshot"},
        {"type": "image", "url": uri, "name": "shot.png"},
    ]
    return {"role": "tool", "tool_call_id": call_id, "name": "shot", "content": json.dumps(blocks)}


def test_tool_images_follow_the_whole_group(store, media, settings):
    """Two parallel calls, the first of which returned an image.

    The image has to travel in a ``user`` message — the API rejects it anywhere
    else — but putting that message between two results ends the run, so the later
    call is left unanswered. It belongs after the last result of the group.
    """
    messages = [
        user("what does it look like"),
        assistant("c1", "c2"),
        tool_message_with_image("c1"),
        result("c2", "nothing visual"),
    ]

    out = rehydrate(messages, media, settings, media_root=store.root)

    assert roles(out) == ["user", "assistant", "tool", "tool", "user"]
    assert_paired(out)
    assert "Images returned by the tool above:" in json.dumps(out[-1]["content"])


# ── media that is meant for the person ────────────────────────────────────────
#
# The second audience. A picture a tool produced *for the user* — the file
# `display_media` was handed — is cut out of the tool result and hung on the assistant
# turn the user reads to, so the frontend can render it above the answer. It must never
# reach the model: it is not the model's to look at unless it asks (`inspect_media`),
# and leaving it in the transcript would charge it for the image on every later
# request, forever.
#
# A page image `web_fetch` downloaded is the awkward third case, and it is the one that
# made a mess: marked like the first (it is the user's, not the model's) but not pinned
# like it, because nobody asked to see it. It stays in the result, which is drawn inside
# the collapsed tool card — see `test_media_that_merely_arrived_stays_in_the_result_it_came_with`.

def frames_of(frames: list[str], event: str) -> list:
    """The decoded payloads of every ``event`` frame in a turn's SSE output."""
    out = []
    for frame in frames:
        lines = frame.splitlines()
        if not lines or lines[0].strip() != f"event: {event}":
            continue
        for line in lines:
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


def last_assistant(messages) -> dict:
    """The assistant turn the user reads to, which is where shown media is parked.

    Not the last *message*: a turn that ends on tool calls has its results after it.
    """
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    raise AssertionError("no assistant turn in the transcript")


def shot_registry(*, display: bool | str = "shown"):
    """A tool that returns one PNG, marked as ``display`` for the user.

    ``"shown"``/``True`` is a picture the user asked to see — it is pinned above the
    reply. ``"inline"`` is a picture that merely arrived with the result, the way a
    page's images arrive with ``web_fetch``: still marked, still never sent upstream,
    but drawn inside the tool card. ``None``/``False`` leaves it unmarked, so it is the
    model's to look at.
    """

    async def shot(**kwargs):
        block = {
            "type": "image",
            "data": base64.b64encode(make_png(16, 8)).decode(),
            "mimeType": "image/png",
        }
        if display:
            block["display"] = display
            block["caption"] = "the monkey"
        return [{"type": "text", "text": "took a picture"}, block]

    registry = ToolRegistry()
    registry.register(
        shot, name="shot", description="takes a picture",
        parameters={"type": "object", "properties": {}},
    )
    return registry


async def test_media_for_the_user_is_cut_out_and_hung_on_the_reply(store, media, settings):
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("show me the monkey"))

    client = FakeClient(assistant_turn(("c1", "shot", "{}")), final_turn("here you go"))
    engine = ChatEngine(settings, store, media, client, tool_registry=shot_registry(display=True))

    frames = [frame async for frame in engine.stream_turn(uuid)]

    written = store.load(uuid).messages
    assert roles(written) == ["user", "assistant", "tool", "assistant"]

    # Out of the tool result: what is left is the tool's own prose plus the path,
    # which is the part the model can act on.
    blocks = json.loads(written[2]["content"])
    assert [b["type"] for b in blocks] == ["text", "text"]
    assert "was shown to the user" in blocks[0]["text"]
    assert blocks[1]["text"] == "took a picture"

    # Onto the reply the user reads, which is the *last* assistant turn and usually
    # not the step that produced the media.
    shown = last_assistant(written)["_display"]
    assert len(shown) == 1
    assert shown[0]["type"] == "image"
    assert shown[0]["display"] is True
    assert shown[0]["caption"] == "the monkey"
    assert shown[0]["url"].startswith(f"/memory/{uuid}/")
    assert (store.root / uuid / shown[0]["url"].rsplit("/", 1)[-1]).is_file()

    # The live paint is told in the same frame as the result, so the picture is not
    # held back until the turn ends.
    results = frames_of(frames, "tool_result")
    assert len(results) == 1
    assert results[0]["media"] == shown
    assert results[0]["blocks"] == blocks


async def test_media_for_the_user_is_never_sent_to_the_model(store, media, settings):
    """The whole point of cutting it out: the API never sees the picture."""
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("show me the monkey"))

    client = FakeClient(
        assistant_turn(("c1", "shot", "{}")),
        final_turn("here you go"),
        final_turn("anything else?"),
    )
    engine = ChatEngine(settings, store, media, client, tool_registry=shot_registry(display=True))

    async for _frame in engine.stream_turn(uuid):
        pass

    assert roles(client.requests[1]) == ["user", "assistant", "tool"]
    # The model is told where the picture is instead.
    assert "was shown to the user" in json.dumps(client.requests[1])

    # A later turn replays the whole transcript — the reply now carries `_display` on
    # disk — and neither the marker nor the image can appear in it.
    await store.append(uuid, user("what did you show me?"))
    async for _frame in engine.stream_turn(uuid):
        pass

    replayed = client.requests[-1]
    assert all("_display" not in message for message in replayed)
    assert "data:image" not in json.dumps(replayed)
    assert "was shown to the user" in json.dumps(replayed)


async def test_media_the_model_asked_for_is_still_inlined(store, media, settings):
    """An unmarked block is the model's to look at, and goes upstream untouched."""
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("what is in this picture"))

    client = FakeClient(assistant_turn(("c1", "shot", "{}")), final_turn("a blue square"))
    engine = ChatEngine(settings, store, media, client, tool_registry=shot_registry(display=False))

    async for _frame in engine.stream_turn(uuid):
        pass

    written = store.load(uuid).messages
    assert [b["type"] for b in json.loads(written[2]["content"])] == ["text", "image"]
    assert "_display" not in last_assistant(written)
    # It travels in a `user` turn, which is the only place the API accepts an image.
    assert roles(client.requests[1]) == ["user", "assistant", "tool", "user"]
    assert "data:image" in json.dumps(client.requests[1])


async def test_media_that_merely_arrived_stays_in_the_result_it_came_with(store, media, settings):
    """``web_fetch`` downloads a page's images because a page has images.

    Nobody asked to see them, so they must not be pinned above the reply for the rest of
    the conversation — that is where a row of page posters ends up standing between the
    reader and every answer that follows. They stay in the result, which the frontend
    draws inside the tool card that carried them, collapsed like the page text beside
    them. Still marked, so keeping them there is not the same as showing them to the
    model.
    """
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("read that page"))

    client = FakeClient(assistant_turn(("c1", "shot", "{}")), final_turn("it says hello"))
    engine = ChatEngine(settings, store, media, client, tool_registry=shot_registry(display="inline"))

    frames = [frame async for frame in engine.stream_turn(uuid)]

    written = store.load(uuid).messages
    blocks = json.loads(written[2]["content"])
    # Left where it was: with the prose that came with it, not cut out of the result.
    assert [b["type"] for b in blocks] == ["text", "image"]
    assert blocks[1]["display"] == "inline"
    assert blocks[1]["url"].startswith(f"/memory/{uuid}/")
    # And not hung on the reply, so nothing renders above the answer.
    assert "_display" not in last_assistant(written)

    # The live paint gets the same split: nothing to pin, and the media in the frame the
    # card is filled from. Compared by url rather than by block, because the frame
    # carries `_display_blocks`' whitelist of the stored block and not the stored block.
    results = frames_of(frames, "tool_result")
    assert results[0]["media"] == []
    assert [b["type"] for b in results[0]["blocks"]] == ["text", "image"]
    assert results[0]["blocks"][1]["url"] == blocks[1]["url"]

    # Marked is not exempt: the model is handed the path in place of the picture.
    assert "data:image" not in json.dumps(client.requests[1])
    assert "was shown to the user" in json.dumps(client.requests[1])


async def test_media_is_still_shown_when_the_step_limit_ends_the_turn(store, media):
    """The turn's last assistant message is a tool call, and the user still sees it."""
    settings = load_settings({}, load_dotenv=False, memory_root=store.root, max_tool_steps=1)
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("show me the monkey"))

    client = FakeClient(assistant_turn(("c1", "shot", "{}")))
    engine = ChatEngine(settings, store, media, client, tool_registry=shot_registry(display=True))

    frames = [frame async for frame in engine.stream_turn(uuid)]

    written = store.load(uuid).messages
    assert "tool_calls" in last_assistant(written)
    assert len(last_assistant(written)["_display"]) == 1
    assert frames_of(frames, "warning"), "the truncated turn says so"


async def test_an_interrupted_turn_still_shows_what_it_had_already_shown(store, media, settings):
    """A half-finished turn is the case where the media matters most."""
    uuid = store.create(title="t").uuid
    await store.append(uuid, user("show me two things"))

    async def shot(**kwargs):
        if kwargs.get("what") == "second":
            raise asyncio.CancelledError()
        return [{
            "type": "image",
            "data": base64.b64encode(make_png(16, 8)).decode(),
            "mimeType": "image/png",
            "display": True,
        }]

    registry = ToolRegistry()
    registry.register(
        shot, name="shot", description="takes a picture",
        parameters={"type": "object", "properties": {"what": {"type": "string"}}},
    )
    client = FakeClient(assistant_turn(
        ("c1", "shot", json.dumps({"what": "first"})),
        ("c2", "shot", json.dumps({"what": "second"})),
    ))
    engine = ChatEngine(settings, store, media, client, tool_registry=registry)

    with pytest.raises(asyncio.CancelledError):
        async for _frame in engine.stream_turn(uuid):
            pass

    written = store.load(uuid).messages
    assert_paired(written)
    shown = last_assistant(written)["_display"]
    assert len(shown) == 1
    assert shown[0]["url"].startswith(f"/memory/{uuid}/")
    # And the next turn can still be served: nothing about the cut broke the group.
    engine.prepare_messages(store.load(uuid))


def test_a_marked_block_costs_nothing_to_keep(store, media, settings):
    """It is never inlined, so it must not be dropped by the media budget either."""
    import base64 as b64

    marked = [{
        "type": "image",
        "url": f"/memory/{'a' * 32}/shown.png",
        "mime": "image/png",
        "display": True,
    }]
    block = tool_message_with_image("c1")
    block["content"] = json.dumps(marked)

    out = rehydrate([user("go"), assistant("c1"), block], media, settings, media_root=store.root)

    assert roles(out) == ["user", "assistant", "tool"]
    text = json.dumps(out[-1]["content"])
    assert "was shown to the user" in text
    assert "data:image" not in text

