/* app.js — the UI.
 *
 * Structure, in order:
 *   1. tiny DOM helpers
 *   2. rendering: transcript, tool cards, media, context meter
 *   3. the live turn (SSE consumer)
 *   4. conversation management
 *   5. composer, attachments and drag-and-drop
 *   6. drawers, settings, boot
 *
 * The one invariant worth stating up front: **the transcript is re-rendered from
 * the server's copy after every turn.** Streaming paints optimistically for
 * responsiveness, then `renderTranscript` replaces it with what was actually
 * persisted. The disk is the source of truth, not the DOM.
 */

import { api, ApiError, streamChat } from './api.js';
import { markdown, toPlainText, escapeHtml } from './markdown.js';
import {
  blocksOf, isMediaBlock, mediaLabel, elementForBlock, mediaRow, kindForFile, uiKindForFile, formatBytes, extOf,
  readFileListing, copyText as copyToClipboard, revokePreview,
} from './media.js';
import { detectLanguage, languageForName, languageLabel } from './highlight.js';
import {
  formatHeaders, parseHeaders, mcpApiKeyOf, mcpHeadersWithoutKey, withApiKey,
} from './mcpheaders.js';
import { prefs, savePrefs, app, currentModel, models, isConfigured, visionBlocked } from './state.js';

const $ = (id) => document.getElementById(id);
const NARROW = () => window.innerWidth <= 820;

/* ── 1. helpers ─────────────────────────────────────────────────────────── */

function h(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function fmtTokens(n) {
  const value = Number(n) || 0;
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}k`;
  return `${(value / 1_000_000).toFixed(2)}M`;
}

function clockTime(seconds) {
  const value = Number(seconds);
  if (!value) return '';
  const date = new Date(value * 1000);
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function relativeTime(seconds) {
  const value = Number(seconds);
  if (!value) return '';
  const delta = Date.now() / 1000 - value;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)} d ago`;
  return new Date(value * 1000).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
}

let toastId = 0;
function toast(message, kind = '', ms = 4200) {
  const node = h('div', `toast ${kind}`.trim(), message);
  node.dataset.id = String(++toastId);
  $('toasts').append(node);
  setTimeout(() => {
    node.style.transition = 'opacity .25s';
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 260);
  }, ms);
}

/** The clipboard lives in media.js so the code chip can use it too; this wraps it to
 *  add the toast that only the transcript wants. */
async function copyText(text, button) {
  const ok = await copyToClipboard(text, button);
  if (!ok) toast('the clipboard is not available in this context', 'warn');
  return ok;
}

/** Shrink a textarea to fit its content, up to a ceiling. */
function autogrow(textarea, max = 240) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, max)}px`;
}

/* ── 2. rendering ───────────────────────────────────────────────────────── */

function openLightbox(url) {
  const box = h('div', 'lightbox');
  const img = h('img');
  img.src = url;
  img.alt = '';
  box.append(img);
  const close = () => box.remove();
  box.addEventListener('click', close);
  document.addEventListener('keydown', function onKey(event) {
    if (event.key === 'Escape') { close(); document.removeEventListener('keydown', onKey); }
  });
  document.body.append(box);
}

/** Prose with click-to-copy on every code block. */
function proseElement(text) {
  const wrapper = h('div', 'prose');
  wrapper.innerHTML = markdown(text);
  wrapper.querySelectorAll('.code-copy').forEach((button) => {
    button.addEventListener('click', () => {
      const block = button.closest('.code-block');
      // `__source` is set when the block is line-numbered, because then
      // `textContent` would hand the clipboard a gutter along with the code.
      const source = block?.__source ?? block?.querySelector('code')?.textContent;
      if (source != null) copyText(source, button);
    });
  });
  wrapper.querySelectorAll('a[href^="/memory/"]').forEach((anchor) => {
    if (/\.(png|jpe?g|gif|webp|mp4|webm|mov|mkv)$/i.test(anchor.getAttribute('href') || '')) {
      anchor.removeAttribute('target');
    }
  });
  return wrapper;
}

function mediaGrid(blocks) {
  // One item is a grid, several are a carousel — see `mediaRow`. Which is the only
  // thing that varies: the elements themselves are built the same way either side.
  return mediaRow(blocks, { onZoom: openLightbox });
}

/**
 * The media a tool showed to *you* on purpose rather than media it merely collected.
 *
 * The server cuts these out of the tool result, stores them on the assistant turn as
 * `_display`, and they render above the reply — the point of asking to see something
 * is to have it be the first thing you look at, not a card you have to open. The key
 * is deliberately underscore-prefixed, which is what keeps it out of the request.
 *
 * Only what was asked for ends up here. An image ``web_fetch`` pulled off the page it
 * read was not asked for, so the server leaves it in the tool result and it is drawn
 * inside that card, which is collapsed — see `fillToolBody`. A reply is what the
 * reader is here for, and a row of page posters parked above it is attention taken
 * from the reply for the entire life of the transcript.
 */
function displayMedia(blocks) {
  if (!Array.isArray(blocks) || !blocks.length) return null;
  const grid = mediaGrid(blocks);
  if (grid) grid.classList.add('shown-media');
  return grid;
}

/**
 * Render a message's blocks into `container`: text first, then media, then code.
 *
 * `mediaFirst` puts the media above the text instead. Only a tool card asks for it:
 * a fetched page arrives as 60 kB of body text followed by whatever pictures came with
 * it, and the pictures at the bottom of that are pictures nobody scrolls to. In the
 * reply the order stays text-first, because there the prose is the thing being read.
 */
function renderBlocks(container, blocks, { mediaFirst = false } = {}) {
  const texts = blocks.filter((b) => b.type === 'text');
  const media = blocks.filter(isMediaBlock);
  const code = blocks.filter((b) => b.type === 'code');
  const others = blocks.filter((b) => b.type !== 'text' && b.type !== 'code' && !isMediaBlock(b));

  const grid = media.length ? mediaGrid(media) : null;
  const text = texts.length ? proseElement(texts.map((b) => b.text).join('\n\n')) : null;
  if (mediaFirst && grid) container.append(grid);
  if (text) container.append(text);
  if (!mediaFirst && grid) container.append(grid);
  if (code.length) {
    // Code gets its own full-width column: a chip is a paragraph, not a thumbnail,
    // and it looks wrong wrapped into the media grid's flex row.
    const stack = h('div', 'code-stack');
    code.forEach((block) => stack.append(elementForBlock(block)));
    container.append(stack);
  }
  if (others.length) {
    const row = h('div', 'media-grid');
    others.forEach((block) => row.append(elementForBlock(block)));
    if (row.children.length) container.append(row);
  }
}

/* Formats whose detector is structural enough that reasonable prose does not trip it.
 * YAML is deliberately missing: `Key: value` is a sentence pattern too, and turning a
 * paragraph of tool output into a code block is a worse failure than not highlighting
 * it. JSON is decided by `JSON.parse` actually succeeding, so it is exact. */
const AUTO_OUTPUT_LANGS = new Set(['json', 'diff', 'sql', 'html', 'css']);

/**
 * Turn a tool result's text into a code block when — and only when — it is code.
 *
 * `read_file` is the common case: its listing is recognised structurally and its
 * gutter is moved into real line numbers. Otherwise the block is highlighted only if
 * it is one of a few formats prose cannot plausibly imitate.
 */
function outputBlocks(blocks, args) {
  return blocks.map((block) => {
    if (block.type !== 'text') return block;
    const listing = readFileListing(block.text);
    if (listing) {
      const path = (args && (args.path || args.file || args.filename)) || listing.name;
      return {
        type: 'code',
        text: listing.code,
        name: path || '',
        lang: languageForName(path) || detectLanguage(listing.code),
        numbered: true,
        startLine: listing.startLine,
        note: listing.note,
      };
    }
    const lang = detectLanguage(block.text);
    if (!AUTO_OUTPUT_LANGS.has(lang)) return block;
    return { type: 'code', text: block.text, lang, name: '' };
  });
}

/* ── collapsed tool cards ────────────────────────────────────────────────
 * A card shows its header and nothing else until it is clicked: the name, whether
 * the call failed and the step it ran on. Everything else in a tool card is
 * arguments and file listings, which is most of a long transcript and read far
 * less often than the answer.
 *
 * The collapse state is a class on the card, not the panes' `hidden` flags, because
 * content arrives after the click: `applyToolCollapsed` is re-run each time a pane
 * gains something, so a card opened while it was still running stays open.
 *
 * It is also remembered per call id. The transcript is re-rendered from disk after
 * every turn and whenever a conversation is opened, so without this a click would be
 * undone by the very next render. (A tool message and the tool card painted live
 * during streaming carry the same call id, so the state survives the swap.)
 */
function toolExpanded(callId) {
  if (!callId) return false;
  const per = prefs.expandedTools && prefs.expandedTools[app.conv?.uuid || ''];
  return Boolean(per && per[callId]);
}

function rememberToolExpanded(callId, open) {
  if (!callId) return;
  const uuid = app.conv?.uuid || '';
  if (!prefs.expandedTools || typeof prefs.expandedTools !== 'object') prefs.expandedTools = {};
  const per = prefs.expandedTools[uuid] || (prefs.expandedTools[uuid] = {});
  if (open) per[callId] = true;
  else delete per[callId];
  if (!Object.keys(per).length) delete prefs.expandedTools[uuid];
  savePrefs();
}

/** Mirror the card's class onto its two collapsible panes. A pane with nothing in it
 *  (`data-has` unset) stays hidden even when the card is open, so clicking a card
 *  that only ever produced arguments does not open an empty box. */
function applyToolCollapsed(card) {
  const open = card.classList.contains('open');
  const args = card.querySelector('.tool-args');
  const body = card.querySelector('.tool-body');
  if (args) args.hidden = !open || !args.dataset.has;
  if (body) body.hidden = !open || !body.dataset.has;
}

function toolCard(name, callId) {
  const card = h('div', 'tool-card');
  card.dataset.callId = callId || '';
  card.innerHTML =
    '<div class="tool-head">'
    + '<span class="spin-dot"></span>'
    + `<span class="tool-name">${escapeHtml(name || 'tool')}</span>`
    + '<span class="tool-status">calling…</span>'
    + '</div>'
    + '<pre class="tool-args" hidden></pre>'
    + '<div class="tool-body" hidden></div>';
  if (toolExpanded(callId)) card.classList.add('open');
  applyToolCollapsed(card);
  card.querySelector('.tool-head').addEventListener('click', () => {
    const open = !card.classList.contains('open');
    card.classList.toggle('open', open);
    rememberToolExpanded(card.dataset.callId, open);
    applyToolCollapsed(card);
  });
  return card;
}

function setToolArguments(card, args) {
  if (!args || !Object.keys(args).length) return;
  const node = card.querySelector('.tool-args');
  node.textContent = JSON.stringify(args, null, 2);
  node.dataset.has = '1';
  // Deliberately not shown: the card decides, and it is closed unless clicked.
  applyToolCollapsed(card);
  // Kept for the result that follows: `read_file`'s output is a line-numbered listing
  // with no filename in it, and the arguments are where the path comes from.
  card.__args = args;
}

function finishToolCard(card, { isError, blocks, step }) {
  card.classList.remove('running');
  if (isError) card.classList.add('error');
  // finishToolCard can be reached twice (a tool_result for a card that was never
  // announced, then the same id again) — the dot may already be gone.
  card.querySelector('.spin-dot')?.remove();

  const status = card.querySelector('.tool-status');
  status.textContent = isError ? 'failed' : (step ? `done · step ${step}` : 'done');
  if (isError) status.classList.add('err');

  fillToolBody(card, blocks);
}

/**
 * Fill a tool card's body from the blocks of its result, and say in the header how
 * much media is in there.
 *
 * The body is emptied rather than removed: `finishToolCard` can be reached twice for
 * one call id, and a *collapsed* card still needs the pane in the DOM for the click
 * handler to have something to reveal.
 *
 * Media that arrived with the result is drawn here, inside the card, and the card is
 * closed by default — so the header has to mention it. A header that says `web_fetch
 * done · step 2` over a pile of page images reads as a call that found nothing worth
 * looking at, which is exactly how the same images became a permanent fixture above
 * the reply: they had to be visible somewhere. Saying so in the header is the visible
 * part; the images themselves wait for the click, like the page text beside them.
 */
function fillToolBody(card, blocks) {
  const body = card.querySelector('.tool-body');
  if (!body) return;
  card.querySelector('.tool-media')?.remove();
  body.textContent = '';
  delete body.dataset.has;

  // `blocksOf` is applied to both callers' shapes: the live card holds the SSE blocks
  // as they came off the wire, the re-rendered one holds blocks already normalised
  // from the stored message, and the pass is idempotent either way.
  const rendered = blocks && blocks.length ? outputBlocks(blocksOf(blocks), card.__args) : [];
  if (rendered.length) {
    renderBlocks(body, rendered, { mediaFirst: true });
    body.dataset.has = '1';
  }
  const label = mediaLabel(rendered);
  if (label) {
    // Left of the status rather than after it: the status is the card's fixed anchor in
    // the right-hand corner, and a badge that came and went beside it would move it.
    const head = card.querySelector('.tool-head');
    const badge = h('span', 'tool-media', label);
    badge.title = `${label} came in with this result — open the card to look`;
    if (head) head.insertBefore(badge, head.querySelector('.tool-status'));
  }
  applyToolCollapsed(card);
}

function turnShell(role, label) {
  const turn = h('section', `turn ${role}`);
  const inner = h('div', 'turn-inner');
  const head = h('div', 'turn-head');
  head.innerHTML = `<span class="role-dot"></span><span>${escapeHtml(label)}</span>`;
  inner.append(head);
  turn.append(inner);
  return { turn, inner, head };
}

function renderMessage(message, { onEdit = null, onRegenerate = null, isLast = false } = {}) {
  const role = String(message.role || 'user');
  const blocks = blocksOf(message.content);

  if (role === 'tool') {
    const { turn, inner } = turnShell('tool', `tool · ${message.name || 'result'}`);
    turn.append(inner);
    const card = toolCard(message.name || 'tool', message.tool_call_id || '');
    card.classList.remove('running');
    card.querySelector('.spin-dot')?.remove();
    const status = card.querySelector('.tool-status');
    status.textContent = 'result';
    // The same pane the live card is filled by, so the streaming paint and the render
    // after it cannot drift — which is where media in the body would visibly jump.
    fillToolBody(card, blocks);
    inner.append(card);
    return turn;
  }

  const label = role === 'assistant' ? 'assistant' : 'you';
  const { turn, inner, head } = turnShell(role === 'assistant' ? 'assistant' : 'user', label);
  const time = h('span', 'head-time', clockTime(message.created));
  head.append(time);

  const bubble = h('div', 'bubble');

  // Reasoning is stored on the assistant turn and echoed back to the API on the
  // next request, so it is replayed here rather than hidden.
  const reasoning = message.reasoning_content || message.reasoning || '';
  if (role === 'assistant' && reasoning) {
    const details = h('details', 'reasoning');
    details.open = prefs.expandReasoning;
    details.innerHTML = '<summary>reasoning</summary>';
    const body = h('div', 'reasoning-body', reasoning);
    details.append(body);
    bubble.append(details);
  }

  const shown = role === 'assistant' ? displayMedia(message._display) : null;
  if (shown) bubble.append(shown);

  if (blocks.length) renderBlocks(bubble, blocks);

  if (role === 'assistant' && Array.isArray(message.tool_calls) && message.tool_calls.length && !blocks.length) {
    bubble.append(h('p', 'dim', `called ${message.tool_calls.map((c) => c.function?.name || c.name || 'tool').join(', ')}`));
  }
  if (!bubble.children.length) bubble.append(h('p', 'dim', role === 'assistant' ? '(empty)' : ''));

  inner.append(bubble);

  /* hover actions */
  const actions = h('div', 'msg-actions');

  const copy = h('button', 'icon-btn');
  copy.title = 'Copy';
  copy.innerHTML = '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5h10"/></svg>';
  copy.addEventListener('click', () => copyText(toPlainText(blocks.filter((b) => b.type === 'text').map((b) => b.text).join('\n\n') || reasoning)));
  actions.append(copy);

  if (role === 'user' && onEdit) {
    const edit = h('button', 'icon-btn');
    edit.title = 'Edit and resend — puts the message back in the box';
    edit.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
    edit.addEventListener('click', () => onEdit(message));
    actions.append(edit);
  }

  if (role === 'assistant' && onRegenerate && isLast) {
    const again = h('button', 'icon-btn');
    again.title = 'Regenerate — drops this reply and asks again';
    again.innerHTML = '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 6.3"/><path d="M20 5v6h-6"/></svg>';
    again.addEventListener('click', onRegenerate);
    actions.append(again);
  }

  if (actions.children.length && blocks.length) {
    bubble.after(actions);
  }
  turn.append(inner);
  return turn;
}

function renderTranscript() {
  const target = $('transcript');
  target.textContent = '';
  const messages = app.conv?.messages || [];

  if (!messages.length) {
    target.hidden = true;
    $('empty-state').hidden = false;
    renderHints();
    renderEmptySetup();
    return;
  }
  target.hidden = false;
  $('empty-state').hidden = true;

  messages.forEach((message, index) => {
    const isLast = index === messages.length - 1;
    target.append(renderMessage(message, {
      onEdit: message.role === 'user' ? editAndResend : null,
      onRegenerate: regenerate,
      isLast,
    }));
  });

  // A failed turn persists the user message but no reply, so the failure notice has
  // to be re-attached on every render or it is wiped the moment the disk is re-read.
  if (app.turnError && app.turnError.uuid === app.conv?.uuid) {
    target.append(turnErrorLine(app.turnError));
  }

  scrollToBottom(true);
}

/** The one-line notice shown when a turn did not complete. */
function turnErrorLine({ message, hint }) {
  const box = h('div', 'error-line', message);
  if (hint) box.append(h('span', 'hint', hint));
  return box;
}

function renderHints() {
  const container = $('empty-hints');
  if (container.dataset.done) return;
  container.dataset.done = '1';
  const model = currentModel();
  const suggestions = [
    'Summarise this screenshot and list anything that looks wrong',
    model?.vision === false ? 'Explain how context caching works' : 'Describe what is happening in this video',
    'Show me three ways to resize an image before upload and their tradeoffs',
  ];
  for (const text of suggestions) {
    const chip = h('button', 'hint', text);
    chip.type = 'button';
    chip.addEventListener('click', () => {
      $('input').value = text;
      autogrow($('input'));
      $('input').focus();
    });
    container.append(chip);
  }
}

/** Swap the empty state's suggestion chips for the "no key yet" instructions. */
function renderEmptySetup() {
  const box = $('empty-setup');
  const blocked = !isConfigured();
  box.hidden = !blocked;
  $('empty-hints').hidden = blocked;
  if (!blocked) return;
  // The server's own message, so a broken providers.json is not reported as a
  // missing key.
  $('empty-setup-why').textContent = app.props?.error || '';
}

function scrollToBottom(force = false) {
  const target = $('transcript');
  if (target.hidden) return;
  const nearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 140;
  if (force || (prefs.autoscroll && nearBottom)) {
    target.scrollTop = target.scrollHeight;
  }
}

/* ── context meter ─────────────────────────────────────────────────────── */

function renderContext(context) {
  const bar = $('context-bar');
  if (!context || !context.limit) {
    bar.hidden = true;
    return;
  }
  app.context = context;
  const ratio = Math.max(0, Math.min(1, Number(context.ratio) || 0));
  const safe = Number(app.props?.limits?.context_safety_ratio) || 0.92;

  bar.hidden = false;
  bar.classList.toggle('warn', ratio >= safe * 0.85 && ratio < safe);
  bar.classList.toggle('hot', ratio >= safe);
  $('context-fill').style.width = `${(ratio * 100).toFixed(1)}%`;
  $('context-text').textContent =
    `${fmtTokens(context.tokens)} / ${fmtTokens(context.limit)} tokens · ${(ratio * 100).toFixed(0)}% · ${context.messages || 0} messages`;
}

function renderTitle() {
  const title = app.conv ? (app.conv.title || 'New conversation') : 'No conversation';
  $('conv-title').textContent = title;
  $('conv-title').title = app.conv ? 'Click to rename' : '';
}

function renderMeta() {
  const model = currentModel();
  const conv = app.conv;
  const bits = [];
  if (model) {
    bits.push(`<b>${escapeHtml(model.id)}</b>`);
    if (model.tool_calling) bits.push('tools');
    if (model.vision) bits.push('vision');
    else bits.push('<span style="color:var(--warn)">no vision</span>');
  }
  if (conv?.updated) bits.push(relativeTime(conv.updated));
  $('conv-meta').innerHTML = bits.join(' · ');
}

/* ── 3. the live turn ──────────────────────────────────────────────────── */

/**
 * Paint one streaming turn.
 *
 * Everything the server can emit is handled here. A failure that happens after
 * HTTP 200 arrives as an `error` event, which is why the error path is a normal
 * branch rather than a catch around the whole thing.
 */
function createTurnPainter() {
  const transcript = $('transcript');
  transcript.hidden = false;
  $('empty-state').hidden = true;

  const { turn, inner, head } = turnShell('assistant', 'assistant');
  head.insertAdjacentHTML('beforeend', '<span class="head-time">streaming</span>');
  transcript.append(turn);
  scrollToBottom(true);

  let reasoningDetails = null;
  let reasoningBody = null;
  let reasoningText = '';
  const reasonings = h('div');
  const stream = h('div', 'prose live-body');
  // Media the tools hand over mid-turn. It has to land above the streaming text,
  // because the disk re-render at the end of the turn puts it there — and a turn that
  // jumped around while it ran would be worse than no live paint at all.
  const shownRow = h('div', 'media-grid shown-media');
  const shownBlocks = [];
  let streamText = '';
  const events = h('div');
  const cards = new Map();
  let usage = null;
  let dirty = false;

  const bubble = h('div', 'bubble');
  bubble.append(events);
  inner.append(bubble);

  const schedule = () => {
    if (dirty) return;
    dirty = true;
    requestAnimationFrame(() => {
      dirty = false;
      if (reasoningBody && reasoningText) reasoningBody.textContent = reasoningText;
      if (streamText) {
        stream.innerHTML = `${markdown(streamText)}<span class="caret"></span>`;
        stream.querySelectorAll('.code-copy').forEach((button) => {
          button.addEventListener('click', () => {
            const code = button.closest('.code-block')?.querySelector('code');
            if (code) copyText(code.textContent, button);
          });
        });
      }
      scrollToBottom();
    });
  };

  const ensureStream = () => {
    if (stream.isConnected) return;
    events.append(stream);
  };

  return {
    get text() { return streamText; },

    step(step, maxSteps) {
      events.append(h('div', 'worked', `step ${step} of ${maxSteps}`));
      scrollToBottom();
    },

    warning(message) {
      events.append(h('div', 'warn-line', message));
      scrollToBottom(true);
    },

    error(message, hint) {
      const box = h('div', 'error-line', message);
      if (hint) box.append(h('span', 'hint', hint));
      // A partial answer plus an error is more useful than losing the turn, so the
      // stream stays above the error box rather than being replaced by it.
      events.append(box);
      scrollToBottom(true);
    },

    reasoning(text) {
      if (!reasoningDetails) {
        reasoningDetails = h('details', 'reasoning live');
        // Collapsed while it streams too. The tinted summary is the signal that the
        // model is thinking; the text itself is only interesting afterwards, and an
        // auto-expanding block fights the answer for the same screen space.
        reasoningDetails.open = prefs.expandReasoning;
        reasoningDetails.innerHTML = '<summary>reasoning…</summary>';
        reasoningBody = h('div', 'reasoning-body');
        reasoningDetails.append(reasoningBody);
        reasonings.append(reasoningDetails);
        events.after(reasonings);
      }
      reasoningText += text;
      schedule();
    },

    content(text) {
      streamText += text;
      ensureStream();
      schedule();
    },

    toolPending(id) {
      if (!id) return;
      const card = toolCard('tool', id);
      card.classList.add('running');
      cards.set(id, card);
      events.append(card);
      scrollToBottom();
    },

    toolCall(id, name, args, rawArguments) {
      const card = cards.get(id) || toolCard(name, id);
      if (!cards.has(id)) {
        card.classList.add('running');
        cards.set(id, card);
        events.append(card);
      }
      const nameNode = card.querySelector('.tool-name');
      if (nameNode) nameNode.textContent = name || 'tool';
      if (args && Object.keys(args).length) setToolArguments(card, args);
      else if (rawArguments) {
        const node = card.querySelector('.tool-args');
        node.textContent = rawArguments;
        node.dataset.has = '1';
        applyToolCollapsed(card);
      }
      scrollToBottom();
    },

    toolResult(id, name, isError, blocks, step, media) {
      let card = cards.get(id);
      if (!card) {
        card = toolCard(name, id);
        cards.set(id, card);
        events.append(card);
      }
      finishToolCard(card, { isError, blocks, step });
      if (Array.isArray(media) && media.length) {
        // Prepended, not appended: the transcript re-render at the end of the turn
        // builds media-before-cards-before-text from disk, and the live paint should
        // not visibly reshuffle when that happens.
        shownBlocks.push(...media.filter(isMediaBlock));
        // Rebuilt rather than appended to, because the row has to become a carousel
        // the moment it holds a second item and there is no way to grow one in place.
        // Every media block a tool returns arrives in the same batch, so this is one
        // rebuild per tool call, and the end-of-turn re-render replaces it anyway.
        const row = mediaRow(shownBlocks, { onZoom: openLightbox });
        if (row) {
          if (!shownRow.isConnected) events.prepend(shownRow);
          // `row.className` is a replacement, so the class that marks this row as tool
          // media rather than something the reply said has to be put back.
          shownRow.className = `${row.className} shown-media`;
          // A snapshot of the children, because moving them mutates the live list.
          shownRow.replaceChildren(...row.children);
        }
      }
      scrollToBottom();
    },

    setUsage(value) {
      usage = value;
    },

    finish({ steps, toolCalls, truncated } = {}) {
      if (reasoningDetails) {
        reasoningDetails.classList.remove('live');
        const summary = reasoningDetails.querySelector('summary');
        if (summary) summary.textContent = 'reasoning';
        reasoningDetails.open = prefs.expandReasoning;
      }
      if (!streamText && !cards.size) {
        stream.innerHTML = '<p class="dim">(no reply)</p>';
        ensureStream();
      }
      if (streamText) schedule();

      const bits = [];
      if (steps) bits.push(`${steps} step${steps > 1 ? 's' : ''}`);
      if (toolCalls) bits.push(`${toolCalls} tool call${toolCalls > 1 ? 's' : ''}`);
      if (usage?.total_tokens) bits.push(`${fmtTokens(usage.total_tokens)} tokens`);
      if (usage?.reasoning_tokens) bits.push(`${fmtTokens(usage.reasoning_tokens)} reasoning`);
      if (truncated) bits.push('stopped early at the step limit');
      if (bits.length) events.append(h('div', 'worked', bits.join(' · ')));

      scrollToBottom();
    },

    fail(message, hint) {
      // Remembered on `app` so the re-render in runTurn's `finally` cannot erase the
      // only evidence that the turn failed.
      app.turnError = {
        uuid: app.conv?.uuid,
        message: message || 'the request failed',
        hint: hint || '',
      };
      events.append(turnErrorLine(app.turnError));
      scrollToBottom(true);
    },
  };
}

/* ── turn lifecycle ────────────────────────────────────────────────────── */

async function runTurn({ content, displayContent, resend = false } = {}) {
  if (!app.conv) return;
  if (app.streaming) { toast('a turn is already running', 'warn'); return; }
  if (!isConfigured()) {
    toast('there is no API key listed for Deepseek — open Settings to paste one in', 'err', 6000);
    return;
  }

  const uuid = app.conv.uuid;
  const model = currentModel();
  const body = {
    model: model?.id || undefined,
    tools: prefs.tools,
    reasoning_effort: prefs.effort || undefined,
  };

  app.turnError = null;
  app.streaming = true;
  setComposerBusy(true);

  // Optimistic: paint the user turn immediately so the send feels instant.
  try {
    if (content !== undefined && content !== null) {
      const optimistic = { role: 'user', content: displayContent ?? content };
      app.conv.messages = [...(app.conv.messages || []), optimistic];
      renderTranscript();
      body.content = content;
    }
  } catch (err) {
    /* rendering must never block the request */
  }

  app.controller = new AbortController();
  const painter = createTurnPainter();

  try {
    await streamChat(uuid, body, (event, data) => {
      switch (event) {
        case 'meta':
          if (data.tools) app.liveTools = data.tools;
          break;
        case 'warning':
          painter.warning(data.message || 'warning');
          break;
        case 'step':
          painter.step(data.step, data.max_steps);
          break;
        case 'reasoning':
          painter.reasoning(data.text || '');
          break;
        case 'content':
          painter.content(data.text || '');
          break;
        case 'tool_pending':
          painter.toolPending(data.id);
          break;
        case 'tool_call':
          painter.toolCall(data.id, data.name, data.arguments, data.raw_arguments);
          break;
        case 'tool_result':
          painter.toolResult(data.id, data.name, Boolean(data.is_error), data.blocks, data.step, data.media);
          break;
        case 'usage':
          painter.setUsage(data);
          break;
        case 'title':
          app.conv.title = data.title || app.conv.title;
          renderTitle();
          break;
        case 'error':
          painter.fail(data.message || 'the request failed', data.hint || '');
          break;
        case 'done':
          painter.finish(data);
          break;
        default:
          break;
      }
    }, { signal: app.controller.signal });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      painter.warning('stopped');
    } else if (err instanceof ApiError) {
      painter.fail(err.message);
      if (err.status === 0) toast('lost contact with the local server', 'err', 6000);
    } else {
      painter.fail(String(err && err.message ? err.message : err));
    }
  } finally {
    app.controller = null;
    app.streaming = false;
    setComposerBusy(false);

    // The disk is the source of truth: replace the optimistic paint with what the
    // server actually persisted, and pick up the new context figures.
    try {
      await loadConversation(uuid, { keepScroll: false, silent: true });
    } catch { /* the transcript stays as painted */ }
    await refreshList({ silent: true });
  }
}

async function send() {
  const input = $('input');
  const text = input.value.trim();
  const files = app.attachments.slice();

  if (app.streaming) return;
  if (!text && !files.length) return;

  // Refuse before uploading files or appending anything. `runTurn` would refuse too,
  // but by then the attachments would already be on disk and the user turn would be
  // persisted with no reply — a stranded conversation.
  if (!isConfigured()) {
    toast('there is no API key listed for Deepseek — open Settings to paste one in', 'err', 8000);
    return;
  }

  if (!app.conv) {
    await createConversation();
    if (!app.conv) return;                      // creation failed and already toasted
  }

  const uuid = app.conv.uuid;

  // An edit rewrites history: the original message and everything after it are
  // dropped before the replacement is appended, so it lands where it was. The
  // rewind is deliberately here and not in the pencil's click handler — clicking
  // only opens the box, and nothing is destroyed until the user actually sends.
  const editing = pendingEditFor(uuid);
  if (editing) {
    clearPendingEdit();
    try {
      await api.truncate(uuid, { keep: editing.index });
      await loadConversation(uuid, { silent: true });
    } catch (err) {
      toast(err.message || 'could not rewind the conversation', 'err');
      return;
    }
    // The message's own media went with it. Re-post it so editing the text of a
    // screenshot does not silently drop the screenshot; files staged in the
    // composer now win over the old ones (the user is replacing, not adding).
    // `editing.media` holds the server's own blocks, not `blocksOf`' output, so
    // the shapes the ingest recognises survive the round trip (see
    // `carriedBlocksOf`).
    if (editing.media.length && !files.length) {
      const carried = editing.media.slice();
      if (text) carried.push({ type: 'text', text });
      try {
        await api.appendMessage(uuid, carried);
        await loadConversation(uuid, { silent: true });
      } catch (err) {
        toast(err.message || 'could not resend', 'err');
        return;
      }
      input.value = '';
      autogrow(input);
      await runTurn({ content: undefined });
      return;
    }
  }

  // Files go up first and the message is assembled from their URLs, so all the
  // attachments in one send land in a single user turn rather than several.
  let blocks = [];
  if (files.length) {
    if (visionBlocked() && files.some((f) => kindForFile(f) === 'image')) {
      toast('this model cannot see images — switch to a vision model or remove the image', 'err', 6000);
      return;
    }
    const button = $('send-btn');
    const original = button.innerHTML;
    button.innerHTML = '<span class="spinner" style="width:15px;height:15px;margin:0"></span>';
    try {
      for (const [index, file] of files.entries()) {
        const kind = kindForFile(file);
        const result = await api.upload(uuid, file, kind);
        blocks.push(result.block);
        button.style.opacity = String(0.6 + (0.4 * (index + 1)) / files.length);
      }
    } catch (err) {
      toast(err.message || 'the upload failed', 'err', 6000);
      button.innerHTML = original;
      button.style.opacity = '';
      return;
    }
    button.innerHTML = original;
    button.style.opacity = '';
    clearAttachments();
  }

  input.value = '';
  autogrow(input);

  let payload;
  let display;
  if (blocks.length) {
    if (text) blocks.push({ type: 'text', text });
    payload = blocks;
    display = blocks;
  } else {
    payload = text;
    display = text;
  }

  if (blocks.length) {
    // Persist the user turn here so `runTurn` only has to stream.
    try {
      await api.appendMessage(uuid, payload);
      await loadConversation(uuid, { silent: true });
      await runTurn({ content: undefined });
      return;
    } catch (err) {
      toast(err.message || 'could not append the message', 'err', 6000);
      return;
    }
  }

  await runTurn({ content: payload, displayContent: display });
}

async function regenerate() {
  if (!app.conv || app.streaming) return;
  // Regenerating cuts the transcript too, so an edit that was waiting to be sent
  // is abandoned rather than left pointing at an index that no longer exists.
  if (app.pendingEdit) clearPendingEdit({ clearInput: true });
  const messages = app.conv.messages || [];
  let cut = messages.length;
  while (cut > 0 && messages[cut - 1].role !== 'user') cut -= 1;
  if (cut === 0) { toast('nothing to regenerate from', 'warn'); return; }

  try {
    await api.truncate(app.conv.uuid, { keep: cut });
    await loadConversation(app.conv.uuid, { silent: true });
  } catch (err) {
    toast(err.message || 'could not rewind the conversation', 'err');
    return;
  }
  await runTurn({ content: undefined });
}

/** The edit waiting to be sent, if it belongs to this conversation. */
function pendingEditFor(uuid) {
  const pending = app.pendingEdit;
  return pending && uuid && pending.uuid === uuid ? pending : null;
}

/**
 * Leave edit mode. Nothing has been written yet, so cancelling is free.
 *
 * The composer is only emptied when it still holds the text the pencil put there:
 * once the user has typed, that text is theirs and must survive a cancel.
 */
function clearPendingEdit({ clearInput = false } = {}) {
  const pending = app.pendingEdit;
  app.pendingEdit = null;
  $('composer').classList.remove('editing');
  if (clearInput && pending && $('input').value === pending.text) {
    $('input').value = '';
    autogrow($('input'));
  }
  updateComposerNote();
}

/**
 * Everything in a message except its prose, exactly as the server sent it.
 *
 * `blocksOf` is a *presentation* normaliser: it rewrites our own `_file` blocks
 * (video, audio, attachments) into bare `video`/`audio`/`file` shapes, turns
 * `image_url` into `image`, and turns a `_file kind=text` into a `code` block with
 * no URL. `media.ingest_user_content` only recognises `_file`/`file`/`image_url`
 * and passes anything else through verbatim, so re-posting normalised blocks loses
 * what the ingest needs: an attached text or code file comes back unrecognised (and
 * its bytes would be dropped), and the media is re-sent in a shape the server has
 * no rule for. The raw blocks keep the round trip faithful; only the text is the
 * user's to rewrite. Display affordances (`_sources`, `_tool`, …) are not the
 * message, and `text` is handled separately.
 */
function carriedBlocksOf(content) {
  if (!Array.isArray(content)) return [];
  return content.filter((block) => {
    if (!block || typeof block !== 'object') return false;
    const type = String(block.type || '');
    if (type === 'text') return false;
    return !(type.startsWith('_') && type !== '_file');
  });
}

/**
 * Put an earlier user message back in the composer instead of sending it again.
 *
 * The pencil used to truncate and resend in one go, which made "edit" a lie: the
 * text was in the box only for the instant it took `send()` to clear it. Nothing is
 * destroyed here — the rewind happens in `send()`, so the message can be rewritten
 * (or the whole thing abandoned) before the transcript is cut.
 */
function editAndResend(message) {
  if (!app.conv || app.streaming) return;
  const messages = app.conv.messages || [];
  const index = messages.indexOf(message);
  if (index < 0) return;

  const blocks = blocksOf(message.content);
  const text = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('\n\n');
  const media = carriedBlocksOf(message.content);
  if (!text && !media.length) { toast('this message has nothing to edit', 'warn'); return; }

  if (!window.confirm('Edit this message? Sending the new text drops this message and everything after it.')) return;

  app.pendingEdit = { uuid: app.conv.uuid, index, text, media };
  const input = $('input');
  input.value = text;
  autogrow(input);
  input.focus();
  // Cursor at the end, so typing continues the message rather than replacing it.
  input.setSelectionRange(text.length, text.length);
  $('composer').classList.add('editing');
  updateComposerNote();
  toast(`editing — press ${prefs.enterSends ? 'Enter' : 'Ctrl+Enter'} to send, Esc to cancel`, 'ok', 5000);
}

/* ── 4. conversations ──────────────────────────────────────────────────── */

async function refreshList({ silent = false } = {}) {
  try {
    const data = await api.listConversations();
    app.conversations = data.conversations || [];
    renderList();
  } catch (err) {
    if (!silent) toast(err.message || 'could not list conversations', 'err');
  }
}

function renderList() {
  const container = $('conv-list');
  const query = ($('search').value || '').trim().toLowerCase();
  container.textContent = '';

  const items = app.conversations.filter((conv) => {
    if (!query) return true;
    return String(conv.title || '').toLowerCase().includes(query)
      || String(conv.preview || '').toLowerCase().includes(query)
      || String(conv.uuid || '').toLowerCase().includes(query);
  });

  if (!items.length) {
    container.append(h('p', 'list-note', query ? 'No matches.' : 'No conversations yet.'));
    return;
  }

  for (const conv of items) {
    const item = h('div', 'conv-item');
    if (app.conv && conv.uuid === app.conv.uuid) item.classList.add('active');
    item.tabIndex = 0;
    item.setAttribute('role', 'button');

    item.append(h('div', 'conv-item-title', conv.title || 'New conversation'));
    const sub = [conv.model, relativeTime(conv.updated)].filter(Boolean).join(' · ');
    item.append(h('div', 'conv-item-sub', sub || conv.uuid.slice(0, 8)));

    const actions = h('div', 'conv-item-actions');

    const rename = h('button', 'icon-btn');
    rename.title = 'Rename';
    rename.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
    rename.addEventListener('click', (event) => {
      event.stopPropagation();
      beginRename(conv);
    });
    actions.append(rename);

    const remove = h('button', 'icon-btn danger');
    remove.title = 'Delete';
    remove.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteConversation(conv);
    });
    actions.append(remove);

    item.append(actions);

    const open = () => loadConversation(conv.uuid);
    item.addEventListener('click', open);
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    });

    container.append(item);
  }
}

async function createConversation({ focus = true } = {}) {
  const model = currentModel();
  if (app.pendingEdit) clearPendingEdit({ clearInput: true });
  try {
    const conv = await api.createConversation({
      model: model?.id || '',
      sys_base: prefs.defaultSystem || '',
    });
    app.conv = conv;
    app.context = null;
    prefs.lastUuid = conv.uuid;
    savePrefs();
    renderTitle();
    renderTranscript();
    renderMeta();
    renderContext(null);
    await refreshList();
    // Narrow viewports show the sidebar as an overlay, so it has to get out of the
    // way of the conversation it just created.
    if (NARROW()) $('shell').classList.remove('mobile-open');
    if (focus) $('input').focus();
    return conv;
  } catch (err) {
    toast(err.message || 'could not create a conversation', 'err');
    return null;
  }
}

async function loadConversation(uuid, { silent = false, keepScroll = false } = {}) {
  if (app.streaming && !silent) { toast('wait for the current turn to finish', 'warn'); return; }
  // An edit belongs to the conversation it was started in. Opening another one
  // abandons it, and nothing has been written, so there is nothing to undo.
  if (app.pendingEdit && app.pendingEdit.uuid !== uuid) clearPendingEdit({ clearInput: true });
  try {
    const conv = await api.getConversation(uuid);
    app.conv = conv;
    app.context = conv.context || null;
    prefs.lastUuid = uuid;
    savePrefs();
    renderTitle();
    renderTranscript();
    renderMeta();
    renderContext(conv.context);
    syncComposerToConversation();
    renderList();
    if (!keepScroll) scrollToBottom(true);
    if (window.innerWidth <= 820) $('shell').classList.remove('mobile-open');
  } catch (err) {
    if (err.status === 404) {
      toast('that conversation no longer exists', 'warn');
      app.conv = null;
      await refreshList();
      renderTitle();
      renderTranscript();
    } else if (!silent) {
      toast(err.message || 'could not open the conversation', 'err');
    }
  }
}

/**
 * Rename in place.
 *
 * ``window.prompt`` is not implemented in VS Code's browser — it throws, which made
 * the title unrenamable there. The title is already styled for the editable state,
 * so editing it directly is both more portable and less jarring than a dialog.
 */
function beginRename(conv) {
  const node = $('conv-title');
  if (!conv || node.isContentEditable) return;

  const original = conv.title || '';
  let settled = false;

  const revert = () => {
    node.setAttribute('contenteditable', 'false');
    node.removeEventListener('keydown', onKey);
    node.removeEventListener('blur', onBlur);
    renderTitle();
  };

  const finish = async (commit) => {
    if (settled) return;
    settled = true;
    const trimmed = node.textContent.trim();
    if (!commit || !trimmed || trimmed === original) { revert(); return; }
    try {
      await api.renameConversation(conv.uuid, trimmed);
      if (app.conv && app.conv.uuid === conv.uuid) app.conv.title = trimmed;
      revert();
      await refreshList();
    } catch (err) {
      revert();
      toast(err.message || 'rename failed', 'err');
    }
  };

  function onKey(event) {
    if (event.key === 'Enter') { event.preventDefault(); finish(true); }
    else if (event.key === 'Escape') { event.preventDefault(); finish(false); }
  }

  function onBlur() { finish(true); }

  node.textContent = original;
  node.removeAttribute('title');
  node.setAttribute('contenteditable', 'true');
  node.focus();

  const range = document.createRange();
  range.selectNodeContents(node);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);

  node.addEventListener('keydown', onKey);
  node.addEventListener('blur', onBlur);
}

async function deleteConversation(conv) {
  if (!window.confirm(`Delete “${conv.title || 'this conversation'}” and all of its media?\n\nThis removes memory/${conv.uuid}/ from disk.`)) return;
  try {
    await api.deleteConversation(conv.uuid);
    // The expansion map is keyed by call id, so it is dead weight once the
    // conversation is gone and there would be nothing left to prune it against.
    if (prefs.expandedTools) delete prefs.expandedTools[conv.uuid];
    if (app.pendingEdit && app.pendingEdit.uuid === conv.uuid) clearPendingEdit({ clearInput: true });
    if (app.conv && app.conv.uuid === conv.uuid) {
      app.conv = null;
      prefs.lastUuid = '';
      savePrefs();
      renderTitle();
      renderTranscript();
      renderMeta();
      renderContext(null);
    }
    await refreshList();
    toast('deleted', 'ok');
  } catch (err) {
    toast(err.message || 'delete failed', 'err');
  }
}

async function exportConversation() {
  if (!app.conv) return;
  try {
    await api.downloadConversation(app.conv.uuid, app.conv.title);
  } catch (err) {
    toast(err.message || 'export failed', 'err');
  }
}

async function importConversation(file) {
  try {
    const text = await file.text();
    const payload = JSON.parse(text);
    const conv = await api.importConversation(payload);
    await refreshList();
    await loadConversation(conv.uuid);
    toast('imported as a new conversation', 'ok');
  } catch (err) {
    toast(err instanceof SyntaxError ? 'that file is not valid JSON' : (err.message || 'import failed'), 'err', 6000);
  }
}

/* ── 5. composer ───────────────────────────────────────────────────────── */

function setComposerBusy(busy) {
  $('send-btn').hidden = busy;
  $('stop-btn').hidden = !busy;
  // The textarea stays editable while streaming so the next message can be typed
  // during a long tool loop; only the controls that would race the turn are locked.
  $('attach-btn').disabled = busy;
  $('model-select').disabled = busy || models().length === 0;
  $('effort-select').disabled = busy;
}

function syncComposerToConversation() {
  const model = currentModel();
  const select = $('model-select');
  select.value = model?.id || '';
  highlightVision();
  updateComposerNote();
}

function highlightVision() {
  const select = $('model-select');
  const model = currentModel();
  select.classList.toggle('warn', Boolean(model && model.vision === false));
}

function updateComposerNote() {
  const note = $('composer-note');

  // A missing key fails every send, so explain that instead of describing scaling
  // ratios the user cannot use yet.
  const blocked = !isConfigured();
  $('send-btn').disabled = blocked;
  $('send-btn').title = blocked ? 'no API key — add one in Settings' : '';
  if (blocked) {
    note.innerHTML = '<b>Sending is disabled</b> — there is no API key listed for Deepseek. '
      + 'If you need one go <a href="https://platform.deepseek.com/sign_up" target="_blank" '
      + 'rel="noopener noreferrer">here</a>, sign up, follow the steps to get a new key and '
      + 'paste it in to the <b>Settings</b> area for Deepseek API key.';
    return;
  }

  if (pendingEditFor(app.conv?.uuid)) {
    note.innerHTML = '<b>Editing an earlier message</b> — sending drops it and everything after it. '
      + '<a href="#" id="edit-cancel">Cancel</a>';
    note.querySelector('#edit-cancel').addEventListener('click', (event) => {
      event.preventDefault();
      clearPendingEdit({ clearInput: true });
    });
    $('send-btn').title = 'Send the edited message';
    return;
  }

  const model = currentModel();
  if (!model) { note.textContent = ''; return; }

  const parts = [];
  if (model.vision === false) parts.push('<b>This model has no vision</b> — images will be rejected.');
  if (!model.tool_calling) parts.push('tools unavailable on this model');
  const limits = app.props?.limits;
  if (limits) {
    parts.push(`images downscaled to ${limits.model_image_max_dim}px`);
    parts.push(`video sampled at ${limits.model_video_fps} fps, ≤${limits.model_video_max_frames} frames`);
    parts.push(`≤${limits.max_tool_steps} tool steps`);
  }
  note.innerHTML = parts.join(' · ');
}

function addFiles(fileList) {
  const incoming = Array.from(fileList || []);
  if (!incoming.length) return;

  const blocked = incoming.filter((file) => kindForFile(file) === 'image' && visionBlocked());
  if (blocked.length) {
    toast(`${currentModel()?.id} cannot see images — ${blocked.length} image${blocked.length > 1 ? 's' : ''} skipped`, 'err', 6000);
  }
  const usable = incoming.filter((file) => !(kindForFile(file) === 'image' && visionBlocked()));

  for (const file of usable) {
    if (file.size > 512 * 1024 * 1024) {
      toast(`${file.name} is larger than 512 MB`, 'err', 6000);
      continue;
    }
    app.attachments.push(file);
  }
  renderAttachments();
}

function clearAttachments() {
  app.attachments.forEach((file) => revokePreview(file.__preview));
  app.attachments = [];
  renderAttachments();
}

/** The descriptor under an attachment's name: a language when we know one. */
function attachmentNote(file) {
  const lang = languageForName(file.name);
  const ext = extOf(file.type, file.name);
  if (lang) return `${languageLabel(lang)} · ${ext}`;
  // `extOf` falls back to the literal string FILE, which is not a fact about the file.
  return ext === 'FILE' ? (file.type || 'file') : ext;
}

function renderAttachments() {
  const container = $('attachments');
  container.textContent = '';
  if (!app.attachments.length) { container.hidden = true; return; }
  container.hidden = false;

  app.attachments.forEach((file, index) => {
    const kind = uiKindForFile(file);
    const chip = h('div', `att ${kind}`);

    // Only an image or a video can be previewed from an object URL. This used to be an
    // `<img>` for everything that was not a video, so attaching a `.py`, a `.zip` or an
    // `.mp3` showed a broken-image glyph instead of the file's name.
    if (kind === 'image' || kind === 'video') {
      const preview = document.createElement(kind === 'video' ? 'video' : 'img');
      if (!file.__preview) file.__preview = URL.createObjectURL(file);
      preview.src = file.__preview;
      chip.append(preview);
    } else {
      chip.append(h('span', 'att-icon', extOf('', file.name).slice(0, 4)));
    }

    const info = h('div');
    info.append(h('div', 'att-name', file.name || 'pasted image'));
    info.append(h('div', 'att-size mono', `${formatBytes(file.size)} · ${attachmentNote(file)}`));
    chip.append(info);

    const remove = h('button', 'att-x', '×');
    remove.type = 'button';
    remove.title = 'Remove';
    remove.addEventListener('click', () => {
      revokePreview(file.__preview);
      app.attachments.splice(index, 1);
      renderAttachments();
    });
    chip.append(remove);
    container.append(chip);
  });
}

/* ── 6. drawers and boot ───────────────────────────────────────────────── */

function openSystemDrawer() {
  const base = app.conv ? (app.conv.sys_base ?? '') : prefs.defaultSystem;
  $('sys-base').value = base;
  $('sys-todo').value = app.conv ? (app.conv.sys_todo ?? '') : '';
  updateSystemPreview();
  $('system-scrim').hidden = false;
  $('system-drawer').hidden = false;
  $('sys-base').focus();
}

function closeSystemDrawer() {
  $('system-scrim').hidden = true;
  $('system-drawer').hidden = true;
}

function updateSystemPreview() {
  const base = $('sys-base').value.trim();
  const todo = $('sys-todo').value.trim();
  // Mirrors server/llm.py's compose_system so the preview is not a guess.
  const preview = todo
    ? `${base}\n\n===== YOUR CURRENT TASK TRACKER =====\n${todo}\n===== END TASK TRACKER =====\n`
    : base;
  $('sys-preview').textContent = preview || '(empty — no system prompt will be sent)';
}

async function saveSystemDrawer() {
  const base = $('sys-base').value;
  const todo = $('sys-todo').value;

  if (!app.conv) {
    prefs.defaultSystem = base;
    savePrefs(true);
    toast('saved as the default for new conversations', 'ok');
    closeSystemDrawer();
    return;
  }
  try {
    const result = await api.setSystem(app.conv.uuid, base, todo);
    app.conv.sys_base = result.sys_base;
    app.conv.sys_todo = result.sys_todo;
    if ($('default-system').checked) {
      prefs.defaultSystem = base;
      savePrefs(true);
    }
    toast('system prompt saved', 'ok');
    closeSystemDrawer();
  } catch (err) {
    toast(err.message || 'could not save the system prompt', 'err');
  }
}

/* ── the DeepSeek API key ────────────────────────────────────────────────────
   The server owns the key. This panel writes it through POST /api/settings/api-key
   and reads back a *masked* status, because the key is never sent to the browser.
   Saving is what makes the change live: the route drops the cached model clients,
   so the next message goes out with the new Authorization header. Nothing here
   needs a restart, and nothing here trusts the file write on its own — `app.props`
   is re-fetched and the whole UI is re-rendered from it. */

function renderApiKeyStatus() {
  const status = app.props?.api_key;
  const line = $('set-key-status');
  if (!status) {
    line.textContent = 'Server did not report a key status — reload the page.';
    return;
  }
  if (status.file) $('set-key-file').textContent = status.file;
  if (status.present) {
    line.innerHTML = `Currently stored: <code>${escapeHtml(status.masked)}</code>`;
  } else {
    line.textContent = 'No key stored yet.';
  }
}

/** Toggle the input between `password` and `text`. Masked unless asked otherwise. */
function revealKey(show) {
  const input = $('set-api-key');
  const button = $('reveal-key');
  input.type = show ? 'text' : 'password';
  button.textContent = show ? 'Hide' : 'Show';
  button.setAttribute('aria-pressed', show ? 'true' : 'false');
  button.title = show ? 'Mask the key again' : 'Show the key while you check what was pasted';
}

/** Re-render every part of the UI that depends on whether the model is usable. */
function applyProps(props) {
  app.props = props;
  renderApiKeyStatus();
  renderToolSteps();
  renderServerInfo();
  populateSelects();
  updateComposerNote();
  renderEmptySetup();
  // A banner telling you to go and set a key is wrong the moment one is set.
  if (isConfigured()) $('setup-banner').hidden = true;
}

async function saveApiKey(button) {
  const input = $('set-api-key');
  const key = input.value.trim();
  if (!key) {
    toast('paste a key first', 'warn');
    input.focus();
    return;
  }

  button.disabled = true;
  try {
    const result = await api.setApiKey(key);
    // Cleared rather than kept: the status line above already reports it in masked
    // form, and a password field holding a live secret is a screenshot hazard.
    input.value = '';
    revealKey(false);

    // Re-read /api/props instead of assuming the write worked: the payload is the
    // server's own verdict on whether the key resolves, and it drives every control.
    applyProps(await api.props());

    if (result && result.client_ready === false) {
      toast(`key saved, but the model is still unusable — ${result.error || 'check providers.json'}`, 'err', 8000);
    } else {
      toast('key saved — it is live for the next message', 'ok');
    }
  } catch (err) {
    toast(err.message || 'could not save the key', 'err', 7000);
  } finally {
    button.disabled = false;
  }
}

/* ── the tool-step ceiling ───────────────────────────────────────────────────
   A server setting written to .env, like the key, and made live the same way: the
   route records an override on the one `Settings` instance the engine reads, so the
   next message uses it. This panel never assumes the write worked — it re-reads
   /api/props, which is the server's own report of the value now in force. */

function renderToolSteps() {
  const limits = app.props?.limits;
  if (!limits) return;
  const input = $('set-max-tool-steps');
  // The bounds come from the server rather than the markup, so the two cannot drift.
  if (limits.min_tool_steps) input.min = limits.min_tool_steps;
  if (limits.max_tool_steps_limit) input.max = limits.max_tool_steps_limit;
  input.value = limits.max_tool_steps;
  $('set-tool-steps-status').textContent = `Currently ${limits.max_tool_steps}.`;
}

async function saveToolSteps(button) {
  const input = $('set-max-tool-steps');
  const value = Number(input.value);
  if (!Number.isInteger(value)) {
    toast('tool steps must be a whole number', 'warn');
    input.focus();
    return;
  }

  button.disabled = true;
  try {
    const result = await api.setLimits({ max_tool_steps: value });
    applyProps(await api.props());
    toast(`tool steps set to ${result.max_tool_steps} — live for the next message`, 'ok');
  } catch (err) {
    // Put the field back to the value still in force rather than leaving the
    // rejected one on screen to be saved again.
    renderToolSteps();
    toast(err.message || 'could not save the tool-step limit', 'err', 7000);
  } finally {
    button.disabled = false;
  }
}

function openSettings() {
  $('set-default-system').value = prefs.defaultSystem || '';
  $('set-expand-reasoning').checked = prefs.expandReasoning;
  $('set-autoscroll').checked = prefs.autoscroll;
  $('set-enter-sends').checked = prefs.enterSends;
  $('set-memory-root').textContent = app.props?.limits ? 'memory/<uuid>/' : 'memory/<uuid>/';
  // Never pre-fill the field from anywhere: the key is write-only from this page.
  // The masked line above it is what tells you one is already stored.
  $('set-api-key').value = '';
  revealKey(false);
  renderApiKeyStatus();
  renderToolSteps();
  renderServerInfo();
  renderMcpStatus({ servers: app.props?.mcp, config: app.props?.mcp_config });
  // Then catch up with anything changed outside the browser, such as a hand edit to
  // mcp.json. Silent: the panel already has something to show either way.
  refreshMcp();
  $('settings-scrim').hidden = false;
  $('settings-drawer').hidden = false;
  $('set-api-key').focus();
}

function closeSettings() {
  if (!$('mcp-editor').hidden) closeMcpEditor();
  $('settings-scrim').hidden = true;
  $('settings-drawer').hidden = true;
}

function infoRow(key, value) {
  const row = h('div', 'info-row');
  row.append(h('span', 'k', key));
  row.append(h('span', 'v', value == null || value === '' ? '—' : String(value)));
  return row;
}

function renderServerInfo() {
  const container = $('server-info');
  container.textContent = '';
  const props = app.props;
  if (!props) { container.append(h('p', 'dim', 'not loaded')); return; }

  container.append(infoRow('version', props.version));
  container.append(infoRow('configured', props.configured ? 'yes' : 'no'));
  if (props.endpoint) container.append(infoRow('endpoint', props.endpoint));
  if (props.provider) container.append(infoRow('provider', props.provider));
  container.append(infoRow('default model', props.default_model));
  container.append(infoRow('models', (props.models || []).map((m) => m.id).join(', ')));

  const tools = $('tools-info');
  tools.textContent = '';
  tools.append(h('span', 'k', `${(props.tools || []).length} tools available`));
  const wrap = h('div', 'tools-wrap');
  (props.tools || []).forEach((name) => wrap.append(h('span', 'tool-pill', name)));
  tools.append(wrap);
}

/* ── MCP servers ─────────────────────────────────────────────────────────────
   Adding a server here writes mcp.json, and hand-editing mcp.json stays supported —
   that is what the Reload button is for. The server owns the file: it preserves its
   shape, its comments and its top-level extras, so this side only ever sends one
   server definition and re-renders from whatever payload comes back. */

let mcpServers = [];          // last payload from the server: the single source of truth
let mcpConfig = null;         // {path, exists, error, editable}
let mcpEditing = null;        // null = editor closed, '' = adding, otherwise the name
const mcpLogOpen = new Set(); // which cards show their connection log

function mcpStore(payload) {
  if (payload && Array.isArray(payload.servers)) mcpServers = payload.servers;
  if (payload && payload.config) mcpConfig = payload.config;
  // The tool list is shown in the Server section, so keep it in step.
  if (payload && Array.isArray(payload.tools) && app.props) app.props.tools = payload.tools;
}

function mcpAction(label, className, onClick) {
  const button = h('button', `mini-btn${className ? ` ${className}` : ''}`, label);
  button.type = 'button';
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      await onClick();
    } finally {
      button.disabled = false;
    }
  });
  return button;
}

function mcpLogBlock(server) {
  const block = h('div', 'mcp-log');
  for (const entry of server.log || []) {
    const bad = /fail|error|could not|not set|refus/i.test(entry.text || '');
    block.append(h('span', `line${bad ? ' bad' : ''}`, `${clockTime(entry.at)}  ${entry.text}`));
  }
  return block;
}

function mcpCard(server) {
  const live = Boolean(server.connected);
  const card = h('div', 'mcp-server');
  if (!server.enabled) card.classList.add('off');
  else if (!live) card.classList.add('down');
  if (mcpEditing === server.name) card.classList.add('editing');

  const row1 = h('div', 'row1');
  row1.append(h('span', 'role-dot'));
  row1.append(h('span', 'name', server.name || '(unnamed)'));
  const label = !server.enabled ? 'disabled'
    : live ? `${(server.tools || []).length} tools` : 'offline';
  row1.append(h('span', `state ${!server.enabled ? '' : live ? 'ok' : 'bad'}`, label));

  const actions = h('div', 'mcp-actions');
  actions.append(mcpAction(server.enabled ? 'Disable' : 'Enable', '', () => setMcpEnabled(server)));
  actions.append(mcpAction('Test', '', () => testMcpServer(server)));
  actions.append(mcpAction('Edit', '', () => openMcpEditor(server.name)));
  if ((server.log || []).length) {
    const open = mcpLogOpen.has(server.name);
    actions.append(mcpAction(open ? 'Hide log' : 'Log', '', () => {
      if (open) mcpLogOpen.delete(server.name);
      else mcpLogOpen.add(server.name);
      renderMcpStatus();
    }));
  }
  actions.append(mcpAction('Delete', 'danger', () => deleteMcpServer(server)));
  row1.append(actions);
  card.append(row1);

  if (server.url) {
    const prefix = server.toolPrefix || server.prefix;
    const line = prefix
      ? `${server.url}  ·  tools appear as ${prefix}__<tool>`
      : server.url;
    // "works but no key" is the one setup mistake a keyed server cannot report on
    // its own, so say whether a key is configured right where the URL is shown.
    card.append(h('div', 'row2', mcpApiKeyOf(server.headers) ? `${line}  ·  key set` : line));
  }
  if (server.error) card.append(h('div', 'mcp-error', server.error));
  if ((server.tools || []).length) {
    const wrap = h('div', 'mcp-tools');
    for (const name of server.tools) wrap.append(h('span', 'tool-pill', name));
    card.append(wrap);
  }
  if (mcpLogOpen.has(server.name)) card.append(mcpLogBlock(server));
  return card;
}

function renderMcpStatus(payload) {
  mcpStore(payload);

  const container = $('mcp-status');
  container.textContent = '';
  const path = $('mcp-path');
  if (path) path.textContent = mcpConfig?.path || 'mcp.json';

  // Two programs edit the same file, so an unreadable config parks the whole panel
  // rather than overwriting whatever the user is in the middle of fixing.
  if (mcpConfig?.error) {
    container.append(h('div', 'mcp-editor-note bad',
      `${mcpConfig.path || 'mcp.json'} could not be read — ${mcpConfig.error}`));
  }

  if (!mcpServers.length) {
    container.append(h('div', 'mcp-empty',
      'No MCP servers configured. Add one below, or write it into the file by hand and press Reload.'));
  }
  for (const server of mcpServers) container.append(mcpCard(server));

  const blocked = Boolean(mcpConfig?.error);
  $('mcp-add').disabled = blocked;
  $('mcp-save').disabled = blocked;
}

function mcpNote(message, kind = '') {
  const note = $('mcp-editor-note');
  note.textContent = message || '';
  note.className = `dim mcp-editor-note${kind ? ` ${kind}` : ''}`;
  note.hidden = !message;
}

function openMcpEditor(name = null) {
  const server = name ? mcpServers.find((entry) => entry.name === name) : null;
  mcpEditing = name || '';
  $('mcp-editor-title').textContent = server ? `Edit ${server.name}` : 'Add an MCP server';
  $('mcp-name').value = server?.name || '';
  $('mcp-url').value = server?.url || '';
  $('mcp-prefix').value = server?.prefix || '';
  // Left blank when it is the default, so the placeholder can say what the default is.
  $('mcp-timeout').value = server?.timeout && server.timeout !== 120 ? String(server.timeout) : '';
  $('mcp-headers').value = formatHeaders(mcpHeadersWithoutKey(server?.headers));
  $('mcp-api-key').value = mcpApiKeyOf(server?.headers);
  $('mcp-tools').value = (server?.allowedTools || []).join(', ');
  $('mcp-enabled').checked = server ? server.enabled !== false : true;
  mcpNote('');
  $('mcp-editor').hidden = false;
  renderMcpStatus();
  $('mcp-name').focus();
}

function closeMcpEditor() {
  mcpEditing = null;
  $('mcp-editor').hidden = true;
  mcpNote('');
  renderMcpStatus();
}

/** The editor as a request body. Throws with a message worth showing the user. */
function mcpFormBody() {
  const name = $('mcp-name').value.trim();
  if (!name) throw new Error('give the server a name');
  const url = $('mcp-url').value.trim();
  if (!url) throw new Error('the Streamable HTTP URL is required');

  const rawTimeout = $('mcp-timeout').value.trim();
  const timeout = rawTimeout ? Number(rawTimeout) : 0;
  if (rawTimeout && (!Number.isFinite(timeout) || timeout <= 0)) {
    throw new Error('the timeout must be a number of seconds');
  }

  // The key field always wins, so clearing it really does clear the key rather than
  // leaving the old one alive in the textarea where nobody would look for it.
  const headers = withApiKey(parseHeaders($('mcp-headers').value), $('mcp-api-key').value);

  const body = {
    name,
    url,
    headers,
    enabled: $('mcp-enabled').checked,
    prefix: $('mcp-prefix').value.trim(),
    // A comma separated list is friendlier to type than a JSON array.
    allowedTools: $('mcp-tools').value.split(',').map((part) => part.trim()).filter(Boolean),
  };
  if (timeout) body.timeout = timeout;
  return body;
}

async function saveMcpServer() {
  let body;
  try {
    body = mcpFormBody();
  } catch (err) {
    mcpNote(err.message, 'bad');
    return;
  }

  const editing = mcpEditing;
  mcpNote('saving…');
  try {
    const result = editing ? await api.mcpUpdateServer(editing, body) : await api.mcpAddServer(body);
    mcpEditing = null;
    $('mcp-editor').hidden = true;
    mcpNote('');
    renderMcpStatus(result);
    renderServerInfo();
    toast(`${body.name} saved`, 'ok');
  } catch (err) {
    mcpNote(err.message || 'could not save the server', 'bad');
  }
}

/** Connect once, without saving, so a URL can be tried before it is committed. */
async function testMcpForm() {
  let body;
  try {
    body = mcpFormBody();
  } catch (err) {
    mcpNote(err.message, 'bad');
    return;
  }

  mcpNote('connecting…');
  try {
    const result = await api.mcpTest(body);
    const tools = result.tools || [];
    if (result.connected) {
      mcpNote(`connected — ${tools.length} tool${tools.length === 1 ? '' : 's'}`
        + `${tools.length ? `: ${tools.join(', ')}` : ''}`, 'ok');
    } else {
      mcpNote(result.error || 'could not connect', 'bad');
    }
  } catch (err) {
    mcpNote(err.message || 'could not connect', 'bad');
  }
}

async function testMcpServer(server) {
  try {
    const result = await api.mcpTest({ name: server.name });
    // Fold the result into the card rather than reconnecting the live session: this
    // is a diagnostic, and it must not disturb an already-working server.
    Object.assign(server, {
      connected: result.connected,
      error: result.error,
      tools: result.tools,
      log: result.log,
    });
    renderMcpStatus();
    const tools = result.tools || [];
    toast(result.connected
      ? `${server.name}: ${tools.length} tool${tools.length === 1 ? '' : 's'}`
      : `${server.name}: ${result.error || 'unreachable'}`,
    result.connected ? 'ok' : 'err', result.connected ? 4200 : 7000);
  } catch (err) {
    toast(err.message || 'the test failed', 'err', 7000);
  }
}

async function setMcpEnabled(server) {
  try {
    const result = await api.mcpSetEnabled(server.name, !server.enabled);
    renderMcpStatus(result);
    renderServerInfo();
    toast(`${server.name} ${server.enabled ? 'disabled' : 'enabled'}`, 'ok');
  } catch (err) {
    toast(err.message || 'could not change the server', 'err');
  }
}

async function deleteMcpServer(server) {
  if (!confirm(`Remove the MCP server "${server.name}"?`)) return;
  try {
    const result = await api.mcpDeleteServer(server.name);
    if (mcpEditing === server.name) {
      mcpEditing = null;
      $('mcp-editor').hidden = true;
      mcpNote('');
    }
    mcpLogOpen.delete(server.name);
    renderMcpStatus(result);
    renderServerInfo();
    toast(`${server.name} removed`, 'ok');
  } catch (err) {
    toast(err.message || 'could not remove the server', 'err');
  }
}

async function reloadMcp(button) {
  if (button) button.disabled = true;
  try {
    const result = await api.mcpReload();
    renderMcpStatus(result);
    renderServerInfo();
    toast(`reloaded — ${result.tools?.length ?? 0} tools`, 'ok');
  } catch (err) {
    toast(err.message || 'MCP reload failed', 'err', 6000);
  } finally {
    if (button) button.disabled = false;
  }
}

/** Pull the current state without reconnecting, so Settings shows what is on disk. */
async function refreshMcp() {
  try {
    renderMcpStatus(await api.mcp());
  } catch {
    /* the Reload button is where a failure gets reported */
  }
}

function applyTheme() {
  document.documentElement.dataset.theme = prefs.theme;
  $('theme-icon').textContent = prefs.theme === 'dark' ? '☾' : '☀';
  $('theme-label').textContent = prefs.theme === 'dark' ? 'Dark' : 'Light';
}

function wire() {
  /* sidebar */
  $('toggle-sidebar').addEventListener('click', () => {
    // On a narrow viewport the sidebar is off-canvas by default, so the button has
    // to slide the whole shell rather than collapse a column that is already gone.
    if (NARROW()) {
      $('shell').classList.remove('mobile-open');
      return;
    }
    $('shell').classList.add('collapsed');
    $('show-sidebar').classList.remove('hidden');
    prefs.sidebarCollapsed = true;
    savePrefs();
  });
  $('show-sidebar').addEventListener('click', () => {
    if (NARROW()) {
      $('shell').classList.add('mobile-open');
      return;
    }
    $('shell').classList.remove('collapsed');
    $('show-sidebar').classList.add('hidden');
    prefs.sidebarCollapsed = false;
    savePrefs();
  });

  $('new-chat').addEventListener('click', () => createConversation());
  $('search').addEventListener('input', renderList);

  $('theme-toggle').addEventListener('click', () => {
    prefs.theme = prefs.theme === 'dark' ? 'light' : 'dark';
    savePrefs(true);
    applyTheme();
  });

  $('open-settings').addEventListener('click', openSettings);
  $('close-settings').addEventListener('click', closeSettings);
  $('settings-scrim').addEventListener('click', closeSettings);
  // Both prompts about the missing key lead to the one place that can fix it.
  $('banner-settings').addEventListener('click', openSettings);
  $('empty-settings').addEventListener('click', openSettings);

  /* the API key field */
  $('reveal-key').addEventListener('click', () => revealKey($('set-api-key').type === 'password'));
  $('save-api-key').addEventListener('click', (event) => saveApiKey(event.currentTarget));
  $('clear-api-key').addEventListener('click', () => {
    $('set-api-key').value = '';
    $('set-api-key').focus();
  });
  // Enter is the natural gesture straight after pasting a key.
  $('set-api-key').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      saveApiKey($('save-api-key'));
    }
  });

  /* the tool-step ceiling */
  $('save-tool-steps').addEventListener('click', (event) => saveToolSteps(event.currentTarget));
  $('set-max-tool-steps').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      saveToolSteps($('save-tool-steps'));
    }
  });

  $('export-conv').addEventListener('click', exportConversation);
  $('delete-conv').addEventListener('click', () => {
    if (!app.conv) { toast('no conversation is open', 'warn'); return; }
    deleteConversation({ uuid: app.conv.uuid, title: app.conv.title });
  });

  /* title rename by clicking it */
  $('conv-title').addEventListener('click', () => {
    if (!app.conv) return;
    beginRename({ uuid: app.conv.uuid, title: app.conv.title });
  });

  /* composer */
  const form = $('composer');
  const input = $('input');

  input.addEventListener('input', () => autogrow(input));
  input.addEventListener('keydown', (event) => {
    // Escape abandons an edit. Nothing has been written yet, so this costs nothing.
    if (event.key === 'Escape' && pendingEditFor(app.conv?.uuid)) {
      event.preventDefault();
      clearPendingEdit({ clearInput: true });
      toast('edit cancelled', 'warn', 1800);
      return;
    }
    if (event.key === 'Enter' && !event.shiftKey && (prefs.enterSends ? !event.ctrlKey : event.ctrlKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener('submit', (event) => { event.preventDefault(); send(); });

  $('stop-btn').addEventListener('click', () => {
    if (app.controller) {
      app.controller.abort();
      toast('stopping…', 'warn', 1600);
    }
  });

  $('attach-btn').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (event) => {
    addFiles(event.target.files);
    event.target.value = '';
  });

  /* paste an image straight from the clipboard */
  input.addEventListener('paste', (event) => {
    const files = Array.from(event.clipboardData?.files || []);
    if (files.length) {
      event.preventDefault();
      addFiles(files);
    }
  });

  /* drag and drop anywhere in the main column */
  const main = $('main');
  let dragDepth = 0;
  main.addEventListener('dragenter', (event) => {
    if (!event.dataTransfer?.types?.includes('Files')) return;
    dragDepth += 1;
    form.classList.add('dragover');
  });
  main.addEventListener('dragover', (event) => {
    if (event.dataTransfer?.types?.includes('Files')) event.preventDefault();
  });
  main.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) form.classList.remove('dragover');
  });
  main.addEventListener('drop', (event) => {
    dragDepth = 0;
    form.classList.remove('dragover');
    const files = Array.from(event.dataTransfer?.files || []);
    if (!files.length) return;
    event.preventDefault();
    addFiles(files);
  });

  /* model + effort */
  $('model-select').addEventListener('change', async (event) => {
    const model = event.target.value;
    highlightVision();
    updateComposerNote();
    if (!app.conv) return;
    try {
      await api.setModel(app.conv.uuid, model);
      app.conv.model = model;
      renderMeta();
      await refreshList({ silent: true });
    } catch (err) {
      toast(err.message || 'could not set the model', 'err');
      syncComposerToConversation();
    }
  });

  $('effort-select').addEventListener('change', (event) => {
    prefs.effort = event.target.value;
    savePrefs();
  });

  $('tools-toggle').addEventListener('change', (event) => {
    prefs.tools = event.target.checked;
    savePrefs();
  });

  $('default-system').addEventListener('change', (event) => {
    if (event.target.checked && app.conv) {
      prefs.defaultSystem = app.conv.sys_base || '';
      savePrefs(true);
      toast('this conversation’s prompt is now the default', 'ok');
    }
  });

  /* system drawer */
  $('system-btn').addEventListener('click', openSystemDrawer);
  $('close-system').addEventListener('click', closeSystemDrawer);
  $('system-scrim').addEventListener('click', closeSystemDrawer);
  $('sys-base').addEventListener('input', updateSystemPreview);
  $('sys-todo').addEventListener('input', updateSystemPreview);
  $('sys-save').addEventListener('click', saveSystemDrawer);
  $('sys-clear').addEventListener('click', () => {
    $('sys-base').value = '';
    $('sys-todo').value = '';
    updateSystemPreview();
  });

  /* settings */
  $('set-default-system').addEventListener('input', (event) => {
    prefs.defaultSystem = event.target.value;
    savePrefs();
  });
  $('set-expand-reasoning').addEventListener('change', (event) => {
    prefs.expandReasoning = event.target.checked;
    savePrefs(true);
  });
  $('set-autoscroll').addEventListener('change', (event) => {
    prefs.autoscroll = event.target.checked;
    savePrefs(true);
  });
  $('set-enter-sends').addEventListener('change', (event) => {
    prefs.enterSends = event.target.checked;
    savePrefs(true);
  });
  $('import-btn').addEventListener('click', () => $('import-file').click());
  $('import-file').addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) importConversation(file);
  });

  /* MCP servers */
  $('mcp-add').addEventListener('click', () => openMcpEditor());
  $('mcp-reload').addEventListener('click', () => reloadMcp($('mcp-reload')));
  $('mcp-cancel').addEventListener('click', () => closeMcpEditor());
  $('mcp-test').addEventListener('click', () => testMcpForm());
  $('mcp-editor').addEventListener('submit', (event) => {
    event.preventDefault();
    saveMcpServer();
  });

  $('dismiss-banner').addEventListener('click', () => {
    $('setup-banner').hidden = true;
    prefs.bannerDismissed = true;
    savePrefs(true);
  });

  /* keyboard */
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      // Escape backs out one layer at a time, so it cannot discard a half-filled
      // server definition by closing the whole drawer underneath it.
      if (!$('mcp-editor').hidden) { closeMcpEditor(); return; }
      closeSettings();
      closeSystemDrawer();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      $('search').focus();
    }
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'o') {
      event.preventDefault();
      createConversation();
    }
  });
}

function populateSelects() {
  const modelSelect = $('model-select');
  modelSelect.textContent = '';
  const catalogue = models();
  if (!catalogue.length) {
    // An empty <select> just looks broken, so say what is actually wrong. This also
    // covers a genuine /api/props failure, not only a missing API key.
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'no models configured';
    modelSelect.append(placeholder);
  }
  for (const model of catalogue) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.name && model.name !== model.id ? `${model.name}` : model.id;
    modelSelect.append(option);
  }
  modelSelect.disabled = catalogue.length === 0;

  const effortSelect = $('effort-select');
  effortSelect.textContent = '';
  const efforts = app.props?.reasoning_efforts || ['none', 'low', 'high', 'max'];
  const automatic = document.createElement('option');
  automatic.value = '';
  automatic.textContent = 'model default';
  effortSelect.append(automatic);
  for (const effort of efforts) {
    const option = document.createElement('option');
    option.value = effort;
    option.textContent = effort;
    effortSelect.append(option);
  }
  // A stored effort the server no longer offers would silently send nothing.
  if (prefs.effort && !efforts.includes(prefs.effort)) prefs.effort = '';
  effortSelect.value = prefs.effort || '';

  $('tools-toggle').checked = prefs.tools;
}

async function boot() {
  applyTheme();
  wire();

  try {
    app.props = await api.props();
  } catch (err) {
    $('boot').innerHTML =
      `<div class="boot-card"><h2 style="color:var(--err)">Cannot reach the server</h2>`
      + `<p>${escapeHtml(err.message || 'unknown error')}</p>`
      + '<p class="dim">Restart it with <code>python run.py</code> and reload this page.</p></div>';
    return;
  }

  populateSelects();
  updateComposerNote();
  renderServerInfo();
  renderApiKeyStatus();
  renderEmptySetup();
  renderMcpStatus({ servers: app.props.mcp, config: app.props.mcp_config });

  // The copy in the banner is the user's own wording; the message line stays the
  // server's, so a broken providers.json is not misreported as a missing key.
  if (app.props.configured === false && !prefs.bannerDismissed) {
    $('setup-message').textContent = app.props.error || 'The provider could not be read.';
    $('setup-banner').hidden = false;
  }

  if (prefs.sidebarCollapsed) {
    $('shell').classList.add('collapsed');
    $('show-sidebar').classList.remove('hidden');
  }

  await refreshList();

  const wanted = prefs.lastUuid || app.conversations[0]?.uuid;
  if (wanted) {
    await loadConversation(wanted, { silent: true });
  } else {
    renderTitle();
    renderTranscript();
    renderMeta();
  }

  app.booted = true;
  $('shell').hidden = false;
  $('boot').classList.add('done');
  setTimeout(() => $('boot').remove(), 350);

  // A poll keeps the context meter and the list honest when a turn outlives the
  // tab it was started in.
  setInterval(() => {
    if (app.streaming || !app.conv) return;
    api.context(app.conv.uuid).then((context) => renderContext(context)).catch(() => {});
  }, 20000);
}

window.addEventListener('DOMContentLoaded', boot);
window.addEventListener('error', (event) => {
  if (event.message) toast(`script error: ${event.message}`, 'err', 6000);
});
