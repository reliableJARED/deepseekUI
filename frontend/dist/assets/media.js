/* media.js — turning stored blocks into elements, and files into uploads.
 *
 * The rule this module exists to enforce: **the browser never receives base64.**
 * The server writes every attachment and every image a tool returns to
 * `memory/<uuid>/` and hands back a `/memory/...` URL. The client only ever emits
 * that URL into an `img` or `video` tag, so a 4 MB screenshot costs one request to
 * the same local server rather than a 5.4 MB JSON field rendered twice.
 */

import { escapeHtml } from './markdown.js';
import { detectLanguage, highlightLines, isTextName, languageForName, languageLabel } from './highlight.js';

const IMAGE_MIMES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
const VIDEO_MIMES = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska'];

/* A code preview is fetched only when a chip is opened, and only this much of it.
   A 300 MB `.log` rendered into spans would lock the main thread for seconds, and
   the reader scrolling to character 100,001 of a log is not a thing that happens. */
const PREVIEW_MAX_BYTES = 512 * 1024;
const PREVIEW_MAX_CHARS = 120_000;
const PREVIEW_REFUSE_BYTES = 16 * 1024 * 1024;

/* `read_file` numbers its gutter, so a listing arrives as `<spaces><number>␠␠<line>`.
   Recognising that prefix is the whole trick: it is also what lets the numbers be
   stripped before highlighting, so the digits are not read as code. The group is what
   the first line's number is read from. */
const NUMBERED = /^ *(\d+) {2}/;

export function formatBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1048576).toFixed(1)} MB`;
  return `${(n / 1073741824).toFixed(2)} GB`;
}

/** The file extension shown in a caption, derived from a mime or a name. */
export function extOf(mime, name = '') {
  const fromName = /\.[a-z0-9]{1,5}$/i.exec(name);
  if (fromName) return fromName[0].slice(1).toUpperCase();
  if (!mime) return 'FILE';
  if (IMAGE_MIMES.includes(mime)) return mime.split('/')[1].replace('jpeg', 'jpg').toUpperCase();
  if (VIDEO_MIMES.includes(mime)) return mime.split('/')[1].toUpperCase();
  return mime.split('/').pop().slice(0, 5).toUpperCase();
}

export function isImageMime(mime) {
  return typeof mime === 'string' && mime.startsWith('image/');
}

export function isVideoMime(mime) {
  return typeof mime === 'string' && mime.startsWith('video/');
}

/** Guess a kind for the `kind` form field from a File. */
export function kindForFile(file) {
  const type = file.type || '';
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('video/')) return 'video';
  if (type.startsWith('audio/')) return 'audio';
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) return 'image';
  if (['mp4', 'webm', 'mov', 'mkv'].includes(ext)) return 'video';
  return 'file';
}

/**
 * How a *pending* attachment should be drawn in the composer.
 *
 * Deliberately not `kindForFile`: that one is the label sent to the server, and
 * widening it would change what the server believes about a file. This is purely
 * presentational, and it exists because the composer used to build an `<img>` for
 * every non-video file — so attaching a `.py`, a `.zip` or an `.mp3` showed a
 * broken-image glyph, since an object URL for something that is not an image
 * cannot decode.
 */
export function uiKindForFile(file) {
  const type = file.type || '';
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('video/')) return 'video';
  if (type.startsWith('audio/')) return 'audio';
  if (type.startsWith('text/') || isTextName(file.name)) return 'code';
  return 'file';
}

/* ── block normalisation ───────────────────────────────────────────────── */

/**
 * Flatten one message's `content` into display blocks.
 *
 * Input arrives in four shapes and they all have to look the same downstream:
 *   - a plain string (the common case)
 *   - user block lists:      {type:'text'} | {type:'image_url',image_url:{url}} | {type:'_file',kind,url}
 *   - tool block lists:      {type:'text'} | {type:'image'|'video'|'audio',url,mime}
 *   - the JSON string a tool message's content is stored as
 */
export function blocksOf(content) {
  if (typeof content === 'string') {
    const trimmed = content.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const decoded = JSON.parse(trimmed);
        if (Array.isArray(decoded)) return normalise(decoded);
      } catch {
        /* a plain string that merely looks like JSON */
      }
    }
    return trimmed ? [{ type: 'text', text: content }] : [];
  }
  if (Array.isArray(content)) return normalise(content);
  if (content == null) return [];
  return [{ type: 'text', text: String(content) }];
}

function normalise(items) {
  const out = [];
  for (const item of items) {
    if (!item || typeof item !== 'object') {
      if (item != null) out.push({ type: 'text', text: String(item) });
      continue;
    }
    const type = item.type;

    if (type === 'text') {
      const text = String(item.text ?? '');
      if (text) out.push({ type: 'text', text });
      continue;
    }

    if (type === 'image_url') {
      const inner = item.image_url || {};
      out.push({
        type: 'image',
        url: String(inner.url || ''),
        mime: String(item.mime || ''),
        width: item.width,
        height: item.height,
        detail: inner.detail || '',
      });
      continue;
    }

    if (type === '_file' || type === '_code' || type === 'code') {
      const kind = String(item.kind || (type === '_file' ? 'file' : 'text'));
      // Text becomes its own block rather than collapsing into the generic file
      // chip. Everything else down here is a byte container whose only affordance is
      // a download link; a `.py` or a `.json` is something you can *read*, and
      // rendering it as the same anonymous grey chip as an unidentified binary is
      // the thing this branch exists to stop doing.
      if (kind === 'text' || kind === 'code' || type === 'code' || type === '_code'
        || (kind === 'file' && isTextName(item.name))) {
        out.push({
          type: 'code',
          url: String(item.url || ''),
          text: typeof item.text === 'string' ? item.text : '',
          name: String(item.name || ''),
          mime: String(item.mime || ''),
          bytes: Number(item.bytes) || 0,
          lines: Number(item.lines) || 0,
          lang: String(item.lang || ''),
          numbered: Boolean(item.numbered),
          startLine: Number(item.startLine) || 1,
          note: String(item.note || ''),
        });
        continue;
      }
      const resolved = kind === 'image' || isImageMime(item.mime) ? 'image'
        : kind === 'video' || isVideoMime(item.mime) ? 'video'
          : kind === 'audio' ? 'audio' : 'file';
      out.push({
        type: resolved,
        url: String(item.url || ''),
        name: String(item.name || ''),
        mime: String(item.mime || ''),
        width: item.width,
        height: item.height,
      });
      continue;
    }

    if (type === 'file' && String(item.kind || '') === 'text') {
      out.push({
        type: 'code',
        url: String(item.url || ''),
        text: typeof item.text === 'string' ? item.text : '',
        name: String(item.name || ''),
        mime: String(item.mime || ''),
        bytes: Number(item.bytes) || 0,
        lines: Number(item.lines) || 0,
      });
      continue;
    }

    if (type === 'image' || type === 'video' || type === 'audio') {
      out.push({
        type,
        url: String(item.url || ''),
        name: String(item.name || ''),
        mime: String(item.mimeType || item.mime || ''),
        width: item.width,
        height: item.height,
      });
      continue;
    }

    // Unknown block: show the text if there is any, otherwise keep the door open
    // rather than silently swallowing content the model was told to expect.
    if (typeof item.text === 'string' && item.text) out.push({ type: 'text', text: item.text });
  }
  return out;
}

/** True when a block is an image or video the user should see rendered. */
export function isMediaBlock(block) {
  return (block.type === 'image' && block.url)
    || (block.type === 'video' && block.url)
    || (block.type === 'audio' && block.url);
}

/** A one-line description of a document, for the `title` attribute. */
export function describeBlock(block) {
  const parts = [];
  if (block.width && block.height) parts.push(`${block.width}×${block.height}`);
  if (block.name) parts.push(block.name);
  if (block.mime) parts.push(block.mime);
  return parts.join(' · ');
}

/* ── element construction ──────────────────────────────────────────────── */

function mediaCaption(block) {
  // `caption` is what the tool said it was showing ('the monkey you asked for'), so it
  // beats the filename as the visible label; the filename stays in the title text.
  const label = block.caption || block.name || (block.type === 'video' ? 'video' : 'image');
  const detail = [extOf(block.mime, block.name)];
  if (block.width && block.height) detail.push(`${block.width}×${block.height}`);
  const title = [describeBlock(block), block.caption].filter(Boolean).join(' · ');
  return `<div class="media-cap"><span title="${escapeHtml(title)}">${escapeHtml(label)}</span>`
    + `<span class="spacer"></span><a href="${escapeHtml(block.url)}" download title="Download">▼</a>`
    + `<span>${escapeHtml(detail.join(' · '))}</span></div>`;
}

/**
 * Build the element for one media block.
 * `onZoom(url)` is called when an image is clicked.
 */
export function elementForBlock(block, { onZoom = null } = {}) {
  if (block.type === 'code') return codeChip(block);

  if (block.type === 'image' && block.url) {
    const figure = document.createElement('figure');
    figure.className = 'media-item';
    figure.innerHTML =
      `<img src="${escapeHtml(block.url)}" alt="${escapeHtml(block.name || 'image')}" loading="lazy"`
      + `${block.width ? ` width="${Number(block.width)}"` : ''}`
      + `${block.height ? ` height="${Number(block.height)}"` : ''}>`
      + mediaCaption(block);
    const img = figure.querySelector('img');
    // A /memory/ URL can 404 if the file was deleted by hand; say so rather than
    // leaving a broken-image glyph in the transcript.
    img.addEventListener('error', () => {
      figure.classList.add('broken');
      img.replaceWith(Object.assign(document.createElement('span'), {
        className: 'file-chip',
        textContent: `⚠ ${block.name || 'image'} is missing from memory/`,
      }));
    });
    if (onZoom) img.addEventListener('click', () => onZoom(block.url));
    return figure;
  }

  if (block.type === 'video' && block.url) {
    const figure = document.createElement('figure');
    figure.className = 'media-item';
    figure.innerHTML =
      `<video src="${escapeHtml(block.url)}" controls preload="metadata"`
      + `${block.width ? ` width="${Number(block.width)}"` : ''}></video>`
      + mediaCaption(block);
    return figure;
  }

  if (block.type === 'audio' && block.url) {
    const figure = document.createElement('figure');
    figure.className = 'media-item';
    figure.innerHTML = `<audio src="${escapeHtml(block.url)}" controls></audio>` + mediaCaption(block);
    return figure;
  }

  const chip = document.createElement('span');
  chip.className = 'file-chip';
  const name = block.name || extOf(block.mime);
  chip.innerHTML = block.url
    ? `<svg viewBox="0 0 24 24"><path d="M14 3v5h5M6 3h8l5 5v13H6z"/></svg><a href="${escapeHtml(block.url)}" download>${escapeHtml(name)}</a>`
    : `<svg viewBox="0 0 24 24"><path d="M14 3v5h5M6 3h8l5 5v13H6z"/></svg>${escapeHtml(name)}`;
  return chip;
}

/* ── code ──────────────────────────────────────────────────────────────── */

/**
 * A highlighted code block, as the transcript and the tool cards both want it.
 *
 * When `numbered`, the gutter numbers live in `<span class="ln">` and the source is
 * kept on the element as `__source`, because `textContent` would otherwise include
 * the gutter — copying a listing should give you the listing's code, not "  42  ".
 */
export function codeBlock(code, lang, { numbered = false, startLine = 1, name = '', note = '', head = true } = {}) {
  const block = document.createElement('div');
  block.className = 'code-block';
  block.dataset.lang = lang || 'text';

  if (head) {
    const bar = document.createElement('div');
    bar.className = 'code-head';
    bar.innerHTML = `<span>${escapeHtml(name || languageLabel(lang))}</span>`
      + '<span class="spacer"></span>'
      + '<button class="code-copy" type="button">copy</button>';
    block.append(bar);
  }

  const pre = document.createElement('pre');
  const codeEl = document.createElement('code');
  // A file that ends in a line break has one more element in `split('\n')` than it has
  // lines, and numbering that phantom line would leave a bare "  43  " at the bottom of
  // every listing. Both endings have to be peeled, because a Windows file's last line
  // break is `\r\n` and dropping only the `\n` leaves a `\r` that the tokenizer then
  // turns back into a line break. Only the numbering is affected: `__source` keeps the
  // text as it came in, so a copy is still faithful.
  const lines = highlightLines(numbered ? code.replace(/\r?\n$/, '') : code, lang);
  if (numbered) {
    const width = String(startLine + lines.length - 1).length;
    codeEl.innerHTML = lines
      .map((line, i) => `<span class="ln">${String(startLine + i).padStart(width)}</span>${line}`)
      .join('\n');
  } else {
    codeEl.innerHTML = lines.join('\n');
  }
  pre.append(codeEl);
  block.append(pre);

  if (note) {
    const foot = document.createElement('div');
    foot.className = 'code-note';
    foot.textContent = note;
    block.append(foot);
  }
  block.__source = code;
  return block;
}

/**
 * Recognise the line-numbered listing `read_file` returns.
 *
 * The shape is `/^ *\d+ {2}/` on every line after an optional header, with an
 * optional bracketed footer. Requiring *every* line to match is what keeps this from
 * firing on prose that merely contains one numbered line — a false positive here
 * would replace readable text with a code block, which is worse than not numbering.
 */
export function readFileListing(text) {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
  if (lines.length < 3) return null;

  let body = lines;
  let note = '';
  if (/^\[.*\]$/.test(body[body.length - 1].trim())) {
    note = body[body.length - 1].trim();
    body = body.slice(0, -1);
  }

  let header = '';
  if (!NUMBERED.exec(body[0])) {
    header = body[0];
    body = body.slice(1);
  }

  // Blank lines around the listing are tolerated: `read_file` joins its header and
  // footer to the gutter with plain newlines, but the text may have been reformatted
  // on the way here, and a blank line must not read as an unnumbered line of prose.
  while (body.length && !body[0].trim()) body = body.slice(1);
  while (body.length && !body[body.length - 1].trim()) body = body.slice(0, -1);

  // Strictness is the safety property here: prose that happens to contain one numbered
  // line must not be rewritten as a file listing, so *every* remaining line has to be
  // numbered. A blank line inside the file still carries its gutter, so this costs
  // nothing on real output.
  if (body.length < 2) return null;
  if (!body.every((line) => NUMBERED.test(line))) return null;

  const first = NUMBERED.exec(body[0]);
  if (!first) return null;
  const name = (/^(.+?)\s+\u2014\s+/.exec(header) || [])[1] || '';
  return {
    name,
    startLine: Number(first[1]),
    note,
    code: body.map((line) => line.replace(NUMBERED, '')).join('\n'),
  };
}

/**
 * A block of code, as a chip: a labelled frame that carries the highlighting.
 *
 * The chip is the container for code in *both* directions, so an attachment and a tool
 * result read as the same kind of thing. What differs is when it opens. Text already in
 * hand — `read_file`'s listing, a tool's JSON — is the answer the reader asked for, so it
 * opens immediately and reads as output. An attachment is a reference instead: opening
 * one fetches at most 512 KB of the file with a ranged request, and nothing is read
 * until the reader asks, because a conversation holds dozens of them and none are
 * needed to draw the transcript.
 */
function codeChip(block) {
  const wrap = document.createElement('div');
  wrap.className = 'code-chip';

  const name = block.name || '';
  const lang = block.lang || languageForName(name) || languageForName(block.mime);
  // A listing brings its own numbering and its own first line number; a whole file is
  // numbered from 1; a blob with no lines to refer to — a tool's JSON — is not numbered
  // at all, because there is no line for a reader to be pointed at.
  const numbered = block.numbered === true || (!block.text && Boolean(block.url));
  const startLine = Number(block.startLine) > 0 ? Number(block.startLine) : 1;
  const label = lang ? languageLabel(lang) : (block.mime || 'text');
  // With a name, the badge is the extension, the way a file manager shows one. Without
  // one there is no extension to show, so the language takes the badge instead and the
  // meta drops it rather than saying the same thing twice.
  const badge = name ? extOf(block.mime, name) : (lang ? languageLabel(lang).toUpperCase() : 'CODE');
  const detail = [];
  if (block.bytes) detail.push(formatBytes(block.bytes));
  if (block.lines) detail.push(`${block.lines.toLocaleString()} lines`);
  if (label && label.toUpperCase() !== badge) detail.push(label);

  const head = document.createElement('div');
  head.className = 'code-chip-head';
  head.innerHTML =
    `<span class="code-badge">${escapeHtml(badge)}</span>`
    + (name ? `<span class="code-chip-name" title="${escapeHtml(describeBlock(block))}">${escapeHtml(name)}</span>` : '')
    + (detail.length ? `<span class="code-chip-meta">${escapeHtml(detail.join(' · '))}</span>` : '')
    + '<span class="spacer"></span>'
    + '<button class="code-chip-toggle" type="button" aria-expanded="false">show</button>'
    + '<button class="code-copy" type="button">copy</button>'
    + (block.url ? `<a class="code-chip-tool" href="${escapeHtml(block.url)}" download title="Download">\u2913</a>` : '');
  wrap.append(head);

  const body = document.createElement('div');
  body.className = 'code-chip-body';
  body.hidden = true;
  wrap.append(body);

  let loaded = null;
  const load = async () => {
    if (loaded) return loaded;
    if (block.text) {                                   // a listing already in hand
      loaded = { text: block.text, truncated: false };
      return loaded;
    }
    if (!block.url) {
      loaded = { error: 'this file has no contents stored' };
      return loaded;
    }
    if (block.bytes > PREVIEW_REFUSE_BYTES) {
      loaded = { error: `too large to preview (${formatBytes(block.bytes)}) \u2014 download it instead` };
      return loaded;
    }
    // A ranged read: the server serves /memory/ files with Accept-Ranges, and this
    // caps both the transfer and the number of spans we are willing to build.
    const response = await fetch(block.url, { headers: { Range: `bytes=0-${PREVIEW_MAX_BYTES - 1}` } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    let text = new TextDecoder('utf-8').decode(buffer);
    // A byte range can end mid-character; drop the replacement char rather than
    // ending the preview with a stray diamond.
    if (text.endsWith('\uFFFD')) text = text.slice(0, -1);
    // The server answers a satisfied-in-full range with a 206 as well as a partial one,
    // so the status says nothing about whether the *file* was cut short. The size that
    // travelled with the block does, and it is only unknown for an older attachment.
    const truncated = (block.bytes ? buffer.byteLength < block.bytes : false)
      || text.length > PREVIEW_MAX_CHARS;
    loaded = { text: text.slice(0, PREVIEW_MAX_CHARS), truncated };
    return loaded;
  };

  const noteIn = (message) => {
    body.textContent = '';
    const note = document.createElement('div');
    note.className = 'code-note';
    note.textContent = message;
    body.append(note);
  };

  const draw = (result) => {
    if (result.error) {
      noteIn(result.error);
      return;
    }
    const text = result.text || '';
    body.textContent = '';
    body.append(codeBlock(text, lang, {
      numbered,
      startLine,
      head: false,
      note: result.truncated
        ? `showing the first ${text.length.toLocaleString()} characters \u2014 download the file for the rest`
        : block.note,
    }));
    body.dataset.ready = '1';
  };

  const toggle = head.querySelector('.code-chip-toggle');
  const copy = head.querySelector('.code-copy');

  const open = async () => {
    body.hidden = false;
    toggle.textContent = 'hide';
    toggle.setAttribute('aria-expanded', 'true');
    if (body.dataset.ready) return;
    if (block.text) {
      // Already in hand, so there is nothing to wait for and no spinner to show.
      draw({ text: block.text, truncated: false });
      return;
    }
    noteIn('loading\u2026');
    try {
      draw(await load());
    } catch (err) {
      noteIn(`could not read this file (${err.message})`);
    }
  };

  toggle.addEventListener('click', () => {
    if (body.hidden) {
      open();
      return;
    }
    body.hidden = true;
    toggle.textContent = 'show';
    toggle.setAttribute('aria-expanded', 'false');
  });

  copy.addEventListener('click', async () => {
    try {
      const result = await load();
      if (result.error) {
        noteIn(result.error);
        return;
      }
      copyText(result.text || '', copy);
    } catch {
      copyText('', copy);
    }
  });

  // Text already in hand is the answer itself — a tool's output — so it opens at once.
  // An attachment is a reference: its bytes are one ranged request away and nothing is
  // read until the reader asks, because a transcript can hold dozens of files.
  if (block.text) open();

  return wrap;
}

/**
 * Copy text, giving the button its own feedback. Returns success.
 *
 * Lives here rather than in app.js so that both the transcript and the code chip can
 * use it — media.js must not import app.js, since app.js imports media.js.
 */
export async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(String(text ?? ''));
    if (button) {
      const original = button.textContent;
      button.textContent = 'copied';
      button.classList.add('done');
      setTimeout(() => { button.textContent = original; button.classList.remove('done'); }, 1400);
    }
    return true;
  } catch {
    return false;
  }
}

/** Read a File into a data URI, for the local preview strip only. */
export function previewUrl(file) {
  return URL.createObjectURL(file);
}

export function revokePreview(url) {
  if (url && url.startsWith('blob:')) URL.revokeObjectURL(url);
}
