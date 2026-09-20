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

    if (type === 'image' || type === 'video' || type === 'audio' || type === 'embed') {
      // The remote fields only exist on a block the server marked `source: 'remote'`,
      // but they are copied unconditionally: `elementForBlock` also receives blocks
      // straight off the SSE wire, which have never been through here at all.
      out.push({
        type,
        url: String(item.url || ''),
        name: String(item.name || ''),
        mime: String(item.mimeType || item.mime || ''),
        width: item.width,
        height: item.height,
        // `stream` is a loopback proxy URL when the browser would be refused by the
        // host directly; it beats `url` for playback and is meaningless for download.
        stream: String(item.stream || ''),
        poster: String(item.poster || ''),
        source: String(item.source || ''),
        note: String(item.note || ''),
        caption: String(item.caption || ''),
        duration: Number(item.duration) || 0,
        embedUrl: String(item.embed_url || item.embedUrl || ''),
        provider: String(item.provider || ''),
        title: String(item.title || ''),
        author: String(item.author || ''),
        frames: Array.isArray(item.frames) ? item.frames.map(String) : [],
      });
      continue;
    }

    // Unknown block: show the text if there is any, otherwise keep the door open
    // rather than silently swallowing content the model was told to expect.
    if (typeof item.text === 'string' && item.text) out.push({ type: 'text', text: item.text });
  }
  return out;
}

/* Lowercase extensions, for deciding rather than for display: `extOf` returns the
   caption's `MP4`/`JPG`, and prefers the name over the mime, which is the wrong way
   round for a question about the bytes. */
const NAME_EXT = /\.([a-z0-9]{1,5})$/i;
const AUDIO_EXT = /^(mp3|wav|m4a|aac|ogg|oga|opus|flac|weba)$/;
const VIDEO_EXT = /^(mp4|m4v|webm|mov|mkv|avi|ogv)$/;
const IMAGE_EXT = /^(png|jpe?g|gif|webp|avif|bmp|svg)$/;

/** The lowercase extension of a file name, or `''`. */
function nameExt(name) {
  const match = NAME_EXT.exec(String(name || ''));
  return match ? match[1].toLowerCase() : '';
}

/**
 * Which element a block wants, so that "what is this" is asked once.
 *
 * The declared `type` normally settles it and is not worth second-guessing. But a
 * block that only says `file`, or says nothing at all, is not a reason to hand over a
 * download chip for something the browser would have played — a URL with no suffix, a
 * still a tool published under a bare name. So the mime is consulted next and the file
 * name after it: a content type is a statement about the bytes, a `.mp4` is a hint.
 */
export function presentationFor(block) {
  if (!block) return 'file';
  const type = String(block.type || '');
  if (type === 'embed' || type === 'image' || type === 'video' || type === 'audio') return type;

  const mime = String(block.mime || '');
  if (isImageMime(mime)) return 'image';
  if (isVideoMime(mime)) return 'video';
  if (mime.startsWith('audio/')) return 'audio';

  const ext = nameExt(block.name);
  if (IMAGE_EXT.test(ext)) return 'image';
  if (VIDEO_EXT.test(ext)) return 'video';
  if (AUDIO_EXT.test(ext)) return 'audio';
  return 'file';
}

/** True when a block is an image or video the user should see rendered. */
export function isMediaBlock(block) {
  if (!block) return false;
  const kind = presentationFor(block);
  // An embed is media the user sees, just not media we hold: the player is a frame
  // from someone else's origin, so the block is a page URL plus an iframe URL. The
  // iframe URL is what makes it playable — a page URL with no player behind it falls
  // through to the file chip, which at least links somewhere.
  if (kind === 'embed') return Boolean(block.embed_url || block.embedUrl);
  if (kind !== 'image' && kind !== 'video' && kind !== 'audio') return false;
  // `stream` counts as much as `url` for a player; `url` is what the caption links to.
  return Boolean(block.url || block.stream);
}

/** True when a block is an embed we should draw a play affordance for. */
export function isEmbedBlock(block) {
  return Boolean(block) && block.type === 'embed';
}

/** `1:23`, `1h 02m 03s` — the compact form of a duration in seconds. */
export function durationText(seconds) {
  const total = Math.round(Number(seconds) || 0);
  if (total <= 0) return '';
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`;
  if (minutes) return `${minutes}:${String(secs).padStart(2, '0')}`;
  return `${secs}s`;
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
  // The fallback label is the *rendered* kind, not `block.type`, so a block that only
  // declared itself a `file` is not captioned as an image it is not.
  const kind = presentationFor(block);
  const remote = block.source === 'remote';
  const label = block.caption || block.title || block.name
    || (kind === 'video' || kind === 'embed' ? 'video' : kind === 'audio' ? 'audio' : 'image');
  const detail = [];
  if (kind !== 'embed') detail.push(extOf(block.mime, block.name));
  if (block.width && block.height) detail.push(`${block.width}×${block.height}`);
  const duration = durationText(block.duration);
  if (duration) detail.push(duration);
  const title = [describeBlock(block), block.caption, block.provider].filter(Boolean).join(' · ');
  // `download` does nothing on a cross-origin URL, so a remote source gets a link
  // that says what it really does instead: open the original.
  const action = remote
    ? `<a href="${escapeHtml(block.url)}" target="_blank" rel="noopener" title="Open the original">↗</a>`
    : `<a href="${escapeHtml(block.url)}" download title="Download">▼</a>`;
  const badge = remote
    ? '<span class="media-badge" title="Streamed from the original URL — nothing was downloaded">remote</span>'
    : '';
  return `<div class="media-cap"><span title="${escapeHtml(title)}">${escapeHtml(label)}</span>`
    + badge
    + `<span class="spacer"></span>${action}`
    + `<span>${escapeHtml(detail.join(' · '))}</span></div>`;
}

const PLAY_ICON = '<svg class="play-icon" viewBox="0 0 24 24" aria-hidden="true">'
  + '<path d="M8 5v14l11-7z"/></svg>';

/**
 * A poster with a play button that becomes the player only when clicked.
 *
 * The iframe is not in the DOM until then, and that is the whole point: an embed
 * loads a complete third-party player, sets cookies and counts a view, and the
 * transcript is re-rendered every time a turn is sent. Building it on click means a
 * conversation can hold ten of these and pay for exactly the ones that got played.
 * It is also why the poster matters — without it there is nothing to look at, so a
 * provider that publishes no still falls back to naming itself.
 */
function embedElement(block) {
  const source = block.embedUrl || block.embed_url || '';
  const label = block.title || block.name || `${block.provider || 'video'}`;

  const figure = document.createElement('figure');
  figure.className = 'media-item media-embed';
  figure.innerHTML =
    `<button type="button" class="embed-play"${source ? '' : ' disabled'}`
    + ` title="Play ${escapeHtml(label)}">`
    + `${block.poster ? `<img src="${escapeHtml(block.poster)}" alt="" loading="lazy">` : ''}`
    + '<span class="play-scrim"></span>'
    + PLAY_ICON
    + '</button>'
    + mediaCaption(block);

  const button = figure.querySelector('.embed-play');
  if (button && source) {
    button.addEventListener('click', () => {
      const frame = document.createElement('iframe');
      frame.className = 'embed-frame';
      frame.src = source;
      frame.title = label;
      frame.setAttribute('referrerpolicy', 'strict-origin-when-cross-origin');
      // `web-share` is deliberately absent: Chrome dropped it from the allowlist and
      // now logs an "Unrecognized feature" warning for every player on the page.
      frame.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture');
      frame.setAttribute('allowfullscreen', '');
      button.replaceWith(frame);
    });
  }
  return figure;
}

/**
 * A codec the browser will not take is not an error the user can act on, and a dead
 * `<video>` is a black rectangle with controls and no explanation. Swapping in the same
 * chip the unknown-type path draws says what the file is and offers the bytes, which is
 * also what a format the machine cannot play deserves.
 */
function fallbackToChip(figure, media, block) {
  if (!media) return;
  media.addEventListener('error', () => {
    figure.classList.add('broken');
    figure.replaceChildren(fileChip(block));
  });
}

/** The last resort: the name we have for the bytes, as a link if there is somewhere to link to. */
function fileChip(block) {
  const chip = document.createElement('span');
  chip.className = 'file-chip';
  const name = block.name || extOf(block.mime);
  chip.innerHTML = block.url
    ? `<svg viewBox="0 0 24 24"><path d="M14 3v5h5M6 3h8l5 5v13H6z"/></svg><a href="${escapeHtml(block.url)}" download>${escapeHtml(name)}</a>`
    : `<svg viewBox="0 0 24 24"><path d="M14 3v5h5M6 3h8l5 5v13H6z"/></svg>${escapeHtml(name)}`;
  return chip;
}

/**
 * Build the element for one media block.
 * `onZoom(url)` is called when an image is clicked.
 *
 * Which element that is comes from `presentationFor`, which reads the mime and the
 * file name as well as the type — so a block that arrived without a usable type still
 * gets a player rather than a chip.
 */
export function elementForBlock(block, { onZoom = null } = {}) {
  // First, before `presentationFor` is asked anything: a code block is not media and
  // has no mime or name to infer from, and a chip of source is what it wants.
  if (block && block.type === 'code') return codeChip(block);

  const kind = presentationFor(block);

  if (kind === 'image' && block.url) {
    const figure = document.createElement('figure');
    figure.className = 'media-item';
    figure.innerHTML =
      `<img src="${escapeHtml(block.url)}" alt="${escapeHtml(block.name || 'image')}" loading="lazy"`
      + `${block.width ? ` width="${Number(block.width)}"` : ''}`
      + `${block.height ? ` height="${Number(block.height)}"` : ''}>`
      + mediaCaption(block);
    const img = figure.querySelector('img');
    // A /memory/ URL can 404 if the file was deleted by hand, and a format the browser
    // cannot decode is not a picture either; say so rather than leaving a broken-image
    // glyph in the transcript.
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

  if (kind === 'video' && (block.url || block.stream)) {
    // `stream` first: a proxy URL exists exactly when the browser could not be
    // pointed at the original. `url` stays the identity of the media — the caption,
    // the download link and the note the model was given all use it.
    const src = block.stream || block.url;
    const figure = document.createElement('figure');
    figure.className = 'media-item';
    figure.innerHTML =
      `<video src="${escapeHtml(src)}" controls preload="metadata"`
      + `${block.poster ? ` poster="${escapeHtml(block.poster)}"` : ''}`
      + `${block.width ? ` width="${Number(block.width)}"` : ''}></video>`
      + mediaCaption(block);
    fallbackToChip(figure, figure.querySelector('video'), block);
    return figure;
  }

  if (kind === 'embed') {
    return embedElement(block);
  }

  if (kind === 'audio' && block.url) {
    const figure = document.createElement('figure');
    figure.className = 'media-item';
    figure.innerHTML = `<audio src="${escapeHtml(block.url)}" controls></audio>` + mediaCaption(block);
    fallbackToChip(figure, figure.querySelector('audio'), block);
    return figure;
  }

  return fileChip(block);
}

/* ── the row: one item, or a carousel of several ───────────────────────── */

/**
 * A horizontal track of media with prev/next arrows and dots.
 *
 * Several things to look at used to wrap onto as many rows as they needed, which for
 * four videos is four screens of scrolling; a carousel is one row whatever it holds.
 * It is a plain `media-grid` for a single item — the arrows would have nowhere to go,
 * and one image should not have to look interactive.
 *
 * What each item *is* stays `elementForBlock`'s business: an image, a player, a poster
 * or a chip. Nothing here decides that, so a new kind of media joins the carousel by
 * being one more element.
 *
 * Stepping is `scrollTo` on the track rather than a re-layout, so the arrows, a
 * trackpad, a wheel and a thumb swipe all move the same element, and the arrows are
 * computed from where the track actually is (`scrollLeft`) rather than from a counter
 * that a smooth scroll can leave out of step.
 */
export function mediaRow(blocks, { onZoom = null } = {}) {
  const items = [];
  for (const block of blocks) {
    if (!isMediaBlock(block)) continue;
    const element = elementForBlock(block, { onZoom });
    if (element) items.push(element);
  }

  const root = document.createElement('div');
  root.className = 'media-grid';
  if (!items.length) return null;
  if (items.length === 1) {
    root.append(items[0]);
    return root;
  }

  root.classList.add('carousel');
  const track = document.createElement('div');
  track.className = 'car-track';
  track.append(...items);

  const count = document.createElement('span');
  count.className = 'car-count';
  const dots = document.createElement('div');
  dots.className = 'car-dots';
  const foot = document.createElement('div');
  foot.className = 'car-foot';
  foot.append(count, dots);

  const nav = [];
  for (const [direction, label] of [[-1, 'Previous'], [1, 'Next']]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = direction < 0 ? 'car-nav prev' : 'car-nav next';
    button.title = label;
    button.setAttribute('aria-label', label);
    button.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="'
      + (direction < 0 ? 'M14.5 5.5L8 12l6.5 6.5' : 'M9.5 5.5L16 12l-6.5 6.5')
      + '"/></svg>';
    nav.push(button);
  }

  root.append(track, nav[0], nav[1], foot);

  let at = 0;
  let fits = false;
  const goTo = (index) => {
    const item = items[Math.max(0, Math.min(items.length - 1, index))];
    // Measured against the track, not the page: both rects change with the scroll.
    if (item) track.scrollTo({ left: item.offsetLeft - track.offsetLeft, behavior: 'smooth' });
  };

  let frame = 0;
  // A disabled arrow is also what the stylesheet hides, so the two agree on the day
  // one of them is looked at without the other.
  const step = () => {
    nav[0].disabled = fits || at <= 0;
    nav[1].disabled = fits || at >= items.length - 1;
  };
  const sync = () => {
    const max = track.scrollWidth - track.clientWidth;
    let nearest = Infinity;
    items.forEach((item, index) => {
      const distance = Math.abs(item.offsetLeft - track.offsetLeft - track.scrollLeft);
      if (distance < nearest) { nearest = distance; at = index; }
    });
    // The end of the track is the one place "nearest item start" cannot reach: the
    // last item's left edge never arrives at the track's left edge, because the
    // scroll stops first. Two 460px items in the transcript's 790px leave the second
    // one 142px short, so the nearest start would still be the first item's — the
    // counter would stall at "1 / 2" with Next lit and nothing left for it to do.
    // A scroll that is as far right as it goes means the last item, whatever the
    // pixels say. (`max <= 2` is a row that fits, where the last item is not current.)
    if (max > 2 && max - track.scrollLeft <= 2) at = items.length - 1;
    // A detached element measures 0, which must not read as "all of it fits".
    fits = track.clientWidth > 0 && max <= 2;
    root.classList.toggle('fits', fits);
    count.textContent = `${at + 1} / ${items.length}`;
    dots.replaceChildren(...items.map((_, index) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = index === at ? 'car-dot on' : 'car-dot';
      dot.title = `Show item ${index + 1} of ${items.length}`;
      dot.setAttribute('aria-label', dot.title);
      dot.addEventListener('click', () => goTo(index));
      return dot;
    }));
    step();
  };

  const schedule = () => {
    if (frame) return;
    // A timer in a hidden tab and a frame in a visible one: `requestAnimationFrame`
    // never fires while the tab is in the background, so a transcript re-rendered
    // there would keep a carousel with no counter, no dots and arrows that have not
    // been disabled yet, and nothing would correct it until something scrolled.
    if (document.hidden) frame = setTimeout(() => { frame = 0; sync(); }, 0);
    else frame = requestAnimationFrame(() => { frame = 0; sync(); });
  };

  nav[0].addEventListener('click', () => { goTo(at - 1); step(); });
  nav[1].addEventListener('click', () => { goTo(at + 1); step(); });
  track.addEventListener('scroll', schedule, { passive: true });
  // An image that finishes loading after the row was built changes the track's width,
  // which is what decides whether the arrows are needed at all. A listener on the row
  // rather than on `window`, because the transcript is re-rendered every turn and a
  // global one would outlive every carousel it was added for.
  root.addEventListener('load', schedule, true);
  root.addEventListener('error', schedule, true);
  schedule();
  return root;
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
