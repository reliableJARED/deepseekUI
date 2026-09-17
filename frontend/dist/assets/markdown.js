/* markdown.js — a small, self-contained Markdown renderer.
 *
 * No dependency and no CDN: a privacy-first tool that phones out to a JS CDN on
 * every load would be a contradiction, and the model's output is untrusted text.
 *
 * The security model is simple and absolute: **everything is HTML-escaped before
 * any markup is generated.** Tags are then produced only from escaped text, so no
 * input can ever become an element. Link targets get a second check on top, because
 * an escaped href can still be `javascript:`.
 */

import { highlight, languageLabel, resolveLanguage } from './highlight.js';

const ESCAPE = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ESCAPE[ch]);
}

/** Only allow schemes that cannot execute. Everything else becomes inert text. */
function safeHref(href) {
  const raw = String(href || '').trim();
  if (!raw) return '';
  // Strip control characters, which browsers ignore when resolving a scheme.
  const cleaned = raw.replace(/[\u0000-\u001f\u007f]/g, '');
  if (/^(https?:|mailto:)/i.test(cleaned)) return cleaned;
  if (/^[/#]/.test(cleaned)) return cleaned;             // local or in-page
  if (/^[\w.\-]+@[\w.\-]+$/.test(cleaned)) return `mailto:${cleaned}`;
  return '';
}

/* ── inline ────────────────────────────────────────────────────────────── */

function renderInline(escaped, codes) {
  let out = escaped;

  // Code spans first: their content must not be touched by any other rule.
  // Placeholders were substituted before escaping, so they are plain ASCII tokens.
  out = out.replace(/\u0000C(\d+)\u0000/g, (_, index) => {
    const code = codes[Number(index)] ?? '';
    return `<code>${code}</code>`;
  });

  // Images before links (identical prefix, different meaning). Escaped `<img>` is
  // intentionally NOT rendered — an inline image in chat text is almost always a
  // hallucinated URL, and a broken tracker is worse than a visible link.
  out = out.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (whole, alt, url) => {
    const href = safeHref(url);
    return href ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${alt || href}</a>` : alt || whole;
  });

  out = out.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (whole, label, url) => {
    const href = safeHref(url);
    return href ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>` : label || whole;
  });

  // Bare URLs, but not the ones inside an href we just wrote.
  out = out.replace(/(^|[\s(])(https?:\/\/[^\s<>")\]]+)/g, (_, lead, url) =>
    `${lead}<a href="${safeHref(url)}" target="_blank" rel="noopener noreferrer">${url}</a>`);

  out = out.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[^*\w])\*([^*\n]+)\*(?=[^*\w]|$)/g, '$1<em>$2</em>');
  out = out.replace(/(^|[^_\w])_([^_\n]+)_(?=[^_\w]|$)/g, '$1<em>$2</em>');
  out = out.replace(/~~([^~]+)~~/g, '<del>$1</del>');

  return out;
}

/* ── blocks ────────────────────────────────────────────────────────────── */

function isTableDivider(line) {
  return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
}

function splitRow(line) {
  let text = line.trim();
  if (text.startsWith('|')) text = text.slice(1);
  if (text.endsWith('|')) text = text.slice(0, -1);
  return text.split('|').map((cell) => cell.trim());
}

function listItem(line) {
  const match = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/.exec(line);
  if (!match) return null;
  return {
    indent: match[1].replace(/\t/g, '    ').length,
    ordered: /\d/.test(match[2]),
    text: match[3],
  };
}

/**
 * Render Markdown to HTML.
 *
 * @param {string} source
 * @returns {string} escaped, safe HTML
 */
export function markdown(source) {
  if (!source) return '';
  const text = String(source).replace(/\r\n?/g, '\n');

  const codes = [];

  /* 1. Fenced and indented code first, replaced by placeholders so nothing below
   *    can reinterpret their contents. Placeholders use \u0000, a character that
   *    cannot appear in the input in any meaningful way and survives escaping. */
  let prepared = text.replace(/```([^\n`]*)\n([\s\S]*?)(?:```|$)/g, (_, info, body) => {
    codes.push({ lang: String(info).trim(), body });
    return `\u0000C${codes.length - 1}\u0000`;
  });
  prepared = prepared.replace(/~~~([^\n~]*)\n([\s\S]*?)(?:~~~|$)/g, (_, info, body) => {
    codes.push({ lang: String(info).trim(), body });
    return `\u0000C${codes.length - 1}\u0000`;
  });

  /* 2. Inline code, before escaping, so its contents can be escaped separately. */
  prepared = prepared.replace(/(`+)([\s\S]*?)\1/g, (_, __, body) => {
    codes.push({ lang: '', body, inline: true });
    return `\u0000C${codes.length - 1}\u0000`;
  });

  // Inline spans keep the plain escape. Fenced blocks are highlighted instead, and
  // `highlight` escapes as it wraps each token — pre-escaping a block here would
  // escape the largest string in the transcript twice and would hand the tokenizer
  // `&amp;` where the source said `&`.
  const escapedCodes = codes.map((entry) => (entry.inline ? escapeHtml(entry.body) : null));

  /* 3. Escape everything that remains, then build structure from escaped text. */
  const lines = escapeHtml(prepared).split('\n');
  const html = [];
  let paragraph = [];
  const listStack = [];       // {tag, indent}
  let inQuote = false;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const joined = paragraph.join('\n');
    html.push(`<p>${renderInline(joined, escapedCodes).replace(/\n/g, '<br>\n')}</p>`);
    paragraph = [];
  };

  const closeLists = (toIndent = -1) => {
    while (listStack.length && listStack[listStack.length - 1].indent > toIndent) {
      html.push(`</${listStack.pop().tag}>`);
    }
  };

  const closeQuote = () => {
    if (inQuote) { html.push('</blockquote>'); inQuote = false; }
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    // A code placeholder occupies a line of its own.
    const placeholder = /^\u0000C(\d+)\u0000$/.exec(line.trim());
    if (placeholder) {
      flushParagraph(); closeLists(); closeQuote();
      const index = Number(placeholder[1]);
      const entry = codes[index];
      if (entry && !entry.inline) {
        // The fence's own info string wins; when it named nothing (or something we
        // do not know) the block itself is asked what it is. The raw hint is still
        // what gets shown when we could not resolve it, so a ````ocaml`` fence does
        // not silently become "text".
        const body = entry.body.replace(/\n+$/, '');
        const lang = resolveLanguage(body, entry.lang);
        const label = lang ? languageLabel(lang) : (entry.lang || 'text');
        const copy = '<button class="code-copy" type="button">copy</button>';
        html.push(
          `<div class="code-block" data-lang="${escapeHtml(lang || 'text')}">`
          + `<div class="code-head"><span>${escapeHtml(label)}</span>`
          + `<span class="spacer"></span>${copy}</div>`
          + `<pre><code>${highlight(body, lang)}</code></pre></div>`);
      } else {
        html.push(`<p>${escapedCodes[index]}</p>`);
      }
      continue;
    }

    if (!line.trim()) { flushParagraph(); closeLists(); closeQuote(); continue; }

    // Horizontal rule
    if (/^\s*([-*_])\s*(?:\1\s*){2,}$/.test(line)) {
      flushParagraph(); closeLists(); closeQuote();
      html.push('<hr>');
      continue;
    }

    // ATX heading
    const heading = /^(#{1,6})\s+(.*)$/.exec(line.trim());
    if (heading) {
      flushParagraph(); closeLists(); closeQuote();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2].replace(/\s+#+\s*$/, ''), escapedCodes)}</h${level}>`);
      continue;
    }

    // Table: a header row followed by a divider row
    if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      flushParagraph(); closeLists(); closeQuote();
      const head = splitRow(line);
      const rows = [];
      let j = i + 2;
      while (j < lines.length && lines[j].includes('|') && lines[j].trim()) {
        rows.push(splitRow(lines[j]));
        j += 1;
      }
      i = j - 1;
      const headHtml = head.map((cell) => `<th>${renderInline(cell, escapedCodes)}</th>`).join('');
      const bodyHtml = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell, escapedCodes)}</td>`).join('')}</tr>`)
        .join('');
      html.push(`<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`);
      continue;
    }

    // Blockquote
    if (/^\s*&gt;\s?/.test(line)) {
      flushParagraph(); closeLists();
      if (!inQuote) { html.push('<blockquote>'); inQuote = true; }
      html.push(`<p>${renderInline(line.replace(/^\s*&gt;\s?/, ''), escapedCodes)}</p>`);
      continue;
    }
    closeQuote();

    // List item
    const item = listItem(line);
    if (item) {
      flushParagraph();
      const tag = item.ordered ? 'ol' : 'ul';
      if (!listStack.length || item.indent > listStack[listStack.length - 1].indent) {
        listStack.push({ tag, indent: item.indent });
        html.push(`<${tag}>`);
      } else {
        closeLists(item.indent);
        if (!listStack.length || listStack[listStack.length - 1].indent < item.indent) {
          listStack.push({ tag, indent: item.indent });
          html.push(`<${tag}>`);
        } else if (listStack[listStack.length - 1].tag !== tag) {
          // Same depth, different marker: a new list rather than a mangled one.
          html.push(`</${listStack.pop().tag}>`);
          listStack.push({ tag, indent: item.indent });
          html.push(`<${tag}>`);
        }
      }
      html.push(`<li>${renderInline(item.text, escapedCodes)}</li>`);
      continue;
    }

    // Continuation of the previous list item, indented.
    if (listStack.length && /^\s{2,}\S/.test(line)) {
      const last = html.length - 1;
      if (html[last] && html[last].endsWith('</li>')) {
        html[last] = html[last].slice(0, -5) + `\n${renderInline(line.trim(), escapedCodes)}</li>`;
      } else {
        html.push(renderInline(line.trim(), escapedCodes));
      }
      continue;
    }

    closeLists();
    paragraph.push(line);
  }

  flushParagraph();
  closeLists();
  closeQuote();

  return html.join('\n');
}

/** Strip formatting down to plain text — used for titles and copy actions. */
export function toPlainText(source) {
  return String(source || '')
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/^```[^\n]*\n?|```$/g, ''))
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s*&gt;\s?/gm, '')
    .replace(/(\*\*|__|~~)/g, '')
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1$2')
    .replace(/(^|[^_\w])_([^_\n]+)_/g, '$1$2')
    .trim();
}

export { escapeHtml, safeHref };
