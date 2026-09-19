/* api.js — the only module that talks to the server.
 *
 * Every call returns parsed JSON or throws an Error carrying a `status` and the
 * server's own message. The server reports failures as `{ok:false, error:"…"}`, so
 * unwrapping happens in exactly one place here rather than at thirty call sites.
 */

class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

/** Pull a human-readable message out of whatever shape the server sent. */
function messageFrom(payload, status, fallbackText) {
  if (payload && typeof payload === 'object') {
    const raw = payload.error ?? payload.message ?? payload.detail;
    if (typeof raw === 'string' && raw) return raw;
    if (raw) {
      try { return JSON.stringify(raw); } catch { /* fall through */ }
    }
  }
  if (fallbackText) {
    const trimmed = fallbackText.trim();
    // Never surface a raw HTML error page into a toast.
    if (trimmed && !/^\s*<(!doctype|html)/i.test(trimmed)) return trimmed.slice(0, 400);
  }
  return `request failed (HTTP ${status})`;
}

async function toJson(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { __raw: text };
  }
}

async function jsonRequest(path, { method = 'GET', body, signal, headers } = {}) {
  const init = { method, signal, headers: { ...(headers || {}) } };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new ApiError('cannot reach the local server — is it still running?', 0);
  }

  const payload = await toJson(response);

  if (!response.ok) {
    throw new ApiError(messageFrom(payload, response.status, payload && payload.__raw), response.status, payload);
  }
  if (payload && payload.ok === false) {
    throw new ApiError(messageFrom(payload, response.status, null), response.status, payload);
  }
  if (payload && typeof payload === 'object' && 'ok' in payload && 'data' in payload) {
    return payload.data;
  }
  return payload;
}

async function formRequest(path, formData, { signal } = {}) {
  let response;
  try {
    response = await fetch(path, { method: 'POST', body: formData, signal });
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new ApiError('the upload was interrupted', 0);
  }
  const payload = await toJson(response);
  if (!response.ok) {
    throw new ApiError(messageFrom(payload, response.status, payload && payload.__raw), response.status, payload);
  }
  if (payload && payload.ok === false) {
    throw new ApiError(messageFrom(payload, response.status, null), response.status, payload);
  }
  return payload && 'data' in payload ? payload.data : payload;
}

/* ── endpoints ─────────────────────────────────────────────────────────── */

export const api = {
  props: () => jsonRequest('/api/props'),
  health: () => jsonRequest('/api/health'),

  /** Masked status of the stored DeepSeek key. The key itself is never returned. */
  apiKey: () => jsonRequest('/api/settings/api-key'),

  /** Store a new key. It is in force for the next message — no restart. */
  setApiKey: (apiKey) =>
    jsonRequest('/api/settings/api-key', { method: 'POST', body: { api_key: apiKey } }),

  /** The limits the tool loop runs under, including `max_tool_steps`. */
  limits: () => jsonRequest('/api/settings/limits'),

  /** Change a limit (currently just `max_tool_steps`). Live for the next message. */
  setLimits: (limits) =>
    jsonRequest('/api/settings/limits', { method: 'POST', body: limits }),

  listConversations: () => jsonRequest('/api/conversations'),
  createConversation: (body = {}) => jsonRequest('/api/conversations', { method: 'POST', body }),
  getConversation: (uuid) => jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}`),
  deleteConversation: (uuid) => jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}`, { method: 'DELETE' }),
  renameConversation: (uuid, title) =>
    jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}/title`, { method: 'POST', body: { title } }),
  setSystem: (uuid, sysBase, sysTodo) =>
    jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}/system`, {
      method: 'POST',
      body: { sys_base: sysBase, sys_todo: sysTodo },
    }),
  setModel: (uuid, model) =>
    jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}/model`, { method: 'POST', body: { model } }),
  appendMessage: (uuid, content, extra = {}) =>
    jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}/messages`, {
      method: 'POST',
      body: { content, ...extra },
    }),
  truncate: (uuid, { keep, dropTail } = {}) =>
    jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}/truncate`, {
      method: 'POST',
      body: keep === undefined ? { drop_tail: dropTail || 0 } : { keep },
    }),
  context: (uuid) => jsonRequest(`/api/conversations/${encodeURIComponent(uuid)}/context`),
  importConversation: (payload) => jsonRequest('/api/conversations/import', { method: 'POST', body: payload }),

  /** Upload without appending — returns `{url, mime, bytes, block}`. */
  upload(uuid, file, kind) {
    const form = new FormData();
    form.append('file', file, file.name);
    if (kind) form.append('kind', kind);
    return formRequest(`/api/conversations/${encodeURIComponent(uuid)}/upload`, form);
  },

  /** Upload *and* append as a user turn in one request. */
  attach(uuid, file, { text = '', kind = '' } = {}) {
    const form = new FormData();
    form.append('file', file, file.name);
    if (kind) form.append('kind', kind);
    if (text) form.append('text', text);
    return formRequest(`/api/conversations/${encodeURIComponent(uuid)}/attach`, form);
  },

  mcp: () => jsonRequest('/api/mcp'),
  mcpReload: () => jsonRequest('/api/mcp/reload', { method: 'POST' }),

  /** Add a server to mcp.json. `body` is `{name, url, headers?, enabled?, prefix?, allowedTools?, timeout?}`. */
  mcpAddServer: (body) => jsonRequest('/api/mcp/servers', { method: 'POST', body }),

  /** Replace a server. Renaming is allowed and stays one operation server-side. */
  mcpUpdateServer: (name, body) => jsonRequest(`/api/mcp/servers/${encodeURIComponent(name)}`, {
    method: 'PUT',
    body,
  }),

  mcpDeleteServer: (name) => jsonRequest(`/api/mcp/servers/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  }),

  /** Turn a server on or off without resubmitting its definition. */
  mcpSetEnabled: (name, enabled) => jsonRequest(
    `/api/mcp/servers/${encodeURIComponent(name)}/enabled`,
    { method: 'POST', body: { enabled } },
  ),

  /** Connect once, without saving. Pass either a full definition or just `{name}`. */
  mcpTest: (body) => jsonRequest('/api/mcp/test', { method: 'POST', body }),

  /** Download a conversation as JSON, then trigger a save in the browser. */
  async downloadConversation(uuid, title) {
    const response = await fetch(`/api/conversations/${encodeURIComponent(uuid)}/export`);
    if (!response.ok) throw new ApiError(`export failed (HTTP ${response.status})`, response.status);
    const blob = await response.blob();
    const safe = (title || uuid).replace(/[^\w \-]+/g, '_').slice(0, 60) || uuid;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${safe}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  },
};

/* ── SSE ───────────────────────────────────────────────────────────────── */

/**
 * Parse an SSE byte stream into events.
 *
 * Frames are delimited by a blank line, but a read() can land anywhere — mid
 * frame, mid JSON, even mid \r\n. The buffer therefore stays outside the loop and
 * only complete frames are consumed. `\r` is normalised first because a proxy is
 * free to re-encode line endings.
 *
 * @param {ReadableStream<Uint8Array>} stream
 */
export async function* iterSse(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, '\n');

      let cut;
      while ((cut = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);

        let event = 'message';
        const dataLines = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith(':')) continue;              // keep-alive comment
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
        }
        if (!dataLines.length) continue;

        const raw = dataLines.join('\n');
        let data;
        try {
          data = JSON.parse(raw);
        } catch {
          data = { text: raw };
        }
        yield { event, data };
      }
    }
    // A well-behaved server ends with a blank line, but handle a truncated tail.
    const tail = buffer.trim();
    if (tail.startsWith('data:')) {
      const raw = tail.split('\n').filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).replace(/^ /, '')).join('\n');
      try {
        yield { event: 'message', data: JSON.parse(raw) };
      } catch { /* ignore a partial frame */ }
    }
  } finally {
    try { reader.releaseLock(); } catch { /* already released */ }
  }
}

/**
 * Stream one chat turn.
 *
 * `onEvent(event, data)` is called for every frame. The server emits `event:` for
 * each — meta, warning, step, reasoning, content, tool_pending, tool_call,
 * tool_result, usage, title, done, error — and onEvent is expected to switch on it.
 *
 * HTTP status is committed before the first frame, so a failure that happens later
 * arrives as an `error` event rather than a rejected promise. Both paths are real.
 */
export async function streamChat(uuid, body, onEvent, { signal } = {}) {
  let response;
  try {
    response = await fetch(`/api/chat/${encodeURIComponent(uuid)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new ApiError('cannot reach the local server', 0);
  }

  if (!response.ok) {
    const payload = await toJson(response);
    throw new ApiError(messageFrom(payload, response.status, payload && payload.__raw), response.status, payload);
  }
  if (!response.body) throw new ApiError('the server sent no stream body', response.status);

  for await (const frame of iterSse(response.body)) {
    if (frame.event !== 'message') onEvent(frame.event, frame.data);
  }
}

export { ApiError };
