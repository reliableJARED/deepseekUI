/* mcpheaders.js — the text<->object rules for an MCP server's headers, plus the
 * one special case in them: the API key.
 *
 * This is a module of its own rather than a handful of helpers buried in app.js
 * because it is the only part of the MCP editor that can be wrong *silently*. A
 * mistyped header name or a key that fails to survive the round trip does not
 * throw — it produces a server that connects and then refuses every call, which
 * looks exactly like a broken server. Pure functions here can be executed by
 * assets/__selftest.html; the same code inside app.js could only be inspected.
 *
 * The key rides in an ordinary header (`x-api-key`) rather than in a field of its
 * own, so nothing downstream needs to know about it: `server/mcp.py` already
 * forwards a spec's headers to every request, and the tool server reads them off
 * the request. The editor gives it a dedicated input anyway, because it is the one
 * header users have to go and look up, and because a key deserves a password box.
 */

/** The header an MCP server's API key is sent in. */
export const API_KEY_HEADER = 'x-api-key';

/** `{a: 1, b: 2}` -> `"a: 1\nb: 2"`, the format the textarea shows. */
export function formatHeaders(headers) {
  return Object.entries(headers || {}).map(([key, value]) => `${key}: ${value}`).join('\n');
}

/**
 * `"Name: Value"` per line to an object. Blank lines and `#` comments are ignored.
 *
 * Throws on a line with no name, because the alternative is dropping it: a header
 * the user typed and the app silently discarded is the failure mode this whole
 * module exists to prevent.
 */
export function parseHeaders(text) {
  const headers = {};
  for (const raw of String(text || '').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const at = line.indexOf(':');
    const name = at < 0 ? '' : line.slice(0, at).trim();
    if (!name) throw new Error(`headers must be "Name: Value" — could not read ${line}`);
    headers[name] = line.slice(at + 1).trim();
  }
  return headers;
}

/** The key as it sits in a server's headers, whatever casing it was stored with. */
export function mcpApiKeyOf(headers) {
  const name = Object.keys(headers || {}).find((key) => key.toLowerCase() === API_KEY_HEADER);
  return name ? headers[name] : '';
}

/**
 * Every header except the key — the key has its own field in the editor, so leaving
 * it in the textarea as well would be a second source of truth, and the one that
 * silently wins is always the one the user is not looking at.
 */
export function mcpHeadersWithoutKey(headers) {
  const rest = {};
  for (const [name, value] of Object.entries(headers || {})) {
    if (name.toLowerCase() !== API_KEY_HEADER) rest[name] = value;
  }
  return rest;
}

/**
 * The key field applied to the parsed headers: the field always wins, so clearing
 * it really does clear the key instead of leaving the old one alive in the textarea.
 * Any casing of the header is removed, and the one we write back is canonical.
 */
export function withApiKey(headers, apiKey) {
  const merged = mcpHeadersWithoutKey(headers);
  const key = String(apiKey || '').trim();
  if (key) merged[API_KEY_HEADER] = key;
  return merged;
}
