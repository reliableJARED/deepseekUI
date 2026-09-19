/* state.js — the browser-local half of the app.
 *
 * Deliberately the *small* half. Conversations, messages and media live on the
 * server's disk where they can be inspected with Explorer; localStorage holds only
 * UI preferences, because losing them costs nothing and syncing them would cost
 * privacy.
 */

const KEY = 'deepseekui.prefs.v1';

const DEFAULTS = {
  theme: 'dark',
  defaultSystem: '',
  effort: '',              // '' means "whatever the model defaults to"
  // Reasoning and tool arguments/results are the bulk of a long transcript and
  // are read far less often than the answer, so they open closed by default.
  // Deliberately not named `showReasoning`: the block is always *shown*, it is
  // only collapsed. An old build's `showReasoning: true` is now an unknown key
  // and is dropped by `read()` below, which is what makes this default stick.
  expandReasoning: false,
  autoscroll: true,
  enterSends: true,
  tools: true,
  lastUuid: '',
  sidebarCollapsed: false,
  bannerDismissed: false,
  expandedTools: {},       // uuid -> { [callId]: bool } — which tool cards were opened
};

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    // Unknown keys are dropped rather than trusted: an old build's leftovers must
    // not be able to reintroduce behaviour this one no longer has.
    const merged = { ...DEFAULTS };
    for (const key of Object.keys(DEFAULTS)) {
      if (key in parsed) merged[key] = parsed[key];
    }
    if (!merged.expandedTools || typeof merged.expandedTools !== 'object') merged.expandedTools = {};
    return merged;
  } catch {
    return { ...DEFAULTS };
  }
}

export const prefs = read();

let saveTimer = 0;

/** Persist, coalescing bursts (typing into a textarea fires on every keystroke). */
export function savePrefs(immediate = false) {
  if (saveTimer) clearTimeout(saveTimer);
  const write = () => {
    saveTimer = 0;
    try {
      localStorage.setItem(KEY, JSON.stringify(prefs));
    } catch {
      /* private mode, or the quota is full — preferences are disposable */
    }
  };
  if (immediate) write();
  else saveTimer = setTimeout(write, 250);
}

export function resetPrefs() {
  Object.assign(prefs, DEFAULTS, { expandedTools: {} });
  savePrefs(true);
}

/* ── runtime state, never persisted ────────────────────────────────────── */

export const app = {
  props: null,             // /api/props
  conversations: [],       // summaries
  conv: null,              // the full record for the selected conversation
  context: null,           // the last context/fill reading
  streaming: false,
  controller: null,        // AbortController for the live turn
  attachments: [],         // staged File objects
  lastUserText: '',
  turnError: null,         // {uuid, message, hint} from the last failed turn
  pendingEdit: null,       // {uuid, index, text, media} — a message loaded for rewriting
  booted: false,
};

/** The model spec currently selected, or null before props have loaded. */
export function currentModel() {
  if (!app.props) return null;
  const wanted = app.conv?.model || app.props.default_model;
  const models = app.props.models || [];
  return models.find((m) => m.id === wanted) || models[0] || null;
}

export function models() {
  return app.props?.models || [];
}

export function isConfigured() {
  return Boolean(app.props && app.props.configured);
}

/** True when the selected model cannot accept an image. */
export function visionBlocked() {
  const model = currentModel();
  return Boolean(model && model.vision === false);
}
