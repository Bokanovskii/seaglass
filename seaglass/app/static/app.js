const token = (() => {
  const hash = window.location.hash.replace(/^#/, '');
  if (hash) {
    sessionStorage.setItem('seaglass-token', hash);
    history.replaceState(null, '', window.location.pathname + window.location.search);
  }
  return sessionStorage.getItem('seaglass-token') || '';
})();

const state = {
  assist: 'auto',
  people: [],
  selectedChat: null,
  datePreset: '',  // '' (All) | 'custom' | a day count, mirrors the preset buttons
  requestId: null,
  assistToken: null,
  results: [],
  lastPayload: null,
  shownResults: 0,
  // pagination
  pageQuery: '',
  pageFilters: {},
  nextOffset: 0,
  hasMore: false,
  loadingMore: false,
};

const els = {
  loading: document.getElementById('loading-screen'),
  buildScreen: document.getElementById('build-screen'),
  buildIdle: document.getElementById('build-idle'),
  buildStart: document.getElementById('build-start'),
  buildProgress: document.getElementById('build-progress'),
  buildStageText: document.getElementById('build-stage-text'),
  buildProgressBar: document.getElementById('build-progress-bar'),
  buildDetail: document.getElementById('build-detail'),
  buildError: document.getElementById('build-error'),
  app: document.getElementById('app'),
  progressBar: document.getElementById('progress-bar'),
  warmupSteps: document.getElementById('warmup-steps'),
  loadingError: document.getElementById('loading-error'),
  query: document.getElementById('query'),
  searchButton: document.getElementById('search-button'),
  results: document.getElementById('results'),
  resultsMeta: document.getElementById('results-meta'),
  statusBar: document.getElementById('status-bar'),
  assistBanner: document.getElementById('assist-banner'),
  syncBanner: document.getElementById('sync-banner'),
  peopleInput: document.getElementById('people-input'),
  peopleSuggestions: document.getElementById('people-suggestions'),
  peopleChips: document.getElementById('people-chips'),
  chatInput: document.getElementById('chat-input'),
  chatSuggestions: document.getElementById('chat-suggestions'),
  drawer: document.getElementById('drawer'),
  drawerContent: document.getElementById('drawer-content'),
  dateFrom: document.getElementById('date-from'),
  dateTo: document.getElementById('date-to'),
  dates: document.querySelector('.dates'),
  dateSummary: document.getElementById('date-summary'),
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, {...options, headers});
  if (response.status === 204) return null;
  const payload = await response.json();
  if (!response.ok) {
    const err = new Error(payload.detail || `Request failed: ${response.status}`);
    err.status = response.status;
    throw err;
  }
  return payload;
}

async function pollHealth() {
  const health = await api('/api/health');
  if (health.state === 'NEEDS_INDEX' && !(health.build && health.build.running)) {
    els.loading.classList.add('hidden');
    els.app.classList.add('hidden');
    els.buildScreen.classList.remove('hidden');
    els.buildIdle.classList.remove('hidden');
    els.buildProgress.classList.add('hidden');
    return;
  }
  if (health.build && health.build.running) {
    showBuildProgress(health.build);
    setTimeout(pollHealth, 500);
    return;
  }
  els.progressBar.style.width = `${Math.round((health.progress || 0) * 100)}%`;
  els.warmupSteps.innerHTML = (health.steps || []).map(step => {
    const stepState = step.state || 'pending';
    const elapsed = step.elapsed_s?.toFixed?.(2) ?? step.elapsed_s ?? 0;
    return `<li class="step step-${escapeHtml(stepState)}"><span class="step-icon">${symbolFor(stepState)}</span><span class="step-name">${escapeHtml(step.name || '')}</span><span class="step-time">${escapeHtml(elapsed)}s</span></li>`;
  }).join('');
  if (health.error) {
    els.loadingError.textContent = health.error;
    els.loadingError.classList.remove('hidden');
  }
  if (health.state === 'READY' || health.state === 'DEGRADED') {
    els.loading.classList.add('hidden');
    els.buildScreen.classList.add('hidden');
    els.app.classList.remove('hidden');
    await refreshStatus();
    if (!window.__statusPollStarted) {
      window.__statusPollStarted = true;
      setInterval(refreshStatus, 60000);
    }
    return;
  }
  setTimeout(pollHealth, 250);
}

function showBuildProgress(build) {
  els.loading.classList.add('hidden');
  els.app.classList.add('hidden');
  els.buildScreen.classList.remove('hidden');
  els.buildIdle.classList.add('hidden');
  els.buildProgress.classList.remove('hidden');
  const stageLabels = {
    snapshotting: 'Snapshotting chat.db (safe, read-only copy)…',
    building: 'Embedding and indexing messages… this can take a while for large histories.',
    idle: 'Starting…',
  };
  els.buildStageText.textContent = stageLabels[build.stage] || build.stage;
  els.buildDetail.textContent = build.chunks_written ? `${build.chunks_written} chunks written so far · ${build.elapsed_s ?? 0}s elapsed` : `${build.elapsed_s ?? 0}s elapsed`;
}

async function startBuild() {
  els.buildError.classList.add('hidden');
  try {
    await api('/api/index/build', {method: 'POST'});
  } catch (err) {
    els.buildError.textContent = err.message;
    els.buildError.classList.remove('hidden');
    return;
  }
  els.buildIdle.classList.add('hidden');
  els.buildProgress.classList.remove('hidden');
  setTimeout(pollHealth, 250);
}

if (els.buildStart) els.buildStart.onclick = startBuild;

function symbolFor(state) {
  return state === 'done' ? '✓' : state === 'failed' ? '⚠' : state === 'running' ? '⟳' : '•';
}

// Owned by pollSyncProgress(): while a sync poll loop is live it owns the
// sync banner, and the 60s refreshStatus() tick must not stomp on it.
let syncInProgress = false;
// Incremented on every syncNow(); a poll loop whose token is stale exits so
// double-clicks / multiple entry points can never stack concurrent loops.
let syncPollToken = 0;
// Last /api/status payload, so other UI can reason about permissions.
let lastStatus = null;

async function refreshStatus() {
  const status = await api('/api/status');
  lastStatus = status;
  const stale = status.n_messages_since_index
    ? `<span class="status-sep">·</span><span class="status-stale">${status.n_messages_since_index.toLocaleString()} msgs since last index build</span>`
    : '';
  // The sync button is always present, not just when new messages are
  // detected: detection needs the live chat.db, and if that is unreadable
  // (or the user just wants to force one) there would otherwise be no way
  // to trigger a sync at all short of restarting the app.
  // Without Contacts access every sender renders as a raw phone number,
  // which reads like a bug rather than a missing permission.
  const contactsWarn = status.contacts_available === false
    ? `<span class="status-sep">·</span><button type="button" id="open-contacts" class="status-warn" title="Grant Contacts access to Seaglass, then restart it. Click to open the setting.">names unavailable</button>`
    : '';
  const syncBtn = `<button type="button" id="status-sync" class="status-sync" title="Re-index new messages now">`
    + `${iconSvg('i-refresh', 'icon icon-sm')}<span>Sync</span></button>`;
  els.statusBar.innerHTML = `<span class="status-dot${status.hydration_available ? '' : ' is-off'}"></span><span>Ready</span><span class="status-sep">·</span><span>${Number(status.n_chunks || 0).toLocaleString()} chunks</span><span class="status-sep">·</span><span>${Number(status.n_chats || 0).toLocaleString()} chats</span>${stale}${contactsWarn}<span class="status-spacer"></span>${syncBtn}`;
  const contactsButton = document.getElementById('open-contacts');
  if (contactsButton) contactsButton.onclick = grantContacts;
  const statusSync = document.getElementById('status-sync');
  if (statusSync) {
    statusSync.disabled = syncInProgress;
    statusSync.onclick = syncNow;
  }
  if (syncInProgress) return;  // the sync poll loop owns the banner right now
  if (!status.live_chat_readable) {
    // Silently reporting "up to date" when we cannot actually see the live
    // database is the one failure mode the user can't debug themselves.
    els.syncBanner.classList.remove('hidden');
    els.syncBanner.innerHTML = `<span class="banner-icon">${iconSvg('i-refresh')}</span>`
      + `<span class="banner-text"><strong>Can't check for new messages.</strong> `
      + `Seaglass needs Full Disk Access to read Messages. Enable Seaglass in the list, then restart it. `
      + `Search still works on what's already indexed.</span>`
      + `<span class="banner-actions">`
      + `<button id="open-fda" class="btn btn-sm btn-primary">Open Settings</button>`
      + `<button id="relaunch" class="btn btn-sm">Relaunch</button></span>`;
    const fdaButton = document.getElementById('open-fda');
    if (fdaButton) fdaButton.onclick = () => openSettings('full_disk_access');
    const relaunchButton = document.getElementById('relaunch');
    if (relaunchButton) relaunchButton.onclick = relaunchApp;
    return;
  }
  if (status.n_messages_since_index > 0) {
    els.syncBanner.classList.remove('hidden');
    const plural = status.n_messages_since_index === 1 ? '' : 's';
    els.syncBanner.innerHTML = `<span class="banner-icon">${iconSvg('i-refresh')}</span><span class="banner-text"><strong>${status.n_messages_since_index.toLocaleString()} new message${plural}</strong> since the index was last built.</span><span class="banner-actions"><button id="sync-now" class="btn btn-sm btn-primary">Sync now</button></span>`;
    document.getElementById('sync-now').onclick = syncNow;
  } else {
    els.syncBanner.classList.add('hidden');
  }
}

// Contacts is the one permission with no manual route: System Settings
// lists only apps that have already asked, with no "+" to add one. So the
// only way to grant it is to make the app ask, and the only way to make it
// ask is from here -- warmup runs before there is an event loop to show a
// prompt on.
async function grantContacts() {
  let result;
  try {
    result = await api('/api/system/request-contacts', {method: 'POST'});
  } catch (err) {
    syncBannerMessage(`Couldn't request Contacts access: ${err.message}`);
    return;
  }
  if (result.granted) {
    // The contact index and every cached chat title were built without
    // names, so they only pick up after a restart.
    syncBannerMessage('Contacts access granted — relaunching to load names…', {spin: true});
    await relaunchApp();
    return;
  }
  if (result.can_prompt === false && result.status !== 0) {
    // Already answered once; macOS never prompts again, so Settings is
    // now the only route.
    await openSettings('contacts');
  }
}

// A privacy grant only applies to a *newly launched* process, so the app
// has to restart before a freshly granted permission does anything. Doing
// that by hand right after granting it is a poor finish to the flow.
async function relaunchApp() {
  syncBannerMessage('Relaunching…', {spin: true});
  try {
    await api('/api/system/relaunch', {method: 'POST'});
  } catch (err) {
    // The server may drop the connection as it exits -- that means it
    // worked, so only a pre-flight failure is worth reporting.
    console.warn('relaunch request ended', err);
  }
}

// macOS can deep-link straight to a Privacy pane, which beats narrating a
// five-step path through System Settings. Best-effort: if it fails the
// banner text still says where to go.
async function openSettings(pane) {
  try {
    await api('/api/system/open-settings', {method: 'POST', body: JSON.stringify({pane})});
  } catch (err) {
    console.warn('could not open System Settings', err);
  }
}

function syncBannerMessage(text, {spin = false} = {}) {
  els.syncBanner.classList.remove('hidden');
  els.syncBanner.innerHTML = `<span class="banner-icon${spin ? ' spin' : ''}">${iconSvg('i-refresh')}</span><span class="banner-text">${escapeHtml(text)}</span>`;
}

async function syncNow() {
  if (syncInProgress) return;  // guard against double-clicks stacking poll loops
  syncInProgress = true;
  const token = ++syncPollToken;
  const button = document.getElementById('sync-now');
  if (button) { button.disabled = true; button.textContent = 'Syncing…'; }
  const statusButton = document.getElementById('status-sync');
  if (statusButton) statusButton.disabled = true;
  syncBannerMessage('Syncing…', {spin: true});
  try {
    await api('/api/index/build', {method: 'POST'});
  } catch (err) {
    if (err.status !== 409) {
      // Hard failure to even start: release ownership and surface it, then
      // let the next refreshStatus() tick restore a clickable banner.
      syncInProgress = false;
      syncBannerMessage(`Sync failed: ${err.message}`);
      return;
    }
    // 409: a build/sync was already in progress -- fine, just watch it.
  }
  pollSyncProgress(token);
}

// `failures` counts consecutive poll errors; transient network blips during a
// long build shouldn't tear down the whole progress UI.
const SYNC_POLL_MAX_FAILURES = 3;

async function pollSyncProgress(token = syncPollToken, failures = 0) {
  if (token !== syncPollToken) return;  // superseded by a newer sync run
  let build;
  try {
    build = await api('/api/index/build');
  } catch (err) {
    if (token !== syncPollToken) return;
    if (failures + 1 < SYNC_POLL_MAX_FAILURES) {
      setTimeout(() => pollSyncProgress(token, failures + 1), 750 * (failures + 1));
      return;
    }
    syncInProgress = false;
    syncBannerMessage(`Lost contact while syncing: ${err.message}`);
    return;
  }
  if (token !== syncPollToken) return;
  if (build.running) {
    const elapsed = build.elapsed_s ?? 0;
    const detail = build.chunks_written
      ? `${build.chunks_written.toLocaleString()} chunks written so far · ${elapsed}s elapsed`
      : `${elapsed}s elapsed`;
    syncBannerMessage(`Syncing… ${detail}`, {spin: true});
    setTimeout(() => pollSyncProgress(token, 0), 750);
    return;
  }
  if (build.stage === 'failed') {
    syncInProgress = false;
    syncBannerMessage(`Sync failed: ${build.error || 'unknown error'}`);
    return;
  }
  // Not running and not failed: either the run finished ('done'), or the
  // background thread hasn't flipped running=true yet (we polled in the gap
  // between POST returning and the worker starting -- in which case `stage`
  // is still 'idle'). Grace-poll a few times before declaring completion.
  if (build.stage === 'idle' && failures < 4) {
    setTimeout(() => pollSyncProgress(token, failures + 1), 750);
    return;
  }
  // Done: release banner ownership and let refreshStatus() render the truth
  // (hidden banner when nothing is stale, a fresh "Sync now" button if not).
  syncInProgress = false;
  try {
    await refreshStatus();
  } catch (err) {
    syncBannerMessage(`Sync finished, but status refresh failed: ${err.message}`);
  }
}

function buildFilters() {
  const groupValue = document.querySelector('input[name="group"]:checked').value;
  return {
    people_handles: state.people.flatMap(person => person.handles),
    is_group: groupValue === '' ? null : groupValue === 'true',
    chat_ids: state.selectedChat ? [state.selectedChat.chat_id] : null,
    date_from: els.dateFrom.value ? new Date(`${els.dateFrom.value}T00:00:00`).getTime() / 1000 : null,
    date_to: els.dateTo.value ? new Date(`${els.dateTo.value}T23:59:59`).getTime() / 1000 : null,
    // No has_media here: a two-state toggle reads as "only with media" vs
    // "only without", which is misleading. Assist can still infer the
    // backend filter from a natural-language query like "photos from Ana".
  };
}

function hasActiveFilters() {
  const filters = buildFilters();
  return Boolean(
    filters.people_handles.length
    || filters.is_group !== null
    || filters.chat_ids
    || filters.date_from
    || filters.date_to
  );
}

async function search() {
  // An empty query with *no* filters has nothing to rank by, so the
  // backend returns an arbitrary slice of the corpus -- confusing. Empty
  // query *with* filters is legitimate though (browse a person/date
  // range), so only the completely-empty case is short-circuited.
  if (!els.query.value.trim() && !hasActiveFilters()) {
    state.results = [];
    els.resultsMeta.innerHTML = '';
    renderIdleState();
    return;
  }
  state.requestId = crypto.randomUUID();
  state.assistToken = null;
  els.assistBanner.classList.add('hidden');
  setBusy(true);
  let payload;
  try {
    payload = await api('/api/search', {
      method: 'POST',
      body: JSON.stringify({ query: els.query.value, filters: buildFilters(), options: {}, assist: state.assist, request_id: state.requestId })
    });
  } finally {
    setBusy(false);
  }
  if (payload.request_id !== state.requestId) return;
  // Remember exactly what produced this page so "load more" pages the same
  // query, not whatever the inputs happen to say by the time it is clicked.
  state.pageQuery = els.query.value;
  state.pageFilters = buildFilters();
  renderResults(payload);
  if (payload.assist_token) {
    state.assistToken = payload.assist_token;
    pollAssist(payload.assist_token);
  }
}

// Paging appends rather than replaces, so results accumulate into one
// scrollable list. The backend caches the ranking per query, so each page
// costs a regroup rather than another embed + cross-encoder pass.
async function loadMore() {
  if (state.loadingMore || !state.hasMore) return;
  state.loadingMore = true;
  const button = els.results.querySelector('.load-more');
  if (button) {
    button.disabled = true;
    button.querySelector('span').textContent = 'Loading…';
  }
  const requestId = state.requestId;
  const offset = state.nextOffset;
  try {
    const payload = await api('/api/search', {
      method: 'POST',
      body: JSON.stringify({
        query: state.pageQuery,
        filters: state.pageFilters,
        options: { offset },
        assist: 'off',
        request_id: requestId,
      })
    });
    // A new search started while this was in flight -- its results own the
    // list now, so discard this page instead of appending it to them.
    if (state.requestId !== requestId) return;
    appendResults(payload);
  } catch (err) {
    if (button) {
      button.disabled = false;
      button.querySelector('span').textContent = 'Could not load more — retry';
    }
    throw err;
  } finally {
    state.loadingMore = false;
  }
}

function setBusy(busy) {
  els.searchButton.classList.toggle('is-busy', busy);
  els.searchButton.disabled = busy;
  els.results.classList.toggle('is-busy', busy);
}

function iconSvg(name, className = 'icon') {
  return `<svg class="${className}" aria-hidden="true"><use href="#${name}"></use></svg>`;
}

// The results pane is the largest surface in the window, so it must never
// sit empty -- an unexplained void reads as a broken app rather than as
// "nothing searched yet".
function renderIdleState() {
  state.results = [];
  state.hasMore = false;
  els.resultsMeta.innerHTML = '';
  els.results.innerHTML = `<div class="empty-state"><div class="empty-icon">${iconSvg('i-search')}</div><div class="empty-title">Search your messages</div><p class="empty-hint">Type what you're looking for — a topic, a phrase, a plan someone mentioned — or narrow things down with the filters on the left.</p></div>`;
}

// Counts describe what is on screen, so they grow as pages are appended.
function updateResultsMeta(payload) {
  const term = payload.effective_filters?.semantic || state.pageQuery || els.query.value;
  const sessions = state.results.length;
  const total = payload.total_sessions || sessions;
  const msgs = state.shownResults || 0;
  els.resultsMeta.innerHTML = [
    term ? `<span class="meta-query">${escapeHtml(term)}</span>` : '',
    `<span class="meta-pill">${sessions}${total > sessions ? ` of ${total}` : ''} session${sessions === 1 ? '' : 's'}</span>`,
    `<span class="meta-pill">${msgs} msg${msgs === 1 ? '' : 's'}</span>`,
    `<span class="meta-pill">${payload.elapsed_s}s</span>`,
  ].filter(Boolean).join('');
}

function renderResults(payload) {
  state.results = payload.sessions || [];
  state.shownResults = payload.n_results || 0;
  state.lastPayload = payload;
  updateResultsMeta(payload);
  if (!payload.sessions?.length) {
    els.results.innerHTML = `<div class="empty-state"><div class="empty-icon">${iconSvg('i-search')}</div><div class="empty-title">No matching messages</div><p class="empty-hint">Nothing matched this search and the filters you have applied. Try loosening a filter, widening the date range, or rephrasing the query.</p></div>`;
    return;
  }
  els.results.innerHTML = payload.sessions.map((session, index) => renderSession(session, index)).join('');
  wireResults(payload);
}

// Second and later pages are appended so the list grows instead of jumping.
function appendResults(payload) {
  const start = state.results.length;
  const sessions = payload.sessions || [];
  // Defensive: the ranking is cached per query so pages cannot normally
  // overlap, but a sync between pages invalidates that cache -- never
  // render the same session twice.
  const seen = new Set(state.results.map(s => `${s.chat_id}|${s.day}`));
  const fresh = sessions.filter(s => !seen.has(`${s.chat_id}|${s.day}`));
  state.results = state.results.concat(fresh);
  state.shownResults = (state.shownResults || 0)
    + fresh.reduce((n, s) => n + (s.messages || []).length, 0);

  const existing = els.results.querySelector('.load-more-row');
  if (existing) existing.remove();
  els.results.insertAdjacentHTML(
    'beforeend',
    fresh.map((session, i) => renderSession(session, start + i)).join('')
  );
  updateResultsMeta(payload);
  wireResults(payload);
}

function wireResults(payload) {
  state.hasMore = Boolean(payload.has_more);
  state.nextOffset = payload.next_offset ?? 0;
  renderLoadMore(payload);
  document.querySelectorAll('[data-chat-id]').forEach(link => {
    link.removeEventListener('click', openConversation);
    link.addEventListener('click', openConversation);
  });
  applyHitClamping();
}

function renderLoadMore(payload) {
  const existing = els.results.querySelector('.load-more-row');
  if (existing) existing.remove();
  if (!state.hasMore) return;
  const shown = state.results.length;
  const total = payload.total_sessions || 0;
  const row = document.createElement('div');
  row.className = 'load-more-row';
  row.innerHTML = `<button type="button" class="load-more">`
    + `${iconSvg('i-chevron-down', 'icon icon-sm')}<span>Load more</span></button>`
    + (total ? `<div class="load-more-count">${shown} of ${total} conversations</div>` : '');
  row.querySelector('.load-more').addEventListener('click', () => { loadMore(); });
  els.results.appendChild(row);
}

const CLAMP_SLACK_PX = 24;
const CLAMP_LEAD_CONTEXT = 3;  // messages of run-up kept above the anchor

// Which message in a hit chunk is the one the user was actually looking
// for. A chunk spans a whole conversation window, so the literal match can
// sit anywhere in it -- including past the clamp, which made a card look
// like it had matched on nothing (searching "how was golf" showed four
// messages whose only tie to the query was the word "was").
//
// Marks are the query terms the server highlighted, so the most-marked
// message is the closest thing to a verbatim match. Ties go to the latest,
// matching the ranking rule that recent verbatim matches dominate.
function bestMatchIndex(msgs) {
  let anchor = -1;
  let best = 0;
  msgs.forEach((msg, i) => {
    const marks = msg.querySelectorAll('mark').length;
    if (marks > 0 && marks >= best) {
      best = marks;
      anchor = i;
    }
  });
  return anchor;
}
  // don't clamp for a sliver -- the toggle would cost more room than it saves

// A hit inside a long group thread can render a card tall enough to push
// every other result off screen. Clamp those, but only after measuring:
// cards that fit get no "Show more" button at all.
//
// Safe to call repeatedly (it re-runs on resize, since a card that fits at
// one window width can overflow at another): any existing toggle is
// discarded and rebuilt, and a card the user explicitly expanded stays
// expanded across re-measurement.
function applyHitClamping() {
  document.querySelectorAll('.card-hits').forEach(hits => {
    // The toggle is overlaid on the fade inside a positioned wrapper so it
    // reads as belonging to the message list, not to the section below it.
    let wrap = hits.parentElement;
    if (!wrap || !wrap.classList.contains('card-hits-wrap')) {
      wrap = document.createElement('div');
      wrap.className = 'card-hits-wrap';
      hits.insertAdjacentElement('beforebegin', wrap);
      wrap.appendChild(hits);
    }
    const existing = wrap.querySelector(':scope > .card-expand');
    if (existing) existing.remove();

    const msgs = Array.from(hits.querySelectorAll(':scope > .msg'));
    msgs.forEach(msg => msg.classList.remove('msg-out'));

    // Measure against the clamped height, then decide the final state.
    hits.classList.add('is-clamped');
    if (hits.scrollHeight <= hits.clientHeight + CLAMP_SLACK_PX) {
      hits.classList.remove('is-clamped');
      delete hits.dataset.expanded;
      return;
    }

    // Clamping cuts from the bottom, so put the best match near the top of
    // what survives by dropping the messages that precede its run-up.
    const anchor = bestMatchIndex(msgs);
    const start = anchor < 0 ? 0 : Math.max(0, anchor - CLAMP_LEAD_CONTEXT);
    const applyWindow = () => {
      const clamped = hits.classList.contains('is-clamped');
      msgs.forEach((msg, i) => msg.classList.toggle('msg-out', clamped && i < start));
    };

    if (hits.dataset.expanded === 'true') hits.classList.remove('is-clamped');
    applyWindow();

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'card-expand';
    // Name what is actually hidden -- a bare "Show more" next to the
    // "Surrounding context" row is ambiguous about which it expands.
    const total = msgs.length;
    const paint = () => {
      const clamped = hits.classList.contains('is-clamped');
      const label = clamped
        ? (total ? `Show all ${total} messages` : 'Show more')
        : 'Show less';
      button.innerHTML = `${iconSvg('i-chevron-down', 'icon icon-sm')} <span>${label}</span>`;
      button.setAttribute('aria-expanded', String(!clamped));
    };
    paint();
    button.addEventListener('click', () => {
      const clamped = hits.classList.toggle('is-clamped');
      hits.dataset.expanded = String(!clamped);
      applyWindow();
      paint();
      // Re-collapsing from far down a long card would otherwise leave the
      // viewport somewhere below the card entirely.
      if (clamped) {
        const card = hits.closest('.result-card');
        if (card && card.getBoundingClientRect().top < 0) card.scrollIntoView({block: 'start'});
      }
    });
    wrap.appendChild(button);
  });
}

function renderSession(session, index) {
  // Sort for display rather than trusting the payload's order: a recency
  // result is ordered newest-first (so a caller reading the top gets the
  // latest message), but a conversation only reads correctly oldest-first.
  const byTime = (a, b) => (a.ts || 0) - (b.ts || 0);
  const hits = (session.messages || []).slice().sort(byTime).map(renderMessage).join('');
  const contextMessages = (session.context_messages || []).slice().sort(byTime);
  const context = contextMessages.map(renderMessage).join('');
  const score = Number(session.score || 0);
  const scoreWidth = Math.max(6, Math.min(100, Math.round(score * 100)));
  const title = escapeHtml(session.title || `Chat ${session.chat_id}`);
  const groupBadge = session.is_group ? `<span class="meta-pill">${iconSvg('i-people', 'icon icon-sm')} ${session.participant_count || ''}</span>` : '';
  const contextBlock = context
    ? `<div class="context"><details><summary>Surrounding context <span class="meta-pill">${contextMessages.length}</span></summary><div class="context-body">${context}</div></details></div>`
    : '';
  return `<article class="result-card" data-index="${index}">`
    + `<header class="card-head"><span class="card-title">${title}</span>${groupBadge}<span class="card-day">${escapeHtml(session.day || '')}</span>`
    + `<span class="score-pill" title="relevance score"><span class="score-bar"><i style="width:${scoreWidth}%"></i></span>${score.toFixed(2)}</span></header>`
    + `<div class="card-hits">${hits}</div>`
    + contextBlock
    + `<div class="card-foot-row"><button class="btn btn-sm" data-chat-id="${session.chat_id}" data-around-ts="${session.messages?.[0]?.ts || ''}">${iconSvg('i-arrow-out', 'icon icon-sm')} Open full conversation</button></div>`
    + `</article>`;
}

function renderMessage(message) {
  const isMe = message.is_from_me || !message.sender;
  const sender = message.sender || 'Me';
  const clip = message.has_attachment ? `<span class="msg-clip" title="has attachment">${iconSvg('i-clip', 'icon icon-sm')}</span>` : '';
  // Attachment-only messages (photos, stickers, audio notes) carry no text
  // at all, so rendering `text` alone produced a blank bubble.
  const hasText = Boolean((message.text || '').trim());
  const body = hasText
    ? highlight(message.text)
    : `<span class="msg-attachment">${iconSvg('i-clip', 'icon icon-sm')} ${escapeHtml(message.attachment_kind || (message.has_attachment ? 'Attachment' : 'No text'))}</span>`;
  return `<div class="msg${isMe ? ' msg-me' : ''}">`
    + `<span class="msg-avatar" style="--avatar-h:${hueFor(sender)}" aria-hidden="true">${escapeHtml(initialsFor(sender))}</span>`
    + `<div class="msg-body"><div class="msg-head"><span class="msg-sender">${escapeHtml(sender)}</span><span class="msg-time">${formatTime(message.ts)}</span>${clip}</div>`
    + `<div class="msg-text">${body}</div></div></div>`;
}

function initialsFor(name) {
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function hueFor(name) {
  let hash = 0;
  for (const char of String(name)) hash = (hash * 31 + char.charCodeAt(0)) % 360;
  return hash;
}

function formatTime(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '';
}

function highlight(text) {
  let output = escapeHtml(text);
  for (const term of (els.query.value.match(/[A-Za-z0-9]{3,}/g) || [])) {
    const re = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'ig');
    output = output.replace(re, '<mark>$1</mark>');
  }
  return output;
}

// The assisted parse is *used*, not offered. Asking the user to press
// "Apply" made Copilot's work optional and, worse, the un-assisted results
// stayed on screen underneath a banner describing filters that had never
// run. Force always applies; auto applies when the backend decided the
// query needed the help. The banner just reports what was applied.
async function pollAssist(tokenValue) {
  const result = await api(`/api/assist/${tokenValue}`);
  if (state.assistToken !== tokenValue) return;  // a newer search owns the page
  if (result === null) {
    setTimeout(() => pollAssist(tokenValue), 1000);
    return;
  }
  if (result.status !== 'ready') return;
  await applyAssist(tokenValue, result);
}

function showAssistBanner(description, onUndo) {
  els.assistBanner.classList.remove('hidden');
  els.assistBanner.innerHTML = `<span class="banner-icon">${iconSvg('i-sparkle')}</span>`
    + `<span class="banner-text">Copilot read this as <span class="assist-parse">${escapeHtml(description)}</span></span>`
    + `<span class="banner-actions"><button id="undo-assist" class="btn btn-sm btn-ghost">Undo</button></span>`;
  document.getElementById('undo-assist').onclick = onUndo;
}

async function applyAssist(tokenValue, result) {
  const baseline = state.lastPayload;
  let payload;
  try {
    payload = await api('/api/search/apply-assist', {
      method: 'POST',
      body: JSON.stringify({
        assist_token: tokenValue,
        query: els.query.value,
        filters: buildFilters(),
        options: {},
        request_id: state.requestId,
      })
    });
  } catch (err) {
    return;  // the deterministic results are already on screen
  }
  // A newer search started while Copilot was thinking; its results own the
  // page now.
  if (state.assistToken !== tokenValue || payload.request_id !== state.requestId) return;
  const baselineQuery = state.pageQuery;
  const baselineFilters = state.pageFilters;
  if (payload.applied_query !== undefined) {
    state.pageQuery = payload.applied_query;
    state.pageFilters = payload.applied_filters;
  }
  renderResults(payload);
  showAssistBanner(payload.assist_description || result.description || '', () => {
    els.assistBanner.classList.add('hidden');
    state.pageQuery = baselineQuery;
    state.pageFilters = baselineFilters;
    if (baseline) renderResults(baseline);
  });
}

async function openConversation(event) {
  const button = event.currentTarget;
  const params = new URLSearchParams({ chat_id: button.dataset.chatId, limit: '50' });
  if (button.dataset.aroundTs) params.set('around_ts', button.dataset.aroundTs);
  const payload = await api(`/api/conversation?${params.toString()}`);
  els.drawer.classList.remove('hidden');
  const count = payload.messages.length;
  const subtitle = payload.is_group
    ? `Group · ${(payload.participants || []).length || ''} people · ${count} messages`
    : `${count} messages`;
  els.drawerContent.innerHTML = `<header class="drawer-head"><div class="drawer-titles"><div class="drawer-title">${escapeHtml(payload.title || `Chat ${payload.chat_id}`)}</div><div class="drawer-sub">${escapeHtml(subtitle)}</div></div><button id="close-drawer" class="icon-btn" aria-label="Close conversation" title="Close (Esc)">${iconSvg('i-close', 'icon')}</button></header><div class="drawer-messages">${payload.messages.map(renderMessage).join('')}</div>`;
  document.getElementById('close-drawer').onclick = () => els.drawer.classList.add('hidden');
  // The messages container is recreated per open, so it starts at scrollTop 0,
  // but be explicit: WebKit can restore scroll offsets on replaced subtrees.
  const messages = els.drawerContent.querySelector('.drawer-messages');
  if (messages) messages.scrollTop = 0;
}

function renderPeopleChips() {
  els.peopleChips.innerHTML = state.people.map((person, index) => `<button class="chip" data-index="${index}" title="Remove ${escapeHtml(person.display_name)}">${escapeHtml(person.display_name)}<span class="chip-x" aria-hidden="true">×</span></button>`).join('');
  els.peopleChips.querySelectorAll('[data-index]').forEach(button => button.addEventListener('click', () => {
    state.people.splice(Number(button.dataset.index), 1);
    renderPeopleChips();
  }));
}

async function suggestPeople() {
  const query = els.peopleInput.value.trim();
  if (!query) { els.peopleSuggestions.innerHTML = ''; return; }
  const suggestions = await api(`/api/contacts/suggest?q=${encodeURIComponent(query)}&limit=10`);
  els.peopleSuggestions.innerHTML = suggestions.map((item, index) => `<button class="suggestion" data-index="${index}"><span class="s-title">${escapeHtml(item.display_name)}</span><span class="s-meta">${item.n_handles} handle${item.n_handles === 1 ? '' : 's'}</span></button>`).join('');
  els.peopleSuggestions.querySelectorAll('[data-index]').forEach(button => button.onclick = () => {
    const item = suggestions[Number(button.dataset.index)];
    state.people.push(item);
    renderPeopleChips();
    els.peopleInput.value = '';
    els.peopleSuggestions.innerHTML = '';
  });
}

async function suggestChats() {
  const query = els.chatInput.value.trim();
  const suggestions = await api(`/api/chats/suggest?q=${encodeURIComponent(query)}&limit=20`);
  els.chatSuggestions.innerHTML = suggestions.map((item, index) => `<button class="suggestion" data-index="${index}"><span class="s-title">${escapeHtml(item.title)}</span><span class="s-meta">${item.is_group ? 'group' : '1:1'}</span></button>`).join('');
  els.chatSuggestions.querySelectorAll('[data-index]').forEach(button => button.onclick = () => {
    state.selectedChat = suggestions[Number(button.dataset.index)];
    els.chatInput.value = state.selectedChat.title;
    els.chatSuggestions.innerHTML = '';
  });
}

function clearFilters() {
  state.people = [];
  state.selectedChat = null;
  renderPeopleChips();
  els.peopleInput.value = '';
  els.chatInput.value = '';
  applyPreset('');
  document.querySelector('input[name="group"][value=""]').checked = true;
}

// `<input type="date">` values are bare calendar dates with no timezone,
// and buildFilters() interprets them as LOCAL midnight/end-of-day. So the
// presets must emit local calendar dates too -- toISOString() emits UTC,
// which west of UTC shifts both ends of the range forward a day (at 7pm
// PDT, "last 7 days" produced a range ending *tomorrow*).
function toLocalDateInputValue(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

const PRESET_LABELS = {'7': 'Last 7 days', '30': 'Last 30 days', '90': 'Last 90 days', '365': 'Last year'};

function formatDateInputValue(value) {
  // Parse as local (see toLocalDateInputValue) so the label can't drift a
  // day from what the input itself displays.
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'});
}

function renderDateSummary() {
  if (state.datePreset === 'custom') {
    const from = els.dateFrom.value ? formatDateInputValue(els.dateFrom.value) : 'Anything';
    const to = els.dateTo.value ? formatDateInputValue(els.dateTo.value) : 'now';
    els.dateSummary.textContent = !els.dateFrom.value && !els.dateTo.value
      ? 'All time'
      : `${from} → ${to}`;
    return;
  }
  els.dateSummary.textContent = PRESET_LABELS[state.datePreset] || 'All time';
}

// `days` is '' (All), 'custom', or a day count. The raw date inputs stay
// hidden unless the range is custom: showing them for "All" meant the
// control displayed a concrete day (today) even though no date filter was
// actually applied, which read as an active filter when it wasn't.
function applyPreset(days) {
  state.datePreset = days;
  document.querySelectorAll('.presets button').forEach(
    button => button.classList.toggle('active', button.dataset.days === days)
  );
  els.dates.classList.toggle('hidden', days !== 'custom');

  if (days === 'custom') {
    renderDateSummary();
    return;
  }
  if (!days) {
    els.dateFrom.value = '';
    els.dateTo.value = '';
  } else {
    els.dateFrom.value = toLocalDateInputValue(new Date(Date.now() - Number(days) * 86400 * 1000));
    els.dateTo.value = toLocalDateInputValue(new Date());
  }
  renderDateSummary();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

document.querySelectorAll('.assist').forEach(button => button.onclick = () => {
  state.assist = button.dataset.assist;
  document.querySelectorAll('.assist').forEach(el => el.classList.toggle('active', el === button));
});
els.searchButton.onclick = search;
els.query.addEventListener('keydown', event => {
  if (event.key === 'Enter') search();
  if (event.key === 'Escape') els.drawer.classList.add('hidden');
});
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    els.query.focus();
  }
  if (event.key === 'Escape') els.drawer.classList.add('hidden');
});
els.peopleInput.addEventListener('input', debounce(suggestPeople, 150));
els.chatInput.addEventListener('input', debounce(suggestChats, 150));
document.getElementById('clear-filters').onclick = clearFilters;
document.querySelectorAll('.presets button').forEach(button => button.onclick = () => applyPreset(button.dataset.days));
[els.dateFrom, els.dateTo].forEach(input => input.addEventListener('change', renderDateSummary));
// A card that fits at one window width can overflow at another.
window.addEventListener('resize', debounce(applyHitClamping, 150));
document.querySelectorAll('[data-drawer-close]').forEach(el => el.addEventListener('click', () => els.drawer.classList.add('hidden')));
document.addEventListener('click', event => {
  const target = event.target;
  if (!(target instanceof Element) || !target.closest('.combo')) {
    els.peopleSuggestions.innerHTML = '';
    els.chatSuggestions.innerHTML = '';
  }
});

function debounce(fn, delay) {
  let handle;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), delay);
  };
}

// Start on "All" (no implicit range filter): this also clears any value
// WebKit's form-state restoration may have applied on load, marks the All
// preset as selected, hides the raw date inputs and renders the "All time"
// summary -- so a fresh search never silently inherits a previous
// session's date range or *looks* like it has one.
applyPreset('');
renderIdleState();

pollHealth();
